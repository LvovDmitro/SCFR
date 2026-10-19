"""
MAIRA-2: Microsoft multimodal model for radiology report generation.
https://huggingface.co/microsoft/maira-2

Uses transformers AutoModelForCausalLM and AutoProcessor with trust_remote_code.
Note: MAIRA-2's model card requires transformers>=4.48.0,<4.52 and says
4.51.3 was last tested. Transformers 5.x changes Llava internals and can
silently collapse evaluation into empty predictions, so this adapter uses the
last-tested version exactly.

Requires: transformers, pillow, protobuf, sentencepiece, torch.
"""
import hashlib
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from PIL import Image
from .base import BaseGenerator

MODEL_ID = "microsoft/maira-2"
MAIRA2_TRANSFORMERS_MIN = (4, 48, 0)
MAIRA2_TRANSFORMERS_MAX_EXCLUSIVE = (4, 52, 0)
MAIRA2_TRANSFORMERS_RECOMMENDED = "4.51.3"
MAIRA2_TRANSFORMERS_RECOMMENDED_TUPLE = (4, 51, 3)
MAIRA2_DEFAULT_PROMPT_STYLE = "default"
MAIRA2_SCFR_SAFETY_PROMPT_STYLE = "scfr_safety_v1"
MAIRA2_SCFR_SAFETY_PROMPT_VERSION = "scfr-maira2-safety-v1"
MAIRA2_SCFR_SAFETY_PROMPT_V1 = (
    "Report the current study using the current images as the primary source "
    "of evidence. Use the prior image and prior report only for comparison. "
    "Do not copy or carry forward a prior finding unless it is supported by "
    "the current images. Do not suppress a current-image finding because it "
    "was absent from the prior. State interval change only when supported by "
    "the available studies. When current-image evidence is equivocal, use "
    "appropriately uncertain language rather than asserting a finding."
)
MAIRA2_PROMPT_STYLES = {
    MAIRA2_DEFAULT_PROMPT_STYLE: None,
    MAIRA2_SCFR_SAFETY_PROMPT_STYLE: MAIRA2_SCFR_SAFETY_PROMPT_V1,
}
MAIRA2_MAX_NEW_TOKENS = 300
MAIRA2_USE_CACHE = True
MAIRA2_DO_SAMPLE = False
MAIRA2_NUM_BEAMS = 1
MAIRA2_TEMPERATURE = 0.0
MAIRA2_SEED_POLICY = "not_applicable_deterministic_greedy"


class MAIRA2Generator(BaseGenerator):
    name = "maira2"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        revision: Optional[str] = None,
        prompt_style: str = MAIRA2_DEFAULT_PROMPT_STYLE,
        local_files_only: bool = False,
        **kwargs,
    ):
        from transformers import AutoModelForCausalLM, AutoProcessor
        import transformers

        self.model_id = model_id
        self.requested_revision = revision
        self.prompt_style = self._normalize_prompt_style(prompt_style)
        self._last_generation_metadata: Dict[str, Any] = {}
        self._last_raw_decoded_text = ""

        print(f"[MAIRA2Generator] Loading {model_id} (bfloat16, no quantization)...")
        print(f"[MAIRA2Generator] transformers={transformers.__version__}, "
              f"torch={torch.__version__}")
        self._require_supported_transformers(transformers.__version__)

        # MAIRA-2 custom model lacks lm_head attr needed by BnB quantizer.
        # Load in bf16 instead — fits on A100 80GB (~16GB VRAM).
        load_kwargs = self._pretrained_load_kwargs(
            revision=revision,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **load_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            **load_kwargs,
        )

        # --- Comprehensive diagnostics for debugging .model issue ---
        model_cls = type(self.model).__name__
        mro = [c.__name__ for c in type(self.model).__mro__[:8]]
        print(f"[MAIRA2Generator] Model class: {model_cls}")
        print(f"[MAIRA2Generator] MRO: {mro}")

        # Log all nn.Module children
        children = {name: type(child).__name__ for name, child in self.model.named_children()}
        print(f"[MAIRA2Generator] nn.Module children: {children}")

        # Log generate/forward methods and their source
        import inspect
        for method_name in ['generate', 'forward', '_generate', 'prepare_inputs_for_generation']:
            method = getattr(self.model, method_name, None)
            if method is not None:
                try:
                    src_file = inspect.getfile(method)
                    # Show which class defines this method
                    for cls in type(self.model).__mro__:
                        if method_name in cls.__dict__:
                            print(f"[MAIRA2Generator] {method_name}() defined in {cls.__name__} "
                                  f"({src_file})")
                            break
                    else:
                        print(f"[MAIRA2Generator] {method_name}() found, source: {src_file}")
                except (TypeError, OSError):
                    print(f"[MAIRA2Generator] {method_name}() exists (built-in/no source)")

        print("[MAIRA2Generator] Using supported MAIRA-2 transformers range; "
              "no 5.x model-internal monkey patch applied.")

        self.device = next(self.model.parameters()).device
        self.model.eval()
        print(f"[MAIRA2Generator] Model loaded on device={self.device}, "
              f"dtype={next(self.model.parameters()).dtype}")

    @staticmethod
    def _pretrained_load_kwargs(
        *,
        revision: Optional[str],
        local_files_only: bool,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if revision:
            kwargs["revision"] = revision
        if local_files_only:
            kwargs["local_files_only"] = True
        return kwargs

    @staticmethod
    def _normalize_prompt_style(prompt_style: Optional[str]) -> str:
        style = str(prompt_style or MAIRA2_DEFAULT_PROMPT_STYLE).strip().lower()
        if style not in MAIRA2_PROMPT_STYLES:
            choices = ", ".join(sorted(MAIRA2_PROMPT_STYLES))
            raise ValueError(
                f"Unknown MAIRA-2 prompt style {prompt_style!r}. "
                f"Available styles: {choices}."
            )
        return style

    def _resolve_prompt_style(self, prompt_style: Optional[str]) -> str:
        if prompt_style is not None:
            return self._normalize_prompt_style(prompt_style)
        return self._normalize_prompt_style(
            getattr(self, "prompt_style", MAIRA2_DEFAULT_PROMPT_STYLE)
        )

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def _compose_reporting_indication(
        cls,
        indication_text: str,
        prompt_style: str,
    ) -> str:
        """Compose the formatter's indication field without changing context."""
        base_indication = indication_text or "No indication provided."
        instruction = MAIRA2_PROMPT_STYLES[prompt_style]
        if instruction is None:
            return base_indication
        return (
            f"{base_indication}\n\n"
            f"Reporting instruction ({MAIRA2_SCFR_SAFETY_PROMPT_VERSION}): "
            f"{instruction}"
        )

    @classmethod
    def prompt_metadata(
        cls,
        prompt_style: str = MAIRA2_DEFAULT_PROMPT_STYLE,
    ) -> Dict[str, Any]:
        """Return stable provenance for a configured reporting instruction."""
        style = cls._normalize_prompt_style(prompt_style)
        instruction = MAIRA2_PROMPT_STYLES[style]
        return {
            "prompt_style": style,
            "prompt_version": (
                MAIRA2_SCFR_SAFETY_PROMPT_VERSION if instruction else None
            ),
            "instruction_sha256": (
                cls._sha256(instruction) if instruction else None
            ),
            "instruction_injection_field": (
                "indication" if instruction else None
            ),
            "decoding": {
                "strategy": "deterministic_greedy",
                "max_new_tokens": MAIRA2_MAX_NEW_TOKENS,
                "use_cache": MAIRA2_USE_CACHE,
                "do_sample": MAIRA2_DO_SAMPLE,
                "num_beams": MAIRA2_NUM_BEAMS,
                # Recorded for a complete human-readable contract but
                # intentionally not passed to transformers when sampling is
                # disabled, avoiding version-dependent warnings/handling.
                "temperature": MAIRA2_TEMPERATURE,
                "temperature_passed_to_generate": False,
                "seed_policy": MAIRA2_SEED_POLICY,
            },
        }

    def get_last_generation_metadata(self) -> Dict[str, Any]:
        """Return per-call prompt provenance without exposing clinical text."""
        if getattr(self, "_last_generation_metadata", None):
            metadata = dict(self._last_generation_metadata)
        else:
            metadata = self.prompt_metadata(
                getattr(self, "prompt_style", MAIRA2_DEFAULT_PROMPT_STYLE)
            )
        config = getattr(getattr(self, "model", None), "config", None)
        resolved_revision = (
            getattr(config, "_commit_hash", None)
            or getattr(getattr(self, "model", None), "_commit_hash", None)
            or getattr(getattr(self, "processor", None), "_commit_hash", None)
        )
        metadata.update(
            {
                "model_id": getattr(self, "model_id", MODEL_ID),
                "requested_model_revision": getattr(
                    self,
                    "requested_revision",
                    None,
                ),
                "resolved_model_revision": resolved_revision,
            }
        )
        return metadata

    def get_last_raw_prediction(self) -> str:
        """Return the decoded model continuation before MAIRA-2 conversion."""
        return str(getattr(self, "_last_raw_decoded_text", "") or "")

    @staticmethod
    def _version_tuple(version: str):
        match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", version)
        if not match:
            return None
        return tuple(int(part or 0) for part in match.groups())

    @staticmethod
    def _is_supported_transformers_version(version: str) -> bool:
        parsed = MAIRA2Generator._version_tuple(version)
        if parsed is None:
            return False
        return (
            MAIRA2_TRANSFORMERS_MIN
            <= parsed
            < MAIRA2_TRANSFORMERS_MAX_EXCLUSIVE
        )

    @staticmethod
    def _require_supported_transformers(version: str):
        parsed = MAIRA2Generator._version_tuple(version)
        if parsed == MAIRA2_TRANSFORMERS_RECOMMENDED_TUPLE:
            return
        if MAIRA2Generator._is_supported_transformers_version(version):
            raise RuntimeError(
                "MAIRA-2 official supported range is transformers>=4.48.0,<4.52, "
                f"but this adapter is pinned to the last-tested "
                f"version {MAIRA2_TRANSFORMERS_RECOMMENDED}; found "
                f"transformers=={version}."
            )
        raise RuntimeError(
            "MAIRA-2 requires transformers>=4.48.0,<4.52 "
            f"(recommended {MAIRA2_TRANSFORMERS_RECOMMENDED}); "
            f"found transformers=={version}. Install "
            f"transformers=={MAIRA2_TRANSFORMERS_RECOMMENDED} before evaluation."
        )

    @staticmethod
    def _nested_module(root, path: str):
        current = root
        for part in path.split("."):
            if not hasattr(current, part):
                return None
            current = getattr(current, part)
        return current if isinstance(current, torch.nn.Module) else None

    @staticmethod
    def _hidden_state_backbone(module):
        """
        Return the decoder module that yields hidden states, not vocab logits.

        MAIRA-2 exposes language_model as LlamaForCausalLM. Llava.forward()
        expects self.model to behave like LlamaModel and then applies
        self.lm_head itself. Pointing self.model at LlamaForCausalLM makes
        Llava.forward apply lm_head twice, producing a shape mismatch.
        """
        if not isinstance(module, torch.nn.Module):
            return module, ""

        has_output_head = (
            isinstance(getattr(module, "lm_head", None), torch.nn.Module)
            or callable(getattr(module, "get_output_embeddings", None))
        )
        for path in ("model", "base_model.model", "transformer", "decoder"):
            child = MAIRA2Generator._nested_module(module, path)
            if child is not None and child is not module and has_output_head:
                return child, path
        return module, ""

    @staticmethod
    def _patch_lm_head_attribute(model, backbone=None):
        """Mirror a nested language-model lm_head onto MAIRA-2's wrapper."""
        if isinstance(getattr(model, "lm_head", None), torch.nn.Module):
            print(
                "[MAIRA2Generator] .lm_head attribute already exists, "
                f"no patch needed ({type(model.lm_head).__name__})."
            )
            return

        candidates = []
        if isinstance(backbone, torch.nn.Module):
            candidates.append(("backbone", backbone))
        for attr_name in ("language_model", "text_model", "model", "lm", "llm"):
            if hasattr(model, attr_name):
                candidate = getattr(model, attr_name)
                if isinstance(candidate, torch.nn.Module):
                    candidates.append((attr_name, candidate))

        seen = set()
        for source_name, candidate in candidates:
            if id(candidate) in seen:
                continue
            seen.add(id(candidate))
            for path in ("lm_head", "base_model.lm_head", "model.lm_head"):
                head = MAIRA2Generator._nested_module(candidate, path)
                if head is not None:
                    model.lm_head = head
                    print(
                        f"[MAIRA2Generator] PATCHED: model.lm_head = "
                        f"{source_name}.{path} (type: {type(head).__name__})"
                    )
                    return
            getter = getattr(candidate, "get_output_embeddings", None)
            if callable(getter):
                try:
                    head = getter()
                except Exception:
                    head = None
                if isinstance(head, torch.nn.Module):
                    model.lm_head = head
                    print(
                        f"[MAIRA2Generator] PATCHED: model.lm_head = "
                        f"{source_name}.get_output_embeddings() "
                        f"(type: {type(head).__name__})"
                    )
                    return

        print(
            "[MAIRA2Generator] WARNING: Could not find language model lm_head "
            "to patch. generate() may fail with transformers 5.x."
        )

    @staticmethod
    def _patch_model_attribute(model):
        """
        Ensure the model object has a .model attribute pointing to its language
        model backbone and an .lm_head output head. Required by transformers 5.x
        LlavaForConditionalGeneration.
        """
        model_cls = type(model).__name__
        print(f"[MAIRA2Generator] Model class: {model_cls}")
        print(f"[MAIRA2Generator] MRO: {[c.__name__ for c in type(model).__mro__[:6]]}")

        # Log all top-level nn.Module children for diagnostics
        if hasattr(model, 'named_children'):
            children = {name: type(child).__name__ for name, child in model.named_children()}
            print(f"[MAIRA2Generator] Model children (nn.Module): {children}")
        else:
            children = {}

        # Log other potentially relevant attributes
        candidate_attrs = [
            'model', 'language_model', 'text_model', 'lm', 'llm',
            'lm_head', 'transformer', 'decoder', 'base_model',
        ]
        existing = {a: type(getattr(model, a, None)).__name__
                     for a in candidate_attrs if hasattr(model, a)}
        print(f"[MAIRA2Generator] Known attrs on model: {existing}")

        if isinstance(getattr(model, 'model', None), torch.nn.Module):
            current_model = model.model
            hidden_backbone, hidden_path = MAIRA2Generator._hidden_state_backbone(current_model)
            if hidden_backbone is not current_model:
                model.model = hidden_backbone
                print(
                    f"[MAIRA2Generator] PATCHED: existing model.model -> "
                    f"model.model.{hidden_path} (type: {type(hidden_backbone).__name__})"
                )
            else:
                print("[MAIRA2Generator] .model attribute already exists, no patch needed.")
            MAIRA2Generator._patch_lm_head_attribute(model, current_model)
            return

        # Try to find the language model backbone and patch it in
        # Priority order: language_model is most common for Llava-family models
        backbone_candidates = ['language_model', 'text_model', 'lm', 'llm',
                               'transformer', 'decoder']
        for attr_name in backbone_candidates:
            if hasattr(model, attr_name):
                backbone = getattr(model, attr_name)
                if isinstance(backbone, torch.nn.Module):
                    hidden_backbone, hidden_path = MAIRA2Generator._hidden_state_backbone(backbone)
                    model.model = hidden_backbone
                    suffix = f".{hidden_path}" if hidden_path else ""
                    print(
                        f"[MAIRA2Generator] PATCHED: model.model = "
                        f"model.{attr_name}{suffix} "
                        f"(type: {type(hidden_backbone).__name__})"
                    )
                    MAIRA2Generator._patch_lm_head_attribute(model, backbone)
                    return

        # If no named attribute found, check nn.Module children by name
        for child_name in ['language_model', 'text_model', 'lm', 'llm',
                           'transformer', 'decoder']:
            if child_name in children:
                backbone = dict(model.named_children())[child_name]
                hidden_backbone, hidden_path = MAIRA2Generator._hidden_state_backbone(backbone)
                model.model = hidden_backbone
                suffix = f".{hidden_path}" if hidden_path else ""
                print(f"[MAIRA2Generator] PATCHED (via children): model.model = "
                      f"model.{child_name}{suffix} (type: {type(hidden_backbone).__name__})")
                MAIRA2Generator._patch_lm_head_attribute(model, backbone)
                return

        # Last resort: if there's exactly one large child module, use it
        if len(children) >= 1:
            # Pick the child that is NOT an embedding/vision component
            non_vision = {n: c for n, c in children.items()
                          if 'vision' not in n.lower() and 'image' not in n.lower()
                          and 'embed' not in n.lower() and 'multi_modal' not in n.lower()}
            if len(non_vision) == 1:
                child_name = list(non_vision.keys())[0]
                backbone = dict(model.named_children())[child_name]
                hidden_backbone, hidden_path = MAIRA2Generator._hidden_state_backbone(backbone)
                model.model = hidden_backbone
                suffix = f".{hidden_path}" if hidden_path else ""
                print(f"[MAIRA2Generator] PATCHED (heuristic): model.model = "
                      f"model.{child_name}{suffix} (type: {type(hidden_backbone).__name__})")
                MAIRA2Generator._patch_lm_head_attribute(model, backbone)
                return

        print("[MAIRA2Generator] WARNING: Could not find language model backbone to "
              "patch .model attribute. generate() may fail with transformers 5.x.")
        MAIRA2Generator._patch_lm_head_attribute(model)

    def _verify_model_attribute(self):
        """
        Pre-check whether self.model.model is accessible, and if not, try
        to fix it by inspecting the forward() source code for attribute access.
        """
        import inspect

        # Check if .model is set and is an nn.Module
        if hasattr(self.model, 'model') and isinstance(self.model.model, torch.nn.Module):
            print(f"[MAIRA2Generator] Verify: .model exists -> {type(self.model.model).__name__}")
            return

        print("[MAIRA2Generator] Verify: .model is NOT set, attempting source-code analysis...")

        # Try to read forward() source to find what self.model references
        try:
            forward_method = type(self.model).forward
            src = inspect.getsource(forward_method)
            print(f"[MAIRA2Generator] forward() source (first 500 chars):\n{src[:500]}")

            # Look for patterns like self.language_model( or self.text_model(
            import re
            refs = re.findall(r'self\.(\w+)\s*[\(\.]', src)
            # Filter to nn.Module children only
            children_dict = dict(self.model.named_children())
            module_refs = [r for r in refs if r in children_dict and r != 'model']
            if module_refs:
                # Use the first referenced module child as .model
                chosen = module_refs[0]
                self.model.model = children_dict[chosen]
                print(f"[MAIRA2Generator] Verify PATCHED from source: model.model = "
                      f"model.{chosen} ({type(children_dict[chosen]).__name__})")
                return
        except (TypeError, OSError) as e:
            print(f"[MAIRA2Generator] Could not read forward() source: {e}")

        # If forward() source didn't help, try to monkey-patch generate() itself
        # to wrap the AttributeError and provide a working .model on the fly
        original_generate = self.model.generate

        def _safe_generate(*args, **kwargs):
            """Wrapper that catches .model AttributeError and patches on the fly."""
            try:
                return original_generate(*args, **kwargs)
            except AttributeError as ae:
                if "'model'" not in str(ae) and "has no attribute 'model'" not in str(ae):
                    raise
                print(f"[MAIRA2Generator] generate() hit .model error: {ae}")
                print("[MAIRA2Generator] Trying all nn.Module children as .model ...")
                children_dict = dict(self.model.named_children())
                for child_name, child_module in children_dict.items():
                    if child_name in ('model',):
                        continue
                    self.model.model = child_module
                    print(f"[MAIRA2Generator] Trying model.model = model.{child_name} "
                          f"({type(child_module).__name__})...")
                    try:
                        result = original_generate(*args, **kwargs)
                        print(f"[MAIRA2Generator] SUCCESS with model.{child_name}!")
                        # Replace generate permanently with the working version
                        self.model.generate = original_generate
                        return result
                    except Exception as retry_e:
                        err_str = str(retry_e)[:150]
                        print(f"[MAIRA2Generator] model.{child_name} failed: {err_str}")
                        if hasattr(self.model, 'model'):
                            del self.model.model
                raise RuntimeError(
                    "[MAIRA2Generator] All children tried as .model, none worked. "
                    f"Original error: {ae}"
                )

        self.model.generate = _safe_generate
        print("[MAIRA2Generator] Verify: installed _safe_generate wrapper as fallback.")

    def generate(
        self,
        image_paths: List[str],
        prior_image_paths: Optional[List[str]] = None,
        context: Optional[str] = None,
        prior_report: Optional[str] = None,
        prompt_style: Optional[str] = None,
    ) -> str:
        effective_prompt_style = self._resolve_prompt_style(prompt_style)
        self._last_raw_decoded_text = ""
        self._last_generation_metadata = self.prompt_metadata(
            effective_prompt_style
        )

        frontal_path = image_paths[0] if image_paths else ""
        if not frontal_path or not Path(frontal_path).exists():
            print(f"[MAIRA2Generator] No valid frontal image: {frontal_path}")
            return ""

        try:
            frontal_image = Image.open(frontal_path).convert("RGB")
        except Exception as e:
            print(f"[MAIRA2Generator] Failed to open frontal image {frontal_path}: {e}")
            return ""

        lateral_image = None
        if len(image_paths) >= 2 and Path(image_paths[1]).exists():
            try:
                lateral_image = Image.open(image_paths[1]).convert("RGB")
            except Exception as e:
                print(f"[MAIRA2Generator] Failed to open lateral image: {e}")

        prior_frontal = None
        if prior_image_paths and len(prior_image_paths) > 0 and Path(prior_image_paths[0]).exists():
            try:
                prior_frontal = Image.open(prior_image_paths[0]).convert("RGB")
            except Exception as e:
                print(f"[MAIRA2Generator] Failed to open prior image: {e}")

        prior_report_text = (prior_report or "").strip() or None
        if prior_report_text is None and context and prior_image_paths:
            prior_report_text = context.strip()

        indication_text = (context or "").strip()
        if prior_report_text and indication_text == prior_report_text:
            indication_text = ""

        indication = self._compose_reporting_indication(
            indication_text,
            effective_prompt_style,
        )
        self._last_generation_metadata["effective_indication_sha256"] = (
            self._sha256(indication)
        )
        technique = "PA and lateral views of the chest." if lateral_image else "Single frontal view of the chest."
        comparison = "Prior study available for comparison." if (prior_frontal or prior_report_text) else "None."
        if not getattr(self, "_logged_input_contract", False):
            print(
                "[MAIRA2Generator] Input contract: "
                f"current_images={len(image_paths or [])}, "
                f"prior_image_paths={len(prior_image_paths or [])}, "
                f"prior_frontal_loaded={prior_frontal is not None}, "
                f"context_chars={len(context or '')}, "
                f"prior_report_chars={len(prior_report_text or '')}, "
                f"prompt_style={effective_prompt_style}"
            )
            self._logged_input_contract = True

        try:
            processed_inputs = self.processor.format_and_preprocess_reporting_input(
                current_frontal=frontal_image,
                current_lateral=lateral_image,
                prior_frontal=prior_frontal,
                indication=indication,
                technique=technique,
                comparison=comparison,
                prior_report=prior_report_text,
                return_tensors="pt",
                get_grounding=False,
            )
        except Exception as e:
            print(f"[MAIRA2Generator] Preprocessing failed: {e}")
            traceback.print_exc()
            return ""

        try:
            processed_inputs = processed_inputs.to(self.device)
        except Exception as e:
            print(f"[MAIRA2Generator] Failed to move inputs to {self.device}: {e}, trying cuda:0")
            try:
                processed_inputs = processed_inputs.to("cuda:0")
            except Exception as e2:
                print(f"[MAIRA2Generator] Failed to move inputs to cuda:0: {e2}")
                return ""

        try:
            with torch.no_grad():
                output_decoding = self.model.generate(
                    **processed_inputs,
                    max_new_tokens=MAIRA2_MAX_NEW_TOKENS,
                    use_cache=MAIRA2_USE_CACHE,
                    do_sample=MAIRA2_DO_SAMPLE,
                    num_beams=MAIRA2_NUM_BEAMS,
                )
        except Exception as e:
            print(f"[MAIRA2Generator] model.generate() failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            # Dump full diagnostics on failure
            print("[MAIRA2Generator] === DIAGNOSTICS ON FAILURE ===")
            all_attrs = [a for a in dir(self.model) if not a.startswith('__')]
            module_attrs = [a for a in all_attrs
                            if isinstance(getattr(self.model, a, None), torch.nn.Module)]
            print(f"[MAIRA2Generator] nn.Module attrs: {module_attrs}")
            generate_methods = [a for a in all_attrs if 'generat' in a.lower()]
            print(f"[MAIRA2Generator] *generat* attrs: {generate_methods}")
            forward_methods = [a for a in all_attrs if 'forward' in a.lower()]
            print(f"[MAIRA2Generator] *forward* attrs: {forward_methods}")
            print(f"[MAIRA2Generator] has .model: {hasattr(self.model, 'model')}")
            if hasattr(self.model, 'model'):
                print(f"[MAIRA2Generator] .model type: {type(self.model.model).__name__}")
            print("[MAIRA2Generator] === END DIAGNOSTICS ===")
            return ""

        prompt_length = processed_inputs["input_ids"].shape[-1]
        generated_ids = output_decoding[0][prompt_length:]
        generated_new_tokens = int(len(generated_ids))
        self._last_generation_metadata["generated_new_tokens"] = (
            generated_new_tokens
        )
        # Conservative cap flag: a continuation of exactly the configured
        # maximum is treated as potentially truncated even if its last token
        # could also be an EOS token.
        self._last_generation_metadata[
            "generation_hit_max_new_tokens"
        ] = generated_new_tokens >= MAIRA2_MAX_NEW_TOKENS
        if generated_new_tokens == 0:
            print("[MAIRA2Generator] WARNING: model generated 0 new tokens")
            return ""

        decoded_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        decoded_text = decoded_text.lstrip()
        self._last_raw_decoded_text = decoded_text

        if not decoded_text:
            print("[MAIRA2Generator] WARNING: decoded text is empty after skip_special_tokens")
            # Try decoding without skipping special tokens for diagnostics
            raw = self.processor.decode(generated_ids, skip_special_tokens=False)
            print(f"[MAIRA2Generator] Raw decoded (first 200 chars): {raw[:200]}")
            return ""

        try:
            prediction = self.processor.convert_output_to_plaintext_or_grounded_sequence(
                decoded_text
            )
        except Exception as e:
            print(f"[MAIRA2Generator] convert_output failed: {e}, using raw decoded text")
            return decoded_text.strip()

        if isinstance(prediction, list):
            result = " ".join(p[0] for p in prediction if p and p[0]).strip()
        else:
            result = str(prediction).strip()

        if not result:
            print(f"[MAIRA2Generator] WARNING: final result empty. decoded_text was: {decoded_text[:200]}")

        return result

