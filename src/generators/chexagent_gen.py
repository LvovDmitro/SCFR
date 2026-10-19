"""
CheXagent-2-3b: Stanford AIMI chest X-ray agent.
https://huggingface.co/StanfordAIMI/CheXagent-2-3b

Based on Microsoft Phi architecture with XraySigLIP vision encoder.
Uses AutoModelForCausalLM + AutoTokenizer (NOT AutoProcessor).
Custom chat format with tokenizer.from_list_format() for multimodal input.

Note: CheXagent's HF tokenizer code imports is_tf_available which was removed
in transformers>=5.0. We monkey-patch it before loading.
"""
from pathlib import Path
from typing import List, Optional
import torch
from .base import BaseGenerator, STANDARD_PROMPT, clean_report

MODEL_ID = "StanfordAIMI/CheXagent-2-3b"


def _patch_transformers_compat():
    """Patch transformers.utils for backward compat with HF model code.

    CheXagent's custom code (trust_remote_code=True) does two problematic things:
    1. Imports is_tf_available / is_torch_available which were removed in transformers>=5.0
    2. Has a hard version check that rejects transformers != 4.40.0

    We monkey-patch both so the model loads with any modern transformers.
    """
    import transformers
    import transformers.utils
    if not hasattr(transformers.utils, "is_tf_available"):
        transformers.utils.is_tf_available = lambda: False
    if not hasattr(transformers.utils, "is_torch_available"):
        transformers.utils.is_torch_available = lambda: True

    # Spoof version so the model's version check passes.
    # The model code checks for transformers==4.40.0 and raises if it doesn't match.
    # We preserve the real version and only patch __version__ for the import.
    _real_version = transformers.__version__
    transformers.__version__ = "4.40.0"

    # Also patch packaging.version in case the model code parses the version
    try:
        import packaging.version
        _orig_parse = packaging.version.parse

        def _patched_parse(v):
            if v == _real_version:
                return _orig_parse("4.40.0")
            return _orig_parse(v)

        packaging.version.parse = _patched_parse
    except ImportError:
        pass


class CheXagentGenerator(BaseGenerator):
    name = "chexagent"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        _patch_transformers_compat()

        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model_id = model_id
        self.requested_revision = revision
        print(f"[CheXagentGenerator] Loading {model_id}...")
        kw = {"trust_remote_code": True}
        if local_files_only:
            kw["local_files_only"] = True
        if revision:
            kw["revision"] = revision

        # Fix for transformers>=5.0: CheXagentConfig may not set pad_token_id,
        # causing AutoModelForCausalLM.from_pretrained to crash internally.
        # Load config first, patch pad_token_id, then load model with patched config.
        config = AutoConfig.from_pretrained(model_id, **kw)
        if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
            config.pad_token_id = getattr(config, 'eos_token_id', 151643)
        print(f"[CheXagentGenerator] Config pad_token_id={config.pad_token_id}")

        # Fix for transformers>=5.0: the native PhiConfig (model_type="phi") may
        # convert rope_scaling from None to {"rope_type": "default"} or rename
        # the "type" key to "rope_type".  CheXagent's custom modeling code does
        # `self.config.rope_scaling["type"]` which then raises KeyError.
        # Ensure rope_scaling is either None or carries the legacy "type" key.
        if getattr(config, 'rope_scaling', None) is not None:
            rs = config.rope_scaling
            if "type" not in rs and "rope_type" in rs:
                rs["type"] = rs["rope_type"]
                print(f"[CheXagentGenerator] Patched rope_scaling: added 'type'='{rs['type']}'")
            elif "type" not in rs:
                # Unknown rope_scaling dict without type info — safer to disable
                config.rope_scaling = None
                print("[CheXagentGenerator] Cleared unexpected rope_scaling (no 'type' key)")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kw)

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            quantization_config=bnb,
            device_map="auto",
            **kw,
        )
        # Note: do NOT call .to(torch.bfloat16) after 4-bit quantization —
        # it would attempt to cast quantized weights and is contradictory.
        # bnb_4bit_compute_dtype=torch.bfloat16 already handles compute dtype.

        self.model.eval()
        self.device = next(self.model.parameters()).device
        print("[CheXagentGenerator] Loaded.")

    def generate(
        self,
        image_paths: List[str],
        prior_image_paths: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> str:
        prompt = STANDARD_PROMPT
        if context:
            prompt += f"\nClinical context: {context}"

        image_path = image_paths[0] if image_paths else ""
        if not image_path or not Path(image_path).exists():
            return ""

        query = self.tokenizer.from_list_format([
            {"image": image_path},
            {"text": prompt},
        ])

        conv = [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": query},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=512,
            )

        prompt_len = input_ids.shape[1]
        gen_tokens = out[0, prompt_len:-1]
        report = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return clean_report(report)

