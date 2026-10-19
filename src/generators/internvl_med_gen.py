"""
Mini-InternVL2-4B-DA-Medical: OpenGVLab medical domain-adapted VLM.
https://huggingface.co/OpenGVLab/Mini-InternVL2-4B-DA-Medical

Note: The model's custom config has a bug with transformers>=5.0 where
__init__() accesses llm_config['architectures'] without key check.
We monkey-patch the config class before loading.
"""
from typing import List, Optional

import torch
from .base import BaseGenerator, STANDARD_PROMPT, clean_report

MODEL_ID = "OpenGVLab/Mini-InternVL2-4B-DA-Medical"


def _patch_internvl_config():
    """Patch InternVL config to handle missing 'architectures' key.

    The model's configuration_internvl_chat.py accesses llm_config['architectures']
    without a .get() guard, causing KeyError with transformers>=5.0 which may omit
    that key from sub-configs.

    Multi-strategy approach:
    1. Patch the downloaded Python source file to use .get() instead of []
    2. Patch config.json to add missing 'architectures' keys to sub-configs
    3. Install a runtime hook on AutoConfig to catch and retry on KeyError
    """
    # --- Strategy 1: patch the source file ---
    try:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(MODEL_ID, "configuration_internvl_chat.py")

        with open(config_file) as f:
            source = f.read()

        if "llm_config['architectures']" in source:
            patched = source.replace(
                "llm_config['architectures']",
                "llm_config.get('architectures', [''])"
            )
            try:
                with open(config_file, "w") as f:
                    f.write(patched)
                print("[InternVLMedGenerator] Patched config source file OK")
            except OSError:
                print("[InternVLMedGenerator] Config file is read-only, skipping source patch")
    except Exception as e:
        print(f"[InternVLMedGenerator] Source patch skipped: {e}")

    # --- Strategy 2: patch config.json to add architectures to sub-configs ---
    try:
        from huggingface_hub import hf_hub_download
        import json
        cfg_path = hf_hub_download(MODEL_ID, "config.json")
        with open(cfg_path) as f:
            cfg_data = json.load(f)

        patched = False
        for key, val in cfg_data.items():
            if isinstance(val, dict) and 'model_type' in val and 'architectures' not in val:
                val['architectures'] = [val.get('model_type', 'Unknown')]
                patched = True

        if patched:
            try:
                with open(cfg_path, "w") as f:
                    json.dump(cfg_data, f, indent=2)
                print("[InternVLMedGenerator] Patched config.json: added missing 'architectures' keys")
            except OSError:
                print("[InternVLMedGenerator] Could not write patched config.json (read-only)")
    except Exception as e:
        print(f"[InternVLMedGenerator] config.json patch skipped: {e}")

    # --- Strategy 3: runtime monkey-patch of AutoConfig resolution ---
    try:
        from transformers import AutoConfig

        _orig_from_pretrained = AutoConfig.from_pretrained.__func__

        @classmethod  # type: ignore[misc]
        def _safe_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            try:
                return _orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
            except (KeyError, ValueError) as e:
                err_msg = str(e)
                if 'architectures' in err_msg or 'Unsupported architecture' in err_msg or "architecture ''" in err_msg:
                    print(f"[InternVLMedGenerator] Caught {type(e).__name__}: {e}, retrying with patched config...")
                    from huggingface_hub import hf_hub_download
                    import json, os, tempfile, shutil

                    # Download config.json and all .py files to a temp dir
                    cfg_path = hf_hub_download(pretrained_model_name_or_path, "config.json")
                    with open(cfg_path) as f:
                        cfg_data = json.load(f)

                    # Inject architectures into all sub-configs
                    for k, v in cfg_data.items():
                        if isinstance(v, dict) and 'model_type' in v and 'architectures' not in v:
                            v['architectures'] = [v.get('model_type', 'Unknown')]

                    # Also ensure top-level architectures is non-empty
                    if not cfg_data.get('architectures'):
                        cfg_data['architectures'] = [cfg_data.get('model_type', 'InternVLChatModel')]

                    try:
                        with open(cfg_path, "w") as f:
                            json.dump(cfg_data, f, indent=2)
                    except OSError:
                        pass

                    return _orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
                raise

        AutoConfig.from_pretrained = _safe_from_pretrained
        print("[InternVLMedGenerator] Installed AutoConfig monkey-patch")
    except Exception as e:
        print(f"[InternVLMedGenerator] AutoConfig patch skipped: {e}")


class InternVLMedGenerator(BaseGenerator):
    name = "internvl_med"

    def __init__(self, model_id: str = MODEL_ID, local_files_only: bool = False, **kwargs):
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

        print(f"[InternVLMedGenerator] Loading {model_id}...")
        kw = {"trust_remote_code": True}
        if local_files_only:
            kw["local_files_only"] = True

        _patch_internvl_config()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kw)

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        try:
            self.model = AutoModel.from_pretrained(
                model_id,
                quantization_config=bnb,
                device_map="auto",
                **kw,
            )
        except Exception as e:
            print(f"[InternVLMedGenerator] 4-bit failed ({e}), trying bf16...")
            self.model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                **kw,
            )

        self.model.eval()
        print("[InternVLMedGenerator] Loaded.")

    def generate(
        self,
        image_paths: List[str],
        prior_image_paths: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> str:
        from PIL import Image
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        image_path = image_paths[0] if image_paths else ""
        if not image_path:
            return ""

        try:
            img = Image.open(image_path).convert("RGB")
            pixel_values = self._load_image(img).to(
                dtype=torch.bfloat16, device=next(self.model.parameters()).device
            )
        except Exception as e:
            print(f"[InternVLMedGenerator] Image load failed: {e}")
            return ""

        prompt = "<image>\n" + STANDARD_PROMPT
        if context:
            prompt += f"\nClinical context: {context}"

        generation_config = {"max_new_tokens": 512, "do_sample": False}

        try:
            response = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config)
            return clean_report(response.strip())
        except Exception as e:
            print(f"[InternVLMedGenerator] Generation failed: {e}")
            return ""

    @staticmethod
    def _load_image(image, input_size=448):
        """Process image for InternVL2."""
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return transform(image).unsqueeze(0)

