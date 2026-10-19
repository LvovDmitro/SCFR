"""
LLaVA-Rad: Microsoft radiology report generation model.
https://huggingface.co/microsoft/llava-rad

Uses LlavaForConditionalGeneration with CLIPImageProcessor (not AutoProcessor,
which fails for this model).

Key detail: model.generate() requires `pixel_values=` (not `images=`).
The `images` kwarg is silently ignored, producing empty output.
"""
from pathlib import Path
from typing import List, Optional
import torch
from .base import BaseGenerator, STANDARD_PROMPT, clean_report

MODEL_ID = "microsoft/llava-rad"
BASE_MODEL = "lmsys/vicuna-7b-v1.5"


class LLaVARadGenerator(BaseGenerator):
    name = "llava_rad"

    def __init__(self, model_id: str = MODEL_ID, **kwargs):
        from transformers import LlavaForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig
        from transformers import CLIPImageProcessor

        print(f"[LLaVARadGenerator] Loading {model_id}...")
        print(f"[LLaVARadGenerator] torch={torch.__version__}, "
              f"CUDA available={torch.cuda.is_available()}, "
              f"CUDA version={getattr(torch.version, 'cuda', 'N/A')}")

        loaded = False

        # Strategy 1: 4-bit quantization with device_map="auto"
        try:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb,
                device_map="auto",
                trust_remote_code=True,
            )
            print("[LLaVARadGenerator] Loaded with 4-bit quantization.")
            loaded = True
        except Exception as e:
            print(f"[LLaVARadGenerator] 4-bit quantization failed ({e})")

        # Strategy 2: bfloat16 with device_map="auto"
        if not loaded:
            try:
                print("[LLaVARadGenerator] Trying bfloat16 with device_map='auto'...")
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                print("[LLaVARadGenerator] Loaded with bfloat16 + device_map=auto.")
                loaded = True
            except (RuntimeError, OSError) as e:
                print(f"[LLaVARadGenerator] bfloat16+auto failed ({e})")

        # Strategy 3: float16 on CPU first, then move to GPU
        # This avoids CUDA allocator issues from torch/CUDA version mismatch
        if not loaded:
            try:
                print("[LLaVARadGenerator] Trying float16 on CPU first, then .to(cuda)...")
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map=None,
                    trust_remote_code=True,
                )
                if torch.cuda.is_available():
                    self.model = self.model.to("cuda:0")
                    print("[LLaVARadGenerator] Loaded float16 on CPU, moved to cuda:0.")
                else:
                    print("[LLaVARadGenerator] Loaded float16 on CPU (no CUDA available).")
                loaded = True
            except (RuntimeError, OSError) as e:
                print(f"[LLaVARadGenerator] float16 CPU->GPU failed ({e})")

        # Strategy 4: float32 on CPU (very slow but functional)
        if not loaded:
            print("[LLaVARadGenerator] Last resort: loading float32 on CPU...")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True,
            )
            print("[LLaVARadGenerator] Loaded float32 on CPU (will be slow).")

        # LLaVA-Rad uses CLIP image processor, not the model's default
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.model.eval()
        print("[LLaVARadGenerator] Ready.")

    def generate(
        self,
        image_paths: List[str],
        prior_image_paths: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> str:
        from PIL import Image

        img_path = image_paths[0] if image_paths else ""
        if not img_path or not Path(img_path).exists():
            print(f"[LLaVARadGenerator] No valid image path: {img_path!r}")
            return ""

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[LLaVARadGenerator] Failed to open image {img_path}: {e}")
            return ""

        prompt = STANDARD_PROMPT
        if context:
            prompt += f"\nClinical context: {context}"

        # LLaVA chat format
        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(device=self.model.device, dtype=self.model.dtype)

        input_ids = self.tokenizer(full_prompt, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(self.model.device)

        print(f"[LLaVARadGenerator] input_ids shape: {input_ids.shape}, "
              f"pixel_values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")

        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=512,
                do_sample=False,
            )

        print(f"[LLaVARadGenerator] output shape: {out.shape}, "
              f"new tokens: {out.shape[1] - input_ids.shape[1]}")

        gen = out[0, input_ids.shape[1]:]
        raw_text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()

        if not raw_text:
            # Fallback: decode the full output (some models don't separate prompt/gen cleanly)
            raw_text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
            # Try to extract text after ASSISTANT:
            if "ASSISTANT:" in raw_text:
                raw_text = raw_text.split("ASSISTANT:")[-1].strip()
            print(f"[LLaVARadGenerator] Used fallback decoding, got {len(raw_text)} chars")

        report = clean_report(raw_text)
        print(f"[LLaVARadGenerator] Generated {len(report)} chars")
        return report

