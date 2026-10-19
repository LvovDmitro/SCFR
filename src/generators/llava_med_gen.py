import sys
from pathlib import Path
from typing import List, Optional
import torch
from .base import BaseGenerator, STANDARD_PROMPT, clean_report

class LLaVAMedGenerator(BaseGenerator):
    """
    LLaVA-Med 7B (chaoyinshe/llava-med-v1.5-mistral-7b-hf) baseline.
    """
    name = "llava_med"
    MODEL_NAME = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"

    def __init__(
        self,
        lora_path: Optional[Path] = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

        kw = {"trust_remote_code": True}
        if local_files_only:
            kw["local_files_only"] = True

        print(f"[LLaVAMedGenerator] Loading {self.MODEL_NAME}...")
        self.processor = AutoProcessor.from_pretrained(self.MODEL_NAME, **kw)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.MODEL_NAME, quantization_config=bnb,
            device_map="auto", **kw,
        )
        if lora_path and Path(lora_path).exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(lora_path))
            print(f"  Loaded LoRA from {lora_path}")
        self.model.eval()

    def generate(self, image_paths: List[str], prior_image_paths: Optional[List[str]] = None, context: Optional[str] = None) -> str:
        from PIL import Image

        try:
            img = Image.open(image_paths[0]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))

        prompt = "USER: <image>\n" + STANDARD_PROMPT
        if context:
            prompt += f"\nClinical context: {context}"
        prompt += "\nASSISTANT:"

        inputs = self.processor(text=prompt, images=img, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=200, do_sample=False,
                repetition_penalty=1.2,
            )
        gen = out[0, inputs["input_ids"].shape[1]:]
        report = self.processor.tokenizer.decode(gen, skip_special_tokens=True).strip()
        return clean_report(report)

