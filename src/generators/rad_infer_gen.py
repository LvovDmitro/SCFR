"""
Radiology-Infer-Mini: Qwen2-VL-2B fine-tuned for radiology report generation.
https://huggingface.co/prithivMLmods/Radiology-Infer-Mini

Lightweight 2B model based on Qwen2-VL-2B-Instruct.
Standard Qwen2VL loading via AutoModelForImageTextToText.

Requires: transformers>=4.50, torch.
"""
from typing import List, Optional

import torch
from .base import BaseGenerator, STANDARD_PROMPT, clean_report

MODEL_ID = "prithivMLmods/Radiology-Infer-Mini"

# More explicit prompt for this small 2B model to encourage detailed multi-sentence output.
_RAD_INFER_PROMPT = (
    "You are a radiologist writing a chest X-ray report. "
    "Provide a detailed findings section covering the following structures: "
    "heart size and silhouette, lungs and pleural spaces, mediastinum and hila, "
    "bones and soft tissues, and any lines, tubes, or devices. "
    "For each structure, state whether it is normal or abnormal and describe any abnormalities. "
    "Write at least 5 sentences. Do not include headers, greetings, or conversational text."
)


class RadInferGenerator(BaseGenerator):
    name = "rad_infer"

    def __init__(self, model_id: str = MODEL_ID, local_files_only: bool = False, **kwargs):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"[RadInferGenerator] Loading {model_id}...")
        kw = {"trust_remote_code": True}
        if local_files_only:
            kw["local_files_only"] = True

        self.processor = AutoProcessor.from_pretrained(model_id, **kw)
        # Small enough (2B) to run in fp16 without quantization
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            **kw,
        )
        self.model.eval()
        print("[RadInferGenerator] Loaded.")

    def generate(
        self,
        image_paths: List[str],
        prior_image_paths: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> str:
        from PIL import Image

        image_path = image_paths[0] if image_paths else ""
        if not image_path:
            return ""

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[RadInferGenerator] Failed to open image {image_path}: {e}")
            img = Image.new("RGB", (224, 224))

        prompt = _RAD_INFER_PROMPT
        if context:
            prompt += f"\nClinical context: {context}"

        # Qwen2VL: use file path string in messages for apply_chat_template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Pass PIL image separately to processor for pixel encoding
        inputs = self.processor(text=[text], images=[img], return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.1,
            )

        gen = out[0, inputs["input_ids"].shape[1]:]
        report = self.processor.tokenizer.decode(gen, skip_special_tokens=True).strip()
        return clean_report(report)

