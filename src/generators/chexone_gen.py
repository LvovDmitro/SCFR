"""CheXOne generator with the frozen SCFR reader loading policy."""

from pathlib import Path
from typing import List, Optional

import torch

from .base import BaseGenerator, STANDARD_PROMPT, clean_report


MODEL_ID = "StanfordAIMI/CheXOne"
PRECISION_POLICY = "bfloat16"
QUANTIZATION_POLICY = "none"
ACCELERATOR_POLICY = "nvidia_a100_80gb"
PROCESSOR_MIN_PIXELS = 4 * 28 * 28
PROCESSOR_MAX_PIXELS = 512 * 512


class CheXOneGenerator(BaseGenerator):
    """Load CheXOne in unquantized BF16 with frozen image bounds."""

    name = "chexone"
    precision_policy = PRECISION_POLICY
    quantization_policy = QUANTIZATION_POLICY
    accelerator_policy = ACCELERATOR_POLICY
    processor_policy = {
        "min_pixels": PROCESSOR_MIN_PIXELS,
        "min_pixels_policy": "explicit_model_card_default",
        "max_pixels": PROCESSOR_MAX_PIXELS,
        "max_pixels_policy": "explicit_training_alignment",
    }

    def __init__(
        self,
        model_id: str = MODEL_ID,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> None:
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
        )

        self.model_id = model_id
        self.requested_revision = revision
        load_kwargs = {}
        if local_files_only:
            load_kwargs["local_files_only"] = True
        if revision:
            load_kwargs["revision"] = revision

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=PROCESSOR_MIN_PIXELS,
            max_pixels=PROCESSOR_MAX_PIXELS,
            **load_kwargs,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **load_kwargs,
        )
        self.model.eval()

    def generate(
        self,
        image_paths: List[str],
        prior_image_paths: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> str:
        from PIL import Image

        ordered_paths = list(image_paths)
        if prior_image_paths:
            ordered_paths.extend(prior_image_paths)
        if not ordered_paths:
            return ""

        images = []
        try:
            for value in ordered_paths:
                path = Path(value)
                if not path.is_file():
                    return ""
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())

            prompt = STANDARD_PROMPT
            if context:
                prompt += f"\nClinical context: {context}"
            content = [
                {"type": "image", "image": Path(path).resolve().as_uri()}
                for path in ordered_paths
            ]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            rendered = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[rendered],
                images=images,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    num_beams=1,
                )
            generated = outputs[0][inputs["input_ids"].shape[-1] :]
            text = self.processor.tokenizer.decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return clean_report(text)
        finally:
            for image in images:
                image.close()


__all__ = [
    "ACCELERATOR_POLICY",
    "CheXOneGenerator",
    "MODEL_ID",
    "PRECISION_POLICY",
    "PROCESSOR_MAX_PIXELS",
    "PROCESSOR_MIN_PIXELS",
    "QUANTIZATION_POLICY",
]

