import os
import sys
from pathlib import Path
from typing import List, Optional

import torch

from .base import BaseGenerator, STANDARD_PROMPT, clean_report

MODEL_ID = "google/medgemma-1.5-4b-it"
PRECISION_POLICY = "bfloat16"
QUANTIZATION_POLICY = "none"
ACCELERATOR_POLICY = "nvidia_a100_80gb"


class MedGemmaGenerator(BaseGenerator):
    """
    MedGemma 1.5 4B baseline loaded in unquantized BF16.
    """

    name = "medgemma"
    precision_policy = PRECISION_POLICY
    quantization_policy = QUANTIZATION_POLICY
    accelerator_policy = ACCELERATOR_POLICY

    def __init__(
        self,
        model_id: str = MODEL_ID,
        revision: Optional[str] = None,
        lora_path: Optional[Path] = None,
        local_files_only: bool = False,
        device: str = "cuda",
        **kwargs,
    ):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.requested_revision = revision
        kw = {"trust_remote_code": True}
        if local_files_only:
            kw["local_files_only"] = True
        if revision:
            kw["revision"] = revision

        self.processor = AutoProcessor.from_pretrained(model_id, **kw)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **kw,
        )

        if lora_path and Path(lora_path).exists():
            try:
                self.model.load_adapter(str(lora_path), adapter_name="default")
                print(f"[MedGemmaGenerator] Loaded LoRA from {lora_path}")
            except Exception as e:
                print(f"[MedGemmaGenerator] LoRA load failed ({e}), using base weights")
        self.model.eval()

    def generate(
        self,
        image_paths: List[str],
        prior_image_paths: Optional[List[str]] = None,
        context: Optional[str] = None,
        prior_report: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        from PIL import Image

        try:
            img = Image.open(image_paths[0]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))

        prior_img = None
        if prior_image_paths:
            try:
                prior_img = Image.open(prior_image_paths[0]).convert("RGB")
            except Exception:
                prior_img = None

        images = [img]
        if prior_img is not None:
            images.append(prior_img)

        clinical_context = (context or "").strip()
        if not clinical_context and prior_report:
            clinical_context = prior_report.strip()

        # Prompt style: "concise" (default) matches MedGemma model card.
        # "report" is a stricter prompt-contract style for artifact ablations.
        # "verbose" uses the legacy STANDARD_PROMPT for backward compatibility.
        prompt_style = os.environ.get("MEDGEMMA_PROMPT_STYLE", "concise").lower()

        if prompt_style == "verbose":
            user_text = STANDARD_PROMPT
            if prior_img is not None:
                user_text += (
                    "\n\nThe first image is the current chest X-ray. "
                    "The second image is a prior chest X-ray for comparison."
                )
            if clinical_context:
                user_text += f"\n\nClinical context: {clinical_context}"
            content = [{"type": "image"}]
            if prior_img is not None:
                content.append({"type": "image"})
            content.append({"type": "text", "text": user_text})
            messages = [
                {
                    "role": "user",
                    "content": content,
                },
            ]
        elif prompt_style == "report":
            if prior_img is not None:
                user_text = (
                    "The first image is the current chest X-ray. "
                    "The second image is a prior chest X-ray for comparison. "
                    "Use the prior only to identify clinically relevant interval change. "
                    "Write a concise report for the current study with Findings and Impression only. "
                    "Do not copy, critique, or continue the prior report. "
                    "Do not include recommendations, meta commentary, or disclaimers."
                )
            else:
                user_text = (
                    "Write a concise chest X-ray report for the current image. "
                    "Use Findings and Impression only. "
                    "Do not include recommendations, meta commentary, or disclaimers."
                )
            if clinical_context:
                user_text += f" Clinical context for comparison only: {clinical_context}"
            content = [{"type": "image"}]
            if prior_img is not None:
                content.append({"type": "image"})
            content.append({"type": "text", "text": user_text})
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are an expert radiologist. "
                                "Return only a concise radiology report for the current study."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": content,
                },
            ]
        else:
            if prior_img is not None:
                user_text = (
                    "The first image is the current chest X-ray. "
                    "The second image is a prior chest X-ray for comparison. "
                    "Describe the current findings and any clinically relevant interval change."
                )
            else:
                user_text = "Describe the findings in this chest X-ray."
            if clinical_context:
                user_text += f" Clinical context: {clinical_context}"
            content = [{"type": "image"}]
            if prior_img is not None:
                content.append({"type": "image"})
            content.append({"type": "text", "text": user_text})
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert radiologist."}],
                },
                {
                    "role": "user",
                    "content": content,
                },
            ]
        text = self.processor.apply_chat_template(messages, tokenize=False)
        inputs = self.processor(text=[text], images=[images], return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            max_new_tokens = int(os.environ.get("MEDGEMMA_MAX_NEW_TOKENS", "300"))
        except ValueError:
            max_new_tokens = 300
        max_new_tokens = max(16, max_new_tokens)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.2,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        if temperature is not None and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        gen = out[0, inputs["input_ids"].shape[1] :]
        report = self.processor.tokenizer.decode(gen, skip_special_tokens=True).strip()
        return clean_report(report)

