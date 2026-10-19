"""
BiomedGPT-LM-7B: General biomedical language model.
https://huggingface.co/PharMolix/BioMedGPT-LM-7B

WARNING: This is a TEXT-ONLY model (Llama-2-7B fine-tune). It has NO vision encoder
and cannot see the input X-ray images. It generates reports purely from a text prompt.
Metrics from this model are NOT comparable to multimodal models — include as a
text-only baseline reference only.

Requires: transformers, torch, bitsandbytes.
"""
from pathlib import Path
from typing import List, Optional
import torch
from .base import BaseGenerator, STANDARD_PROMPT_TEXT_ONLY, clean_report

MODEL_ID = "PharMolix/BioMedGPT-LM-7B"


class BiomedGPTGenerator(BaseGenerator):
    name = "biomedgpt"

    def __init__(self, model_id: str = MODEL_ID, local_files_only: bool = False, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"[BiomedGPTGenerator] Loading {model_id}...")
        kw = {"trust_remote_code": True}
        if local_files_only:
            kw["local_files_only"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kw)

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            **kw,
        )
        self.model.eval()
        print("[BiomedGPTGenerator] Loaded.")

    def generate(
        self,
        image_paths: List[str],
        prior_image_paths: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> str:
        # NOTE: BiomedGPT-LM-7B is a language-only model. It does not have a
        # native vision encoder, so we generate reports based on a text prompt
        # describing the task. If a multimodal variant becomes available,
        # update this to include image processing.
        prompt = STANDARD_PROMPT_TEXT_ONLY
        if context:
            prompt += f"\nClinical context: {context}"
        prompt += "\n\nFindings:"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        gen_tokens = out[0, prompt_len:]
        report = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return clean_report(report)

