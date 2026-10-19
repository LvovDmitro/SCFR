"""
NV-Reason-CXR-3B: NVIDIA chest X-ray reasoning model.
https://huggingface.co/nvidia/NV-Reason-CXR-3B

Based on Qwen2.5-VL-3B-Instruct, fine-tuned with SFT + GRPO for CXR interpretation.
Outputs structured <thinking> and <answer> tags — we extract the <answer> part.

The frozen SCFR reader loads the original weights in unquantized BF16.
"""
import re
from typing import List, Optional

import torch
from .base import BaseGenerator, STANDARD_PROMPT, clean_report

MODEL_ID = "nvidia/NV-Reason-CXR-3B"
PRECISION_POLICY = "bfloat16"
QUANTIZATION_POLICY = "none"
ACCELERATOR_POLICY = "nvidia_a100_80gb"


class NVReasonCXRGenerator(BaseGenerator):
    name = "nv_reason_cxr"
    precision_policy = PRECISION_POLICY
    quantization_policy = QUANTIZATION_POLICY
    accelerator_policy = ACCELERATOR_POLICY

    def __init__(
        self,
        model_id: str = MODEL_ID,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.requested_revision = revision
        print(f"[NVReasonCXRGenerator] Loading {model_id}...")
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
        self.model.eval()
        print("[NVReasonCXRGenerator] Loaded.")

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
            print(f"[NVReasonCXRGenerator] Failed to open image {image_path}: {e}")
            img = Image.new("RGB", (224, 224))

        prompt = STANDARD_PROMPT
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
            )

        gen = out[0, inputs["input_ids"].shape[1]:]
        report = self.processor.tokenizer.decode(gen, skip_special_tokens=True).strip()

        # Extract <answer> content if present (model outputs <thinking>...<answer>...)
        report = self._extract_answer(report)
        return clean_report(report)

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Extract content from <answer> tags, falling back to full text."""
        if not text:
            return text
        match = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # If no tags, try removing <thinking> block
        cleaned = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()
        # Also remove any remaining XML-like tags
        cleaned = re.sub(r"<[^>]+>", "", cleaned).strip()
        cleaned = NVReasonCXRGenerator._strip_reasoning_preamble(cleaned)
        return cleaned if cleaned else text

    @staticmethod
    def _strip_reasoning_preamble(text: str) -> str:
        """Remove reasoning/chain-of-thought preamble that precedes clinical findings.

        The model sometimes outputs conversational reasoning text before the
        actual radiology findings, e.g.:
          "We'll begin with the quality assessment. The image is well-exposed...
           Now let's look at the findings.\n\nThe heart is normal in size..."
        This method strips that preamble so only the clinical content remains.
        """
        if not text:
            return text

        # 1. If there is a clear transition phrase, take everything after it
        transition_patterns = [
            r"(?:Now\s+)?(?:let(?:'s| us)|I(?:'ll| will))\s+(?:look at|describe|present|provide|summarize|write)\s+(?:the\s+)?(?:actual\s+)?(?:findings|report|results)[.:]?\s*\n*",
            r"(?:Here\s+(?:are|is)\s+(?:the\s+)?(?:findings|report))[.:]?\s*\n*",
            r"(?:The\s+(?:findings|report)\s+(?:is|are)\s+as\s+follows)[.:]?\s*\n*",
        ]
        for pat in transition_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                after = text[m.end():].strip()
                if after:
                    return after

        # 2. Strip leading reasoning sentences (before any clinical content).
        #    Reasoning lines typically start with phrases like these:
        reasoning_start = re.compile(
            r"^("
            r"(?:We(?:'ll| will| shall)\s+(?:begin|start|first|now))"
            r"|(?:Let(?:'s| us| me)\s+(?:begin|start|first|analyze|examine|assess|look|evaluate|review))"
            r"|(?:I(?:'ll| will| need to| should| am going to)\s+(?:begin|start|first|analyze|examine|assess|look|evaluate|review))"
            r"|(?:Looking at (?:this|the))"
            r"|(?:First,?\s+(?:I(?:'ll| will)?|let(?:'s| me)|we))"
            r"|(?:To (?:begin|start|analyze|provide))"
            r"|(?:Before (?:I |we )?(?:begin|start|describe|provide|write))"
            r"|(?:Starting with)"
            r"|(?:Upon (?:initial |first |careful )?(?:review|examination|inspection|assessment|observation))"
            r")",
            re.IGNORECASE,
        )

        if not reasoning_start.match(text):
            return text

        # The text starts with reasoning. Try to find where clinical content begins.
        # Clinical findings usually start after a blank line or with anatomical terms.
        clinical_start = re.compile(
            r"(?:^|\n\n)"
            r"(?="
            r"(?:The (?:heart|lungs?|cardiomediastinal|mediastin|trachea|hilar|pleural|costophrenic|diaphragm|osseous|bony|soft tissue|airspace|pulmonary|aorta|aortic))"
            r"|(?:(?:No|There (?:is|are)) (?:acute|focal|significant|definite|pleural|pulmonary))"
            r"|(?:(?:Heart|Lungs?|Mediastin|Cardiomediastinal|Pleural|Osseous|Trachea|Hilar) )"
            r"|(?:(?:PA|AP|Frontal|Lateral) (?:and |view |radiograph|chest))"
            r"|(?:(?:Normal|Stable|Unchanged|Clear) )"
            r"|(?:(?:FINDINGS|IMPRESSION|Findings|Impression)[:\s])"
            r")",
            re.IGNORECASE,
        )
        m = clinical_start.search(text)
        if m:
            after = text[m.start():].lstrip("\n")
            if after:
                return after

        # 3. Fallback: remove all leading sentences that look like reasoning
        #    (sentences that don't contain anatomical/clinical terms)
        sentences = re.split(r'(?<=[.!])\s+', text, maxsplit=20)
        clinical_terms = re.compile(
            r"\b(?:heart|lung|cardio|mediast|pleural|trachea|hilar|diaphragm|"
            r"osseous|pulmonary|aort|opaci|effusion|consolidat|infiltrat|"
            r"atelecta|pneumo|nodul|mass|cardiomegaly|edema|normal|clear|"
            r"costophrenic|silhouette|vasculature|calcif|fractur|catheter|"
            r"sternotomy|pacer|device|tube|line)\b",
            re.IGNORECASE,
        )
        for i, sent in enumerate(sentences):
            if clinical_terms.search(sent):
                return " ".join(sentences[i:])

        # Nothing matched — return original
        return text

