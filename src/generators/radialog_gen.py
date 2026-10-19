"""
RaDialog v2: Dialogue-based radiology report generation.
https://huggingface.co/ChantalPellegrini/RaDialog-interactive-radiology-report-generation

Based on LLaVA-v1.5-7B with BioViL-T vision encoder + LoRA adapters.
Requires downloading the full HF repo (includes LLAVA_Biovil code + CheXpert classifier).

Requires: transformers, peft, pillow, torch, huggingface_hub, pytorch-lightning.
"""
import sys
from pathlib import Path
from typing import List, Optional
import torch
import numpy as np

# RaDialog's HF repo code imports pytorch_lightning internally.
# Check early and give a clear error message.
try:
    import pytorch_lightning  # noqa: F401
except ImportError:
    raise ImportError(
        "RaDialog requires pytorch-lightning. "
        "Install it with: pip install pytorch-lightning"
    )

from .base import BaseGenerator, clean_report

MODEL_REPO = "ChantalPellegrini/RaDialog-interactive-radiology-report-generation"
MODEL_BASE = "liuhaotian/llava-v1.5-7b"
MODEL_NAME = "llava-v1.5-7b-task-lora_radialog_instruct_llava_biovil_unfrozen_2e-5_5epochs_v5_checkpoint-21000"


class RaDialogGenerator(BaseGenerator):
    name = "radialog"

    def __init__(self, device="cuda", **kwargs):
        from huggingface_hub import snapshot_download

        print(f"[RaDialogGenerator] Downloading {MODEL_REPO}...")
        self.model_dir = Path(snapshot_download(repo_id=MODEL_REPO))
        print(f"[RaDialogGenerator] Model dir: {self.model_dir}")

        # Add the HF repo to sys.path so LLAVA_Biovil imports work
        if str(self.model_dir) not in sys.path:
            sys.path.insert(0, str(self.model_dir))

        # Fix for transformers>=5.0: AdamW was removed from transformers.
        # RaDialog's LLAVA_Biovil code may import it. Monkey-patch before import.
        import transformers
        if not hasattr(transformers, "AdamW"):
            transformers.AdamW = torch.optim.AdamW

        from LLAVA_Biovil.llava.model.builder import load_pretrained_model
        from utils import create_chest_xray_transform_for_inference, init_chexpert_predictor

        print("[RaDialogGenerator] Loading LLaVA base + LoRA adapters...")
        self.tokenizer, self.model, _, self.context_len = load_pretrained_model(
            self.model_dir, model_base=MODEL_BASE,
            model_name=MODEL_NAME,
            load_8bit=False, load_4bit=False,
        )
        self.model.config.tokenizer_padding_side = "left"

        self.vis_transforms = create_chest_xray_transform_for_inference(512, center_crop_size=448)

        print("[RaDialogGenerator] Loading CheXpert predictor...")
        self.cp_model, self.cp_class_names, self.cp_transforms = init_chexpert_predictor()

        print("[RaDialogGenerator] Loaded.")

    def generate(
        self,
        image_paths: List[str],
        prior_image_paths: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> str:
        from PIL import Image
        from LLAVA_Biovil.llava.mm_utils import (
            tokenizer_image_token, KeywordsStoppingCriteria, remap_to_uint8,
        )
        from LLAVA_Biovil.llava.conversation import SeparatorStyle, conv_vicuna_v1
        from LLAVA_Biovil.llava.constants import IMAGE_TOKEN_INDEX

        image_path = image_paths[0] if image_paths else ""
        if not image_path or not Path(image_path).exists():
            return ""

        # Load and preprocess image (grayscale for CXR)
        image = Image.open(image_path)
        image = Image.fromarray(remap_to_uint8(np.array(image))).convert("L")

        # Get CheXpert predicted findings
        cp_image = self.cp_transforms(image)
        with torch.no_grad():
            logits = self.cp_model(cp_image[None].half().cuda())
        preds = (torch.sigmoid(logits) > 0.5)[0].cpu().numpy()
        findings = ", ".join(self.cp_class_names[preds]).lower().strip()

        # Build prompt with Vicuna v1 conversation template
        conv = conv_vicuna_v1.copy()
        prompt = (
            f"<image>. Predicted Findings: {findings}. "
            f"Describe the findings in this chest X-ray image based on the predicted findings above. "
            f"Provide a detailed radiology report covering all visible abnormalities and normal findings. "
            f"Write one fluent text without enumeration. "
            f"Do not include headers, greetings, or conversational text."
        )
        conv.append_message("USER", prompt)
        conv.append_message("ASSISTANT", None)
        text_input = conv.get_prompt()

        # Prepare image tensor
        image_tensor = self.vis_transforms(image).unsqueeze(0).to(
            self.model.device, dtype=torch.bfloat16
        )
        input_ids = tokenizer_image_token(
            text_input, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                use_cache=True,
                max_new_tokens=300,
                stopping_criteria=[stopping_criteria],
                pad_token_id=self.tokenizer.pad_token_id,
            )

        report = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip().replace("</s>", "")
        return clean_report(report)

