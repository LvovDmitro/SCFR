"""
LLM-RG4 generator (AAAI 2025). Uses Vicuna-7b, rad-dino, BiomedVLP-CXR-BERT.
"""
import sys
from pathlib import Path
from typing import List, Optional
import torch
from .base import BaseGenerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LLM_RG4_ROOT = PROJECT_ROOT / "external" / "LLM-RG4"
HF_DIR = LLM_RG4_ROOT / "hf"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "llm_rg4"


def _ensure_path(p: Path) -> str:
    return str(p.resolve())


class LLMRG4Generator(BaseGenerator):
    name = "llm_rg4"

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        **kwargs,
    ):
        if str(LLM_RG4_ROOT) not in sys.path:
            sys.path.insert(0, str(LLM_RG4_ROOT))

        ckpt = checkpoint_path or _ensure_path(CKPT_DIR / "LLM_RG4_OF_checkpoint_epoch1_step33464_bleu0.258413_cider0.777104_chexbert0.554402.pth")
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"LLM-RG4 checkpoint not found: {ckpt}")

        self.device = device
        self._model = None
        self._processor = None
        self._ckpt = ckpt
        self._load_model()

    def _load_model(self):
        from transformers import AutoImageProcessor

        # Fix for transformers>=5.0: AdamW was removed from transformers.
        # LLM-RG4's DiscBERT code imports it. Monkey-patch before import.
        import transformers
        if not hasattr(transformers, "AdamW"):
            transformers.AdamW = torch.optim.AdamW

        vicuna_path = _ensure_path(HF_DIR / "vicuna-7b-v1.5")
        rad_dino_path = _ensure_path(HF_DIR / "rad-dino")
        cxr_bert_path = _ensure_path(HF_DIR / "BiomedVLP-CXR-BERT-specialized")
        bert_path = _ensure_path(HF_DIR / "bert-base-uncased")

        if not all(Path(p).exists() for p in [vicuna_path, rad_dino_path, cxr_bert_path]):
            raise FileNotFoundError(
                "LLM-RG4 HF models missing. Run: download LLM-RG4 dependencies to external/LLM-RG4/hf/"
            )

        # Check that source files are not empty (Git LFS stubs)
        model_src = LLM_RG4_ROOT / "models" / "LLM_RG4.py"
        if model_src.exists() and model_src.stat().st_size == 0:
            raise ImportError(
                f"LLM-RG4 source file is empty (0 bytes): {model_src}. "
                f"This likely means Git LFS files were not pulled. "
                f"Run: cd {LLM_RG4_ROOT} && git lfs pull"
            )

        self._processor = AutoImageProcessor.from_pretrained(rad_dino_path)

        # Build minimal args for LLM_RG4 (use Namespace for save_hyperparameters)
        from argparse import Namespace
        args = Namespace(
            rad_dino_path=rad_dino_path,
            cxr_bert_path=cxr_bert_path,
            vicuna_model=vicuna_path,
            bert_path=bert_path,
            chexbert_path=_ensure_path(CKPT_DIR / "discbert.pth"),
            freeze_vm=True,
            stage_class=2,
            llm_use_lora=True,
            llm_r=32,
            llm_alpha=64,
            lora_dropout=0.1,
            visual_token_number=128,
            max_length=100,
            beam_size=3,
            do_sample=False,
            min_new_tokens=50,
            max_new_tokens=200,
            repetition_penalty=2.0,
            length_penalty=2.0,
            temperature=0.0,
            delta_file=self._ckpt,
            visual_delta_file=None,
            low_resource=False,
            end_sym="</s>",
        )

        from models.LLM_RG4 import LLM_RG4
        self._model = LLM_RG4(args)
        self._model.eval()
        self._model = self._model.to(self.device)
        print(f"[LLMRG4Generator] Loaded from {self._ckpt}")

    def _prepare_batch(self, image_paths: List[str], sample_id: str = "0"):
        from PIL import Image

        img_path = image_paths[0] if image_paths else ""
        if not img_path or not Path(img_path).exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt")
        pixel_values = inputs.data["pixel_values"]
        image_tensor = pixel_values.to(self.device)

        batch = {
            "id": [sample_id],
            "dataset_id": ["sn"],
            "finding_flag": [True],
            "impression_flag": [True],
            "APPA_flag": ["PA"],
            "h_i": [""],
            "input_text": [""],
            "image": image_tensor,
        }
        return batch

    def generate(
        self,
        image_paths: List[str],
        prior_image_paths: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> str:
        batch = self._prepare_batch(image_paths, sample_id="0")
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            hypo, _ = self._model.test_step(batch, 0)
        return hypo[0] if hypo else ""

