import sys
from pathlib import Path
from typing import List, Optional
import torch
from .base import BaseGenerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class R2GenCMNGenerator(BaseGenerator):
    """
    R2GenCMN baseline.
    """
    name = "r2gencmn"

    def __init__(self, checkpoint_path: Optional[Path] = None, device: str = "cuda", **kwargs):
        self.device = device
        self.dataset_name = kwargs.get("dataset_name", "mimic")
        self._load(checkpoint_path)

    def _load(self, checkpoint_path: Optional[Path]):
        from mdcm.r2gencmn import R2GenCMNModel, build_r2gencmn_args

        if self.dataset_name == "iuxray":
            default_ckpt = PROJECT_ROOT / "checkpoints" / "r2gencmn" / "model_iu_xray.pth"
        else:
            default_ckpt = PROJECT_ROOT / "checkpoints" / "r2gencmn" / "model_mimic_cxr.pth"

        cmn_ckpt = checkpoint_path or default_ckpt
        args = build_r2gencmn_args(dataset_name=self.dataset_name)

        sys.path.insert(0, str(PROJECT_ROOT / "external" / "R2Gen"))
        from modules.tokenizers import Tokenizer

        tokenizer = Tokenizer(args)
        vocab_size = len(tokenizer.token2idx)
        print(f"[R2GenCMNGenerator] Tokenizer vocab size: {vocab_size}, dataset: {self.dataset_name}")
        self.model = R2GenCMNModel(args, tokenizer)

        if cmn_ckpt.exists():
            import os
            print(f"[R2GenCMNGenerator] Found checkpoint at {cmn_ckpt}, size: {os.path.getsize(cmn_ckpt)} bytes")
            try:
                try:
                    state = torch.load(cmn_ckpt, map_location="cpu", weights_only=False)
                except TypeError:
                    state = torch.load(cmn_ckpt, map_location="cpu")
                sd = state.get("state_dict", state)
                # Check vocab compatibility (model uses tgt_vocab = vocab_size + 1)
                if "encoder_decoder.logit.weight" in sd:
                    ckpt_vocab = sd["encoder_decoder.logit.weight"].shape[0]
                    model_vocab = vocab_size + 1
                    if ckpt_vocab != model_vocab:
                        raise RuntimeError(
                            f"Vocab mismatch: checkpoint expects tgt_vocab={ckpt_vocab} "
                            f"but tokenizer built tgt_vocab={model_vocab}. "
                            f"Need correct vocab JSON or annotation.json matching the training data."
                        )
                    else:
                        print(f"[R2GenCMNGenerator] Vocab OK: ckpt tgt_vocab={ckpt_vocab} == model tgt_vocab={model_vocab}")
                self.model.load_state_dict(sd, strict=False)
                print(f"[R2GenCMNGenerator] Loaded CMN checkpoint {cmn_ckpt}")
            except Exception as e:
                print(f"[R2GenCMNGenerator] Failed to load checkpoint: {e}")
                print("[R2GenCMNGenerator] WARNING: using random init instead")
        else:
            print("[R2GenCMNGenerator] WARNING: No checkpoint found, using random init")

        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(self, image_paths: List[str], prior_image_paths: Optional[List[str]] = None, context: Optional[str] = None) -> str:
        from torchvision import transforms
        from PIL import Image

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        imgs = []
        for p in image_paths[:2]:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(transform(img))
            except Exception as e:
                print(f"[R2GenCMNGenerator] Failed to open image {p}: {e}")
                imgs.append(torch.zeros(3, 224, 224))

        if not imgs:
            print("[R2GenCMNGenerator] No images loaded, returning empty")
            return ""

        # IU-Xray forward expects [B, 2, C, H, W] (two views: frontal + lateral)
        # MIMIC forward expects [B, C, H, W] (single image, no view dimension)
        if self.dataset_name == "iuxray":
            # Pad to 2 images for IU-Xray (duplicate if only 1)
            while len(imgs) < 2:
                imgs.append(imgs[0].clone())
            img_tensor = torch.stack(imgs).unsqueeze(0).to(self.device)  # [1, 2, C, H, W]
        else:
            # MIMIC: use only the first (frontal) image, no view dimension
            img_tensor = imgs[0].unsqueeze(0).to(self.device)  # [1, C, H, W]

        try:
            with torch.no_grad():
                out = self.model(img_tensor, mode="sample")
                reports = self.model.tokenizer.decode_batch(out.cpu().numpy())
        except Exception as e:
            print(f"[R2GenCMNGenerator] Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return ""

        report = reports[0] if reports else ""
        if not report.strip():
            print(f"[R2GenCMNGenerator] WARNING: empty report generated (dataset={self.dataset_name})")

        seen = set()
        clean_sents = []
        for s in report.split(". "):
            s = s.strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                clean_sents.append(s)
        return ". ".join(clean_sents)

