import sys
from pathlib import Path
from typing import List, Optional
import torch
from .base import BaseGenerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
R2GEN_ROOT = PROJECT_ROOT / "external" / "R2Gen"
sys.path.insert(0, str(R2GEN_ROOT))

class R2GenGenerator(BaseGenerator):
    """
    R2Gen baseline.
    """
    name = "r2gen"

    def __init__(self, checkpoint_path: Optional[Path] = None, device: str = "cuda", **kwargs):
        from models.r2gen_generator import load_r2gen_generator
        
        dataset_name = kwargs.get("dataset_name", "mimic")
        if dataset_name == "iuxray":
            default_ckpt = PROJECT_ROOT / "checkpoints" / "r2gen" / "model_iu_xray.pth"
        else:
            default_ckpt = PROJECT_ROOT / "checkpoints" / "r2gen" / "model_mimic_cxr.pth"
            
        ckpt = checkpoint_path or default_ckpt
        self._fn = load_r2gen_generator(checkpoint_path=ckpt, device=device, dataset_name=dataset_name)
        print(f"[R2GenGenerator] Loaded from {ckpt}")

    def generate(self, image_paths: List[str], prior_image_paths: Optional[List[str]] = None, context: Optional[str] = None) -> str:
        # R2Gen doesn't use prior images or context by default
        return self._fn(image_paths)

