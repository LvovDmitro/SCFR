"""
RadFM: Radiology Foundation Model (LLaMA-13B + 3D ViT).
https://huggingface.co/chaoyi-wu/RadFM
https://github.com/chaoyi-wu/RadFM

STATUS: DEFERRED — requires significant setup:
  1. ~100GB download (split zip archives on HF)
  2. Custom model architecture (MultiLLaMAForCausalLM)
  3. Requires transformers==4.28.1 (conflicts with other generators)
  4. Needs A100 80GB (50GB model at fp16)
  5. Custom image preprocessing (3D ViT, 512x512x4 tensors)

To implement:
  - Download & extract RadFM.z01-z04 + RadFM.zip (Language_files/)
  - Download & extract pytorch_model.zip (pytorch_model.bin, ~50GB)
  - Clone github.com/chaoyi-wu/RadFM, add src/Model/RadFM/ to sys.path
  - Load with MultiLLaMAForCausalLM + custom tokenizer with image tokens
"""
from typing import List, Optional
from .base import BaseGenerator


class RadFMGenerator(BaseGenerator):
    name = "radfm"

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "RadFM generator requires ~100GB download + transformers==4.28.1 "
            "(conflicts with other generators). "
            "See radfm_gen.py docstring for implementation plan. "
            "Remove 'radfm' from generator list to skip."
        )

    def generate(
        self,
        image_paths: List[str],
        prior_image_paths: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> str:
        raise NotImplementedError("RadFM not implemented")

