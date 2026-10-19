import sys
from pathlib import Path
from typing import List, Optional
import torch
from .base import BaseGenerator, STANDARD_PROMPT, clean_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LIBRA_ROOT = PROJECT_ROOT / "external" / "Libra"

class LibraGenerator(BaseGenerator):
    """
    Libra baseline.
    Requires setting up the Libra repository in external/Libra.
    """
    name = "libra"

    def __init__(self, model_path: str = "X-iZhang/libra-v1.0-7b", model_base: Optional[str] = None, **kwargs):
        if str(LIBRA_ROOT) not in sys.path:
            sys.path.insert(0, str(LIBRA_ROOT))

        # Check that source files are not empty (Git LFS stubs)
        constants_file = LIBRA_ROOT / "libra" / "constants.py"
        if constants_file.exists() and constants_file.stat().st_size == 0:
            raise ImportError(
                f"Libra source files are empty (0 bytes): {constants_file}. "
                f"This likely means Git LFS files were not pulled. "
                f"Run: cd {LIBRA_ROOT} && git lfs pull"
            )

        # Fix for transformers>=5.0: AdamW was removed from transformers.
        import transformers
        if not hasattr(transformers, "AdamW"):
            transformers.AdamW = torch.optim.AdamW

        try:
            from libra.eval.run_libra import load_model
        except ImportError:
            raise ImportError(f"Cannot import Libra. Ensure it is cloned at {LIBRA_ROOT} and dependencies are installed.")
        
        print(f"[LibraGenerator] Loading model from {model_path}...")
        self.reuse_model = load_model(model_path, model_base=model_base)
        print("[LibraGenerator] Model loaded successfully.")

    def generate(self, image_paths: List[str], prior_image_paths: Optional[List[str]] = None, context: Optional[str] = None) -> str:
        from libra.eval import libra_eval
        
        # Libra expects up to two images: [current, prior]
        images_to_pass = [image_paths[0]]
        if prior_image_paths and len(prior_image_paths) > 0:
            images_to_pass.append(prior_image_paths[0])
            
        prompt = STANDARD_PROMPT
        if context:
            prompt += f"\nClinical context: {context}"

        # Inference
        result = libra_eval(
            libra_model=self.reuse_model,
            image_file=images_to_pass,
            query=prompt,
            temperature=0.0,  # greedy
            max_new_tokens=512,
            conv_mode="libra_v1"
        )
        return clean_report(result)

    def generate_batch(self, batch_data: List[dict]) -> List[str]:
        from libra.eval import libra_eval_batch
        from PIL import Image
        
        images_list = []
        queries = []
        for item in batch_data:
            curr_img = Image.open(item["image_paths"][0]).convert("RGB")
            prior_paths = item.get("prior_image_paths", [])
            
            if prior_paths and len(prior_paths) > 0:
                prior_img = Image.open(prior_paths[0]).convert("RGB")
                images_list.append([curr_img, prior_img])
            else:
                images_list.append([curr_img])
                
            prompt = STANDARD_PROMPT
            context = item.get("context")
            if context:
                prompt += f"\nClinical context: {context}"
            queries.append(prompt)
            
        results = libra_eval_batch(
            libra_model=self.reuse_model,
            images=images_list,
            queries=queries,
            temperature=0.0,
            max_new_tokens=512,
            conv_mode="libra_v1"
        )
        return [clean_report(r) for r in results]

