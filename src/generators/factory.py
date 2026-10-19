from typing import Any
from .base import BaseGenerator

def get_generator(name: str, **kwargs: Any) -> BaseGenerator:
    """
    Factory function to get a generator by name.
    
    Supported generators (14 functional):
    - 'r2gen', 'r2gencmn'
    - 'medgemma', 'llava_med', 'llava_rad'
    - 'maira2', 'chexagent'
    - 'libra', 'llm_rg4', 'biomedgpt', 'radialog'
    - 'radfm' (registered but raises NotImplementedError)
    - 'nv_reason_cxr', 'internvl_med', 'rad_infer'

    Removed stubs (no public weights): priorrg, diffrrg, a3net, multip_r2gen
    """
    name = name.lower()
    
    if name == 'r2gen':
        from .r2gen_gen import R2GenGenerator
        return R2GenGenerator(**kwargs)
    elif name == 'r2gencmn':
        from .r2gencmn_gen import R2GenCMNGenerator
        return R2GenCMNGenerator(**kwargs)
    elif name == 'medgemma':
        from .medgemma_gen import MedGemmaGenerator
        return MedGemmaGenerator(**kwargs)
    elif name == 'llava_med':
        from .llava_med_gen import LLaVAMedGenerator
        return LLaVAMedGenerator(**kwargs)
    elif name == 'llava_rad':
        from .llava_rad_gen import LLaVARadGenerator
        return LLaVARadGenerator(**kwargs)
    elif name == 'libra':
        from .libra_gen import LibraGenerator
        return LibraGenerator(**kwargs)
    # Stubs removed: priorrg, diffrrg, a3net (no public weights / not functional)
    elif name == 'llm_rg4':
        from .llm_rg4_gen import LLMRG4Generator
        return LLMRG4Generator(**kwargs)
    elif name == 'maira2':
        from .maira2_gen import MAIRA2Generator
        return MAIRA2Generator(**kwargs)
    # Stub removed: multip_r2gen (not functional)
    elif name == 'chexagent':
        from .chexagent_gen import CheXagentGenerator
        return CheXagentGenerator(**kwargs)
    elif name == 'radfm':
        from .radfm_gen import RadFMGenerator
        return RadFMGenerator(**kwargs)
    elif name == 'biomedgpt':
        from .biomedgpt_gen import BiomedGPTGenerator
        return BiomedGPTGenerator(**kwargs)
    elif name == 'radialog':
        from .radialog_gen import RaDialogGenerator
        return RaDialogGenerator(**kwargs)
    elif name == 'nv_reason_cxr':
        from .nv_reason_cxr_gen import NVReasonCXRGenerator
        return NVReasonCXRGenerator(**kwargs)
    elif name == 'internvl_med':
        from .internvl_med_gen import InternVLMedGenerator
        return InternVLMedGenerator(**kwargs)
    elif name == 'rad_infer':
        from .rad_infer_gen import RadInferGenerator
        return RadInferGenerator(**kwargs)
    elif name == 'mock':
        from .mock_gen import MockGenerator
        return MockGenerator(**kwargs)
    else:
        available = [
            'r2gen', 'r2gencmn', 'medgemma', 'llava_med', 'llava_rad',
            'maira2', 'llm_rg4', 'chexagent', 'radfm', 'biomedgpt', 'radialog',
            'libra', 'nv_reason_cxr', 'internvl_med', 'rad_infer', 'mock',
        ]
        raise ValueError(f"Unknown generator: {name}. Available: {available}")

