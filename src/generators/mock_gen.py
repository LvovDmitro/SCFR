import time
from typing import List, Optional
from .base import BaseGenerator

class MockGenerator(BaseGenerator):
    """
    Mock generator for testing the evaluation pipeline and metric calculation.
    """
    name = "mock"

    def __init__(self, **kwargs):
        print("[MockGenerator] Loaded mock generator. Will return dummy text.")

    def generate(self, image_paths: List[str], prior_image_paths: Optional[List[str]] = None, context: Optional[str] = None) -> str:
        # Simulate small delay
        time.sleep(0.01)
        return "The heart, pulmonary vasculature and mediastinum are within normal limits. There is no pleural effusion or pneumothorax. No acute cardiopulmonary findings."

