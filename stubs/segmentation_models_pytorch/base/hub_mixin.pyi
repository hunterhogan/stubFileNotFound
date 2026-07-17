import torch
from huggingface_hub import ModelCard, PyTorchModelHubMixin
from pathlib import Path

MODEL_CARD: str

class SMPHubMixin(PyTorchModelHubMixin):
    def generate_model_card(self, *args, **kwargs) -> ModelCard: ...
    def save_pretrained(self, save_directory: str | Path, *args, **kwargs) -> str | None: ...
    @property
    @torch.jit.unused
    def config(self) -> dict: ...

def from_pretrained(pretrained_model_name_or_path: str, *args, strict: bool = True, **kwargs): ...
def supports_config_loading(func):
    """Decorator to filter special config kwargs"""
