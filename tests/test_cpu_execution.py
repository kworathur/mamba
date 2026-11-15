"""
Test that Mamba PyTorch code runs natively on TPU with minimal modifications and 
profile a naive forward pass of a single mamba block.
"""
from pathlib import Path
import torch
from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams

log_dir = "./profiles/mamba"

seed = 42
torch.manual_seed(seed)

with torch.no_grad():
    batch, length, dim = 2, 16, 16
    x = torch.randn(batch, length, dim)
    model = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim, # Model dimension d_model
        d_state=16,
        d_conv=4,    # Local convolution width
        expand=2,
        use_fast_path=False,
        layer_idx=0)
    inference_params = InferenceParams(max_seqlen=length, max_batch_size=batch, seqlen_offset=1)
    
    # warm-up
    model(x)
