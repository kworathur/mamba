"""
Test that Mamba PyTorch code runs natively on TPU with minimal modifications
"""

import torch
from mamba_ssm import Mamba

def test_tpu_execution():
    batch, length, dim = 2, 64, 16
    x = torch.randn(batch, length, dim).to("tpu")
    model = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim, # Model dimension d_model
        d_state=16,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
    ).to("tpu")
    y = model(x)
    assert y.shape == x.shape