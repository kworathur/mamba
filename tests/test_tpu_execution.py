"""
Test that Mamba PyTorch code runs natively on TPU with minimal modifications
"""

import torch
from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams
seed = 42


torch.manual_seed(seed)
batch, length, dim = 2, 128, 16
x = torch.randn(batch, length, dim).to("tpu")
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,
    use_fast_path=False,
    layer_idx=0).to("tpu")
inference_params = InferenceParams(max_seqlen=length, max_batch_size=batch, seqlen_offset=1)
for i in range(length):
    x_t = x[:,i,:]
    model(x_t[:, None, :], inference_params)

_, final_hidden = inference_params.key_value_memory_dict[0]
print(final_hidden.shape)
assert final_hidden.shape == (batch, 2 * dim, dim)


