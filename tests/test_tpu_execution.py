"""
Test that Mamba PyTorch code runs natively on TPU with minimal modifications and 
profile a naive forward pass of a single mamba block.
"""
from pathlib import Path
import torch
import torch_tpu
from torch.tpu.profiler import TpuProfilerConfig
from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams

log_dir = "./profiles/mamba"

seed = 42
torch.manual_seed(seed)

with torch.no_grad():
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
    # warm-up
    for i in range(length):
            x_t = x[:,i,:]
            model(x_t[:, None, :], inference_params)
    
    torch.accelerator.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.TPU
        ],
        experimental_config=TpuProfilerConfig(
            run_dir=Path(log_dir),
            host_tracer_level=2,
            device_tracer_level=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir)
    ) as prof:
        for _ in range(10):
            inference_params = InferenceParams(max_seqlen=length, max_batch_size=batch, seqlen_offset=1)
            with torch.profiler.record_function("mamba_forward_linear"):
                for i in range(length):
                    x_t = x[:,i,:]
                    model(x_t[:, None, :], inference_params)
            prof.step()

    _, final_hidden = inference_params.key_value_memory_dict[0]
    assert final_hidden.shape == (batch, dim * 2, dim)