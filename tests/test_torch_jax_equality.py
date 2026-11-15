
import torch
import numpy as np
import jax

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba_jax_equiv import mamba_forward

# Float32 simplifies initial numerical validation.
model = Mamba(
    d_model=32,
    d_state=16,
    d_conv=4,
    expand=2,
    use_fast_path=False,
).eval()

torch_input = torch.randn(2, 128, 32)
params = torch_params_to_jax(model)
jax_input = jnp.asarray(torch_input.numpy())

with torch.no_grad():
    torch_output = model(torch_input)

jax_output = mamba_forward(
    jax_input, params, scan_impl="unrolled"
)
jax.block_until_ready(jax_output)

np.testing.assert_allclose(
    np.asarray(jax_output),
    torch_output.detach().cpu().numpy(),
    rtol=1e-4,
    atol=1e-4,
)

# Confirm equivalence of the two JAX scans.
lax_output = mamba_forward(
    jax_input, params, scan_impl="lax"
)

np.testing.assert_allclose(
    np.asarray(lax_output),
    np.asarray(jax_output),
    rtol=1e-4,
    atol=1e-4,
)

print("Outputs match")
