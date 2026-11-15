
import jax
import jax.numpy as jnp
from mamba_ssm.modules.mamba_jax_equiv import mamba_forward
# Use your actual profiling dimensions.
batch = 32
seqlen = 128
d_model = 32


with np.load("mamba_params.npz") as saved:
    jax_params = {
        name: jnp.asarray(saved[name])
        for name in saved.files
    }

jax_x = jnp.asarray(np.load("mamba_input.npy"))

# Warm up and compile outside the trace.
y = mamba_forward(jax_x, jax_params, scan_impl="unrolled")
jax.block_until_ready(y)

# Profile repeated forward passes.
with jax.profiler.trace(
    "/tmp/jax_mamba_profile",
    create_perfetto_link=False,
):
    for _ in range(20):
        y = mamba_forward(x, params, scan_impl="unrolled")
        jax.block_until_ready(y)
