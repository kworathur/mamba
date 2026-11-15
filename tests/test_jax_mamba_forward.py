
import jax
import jax.numpy as jnp
from mamba_ssm.modules.mamba_jax_equiv import mamba_forward
# Use your actual profiling dimensions.
batch = 32
seqlen = 128
d_model = 32

x = jnp.ones((batch, seqlen, d_model), dtype=jnp.float32)

# Warm up and compile outside the trace.
y = mamba_forward(x, params, scan_impl="unrolled")
jax.block_until_ready(y)

# Profile repeated forward passes.
with jax.profiler.trace(
    "/tmp/jax_mamba_profile",
    create_perfetto_link=False,
):
    for _ in range(20):
        y = mamba_forward(x, params, scan_impl="unrolled")
        jax.block_until_ready(y)
