
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial


def silu(x):
    return x * jax.nn.sigmoid(x)


def selective_scan_unrolled(dA, h, dB, x, C, D, z, out_w, out_b):
    """
    Direct translation of seq_encode_scan().

    dA, dB: [B, L, H, N]
    h:      [B, H, N]
    x, z:   [B, H, L]
    C:      [B, L, N]
    D:      [H]
    """
    seqlen = dA.shape[1]
    intermediate_states = []

    for t in range(seqlen):
        h = (
            dA[:, t, :, :] * h
            + dB[:, t, :, :] * x[:, :, t, None]
        )
        intermediate_states.append(h)

    h_all = jnp.stack(intermediate_states, axis=1)

    y = jnp.einsum("bldn,bln->bdl", h_all, C)
    y = y + D[None, :, None] * x
    y = y * silu(z)

    y = jnp.transpose(y, (0, 2, 1))
    out = y @ out_w.T

    if out_b is not None:
        out = out + out_b

    return out


def selective_scan_lax(dA, h, dB, x, C, D, z, out_w, out_b):
    """
    Same recurrence, expressed with lax.scan instead
    of a Python loop.
    """
    # Move sequence dimension to leading axis.
    dA_t = jnp.swapaxes(dA, 0, 1)
    dB_t = jnp.swapaxes(dB, 0, 1)
    x_t = jnp.transpose(x, (2, 0, 1))

    def step(h, inputs):
        a_t, b_t, x_t = inputs
        h = a_t * h + b_t * x_t[..., None]
        return h, h

    _, states = lax.scan(
        step, h, (dA_t, dB_t, x_t)
    )

    h_all = jnp.swapaxes(states, 0, 1)

    y = jnp.einsum("bldn,bln->bdl", h_all, C)
    y = y + D[None, :, None] * x
    y = y * silu(z)

    y = jnp.transpose(y, (0, 2, 1))
    out = y @ out_w.T

    if out_b is not None:
        out = out + out_b

    return out


@partial(jax.jit, static_argnames=("scan_impl",))
def mamba_forward(hidden_states, p, scan_impl="unrolled"):
    """
    JAX translation of Mamba.forward().

    hidden_states: [B, L, d_model]

    p: dictionary of PyTorch Mamba parameters.
    """
    batch, seqlen, d_model = hidden_states.shape
    d_inner = p["in_proj.weight"].shape[0] // 2
    d_state = p["A_log"].shape[1]
    dt_rank = p["dt_proj.weight"].shape[1]

    # 1. Input projection
    # Original: in_proj.weight @ rearrange(hidden_states, ...)
    xz = hidden_states @ p["in_proj.weight"].T
    xz = jnp.transpose(xz, (0, 2, 1))

    if "in_proj.bias" in p:
        xz = xz + p["in_proj.bias"][None, :, None]

    A = -jnp.exp(p["A_log"].astype(jnp.float32))

    x, z = jnp.split(xz, 2, axis=1)

    # 2. Depthwise causal convolution
    # PyTorch Conv1d weight: [d_inner, 1, d_conv]
    # PyTorch padding=d_conv-1, followed by [:seqlen]
    d_conv = p["conv1d.weight"].shape[-1]

    x = lax.conv_general_dilated(
        lhs=x,
        rhs=p["conv1d.weight"],
        window_strides=(1,),
        padding=((d_conv - 1, d_conv - 1),),
        dimension_numbers=("NCW", "OIW", "NCW"),
        feature_group_count=d_inner,
    )

    x = x[..., :seqlen]

    if "conv1d.bias" in p:
        x = x + p["conv1d.bias"][None, :, None]

    x = silu(x)

    # 3. Input-dependent SSM parameters
    x_flat = jnp.transpose(x, (0, 2, 1))
    x_flat = x_flat.reshape(batch * seqlen, d_inner)

    x_dbl = x_flat @ p["x_proj.weight"].T

    dt, B, C = jnp.split(
        x_dbl,
        [dt_rank, dt_rank + d_state],
        axis=-1,
    )

    dt = dt @ p["dt_proj.weight"].T
    dt = dt.reshape(batch, seqlen, d_inner)

    B = B.reshape(batch, seqlen, d_state)
    C = C.reshape(batch, seqlen, d_state)

    # 4. Discretization
    dt = jax.nn.softplus(
        dt + p["dt_proj.bias"][None, None, :]
    )

    dA = jnp.exp(jnp.einsum("bld,dn->bldn", dt, A))
    dB = jnp.einsum("bld,bln->bldn", dt, B)

    # 5. Selective scan
    h = jnp.zeros(
        (batch, d_inner, d_state),
        dtype=hidden_states.dtype,
    )

    scan_fn = (
        selective_scan_unrolled
        if scan_impl == "unrolled"
        else selective_scan_lax
    )

    out = scan_fn(
        dA, h, dB, x, C,
        p["D"],
        z,
        p["out_proj.weight"],
        p.get("out_proj.bias"),
    )

    return out
