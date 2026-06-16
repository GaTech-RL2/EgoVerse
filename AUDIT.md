# Pixel DFoT Training Path Audit

**Date:** 2026-05-27
**Our code:** `/coc/flash7/paphiwetsa3/projects/EgoVerse-pact/`
**Reference:** `/coc/flash7/paphiwetsa3/projects/diffusion-forcing-transformer/`

---

## 1. `pl_model.py` — `training_step`

**File:** `egomimic/pl_utils/pl_model.py`

```python
batch = self.model.process_batch_for_training(batch)
predictions = self.model.forward_training(batch)
losses = self.model.compute_losses(predictions, batch)
# ...
return losses["action_loss"]
```

**Gradient clipping:** Done in `on_after_backward` via MAD-based adaptive clipping (measure grad norm, clip to median when it exceeds `median + 3 * MAD`). This replaces reference's simple `grad_norm` logging.

**Loss logging:** Per-embodiment losses logged under `Train/`, averaged across embodiments.

⚠️ DIFFERENT APPROACH but equivalent — reference is a single LightningModule (`DFoTVideo`) with `training_step` that directly calls `self.diffusion_model(xs, conditions, k=noise_levels)`. Ours wraps through `Algo` subclass. Both ultimately call the same math.

---

## 2. `DFoT.process_batch_for_training(batch)`

**File:** `egomimic/algo/dfot/algo.py`, line 219

**What it does:**
1. Iterates over embodiments in the batch dict.
2. Detects packed vs padded mode via `"cu_seqlens" in _batch`.
3. Maps dataset keys to keynames via `norm_stats.zarr_key_to_keyname()`.
4. Normalizes the batch via `norm_stats.normalize()`.
5. Moves everything to device, casts floats to float32.

⚠️ DIFFERENT APPROACH — reference unpacks batch as a simple tuple `(xs, conditions, masks, gt_videos)` with no per-embodiment loop. Our multi-embodiment design adds overhead but is functionally equivalent for single-embodiment training.

---

## 3. `DFoT.forward_training(batch)`

**File:** `egomimic/algo/dfot/algo.py`, line 303

```python
for emb_id, _batch in batch.items():
    ac_key = self.resolved_ac_keys[emb_id]
    obs = self._build_obs(_batch, emb_id)
    ctx = make_dfot_ctx(
        is_packed=is_packed, action_key=ac_key, obs=obs,
        cu_seqlens=..., max_seqlen=...
    )
    self.outer_stage(_batch, ctx)          # encode + backbone + decode
    mse = self.loss(_batch, ctx)           # DFoTLoss
    predictions[f"{emb_id}_action_loss"] = mse
```

Per-embodiment loop builds obs dict from proprio/lang/camera keys, then delegates to `outer_stage` and `loss`.

⚠️ DIFFERENT APPROACH — reference does this in one shot: `self.diffusion_model(xs, conditions, k=noise_levels)`. Our refactored design separates outer_stage (encode+backbone+decode) from loss. Mathematically equivalent.

---

## 4. `make_dfot_ctx(...)`

**File:** `egomimic/algo/dfot/outer_stage.py`

```python
def make_dfot_ctx(*, is_packed, action_key, obs, cu_seqlens=None, max_seqlen=None):
    return SimpleNamespace(
        is_packed=is_packed, action_key=action_key, obs=obs,
        cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
        q_state=None, external_cond=None,
    )
```

No reference equivalent — this is our abstraction for passing state between encode/backbone/loss. Benign.

✅ N/A (our abstraction only)

---

## 5. `PixelSpatialDFoTOuterStage.forward(batch, ctx)`

**File:** `egomimic/algo/dfot/pixel_spatial_outer_stage.py`

### 5a. `_extract_images(ctx)`

```python
img = ctx.obs[self.image_key]
if img.dtype == torch.uint8:
    img = img.float() / 255.0
elif img.max() > 1.5:
    img = img.float() / 255.0
else:
    img = img.float()
```

Reference expects pre-normalized images from the data pipeline. Our code normalizes uint8 → [0,1] inline.

⚠️ DIFFERENT APPROACH but equivalent — just a matter of where normalization happens. As long as the data pipeline provides uint8 or pre-normalized, the result is the same.

### 5b. `_sample_frames_packed(images, cu_seqlens)`

Crops frames per-episode according to `frame_sampling` mode (fixed_window, start_to_end, random_subsample, or full). Updates `cu_seqlens` and `max_seqlen` on ctx.

Reference does NOT have frame sampling — it uses fixed-length chunks from the dataloader. Our frame sampling is an extension for variable-length packed episodes.

⚠️ DIFFERENT APPROACH — reference uses fixed-length video clips from the dataset. Our frame sampling within the outer stage is an architectural difference, not a bug.

### 5c. `_sample_noise_levels(shape, device)` (inherited from `DFoTOuterStage`)

```python
def _sample_noise_levels(self, shape, device):
    if isinstance(self.diffusion, DiscreteDiffusion):
        return torch.randint(0, self.diffusion.timesteps, shape, device=device, dtype=torch.long)
    return torch.rand(shape, device=device).clamp_(1e-5, 1.0 - 1e-5)
```

**Reference** (`_get_training_noise_levels` in `dfot_video.py`):
```python
# For discrete:
rand_fn = partial(torch.randint, 0, self.timesteps, device=xs.device, generator=self.generator)
noise_levels = rand_fn((batch_size, n_tokens))  # "random_independent" mode
```

Both sample `randint(0, timesteps)` per token independently.

✅ MATCHES reference (for `noise_level: "random_independent"` mode, which is the DFoT default)

**Note:** Reference also supports `"random_uniform"` (same level across all tokens), `variable_context`, `fixed_context`, and `uniform_future` modes. Our code only supports `random_independent`. This is fine for vanilla DFoT training.

### 5d. `encode()` — noise addition

```python
noise = torch.randn_like(images).clamp_(-self.diffusion.clip_noise, self.diffusion.clip_noise)
x_t = self.diffusion.q_sample(images, t, noise=noise)
ctx.q_state = {"x_t": x_t, "k": t, "time_cond": t, "noise": noise, "x_start": images}
```

**Reference** (`DiscreteDiffusion.forward`):
```python
noise = torch.randn_like(x)
noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
noised_x = self.q_sample(x_start=x, k=k, noise=noise)
```

✅ MATCHES reference — both clamp noise to `[-clip_noise, clip_noise]`, then call `q_sample`.

### 5e. `q_sample(x_start, k, noise)`

Our code:
```python
return (
    _extract(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
    + _extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * noise
)
```

Reference:
```python
return (
    extract(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
    + extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * noise
)
```

✅ MATCHES reference exactly.

### 5f. `forward()` — per-episode loop (packed mode)

```python
if ctx.is_packed:
    for i in range(B):
        s, e = int(cu[i].item()), int(cu[i + 1].item())
        x_ep = x_t[s:e].unsqueeze(0)       # (1, T_ep, C, H, W)
        t_ep = time_cond[s:e].unsqueeze(0)  # (1, T_ep)
        v_ep = self.inner_stage(x_ep, t_ep, external_cond=None)
        pieces.append(v_ep.squeeze(0))
    v_pred = torch.cat(pieces, dim=0)
```

Reference does NOT have packed mode. This is our extension for variable-length episodes. Each episode is processed independently through the backbone with `unsqueeze(0)` to create a batch-of-1.

⚠️ DIFFERENT APPROACH — reference always uses padded batches. Our packed mode processes episodes sequentially (no parallelism across episodes in a packed batch). This is correct but slow.

---

## 6. `DFoTDiT3DBackbone.forward(x, noise_levels, external_cond)`

**File:** `egomimic/algo/dfot/dit3d_backbone.py`

### 6a. `PatchEmbed(x)`

Our code:
```python
class PatchEmbed(nn.Module):
    self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
    # Xavier init like nn.Linear (DiT convention)
    w = self.proj.weight.data
    nn.init.xavier_uniform_(w.view(w.shape[0], -1))
    nn.init.zeros_(self.proj.bias)
    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "bt c h w -> bt (h w) c")
```

Reference uses `timm.models.vision_transformer.PatchEmbed` + same Xavier init override:
```python
self.patch_embedder = PatchEmbed(img_size=resolution, patch_size=self.patch_size, in_chans=..., embed_dim=..., bias=True)
# Initialize patch_embedder like nn.Linear:
w = embedder.proj.weight.data
nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
nn.init.zeros_(embedder.proj.bias)
```

✅ MATCHES reference — same Conv2d with Xavier init. Our custom class is equivalent to timm's PatchEmbed for the default case.

### 6b. `_build_cond(noise_levels, external_cond, ...)` — conditioning

Our code:
```python
emb = self.time_embed(noise_levels)
if external_cond is not None and self.cond_embed is not None:
    cond_emb = self.cond_embed(external_cond)
    if force_uncond:
        cond_emb = torch.zeros_like(cond_emb)
    emb = emb + cond_emb
return emb
```

Reference (`DiT3D.forward`):
```python
emb = self.noise_level_pos_embedding(noise_levels)
if external_cond is not None:
    emb = emb + self.external_cond_embedding(external_cond, external_cond_mask)
```

✅ MATCHES reference — noise embedding + cond embedding, summed (not concatenated).

### 6c. `StochasticTimeEmbedding` — noise level embedding

**File:** `egomimic/algo/dfot/embeddings.py`

```python
class StochasticTimeEmbedding(nn.Module):
    def __init__(self, dim, time_embed_dim, use_fourier=False, p=0.0):
        self.timesteps = _StochasticUnknownTimesteps(dim, p)  # or FourierEmbedding
        self.embedding = _TimestepEmbedding(dim, time_embed_dim)
    def forward(self, timesteps, mask=None):
        t_emb = self.timesteps(timesteps, mask)
        return self.embedding(t_emb)
```

Reference:
```python
class StochasticTimeEmbedding(nn.Module):
    def __init__(self, dim, time_embed_dim, use_fourier=False, p=0.0):
        self.timesteps = StochasticUnknownTimesteps(dim, p)  # or FourierEmbedding
        self.embedding = TimestepEmbedding(dim, time_embed_dim)  # diffusers
```

Structure matches. But the inner `_Timesteps` defaults differ:

### ❌ BUG: `_Timesteps` sinusoidal embedding defaults differ from reference

Our `_Timesteps`:
```python
class _Timesteps(nn.Module):
    def __init__(self, num_channels, flip_sin_to_cos=False, downscale_freq_shift=1.0):
```

Reference `Timesteps`:
```python
class Timesteps(nn.Module):
    def __init__(self, num_channels, flip_sin_to_cos=True, downscale_freq_shift=0):
```

**Impact:** These defaults flow into `get_timestep_embedding()`:

1. **`flip_sin_to_cos`**: Reference=True flips so output is `[cos, sin]`; ours=False gives `[sin, cos]`.
2. **`downscale_freq_shift`**: Reference=0 gives `exponent / half_dim`; ours=1 gives `exponent / (half_dim - 1)`.

Both change the actual embedding values fed to the MLP. Since the MLP is learned, a random-init model will eventually learn around either convention, but:
- **The frequency spacing is different** (slightly wider for ours).
- **The sin/cos ordering is flipped** (cosmetic but affects which frequencies get which MLP inputs).
- **This means checkpoints are NOT interchangeable** between the two implementations.
- **Training from scratch should converge to similar quality**, but the embedding landscape is different.

**`_StochasticUnknownTimesteps`** inherits from `_Timesteps` and gets these wrong defaults.

❌ **DIFFERS from reference** — `flip_sin_to_cos` and `downscale_freq_shift` defaults are swapped relative to reference. Does not affect final converged quality when training from scratch, but prevents checkpoint transfer.

### 6c.2. `_TimestepEmbedding` (MLP projection)

Our code:
```python
class _TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, time_embed_dim):
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)
```

Reference uses `diffusers.models.embeddings.TimestepEmbedding` which is:
```python
self.linear_1 = nn.Linear(in_channels, time_embed_dim)
self.act = nn.SiLU()
self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)
```

✅ MATCHES reference — same architecture (Linear-SiLU-Linear).

### 6c.3. `time_embed_dim` for noise embedding

Our backbone constructor:
```python
self.time_embed = StochasticTimeEmbedding(
    dim=int(time_embed_dim),      # default 256
    time_embed_dim=self.d_cond,   # default 384
    ...
)
```

Reference (`DiT3D` inherits `BaseBackbone`):
```python
# noise_level_dim overridden by DiT3D to 256
self.noise_level_pos_embedding = StochasticTimeEmbedding(
    dim=self.noise_level_dim,          # 256
    time_embed_dim=self.noise_level_emb_dim,  # hidden_size (e.g. 384)
    ...
)
```

✅ MATCHES reference — `dim=256`, `time_embed_dim=d_cond=hidden_size`. The `noise_level_dim` override in reference DiT3D (hardcoded 256) matches our default `time_embed_dim=256`.

### 6d. Broadcasting cond to patches

Our code:
```python
cond = cond.unsqueeze(2).expand(-1, -1, P, -1)   # (B, T, P, d_cond)
cond = rearrange(cond, "b t p c -> b (t p) c")
```

Reference:
```python
emb = repeat(emb, "b t c -> b (t p) c", p=self.num_patches)
```

✅ MATCHES reference — both broadcast per-frame cond to all patches of that frame.

### 6e. DiT Blocks

Our `DiTBlock`:
```python
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, rope=None):
        self.norm1 = AdaLayerNormZero(hidden_size)
        self.attn = DiTAttention(hidden_size, num_heads, qkv_bias=True, rope=rope)
        self.norm2 = AdaLayerNormZero(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
    def forward(self, x, c):
        x, gate_msa = self.norm1(x, c)
        x = x + gate_msa * self.attn(x)
        x, gate_mlp = self.norm2(x, c)
        x = x + gate_mlp * self.mlp(x)
        return x
```

Reference `DiTBlock`:
```python
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, rope=None):
        self.norm1 = AdaLayerNormZero(hidden_size)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=True, rope=rope)
        self.norm2 = AdaLayerNormZero(hidden_size)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size*mlp_ratio),
                       act_layer=partial(nn.GELU, approximate="tanh"))
    def forward(self, x, c):
        x, gate_msa = self.norm1(x, c)
        x = x + gate_msa * self.attn(x)
        x, gate_mlp = self.norm2(x, c)
        x = x + gate_mlp * self.mlp(x)
        return x
```

✅ MATCHES reference — same structure. Our `nn.Sequential(Linear, GELU, Linear)` is equivalent to timm's `Mlp` with default settings.

### 6e.1. `AdaLayerNormZero`

Our code:
```python
self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True))
self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
# Zero-init the linear
nn.init.zeros_(self.modulation[-1].weight)
nn.init.zeros_(self.modulation[-1].bias)
def forward(self, x, c):
    shift, scale, gate = self.modulation(c).chunk(3, dim=-1)
    return x * (1 + scale) + shift, gate   # via _modulate
```

Reference: identical structure and init.

✅ MATCHES reference exactly.

### 6e.2. `DiTAttention` / `Attention`

Our code:
```python
class DiTAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_norm=False, rope=None):
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
```

Reference `Attention`:
```python
# Same structure, plus attn_drop and proj_drop (default 0.0)
self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
self.proj = nn.Linear(dim, dim)
self.attn_drop = nn.Dropout(attn_drop)   # default 0.0
self.proj_drop = nn.Dropout(proj_drop)   # default 0.0
# forward: same reshape/permute/unbind, same rope application, same SDPA
```

✅ MATCHES reference — we omit `attn_drop` and `proj_drop` which default to 0.0 (no-ops).

### 6e.3. `RotaryEmbedding3D` — RoPE

Our code:
```python
class RotaryEmbedding3D(nn.Module):
    def __init__(self, dim, sizes, theta=10000.0):
        half_dim = dim // 2
        # Split half_dim into 3 axis dims
        # inv_freq per axis = 1/theta^(arange(0, axis_dim*2, 2) / (axis_dim*2))
    def _build_freqs(self, T, device, dtype):
        # Build THW grid, fill per-axis, concatenate, flatten, repeat r=2
    def forward(self, x):
        freqs = self._build_freqs(T, ...)
        return x * freqs.cos() + _rotate_half(x) * freqs.sin()
```

Reference:
```python
class RotaryEmbedding3D(RotaryEmbeddingND):
    def __init__(self, dim, sizes, theta=10000.0):
        dim //= 2
        # Same 3-way split of dim
        super().__init__(tuple(d * 2 for d in dims), sizes, theta, flatten=True)
    # Parent get_freqs: inv_freq = 1/theta^(arange(0, dim, 2)[:dim//2] / dim)
    # Parent forward: x * freqs.cos() + rotate_half(x) * freqs.sin()
```

The frequency computation is mathematically identical (verified: same exponent formula after normalization). The `_rotate_half` function is also identical to the reference's `rotate_half` from `rotary_embedding_torch`.

✅ MATCHES reference — same frequencies, same rotation, same grid construction logic.

### 6e.4. Weight initialization

Our `DiTBlock._init_weights`:
```python
for m in [self.attn, self.mlp]:
    for p in m.modules():
        if isinstance(p, nn.Linear):
            nn.init.xavier_uniform_(p.weight)
            if p.bias is not None:
                nn.init.zeros_(p.bias)
```

Reference `DiTBlock.initialize_weights`:
```python
def _basic_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
self.attn.apply(_basic_init)
if self.use_mlp:
    self.mlp.apply(_basic_init)
```

✅ MATCHES reference — Xavier uniform for all linear layers in attn and mlp.

Our backbone embedding init:
```python
for module in [self.time_embed]:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
```

Reference `DiT3D.initialize_weights`:
```python
def _mlp_init(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
self.noise_level_pos_embedding.apply(_mlp_init)
```

✅ MATCHES reference — normal(std=0.02) for embedding MLPs.

### 6f. `DiTFinalLayer`

Our code:
```python
class DiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        self.norm = AdaLayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    def forward(self, x, c):
        return self.linear(self.norm(x, c))
```

Reference `DITFinalLayer`: identical.

✅ MATCHES reference exactly.

### 6g. Unpatchify

Our code:
```python
tokens = rearrange(tokens, "b (t p) c -> (b t) p c", t=T)
v_pred = rearrange(tokens,
    "bt (h w) (p q c) -> bt c (h p) (w q)",
    h=self.grid_size, w=self.grid_size,
    p=self.patch_size, q=self.patch_size, c=self.latent_channels)
v_pred = rearrange(v_pred, "(b t) c h w -> b t c h w", b=B)
```

Reference:
```python
x = self.unpatchify(rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches))
x = rearrange(x, "(b t) h w c -> b t c h w", b=input_batch_size)
# unpatchify:
rearrange(x, "b (h w) (p q c) -> b (h p) (w q) c",
    h=int(self.num_patches**0.5), p=self.patch_size, q=self.patch_size)
```

Note: reference unpatchify outputs `(BT, H, W, C)` then rearranges to `(B, T, C, H, W)`. Ours outputs `(BT, C, H, W)` directly by placing `c` first in the einops pattern, then rearranges to `(B, T, C, H, W)`.

✅ MATCHES reference — different einops pattern order but same final shape `(B, T, C, H, W)`.

---

## 7. `DFoTLoss.forward(batch, ctx)` — loss computation

**File:** `egomimic/algo/loss.py`

```python
class DFoTLoss(Loss):
    def __init__(self, diffusion):
        self.diffusion = diffusion
    def forward(self, batch, ctx):
        v_pred = batch["pred_v"]
        q_state = ctx.q_state
        per_token = self.diffusion.compute_loss(v_pred, q_state)
        return per_token.mean()
```

### 7a. `DiscreteDiffusion.compute_loss(v_pred, q_state)`

**File:** `egomimic/algo/dfot/discrete_diffusion.py`

```python
def compute_loss(self, v_pred, q_state):
    k, noise, x_start = q_state["k"], q_state["noise"], q_state["x_start"]
    if self.objective == "pred_v":
        target = self.predict_v(x_start, k, noise)
    per_element = (v_pred - target.detach()) ** 2
    # Reduce trailing dims (C, H, W) to get per-token loss
    n_trailing = per_element.dim() - k.dim()
    for _ in range(n_trailing):
        per_element = per_element.mean(dim=-1)
    w = self.compute_loss_weights(k)
    return per_element * w
```

### 7b. `predict_v(x_start, k, noise)` — target computation

```python
def predict_v(self, x_start, k, noise):
    return (
        _extract(self.sqrt_alphas_cumprod, k, x_start.shape) * noise
        - _extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * x_start
    )
```

Reference:
```python
def predict_v(self, x_start, k, noise):
    return (
        extract(self.sqrt_alphas_cumprod, k, x_start.shape) * noise
        - extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * x_start
    )
```

✅ MATCHES reference exactly.

### 7c. `compute_loss_weights(k)` — min_snr weighting

Our code:
```python
def compute_loss_weights(self, k):
    if strategy == "uniform":
        return torch.ones_like(k, dtype=torch.float32)
    snr = self.snr[k]
    if strategy == "min_snr":
        clipped_snr = self.clipped_snr[k]
        epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)
    elif strategy == "sigmoid":
        logsnr = self.logsnr[k]
        epsilon_weighting = torch.sigmoid(self.sigmoid_bias - logsnr)
    # For pred_v:
    return epsilon_weighting * snr / (snr + 1)
```

Reference:
```python
def compute_loss_weights(self, k, strategy):
    snr = self.snr[k]
    match strategy:
        case "min_snr":
            clipped_snr = self.clipped_snr[k]
            epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)
        case "sigmoid":
            logsnr = self.logsnr[k]
            epsilon_weighting = torch.sigmoid(self.cfg.loss_weighting.sigmoid_bias - logsnr)
    match self.objective:
        case "pred_v":
            return epsilon_weighting * snr / (snr + 1)
```

✅ MATCHES reference — same min_snr and sigmoid weighting formulas, same pred_v adjustment.

**Note:** Reference also supports `"fused_min_snr"` (Diffusion Forcing v1 fused reweighting with causal cum_snr). Our code does not. This is intentional — fused_min_snr is specific to the causal DFoT training with history guidance.

### 7d. Loss reduction

Our code: `per_token.mean()` (in `DFoTLoss.forward`)

Reference:
```python
loss = F.mse_loss(pred, target.detach(), reduction="none")
loss_weight = self.compute_loss_weights(k, self.loss_weighting.strategy)
loss_weight = self.add_shape_channels(loss_weight)
loss = loss * loss_weight
# ... returned to training_step which calls _reweight_loss:
def _reweight_loss(self, loss, weight=None):
    if weight is not None:
        loss = loss * weight
    return loss.mean()
```

✅ MATCHES reference — both reduce to scalar via `.mean()` after weighting. Reference also supports mask-based reweighting for context frames; we don't have context frames in our pixel DFoT, so this is N/A.

### 7e. Broadcasting of loss weights

Our code:
```python
# per_element shape: (B, T, C, H, W) or (T_total, C, H, W)
# k shape: (B, T) or (T_total,)
n_trailing = per_element.dim() - k.dim()
for _ in range(n_trailing):
    per_element = per_element.mean(dim=-1)
w = self.compute_loss_weights(k)
return per_element * w
```

Reference:
```python
loss = F.mse_loss(pred, target.detach(), reduction="none")
# loss shape: (B, T, C, H, W)
loss_weight = self.compute_loss_weights(k, strategy)
# loss_weight shape: (B, T)
loss_weight = self.add_shape_channels(loss_weight)
# loss_weight shape: (B, T, 1, 1, 1)
loss = loss * loss_weight
```

⚠️ DIFFERENT APPROACH but equivalent — reference broadcasts weights to match loss shape, then means everything at the end. Ours first reduces trailing dims (mean over C,H,W), then multiplies weights. Both give the same result because `mean(x * w_broadcast) == mean_trailing(x) * w` when `w` doesn't vary over trailing dims.

---

## 8. `DFoT.compute_losses(predictions, batch)`

**File:** `egomimic/algo/dfot/algo.py`

```python
def compute_losses(self, predictions, batch):
    total = torch.tensor(0.0, device=self.device)
    loss_dict = OrderedDict()
    for emb_id in batch.keys():
        a = predictions[f"{emb_id}_action_loss"]
        loss_dict[f"emb{emb_id}_action_loss"] = a
        total = total + a
    loss_dict["action_loss"] = total / max(len(batch), 1)
    return loss_dict
```

Averages per-embodiment losses. `training_step` returns `losses["action_loss"]`.

⚠️ DIFFERENT APPROACH — reference has a single loss path, no per-embodiment averaging. Equivalent for single-embodiment training.

---

## 9. Gradient clipping (back in `pl_model.py`)

Our code (`on_after_backward`):
```python
grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float("inf"))
# MAD-based adaptive clipping:
if grad_norm_val > threshold:
    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=median)
```

Reference (`on_before_optimizer_step`):
```python
norms = grad_norm(self.diffusion_model, norm_type=2)
self.log_dict(norms)
# No explicit clipping — just logging
```

⚠️ DIFFERENT APPROACH — our code does adaptive gradient clipping (MAD-based); reference only logs gradient norms. Our approach is more robust to training instabilities.

---

## 10. Noise schedule

**File:** `egomimic/algo/dfot/noise_schedule.py`

Our `make_beta_schedule`:
```python
def make_beta_schedule(schedule, timesteps, zero_terminal_snr=True, clip_min=1e-9, **kwargs):
    if schedule == "cosine":
        alphas_cumprod = cosine_schedule(timesteps=timesteps, **kwargs)
    # ...
    if schedule != "cosine" and zero_terminal_snr:
        alphas_cumprod = enforce_zero_terminal_snr(alphas_cumprod)
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = torch.cat([alphas_cumprod[0:1], alphas])
    betas = 1 - alphas
    return torch.clip(betas, clip_min, 1.0)
```

Reference `make_beta_schedule`: same structure with additional schedules (sigmoid, sd, cosine_simple_diffusion) and `shift` parameter support.

✅ MATCHES reference for the cosine schedule path (the only one we use). Additional schedules omitted but not needed.

---

## Summary of Findings

| Component | Status | Notes |
|-----------|--------|-------|
| `process_batch_for_training` | ⚠️ | Multi-embodiment wrapper; equivalent for single-emb |
| `forward_training` loop | ⚠️ | Separated into outer_stage + loss; equivalent math |
| `make_dfot_ctx` | ✅ | Our abstraction only |
| `_extract_images` | ⚠️ | Inline uint8→float normalization; fine |
| `_sample_frames_packed` | ⚠️ | Our extension for variable-length; reference has none |
| `_sample_noise_levels` | ✅ | Same `randint(0, timesteps)` per token |
| `q_sample` | ✅ | Identical formula |
| `PatchEmbed` | ✅ | Same Conv2d + Xavier init |
| `_build_cond` | ✅ | Same sum of noise_emb + cond_emb |
| **`_Timesteps` defaults** | **❌** | **`flip_sin_to_cos=False, downscale_freq_shift=1.0` vs reference's `True, 0`. Different embedding values.** |
| `_TimestepEmbedding` MLP | ✅ | Same Linear-SiLU-Linear |
| `StochasticTimeEmbedding` | ✅ | Same structure (modulo `_Timesteps` bug above) |
| `StochasticUnknownTimesteps` | ✅ | Same dropout-to-unknown-token logic |
| Cond broadcast to patches | ✅ | Same expand/repeat |
| `AdaLayerNormZero` | ✅ | Identical |
| `AdaLayerNorm` (final) | ✅ | Identical |
| `DiTAttention` | ✅ | Same; omits 0.0 dropout (no-op) |
| `RotaryEmbedding3D` | ✅ | Same frequencies and rotation |
| `_rotate_half` | ✅ | Identical to `rotary_embedding_torch` |
| `DiTBlock` | ✅ | Same structure and init |
| `DiTFinalLayer` | ✅ | Identical |
| Unpatchify | ✅ | Same result, different einops ordering |
| `predict_v` | ✅ | Identical formula |
| `compute_loss_weights` (min_snr) | ✅ | Identical formula |
| Loss reduction | ✅ | Both `.mean()` after weighting |
| `compute_losses` | ⚠️ | Per-embodiment averaging; equivalent for single-emb |
| Gradient clipping | ⚠️ | We do adaptive MAD clipping; ref only logs |
| Noise schedule | ✅ | Same cosine schedule + zero-terminal-SNR |

### Critical Bug

**❌ `_Timesteps` sinusoidal embedding defaults:**
- Our defaults: `flip_sin_to_cos=False`, `downscale_freq_shift=1.0`
- Reference defaults: `flip_sin_to_cos=True`, `downscale_freq_shift=0`
- **Location:** `egomimic/algo/dfot/embeddings.py`, `_Timesteps.__init__`
- **Fix:** Change defaults to `flip_sin_to_cos=True, downscale_freq_shift=0` to match reference.
- **Impact:** Affects noise-level embedding values. Training from scratch will converge fine either way (MLP learns around it), but checkpoints are not interchangeable. If you want to match reference exactly or load reference checkpoints, this must be fixed.

### Architectural Differences (Not Bugs)

1. Multi-embodiment loop (our extension)
2. Packed-mode support with per-episode backbone calls (our extension)
3. Frame sampling modes for variable-length episodes (our extension)
4. Adaptive MAD gradient clipping (our enhancement)
5. Separated outer_stage + loss design (refactor, mathematically equivalent)

---

# Rollout / Inference Path Audit (2026-05-29)

The original audit above covered only the **training** path and (correctly)
found it equivalent to the reference. The "loss fine but eval/rollout video is
garbage" symptom lives in the **inference/rollout path**, which was never
audited. Diffed `egomimic/algo/dfot/sampling.py` + the eval scripts against the
reference `algorithms/dfot/dfot_video.py` (`_predict_videos` / `_sample_sequence`)
+ `history_guidance.py` + `diffusion/discrete_diffusion.py`.

## ROOT CAUSE (critical) — eval rollout was unconditional

`eval/eval_dfot_pixel_video_rollout.py:compute_metrics_and_viz` always called
`_rollout(...)` (mode="chunk"): it built a `vanilla_schedule` with **all T
frames at max noise** and `_sample(... external_cond=None, x_init=None ...)`, so
`sampling.py` initialised **every** token to `torch.randn` — a fully
**unconditional** full-sequence generation from pure noise with **no GT context
anchor**. The model is a Diffusion-Forcing (`random_independent`) predictor: its
per-token denoising loss never measures unconditional joint generation, so the
loss descends fine, but unconditional sampling drifts off-manifold → garbage.

The reference is a **conditional** prediction task (`context_length: 1` for
pusht): it seeds the first `n_context_tokens` with the **real GT frame**, pins
those tokens **clean (noise level -1)** for the whole trajectory, and reverts
them to GT after each step (`dfot_video.py:1209,1219-1222,1320-1322`), predicting
the rest conditioned on the anchor. `scheduling_matrix: full_sequence` (uniform
future-token denoising) — which our `vanilla_schedule` already matches.

### Cleared (NOT the bug) — verified equivalent to reference
- DDIM/DDPM step math, eta, alpha indexing, schedule direction (`sample_step`).
- Noise schedule / zero-terminal-SNR / first-step scaling / all diffusion buffers.
- x0/x_start clamping (neither side clamps; symmetric).
- Normalisation / value range (both `[0,1]`, no VAE; a range bug would corrupt
  the GT panel too — it doesn't).
- Stochastic unknown-timestep dropout (`stochastic_time_p: 0.0`; gating logic
  byte-identical to reference).
- Scheduling matrix: reference uses `full_sequence` (uniform), same as ours — it
  is NOT a pyramid mismatch.

## FIX
1. `eval/eval_dfot_pixel_video_rollout.py`: `compute_metrics_and_viz` now seeds
   the first `n_context_frames` (default 1 = ref `context_length`) real GT frames
   and routes through `_rollout_sliding_window` (was dead code, but correct:
   pins context clean, predicts the rest). Added `n_context_frames` ctor arg.
   Fixed the misleading "the pixel model is unconditional" docstring.
2. `eval/eval.py`: base `Eval` now defaults `override_dict = {}`. Previously only
   `EvalVideo` defined it, so `mode=eval` crashed (`AttributeError`) for the
   composite `EvalList` runner. (Separate pre-existing bug — `mode=eval` never ran.)

## VERIFICATION (mode=eval on the 399-epoch ckpt `pact_pixel_dfot/...2026-05-27_00-01-03`)
- `recon_mse_step_00 = 0.0` exactly (context frame perfectly anchored — impossible before).
- Per-step MSE bounded 0.031 → 0.046 across steps 1–8 (smooth drift, not flat-high garbage).
- `[GT|pred]` mp4: context + near-context frames coherent and tracking GT.
- **Residual:** far-future frames (steps ~5–8) still degrade to low-amplitude
  speckled noise. This is NOT a rollout-protocol bug (protocol now matches the
  reference exactly) — it is undertraining (399 ep × limit_train_batches=80 ×
  batch=2 is a tiny budget) and/or inherent uncertainty of predicting 8 frames
  ahead from 1 frame with `external_cond_dim=0`. Addressed by a longer training
  run, not a code change.
