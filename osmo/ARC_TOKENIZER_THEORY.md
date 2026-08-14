# Arc-Length Action Tokenization — Formal Setup and Open Proofs

A reference for someone picking this up to formalize it. Everything below is
either **proved**, **verified numerically** (with the measurement stated), or
**conjectured**. The distinction is marked throughout — the point of this
document is to convert the second and third categories into the first.

Code:
- `egomimic/rldb/zarr/arc_length_tokenizer.py` — cartesian (wrist) tokenizer
- `egomimic/rldb/zarr/keypoint_arc_tokenizer.py` — 21-keypoint + hybrid two-stream
- `egomimic/eval/eval_hpt.py` — `arc_matched_resample`, the equal-distance metric

---

## 1. Setup

A trajectory is a sequence $x_0, \dots, x_{T-1}$ with $x_t \in \mathcal{X}$,
sampled at a fixed control period $\Delta t$. In this codebase $\mathcal{X}$ is
one of

| space | dim | contents |
|---|---|---|
| cartesian bimanual | 14 | $[L\,xyz\,ypr\,g \mid R\,xyz\,ypr\,g]$ |
| cartesian human | 12 | as above, no gripper |
| keypoint bimanual | 138 | $[L\,\text{wrist}_6 \mid L\,\text{kp}_{63} \mid R\,\text{wrist}_6 \mid R\,\text{kp}_{63}]$ |

Fix a **step distance** $\delta: \mathcal{X} \times \mathcal{X} \to
\mathbb{R}_{\ge 0}$. Define **cumulative arc length**

$$s_0 = 0, \qquad s_t = \sum_{\tau=1}^{t} \delta(x_{\tau-1}, x_\tau).$$

A **token** of size $D$ starting at $s=0$ stores $M$ waypoints sampled
uniformly in $s$ over $[0, \min(D, s_{T-1})]$, plus a scalar velocity
$v = \tfrac{\text{end}_s}{n\Delta t}$ where $n$ is the number of control steps
spanned. Reconstruction (`detokenize`) emits $H$ samples by interpolating the
waypoints at $s(t) = v\,t$ clipped to $D$.

Write $h = \dfrac{D}{M-1}$ for the **waypoint spacing**.

---

## 2. The rotation metric

For $R, \hat R \in SO(3)$,

$$\mu(R,\hat R) \;=\; \tfrac12 \left\| (R^\top \hat R)^{1/2} - I \right\|_F .$$

### Claim 2.1 (closed form) — *verified numerically, proof wanted*

$$\mu(R,\hat R) = \sqrt{2}\,\sin(\theta/4)$$

where $\theta \in [0,\pi]$ is the geodesic angle of $R^\top \hat R$.

*Evidence.* Agrees to 5 decimals at $\theta \in \{0,15,30,60,90,120,150,180\}°$;
$\mu = 1$ exactly at $\theta = \pi$, confirming the stated range $[0,1]$.

*Sketch.* $Q = R^\top\hat R$ is a rotation by $\theta$; $Q^{1/2}$ is a rotation
by $\theta/2$ about the same axis. For a rotation $R_\varphi$,
$\|R_\varphi - I\|_F^2 = 6 - 2\,\mathrm{tr}(R_\varphi) = 4 - 4\cos\varphi
= 8\sin^2(\varphi/2)$, so $\|R_\varphi - I\|_F = 2\sqrt2\,|\sin(\varphi/2)|$.
Substituting $\varphi = \theta/2$ gives the result. **Care needed:** the
principal square root is only well defined for $\theta < \pi$; the $\theta = \pi$
case needs separate treatment.

### Claim 2.2 (metric) — *verified numerically, proof wanted*

$\mu$ is a metric on $SO(3)$.

*Evidence.* 0 triangle-inequality violations over 3000 uniformly random triples;
worst slack exactly $0$ (equality is attained, consistent with a geodesic-like
metric).

*Sketch.* $\mu = f(d_g)$ where $d_g(R,\hat R) = \theta$ is the geodesic metric
and $f(\theta) = \sqrt2\sin(\theta/4)$. On $[0,\pi]$, $f$ is continuous,
$f(0)=0$, $f$ is strictly increasing and **concave**. A concave increasing $f$
with $f(0)=0$ is subadditive, and subadditive monotone functions of a metric
preserve the triangle inequality. Prove the general lemma, then verify $f$'s
hypotheses on $[0,\pi]$.

### Claim 2.3 (arc-length calibration) — *derivation wanted*

With $\lambda = 2\sqrt2\,r$, the term $\lambda\mu$ equals, to first order, the
arc traced by a point at radius $r$ from the rotation centre.

*Evidence.* $\mu \approx \theta/(2\sqrt2) = 0.35355\,\theta$ for small $\theta$
(matches to 4 decimals at $1°, 5°, 10°$); setting $\lambda\theta/(2\sqrt2) =
r\theta$ gives $\lambda = 2\sqrt2 r$. **Quantify the error term** — for what
$\theta$ does the concavity of $\sin$ make this underestimate by more than
$\epsilon$?

### Claim 2.4 (product metric) — *standard, state it*

If $\delta_1$ is a metric on $\mathcal{X}_1$ and $\delta_2$ on $\mathcal{X}_2$,
then $\delta((a_1,a_2),(b_1,b_2)) = \delta_1 + \lambda\delta_2$ is a metric on
the product for any $\lambda > 0$. Hence $\|\Delta p\| + \lambda\mu(\Delta R)$
is a metric on $SE(3)$, and cumulative sums of it are a valid arc length.

---

## 3. Reconstruction error

### Claim 3.1 (spacing invariance) — *verified numerically, proof wanted*

Reconstruction error depends on $D$ and $M$ **only through** $h = D/(M-1)$.

*Evidence.* Mean per-keypoint error, real 94 s episode:

| $h$ | $D{=}0.13$ | $D{=}0.26$ | $D{=}0.52$ | $D{=}1.04$ |
|---|---|---|---|---|
| 20 mm | 1.848 | 1.848 | 1.847 | 1.845 |
| 10 mm | 1.150 | 1.150 | 1.149 | 1.149 |

(mm, identical to 3 decimals across an 8× range of $D$.)

*Why it should hold.* Resampling uniformly in $s$ then linearly interpolating
is invariant to affine reparameterization of $s$; the error is a functional of
the path restricted to each inter-waypoint interval, and those intervals all
have arc length $h$. **Make precise**: state the regularity needed on the path,
and identify where it fails (finite $T$ means intervals are not exactly $h$).

### Claim 3.2 (error scaling) — *measured, explanation wanted*

For a $C^2$ path, chord-vs-arc gives sagitta $\approx h^2/(8\rho)$, so error
$\Theta(h^2)$ — a **4× drop per doubling of $M$**. Measured on real keypoint
data:

| $M$ | 8 | 15 | 30 | 60 | 120 | 240 | 480 |
|---|---|---|---|---|---|---|---|
| err (mm) | 2.818 | 1.749 | 1.069 | 0.589 | 0.301 | 0.153 | 0.074 |
| ratio | — | 1.61× | 1.64× | 1.81× | 1.96× | 1.97× | 2.06× |

Ratios converge to **2×, not 4×** — so error is $\Theta(h)$, with
$\mathrm{err} \approx 0.137\,h$ asymptotically.

*Diagnosis.* The path is not $C^2$ at these scales. Measured path length is not
stride-invariant — it collapses to 92% at 100 ms subsampling and 33% at 1.33 s
(coastline behaviour), and fingertip and wrist degrade identically, indicating
global tracking jitter rather than articulation. Low-passing at 167 ms restores
ratios toward quadratic (1.64 → 2.33) and cuts error 3.2×.

**To prove:** for a path $= $ smooth signal $+$ i.i.d. noise of scale $\sigma$,
show the piecewise-linear interpolation error is $\Theta(\sigma)$ independent of
$h$ once $h$ is below the correlation length, and $\Theta(h^2/\rho)$ above it;
identify the crossover. This predicts an error floor, which is what
$\mathrm{err}/h \to 0.137$ is measuring.

---

## 4. Norms over per-part distances

For $N$ parts with per-step displacement vector
$\mathbf d = (\|\Delta p_1\|, \dots, \|\Delta p_N\|)$, the scalar step is
$\|\mathbf d\|_p$. The screenshot metric $\mathcal E_{\text{pose}} = \sum_i
[\,\|\hat p_i - p_i\| + \mu_m(R_i,\hat R_i)\,]$ is the $p{=}1$ case.

### Claim 4.1 ($L^\infty$ bounds per-part error) — *proof wanted*

If tokens advance when $\|\mathbf d\|_\infty$ accumulates $D$, then **no part
moves more than $D$ within a token**, hence per-part resampling error is bounded
independently of $N$.

*Evidence.* At equal token budget, worst-case reconstruction error:
wrist-only 16.87 mm → $L^\infty$ **9.11 mm** (max), and p99.9 improves 23%,
while *mean* error is nearly identical across all norms (1.75–1.81 mm). The
advantage **grows** with budget: $L^\infty$/wrist p99.9 ratio is 0.93× at 69
tokens, 0.83× at 138, 0.77× at 276.

### Claim 4.2 ($L^1$ inflation) — *quantify*

$\|\mathbf d\|_1$ inflates total path by $\approx N$ when parts move coherently.

*Evidence.* Measured 22.6× for $N = 21$ (total path 1248.1 m vs wrist 55.2 m);
$L^2$ gives 4.98×, $L^\infty$ 1.30×. **Prove** the coherent-motion bound
$\|\mathbf d\|_1 \le N\|\mathbf d\|_\infty$ with equality iff all parts move
equally, and relate the observed 22.6/21 ≈ 1.08 to a coherence coefficient.

---

## 5. The hybrid two-stream tokenizer

$N$ streams, each with its own column set, distance $\delta_s$, and $D_s$;
all resampled to a shared $M$; each carries its own velocity $v_s$.

Reconstruction emits only

$$H_{\text{valid}} \;=\; \min_{s\,:\,v_s > 0} \frac{D_s}{v_s}.$$

### Claim 5.1 (in-token guarantee) — *proof wanted, and it is short*

For all $t \in [0, H_{\text{valid}}]$ and every stream $s$,
$\;s_s(t) = v_s t \le D_s$, so every stream interpolates strictly within its
stored waypoints — no extrapolation, and the streams stay time-consistent.

*Proof sketch.* $t \le H_{\text{valid}} \le D_s/v_s \Rightarrow v_s t \le D_s$.
The content is that $\min$ is the **largest** horizon with this property:
for $t > H_{\text{valid}}$ the argmin stream exceeds its $D$.

### Claim 5.2 (horizon identity) — *observed, prove*

$H_{\text{valid}} = D/\max_s v_s$ when all $D_s = D$ — i.e. the fastest stream
sets the horizon, which is exactly the quantity $L^\infty$ bounds. Measured:
hybrid and scalar-$L^\infty$ horizons agree to 4 decimals (0.6333 s).

### Claim 5.3 ($M$ decouples from horizon) — *observed, prove*

$H_{\text{valid}}$ depends only on $(D_s, v_s)$, so raising $M$ refines every
stream at **no cost to horizon**. Measured at fixed 0.700 s horizon: eef
reconstruction 1.390 / 0.412 / 0.011 mm for $M = 15/30/60$.

This is the main structural advantage over a single shared parameterization,
where $D$ and $M$ are coupled through $h$.

### Claim 5.4 (balancing the $D_s$) — *open*

Whichever stream exhausts $D_s$ first caps the window for all. Measured path
lengths on one episode: eef 48.71 m, wrist-frame keypoint $L^\infty$ 44.51 m,
suggesting $D_{kp} \approx 0.91\,D_{eef}$. Dropping $D_{kp}$ to 0.20 shortened
the horizon 0.700 → 0.533 s and cut eef coverage to 0.667.

**Open:** given per-stream velocity distributions, choose $\{D_s\}$ to maximize
expected $H_{\text{valid}}$ subject to a total token-width budget $NM$. Is the
equal-token-count heuristic optimal, and under what distributional assumption?

### Degenerate streams

$v_s = 0$ must be **excluded** from the $\min$ (else $D/0$), and such a stream
holds its first waypoint. If every stream is degenerate the token is all-zero
and reconstruction holds row 0. Verified: with eef frozen the horizon is still
finite (2.267 s, set by keypoints) and the wrist correctly holds.

---

## 6. The comparability problem

Two models predicting in different action spaces cannot be compared by raw
per-timestep MSE, because their chunks span **different amounts of motion**.
Measured: a baseline 100-step chunk travels 0.669 m while a $D{=}0.20$ arc token
spans 0.183 m — a 3.66× mismatch that alone produced an implausible "130×
better" reading.

`arc_matched_resample` fixes this: resample **both** to $M$ points spaced
uniformly in arc length over the first $D$ metres, per arm, then compare.

### Claim 6.1 — *state precisely*

Arc-matched MSE is invariant to time reparameterization and measures **path
geometry at matched distance**. It is therefore **blind to timing** by
construction — a model tracing the right path at the wrong speed scores well.
Timing must be evaluated separately (via the velocity channel). Formalize:
show the metric is a function of the two paths' images, not their
parameterizations.

---

## 7. Reproducing the measurements

All numbers above come from one real human episode,
`s3://rldb/processed_v3/aria/2025-09-20-17-42-51-000000.zarr`, 2808 valid
frames at 30 Hz (94 s), left hand unless stated.

**Caveat that matters:** every empirical claim is from **one episode of one
task** (`fold_clothes`), which is transport-dominated — only 3.4% of hand-shape
motion occurs while the wrist is nearly still. Claims 4.1 and 5.4 in particular
should be re-measured on a manipulation-heavy task before being treated as
general.

---

## 8. Summary of what to prove

1. **2.1** closed form $\mu = \sqrt2\sin(\theta/4)$, incl. $\theta = \pi$
2. **2.2** $\mu$ is a metric (via concave-subadditive lemma)
3. **2.3** error term in the $\lambda = 2\sqrt2 r$ calibration
4. **3.1** error depends on $D, M$ only via $h$ — state regularity conditions
5. **3.2** $\Theta(h)$ vs $\Theta(h^2)$ crossover for signal + noise
6. **4.1** $L^\infty$ per-part error bound
7. **4.2** $L^1$ inflation and its coherence coefficient
8. **5.1** in-token guarantee, and maximality of $\min$
9. **5.3** $M$–horizon decoupling
10. **5.4** optimal $\{D_s\}$ under a token-width budget
11. **6.1** arc-matched MSE is reparameterization-invariant
