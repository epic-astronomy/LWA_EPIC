# 1. Formalism

## 1.1 Measurement model and the EPIC image operator

The NDP F-engine delivers, for stand $a$, polarisation $p$, channel $f$ and sequence $t$, a 4+4-bit complex voltage. In the far field,

$$
x_a(f,t) = g_a(f,t)\int W_a(\hat s,f)\,E(\hat s,f,t)\,e^{+2\pi i f\,\mathbf r_a\cdot\hat s/c}\,d\Omega + n_a(f,t),
$$

where $g_a$ is the complex gain of the signal path (front end, ARX, cable, digitiser), $W_a$ the dipole pattern, $\mathbf r_a$ the stand position (east, north, up) and $n_a$ the receiver noise. The sign convention follows `gen_phases_lwasv`. It was confirmed on LWA-SV TBF data: the conjugate convention fits the forward-modelled sky markedly worse (section 3.6).

The array covariance is

$$
R_{ab} = \langle x_a x_b^*\rangle = g_a g_b^*\,M_{ab} + \delta_{ab}\,\sigma_a^2,
\qquad
M_{ab}(f) = \int B_{ab}(\hat s,f)\,I(\hat s,f)\,e^{2\pi i f(\mathbf r_a-\mathbf r_b)\cdot\hat s/c}\,d\Omega .
$$

epic++ multiplies every voltage by a per-antenna complex weight before gridding:

$$
w_a = \phi_a\,\kappa_a\,h_a,\qquad
\phi_a = e^{-2\pi i f\,\hat p\cdot\mathbf r_a/c},\quad
\kappa_a = \frac{e^{2\pi i f\tau^{\rm cable}_a}}{\sqrt{G^{\rm cable}_a}} .
$$

Here:

- $\phi_a$ is the geometric phase toward the stand-plane normal $\hat p$.
- $\kappa_a$ is the LSL cable model.
- $h_a$ is the calibration correction.

The product $\phi_a\kappa_a$ is `phases_g` as built by `gen_phases_lwasv`. Write $\varepsilon_a = \kappa_a g_a$ for the residual gain that the cable model leaves. Calibration then amounts to choosing $h_a = 1/\varepsilon_a$.

The voltages are then gridded with a per-stand 5×5 kernel $K_a$: a prolate spheroid integrated over each cell, w-projected when `--wproj` is set. Each sequence is Fourier transformed and squared, and the squares are accumulated. Thyagarajan et al. (2017, eq. 11–13) show that the accumulated image equals software holography of the gain-weighted covariance. With the discrete epic++ gridder, the Fourier transform of the accumulated image (before the gridding correction) is

$$
D(\mathbf u) = \sum_{a,b} w_a w_b^*\,\hat R_{ab}\;Q_{ab}\!\left(\mathbf u - (\mathbf c_a-\mathbf c_b)\right)\quad(\mathrm{mod}\ N),
\qquad
Q_{ab}(\mathbf v) = \sum_{\mathbf r} K_a(\mathbf r+\mathbf v)\,K_b^*(\mathbf r),
$$

with the following ingredients:

- $\mathbf c_a$ is the anchor cell of stand $a$: `trunc(pos + d)` in `ComputeGcfElements`.
- $Q_{ab}$ is a 9×9 patch.
- The sum is circular on the $N=128$ grid. Baselines longer than half the grid wrap, as they do on the GPU.
- The $a=b$ terms are the autocorrelation pedestal that `--remove_autocorr` subtracts.

This defines a linear operator $\mathcal G$ from the covariance matrix to the image:

$$
\bar I = \mathrm{IFFT}\big[\mathcal G(W\hat R W^H)\big]\times C_{\rm gcf},
$$

where $C_{\rm gcf}$ is the gridding correction, which is known exactly. `streamcal/epicop.py` implements $\mathcal G$ (scatter) and its adjoint $\mathcal G^\dagger$ (gather), which degrids a uv-plane array onto baselines. On 229 stands at 75.8 MHz with w-projected kernels, the image it produces agrees with a literal grid-FFT-square-accumulate MOFF imager to a relative error of $2.5\times10^{-8}$ (`research/test_epicop.py`).

## 1.2 Why calibration must precede gridding

$\mathcal G$ sums several baselines into every uv cell, and the squaring mixes all antennas before any time averaging. The per-antenna gains therefore have to be applied to the voltages (Thyagarajan et al. 2017, §3; Beardsley et al. 2017, §1). There are three ways to estimate them:

1. From the EPIC images themselves, by inverting $\mathcal G$ (section 1.4).
2. From a correlation of antennas with an image pixel, as in EPICal (section 1.3).
3. From a covariance formed on the side from the same voltages (section 1.5, adopted).

## 1.3 EPICal as a projection of the covariance

EPICal correlates each antenna with the complex E-field image at a calibrator pixel $\hat s_0$. That pixel is a beam $e(t) = \sum_b w_b k_b(\hat s_0)\,x_b(t)$, where $k_b$ is the holographic beam of stand $b$ (the Fourier transform of its gridded kernel). The correlation is

$$
K_a = \langle x_a e^*\rangle = \sum_b R_{ab}\,(w_b k_b)^* = (R\,\mathbf c)_a,\qquad \mathbf c = (W\mathbf k)^* .
$$

So EPICal observes a single projection $R\mathbf c$ of the $N\times N$ covariance onto one vector. After the self term is removed, the estimate is

$$
\hat g_a = \frac{K_a - R_{aa}c_a}{\sum_{b\ne a} M_{ab}\,(\phi_b k_b)^*},
$$

because $w_b = \phi_b/\hat g_b$ makes the current estimates cancel from the denominator. Its fixed point is the true gain only if $M$ is the complete sky through the same projection. The published implementation modelled Cyg A and Cas A only (Beardsley et al. 2017, §5). The damped update $g^{(n+1)} = (1-\gamma)\hat g + \gamma g^{(n)}$ has an effective memory of $(1+\gamma)/(1-\gamma)$ loops.

In the closed-loop simulation of section 3.4, EPICal converges, but it reaches 3.0° rms phase (against 2.5° for the adopted method), with a 27% amplitude error and an image residual twice as large. The amplitude bias comes from the point-source model.

The decisive obstacle for epic++ is structural. $e(t)$ must be available as a complex number for every sequence, inside `BlockEpicImager`, and it must be correlated with every raw voltage. The kernel accumulates only $|e|^2$. It runs at the 64-register cap with spills, and it uses about 95% of the 40 ms gulp budget. Exporting per-sequence complex pixels and their correlations is not possible without breaking the real-time budget. The same projection can be formed on the CPU from tapped voltages, without the imager: this is self-holography (section 5.4). It is strictly less informative than the full covariance from the same samples.

## 1.4 Calibration from the EPIC images (grid-domain solver)

Since $\mathcal G$ is known exactly, the gains still uncorrected in an image, $\delta_a = \varepsilon_a h_a$, can be fitted to that image:

$$
\min_{\boldsymbol\delta}\ \big\|\, \Omega\odot\big(D - \mathcal G(\Phi\,\Delta\,M\,\Delta^H\Phi^H)\big)\big\|^2,
\qquad \Delta = \mathrm{diag}(\boldsymbol\delta),\ \Phi=\mathrm{diag}(\boldsymbol\phi),
$$

where $D$ is the Fourier transform of the accumulated image divided by $C_{\rm gcf}$, and $\Omega$ the cell weights. The solver in `solvers.gridcal` works as follows:

- **Iteration:** Gauss-Newton with a matrix-free conjugate gradient on $J^\dagger J$, using the Wirtinger Jacobian $J(\mathbf x) = \mathcal G\big((x_a\delta_b^* + \delta_a x_b^*)\,\tilde V_{ab}\big)$. Each CG step costs one scatter and one gather: 19 ms and 7 ms per channel at 229 stands on one Xeon core.
- **Preconditioner:** the antenna-diagonal of $J^\dagger J$.
- **Weights:** $\Omega = 1/\sqrt{\sum_{ab}|Q_{ab}|^2}$, the inverse noise per cell. Cells near the origin, where the autocorrelations and the shortest baselines fall, are masked.
- **Free fluxes:** the Sun can be solved as a free scale.

The problem is well posed. $J^\dagger J$ has a single null direction (the overall phase), and its condition number is 118 at 75.8 MHz. From 40° initial phase errors it converges to machine precision when the sky model is exact.

It is not adopted as the primary method, for two measured reasons:

1. **Information.** Each uv cell is a sum of several baselines, so the images carry less information about the gains than the covariance does. With an exact model, images at 100% duty cycle constrain the gains about as well as a 10% voltage tap (section 3.3).
2. **Model error.** The same sky-model error biases the grid-domain solution 5–10 times more than the visibility-domain solution: 2.2° against 0.4° when only small-scale diffuse structure is missing, and 1.7° against 0.2° for missing faint sources and A-team flux errors (section 3.2).

The grid solver remains useful as an end-to-end monitor of what the GPU applied (section 4.6).

## 1.5 Adopted formalism: streaming covariance tap, forward-modelled sky, tracked gains

### Data

For each interval $k$ of length $T_{\rm cal}$ (10 s by default), the calibration tap correlates the first $n_{\rm tap}$ of every 1000 sequences of each gulp on the CPU:

$$
\hat R^{(k)} = \frac{1}{N_k}\sum_{t\in\mathcal T_k} \mathbf x(t)\,\mathbf x^H(t),\qquad
N_k \approx \frac{n_{\rm tap}}{1000}\,\frac{T_{\rm cal}}{8192/196\,\mathrm{MHz}} .
$$

The voltages are raw F-engine samples, taken before any phase or gain is applied. **The estimate is therefore independent of the gains the imager uses.** There is no feedback loop, so no damping is needed and a wrong solution cannot reinforce itself.

### Sky

The forward model is evaluated every 60 s at the interval's mid time:

$$
M(f) = A\,\mathrm{diag}(B\odot S)\,A^H,\qquad A_{as} = e^{2\pi i f\,\mathbf r_a\cdot\hat s/c},
$$

where the components $s$ are:

- the A-team point sources (Cyg A, Cas A, Tau A, Vir A), with the Cas A secular decrease;
- the LFSM diffuse sky (Dowell et al. 2017) on HEALPix nside 64;
- the Sun, $M_\odot$, whose real scale $s_\odot$ is solved.

The beam $B$ is the LSL dipole model for the polarisation in question. The diffuse sky must be represented on pixels finer than the fringe spacing of the longest baseline (about 2° at 75 MHz). With nside 16–32, pixelisation alone biased the gains by 11°; nside 64 reduces the pixelisation error to 0.01–0.15° (section 3.1).

### Snapshot solution

The snapshot solution, per channel and polarisation, solves

$$
\min_{\mathbf g,\,s_\odot}\ \big\|\,\mathcal W\odot\big(\kappa\hat R\kappa^H - G(M_0 + s_\odot M_\odot)G^H\big)\big\|_F^2
$$

by StEFCal (Salvini & Wijnholds 2014), alternating with linear least squares for $s_\odot$. The details:

- **Warm start:** from the tracker's prediction. AARTFAAC reports a speed-up above 250× from warm starting.
- **Baseline weights:** $\mathcal W$ removes the autocorrelations, baselines shorter than 1λ, and inputs whose power lies outside 0.3–3× the median.
- **Error estimate:** the per-antenna variance is $\sigma^2/\sum_b \mathcal W_{ab}|g_bM_{ab}|^2$, with $\sigma^2$ the residual variance.
- **Fit quality:** measured as

$$
\chi^2_\nu = \mathrm{median}_a\ \frac{1}{\sum_b\mathcal W_{ab}}\sum_b \mathcal W_{ab}\,\frac{|\hat R_{ab}-g_ag_b^*M_{ab}|^2}{\hat R_{aa}\hat R_{bb}/N_k}.
$$

  It is 1 for a perfect model limited by thermal noise. On LWA-SV data it is 1.1–1.8 in clean channels and 10–5000 in channels dominated by RFI (section 3.6). Channels with $\chi^2_\nu > 3$ are left out of the update.

### Tracking

Per stand and polarisation, a two-state Kalman filter tracks $\mathbf x = (\varphi_0, \tau)$, so that the gain phase is $\varphi(f) = \varphi_0 + 2\pi(f-f_c)\tau + \beta(f)$:

- **Process model:** a random walk with $Q = \mathrm{diag}\big((2^\circ)^2, (0.05\ \mathrm{ns})^2\big)$ per second.
- **Measurements:** the per-channel phases of the snapshot solutions, with variances from the snapshot error estimate.
- **Wrapping:** innovations are wrapped to $(-\pi,\pi]$.
- **Jumps:** a jump, such as a power cycle or a cable change, is accepted after three consecutive innovations with normalised innovation squared above $5^2$. The state covariance of that stand is then reset.
- **Slow terms:** the per-channel phase bandpass $\beta(f)$ (zero mean and zero slope, time constant 600 s) and the log-amplitude (60 s) are exponentially averaged.

The filter is a per-antenna, diagonal reduction of the non-linear Kalman calibration of Tasse (2014). It halves the per-interval StEFCal error in the closed-loop simulation (section 3.4). It also interpolates the gains to the channels that were not solved: one in four is solved by default, and the delay model carries the rest.

### Output

The output is a table of corrections for all channels:

$$
c_a(f) = \frac{1}{\hat g_a(f)}\cdot\frac{\overline{\sum_b c_b}}{|\sum_b c_b|},
$$

with $c_a=0$ for flagged stands. A stand is flagged when its power is abnormal in most channels, when it was never solved, or when its amplitude falls below 0.15 of the median. The second factor fixes the gauge, so that the corrections have zero mean phase in every channel and polarisation. The X–Y phase of the cable model is therefore left unchanged (section 4.7).

### Application on the GPU

epic++ sets `phases_g` ← $\phi\kappa\odot c$ at the next image boundary. The kernel already multiplies every voltage by `phases_g`, so calibration adds no work to `BlockEpicImager`. The autocorrelation removal takes $|g|^2$ from `phases_g` (`fft_dx.cuh`), so it stays consistent with the calibrated weights automatically.

## 1.6 Budgets

**CPU.** The X-engine (`src/ex/xengine.hpp`, AVX2 `vpmaddwd` on 4+4-bit samples) correlates XX and YY for 256 stands at 13.1 GCMAC/s per Zen 3 core, measured on epic1:

| Quantity | Value |
|---|---|
| Full-rate requirement, 128 channels | $128\times 23\,925\ \mathrm{s^{-1}}\times 2\times 32\,896 = 2.0\times10^{11}$ CMAC/s |
| Share of samples one core correlates | 7% |
| `--caltap_nseq 50` (5% of samples) | $1.0\times10^{10}$ CMAC/s, 0.77 of a core |
| `--caltap_nseq 25` (2.5%) | 0.38 of a core |

The forward model costs one complex64 GEMM of $229\times n_{\rm pix}\times229$ per solved channel and polarisation per refresh: about 25k pixels above the horizon at nside 64. StEFCal costs about $20N^2$ flops per iteration, which is negligible.

**GPU.** Gains are applied at no cost. Uploading the table is a synchronous 512 kB copy between gulps, about 50 µs.

**Accuracy against data volume.** These are single-channel, single-interval figures at 75.8 MHz with a perfect model (section 3.3):

| Data | 1 s | 5 s | 20 s |
|---|---|---|---|
| Tap 1% | 23° | 8.6° | 4.3° |
| Tap 3% | 11.5° | 5.1° | 2.4° |
| Tap 10% | 5.9° | 2.9° | 1.3° |

A delay fit across 128 channels reduces the per-stand phase error by a further factor of up to about $\sqrt{128/4}$ for the solved subset. Real gains change by 5–10° over hours (section 3.6), so a 2.5–5% tap with 10 s intervals and Kalman tracking is sufficient.
