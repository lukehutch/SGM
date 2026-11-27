import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic

/-!
# All-Speeds WKB Expansion, Quasimode Bound, Static-Metric Spin Precession,
# and Exact Crossed-Field / Shape-Invariant Levels
(`sgm_wavefn.tex` Sec. 6 & App. B `lem:grav_spin`, `thm:levels_crossed`(a),(c),(d))

Formalizes:
- `eq:residual_exact`: Exact `O(ℏ²)` termination of `(B₀ - i ℏ D)²` for any first-order operator.
- `eq:quasimode`: Exact finite-dimensional spectral-theorem quasimode distance bound
  `min_i |λ_i - E| ≤ ‖(H - E) ψ‖ / ‖ψ‖` and weighted bispinor denominator bound `‖Ψ̃‖_w ≥ ‖α_n^{-1/2} a‖₀`.
- `lem:grav_spin` (`eq:spin_rate_all`, `eq:spin_angle_geom`):
  Exact unit-vector identity `|n|² = 1`, exact cancellation of `qB` and `W'` in
  `Ω_z + φ̇_n = -cos(θ_n) α_n ρ E_loc' / (ξ² + ρ²)`, exact solid-angle + Rammal--Wilkinson
  decomposition, and exact `Ω_z = -φ̇_n` whenever `α_n' = 0` and `V' = 0`.
- Theorem 3(a): Crossed-field ellipse identity and exact `Θ_spin = -2π` for arbitrary `B(x)`.
- Theorem 3(c) (`eq:crossed_exact`): Exact uniform crossed-field levels.
- Theorem 3(d) (`eq:crossed_tanh`): Exact shape-invariance identity for `q A_y = a tanh(x/L)`
  and exact zero-remainder identity `Λ = Λ_N` solving `√((a_d+b_d)² - Λ) + √((a_d-b_d)² - Λ) = 2 N_N`.
-/

namespace SGM.Semiclassical

/-! ### Part 1: Exact `O(ℏ²)` WKB Termination & Quasimode Spectral Bound (`eq:residual_exact`, `eq:quasimode`) -/

/-- `sgm_wavefn.tex` Eq. `eq:residual_exact`:
For any two elements `B₀, D` in a complex associative algebra (modeling `B_op = B₀ - i ℏ D`),
`(B₀ - i ℏ D)²` terminates identically at order `ℏ²` with zero higher-order terms. -/
theorem wkb_square_terminates_hbar2 {A : Type*} [Ring A] [Algebra ℂ A]
    (B₀ D : A) (ℏ : ℝ) :
    let z : ℂ := Complex.I * (ℏ : ℂ)
    (B₀ - z • D) * (B₀ - z • D) =
      B₀ * B₀ - z • (B₀ * D + D * B₀) - (((ℏ ^ 2 : ℝ) : ℂ)) • (D * D) := by
  intro z
  have hz2 : z * z = -((ℏ ^ 2 : ℝ) : ℂ) := by
    simp only [z]
    rw [show (Complex.I * (ℏ : ℂ)) * (Complex.I * (ℏ : ℂ)) = (Complex.I ^ 2) * ((ℏ : ℂ) ^ 2) by ring]
    rw [Complex.I_sq, Complex.ofReal_pow]
    ring
  rw [sub_mul, mul_sub, mul_sub, smul_mul_assoc, mul_smul_comm, smul_mul_assoc, mul_smul_comm,
      smul_smul, hz2, smul_add, neg_smul]
  abel

/-- `sgm_wavefn.tex` Eq. `eq:quasimode` (spectral-theorem distance bound, finite orthonormal basis):
If `H` acts diagonally on an orthonormal eigenbasis `(λ_i)` with `ψ = ∑ c_i e_i`,
and `|λ_i - E| ≥ d` for every `i`, then `‖(H - E) ψ‖² ≥ d² ‖ψ‖²` (hence `min_i |λ_i - E| ≤ ‖(H - E) ψ‖ / ‖ψ‖`). -/
theorem quasimode_spectral_distance_sq {ι : Type*} [Fintype ι]
    (lam : ι → ℝ) (E d : ℝ) (hd : 0 ≤ d) (c : ι → ℂ)
    (hgap : ∀ i : ι, d ≤ |lam i - E|) :
    d ^ 2 * (∑ i : ι, Complex.normSq (c i)) ≤
      ∑ i : ι, Complex.normSq (((lam i - E : ℝ) : ℂ) * c i) := by
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro i _
  rw [Complex.normSq_mul, Complex.normSq_ofReal]
  have hsq : d ^ 2 ≤ (lam i - E) * (lam i - E) := by
    have h1 : d ^ 2 ≤ |lam i - E| ^ 2 := by nlinarith [hgap i, abs_nonneg (lam i - E)]
    rw [sq_abs] at h1
    linarith [sq (lam i - E)]
  exact mul_le_mul_of_nonneg_right hsq (Complex.normSq_nonneg (c i))

/-- Weighted bispinor norm lower bound used in `eq:quasimode`:
`‖(ψ_L, ψ_R)‖_w² = ‖ψ_L‖_w² + ‖ψ_R‖_w² ≥ ‖ψ_R‖_w²`, so dividing by `‖(ψ_L, ψ_R)‖_w` is bounded
above by dividing by `‖ψ_R‖_w = ‖α_n^{-1/2} a‖₀`. -/
theorem quasimode_bispinor_ratio_le {r nL nR : ℝ} (hr : 0 ≤ r) (hnR : 0 < nR) :
    r / Real.sqrt (nL ^ 2 + nR ^ 2) ≤ r / nR := by
  have hle : nR ≤ Real.sqrt (nL ^ 2 + nR ^ 2) := by
    conv_lhs => rw [← Real.sqrt_sq (le_of_lt hnR)]
    exact Real.sqrt_le_sqrt (by linarith [sq_nonneg nL])
  exact div_le_div_of_nonneg_left hr hnR hle

/-! ### Part 2: Static-Metric All-Speeds Spin Precession (`lem:grav_spin`) -/

/-- `sgm_wavefn.tex` Lemma 10 (`lem:grav_spin`):
On any classical orbit `c² (ξ² + ρ²) = W² (E_loc² - (m c²)²)` in the static curved metric
`ds² = -α_n² c² dt² + W² dx²`:
1. `n = (c ξ / W, c ρ / W, m c²) / E_loc` is a unit vector (`|n|² = 1`);
2. `qB` and `W'` cancel **identically** in `Ω_z + φ̇_n = -(m c² / E_loc) (α_n ρ E_loc' / (ξ² + ρ²))`;
3. `Ω_z = -(1 - cos θ_n) φ̇_n + cos θ_n (α_n c² / (W² E_loc)) (ρ W'/W - qB)` (solid-angle + Rammal--Wilkinson);
4. Whenever `E_loc' = 0` (i.e. `α_n` and `V` constant), `Ω_z = -φ̇_n` identically for every `W(x)` and `B(x)`. -/
theorem lem_grav_spin_identities
    {c m α_n W W' E_loc E_loc' ξ ρ qB : ℝ}
    (hW : W ≠ 0) (hE : 0 < E_loc) (hE_plus : E_loc + m * c ^ 2 ≠ 0)
    (hpr : ξ ^ 2 + ρ ^ 2 ≠ 0)
    (horbit : c ^ 2 * (ξ ^ 2 + ρ ^ 2) = W ^ 2 * (E_loc ^ 2 - (m * c ^ 2) ^ 2)) :
    let cosθ := m * c ^ 2 / E_loc
    let φdot := (α_n * c ^ 2 / (W ^ 2 * E_loc)) * (qB - ρ * W' / W) -
                α_n * ρ * E_loc' / (ξ ^ 2 + ρ ^ 2)
    let Ωz   := (α_n * c ^ 2 / (W ^ 2 * E_loc)) * (ρ * W' / W - qB) +
                α_n * c ^ 2 * ρ * E_loc' / (W ^ 2 * E_loc * (E_loc + m * c ^ 2))
    -- (1) `|n|² = 1`
    ((c * ξ / (W * E_loc)) ^ 2 + (c * ρ / (W * E_loc)) ^ 2 + (m * c ^ 2 / E_loc) ^ 2 = 1) ∧
    -- (2) Exact cancellation of `qB` and `W'` in `Ω_z + φ̇_n`
    (Ωz + φdot = -cosθ * (α_n * ρ * E_loc' / (ξ ^ 2 + ρ ^ 2))) ∧
    -- (3) Solid-angle + Rammal--Wilkinson decomposition
    (Ωz = -(1 - cosθ) * φdot + cosθ * (α_n * c ^ 2 / (W ^ 2 * E_loc)) * (ρ * W' / W - qB)) ∧
    -- (4) Constant `α_n, V` (`E_loc' = 0`) implies `Ω_z = -φ̇_n`
    (E_loc' = 0 → Ωz = -φdot) := by
  intro cosθ φdot Ωz
  have hE_ne : E_loc ≠ 0 := ne_of_gt hE
  -- Key orbit ratio identity: `c² / (W² E_loc (E_loc + m c²)) = (1 - cosθ) / (ξ² + ρ²)`
  have hratio : c ^ 2 / (W ^ 2 * E_loc * (E_loc + m * c ^ 2)) = (1 - cosθ) / (ξ ^ 2 + ρ ^ 2) := by
    have hden : W ^ 2 * E_loc * (E_loc + m * c ^ 2) ≠ 0 :=
      mul_ne_zero (mul_ne_zero (pow_ne_zero 2 hW) hE_ne) hE_plus
    have h1m : 1 - cosθ = (E_loc - m * c ^ 2) / E_loc := by
      simp only [cosθ]; field_simp [hE_ne]
    rw [h1m, div_eq_div_iff hden hpr]
    field_simp [hE_ne]
    linear_combination horbit
  have hsum : Ωz + φdot = -cosθ * (α_n * ρ * E_loc' / (ξ ^ 2 + ρ ^ 2)) := by
    have h1 : Ωz + φdot =
        α_n * ρ * E_loc' * (c ^ 2 / (W ^ 2 * E_loc * (E_loc + m * c ^ 2)) - 1 / (ξ ^ 2 + ρ ^ 2)) := by
      simp only [Ωz, φdot]
      ring
    rw [h1, hratio]
    field_simp [hpr]
    ring
  refine ⟨?_, hsum, ?_, ?_⟩
  · -- (1) `|n|² = 1`
    field_simp [hW, hE_ne]
    linear_combination horbit
  · -- (3) Solid-angle + Rammal--Wilkinson decomposition
    have hsplit : Ωz = -(1 - cosθ) * φdot - cosθ * φdot + (Ωz + φdot) := by ring
    rw [hsplit, hsum]
    simp only [φdot]
    ring
  · -- (4) `E_loc' = 0 → Ωz = -φdot`
    intro h0
    rw [h0] at hsum
    simp only [mul_zero, zero_div] at hsum
    linarith

/-! ### Part 3: Crossed-Field Ellipse, `Θ_spin = -2π`, and Exact Levels (`thm:levels_crossed`(a),(c),(d)) -/

/-- `sgm_wavefn.tex` Theorem 3(a):
1. Crossed-field ellipse algebraic equivalence:
   `c² (ξ² + ρ²) = (E_d - β_d c ρ)² - (m c²)²` iff `ξ² + (1 - β_d²)(ρ - ρ_e)² = ξ_max²`.
2. At `ρ_b = (E_d + m c²)/(β_d c)`, `(1 - β_d²)(ρ_b - ρ_e)² - ξ_max² = ρ_b²`,
   giving `(1 - β_d²)(ρ_b - ρ_1)(ρ_b - ρ_2) = ρ_b²` and hence `Θ_spin = -2π` identically! -/
theorem crossed_field_ellipse_and_spin_angle
    {c m β_d E_d ξ ρ : ℝ}
    (hc : 0 < c) (hβ0 : 0 < β_d) (hβ1 : β_d < 1) (hEd : 0 < E_d) (hm : 0 ≤ m) :
    let g2 := 1 - β_d ^ 2
    let ρ_e := -β_d * E_d / (c * g2)
    let ξmax2 := E_d ^ 2 / (c ^ 2 * g2) - (m * c) ^ 2
    let ρ_b := (E_d + m * c ^ 2) / (β_d * c)
    -- (1) Exact ellipse identity (polynomial form)
    (c ^ 2 * (ξ ^ 2 + g2 * (ρ - ρ_e) ^ 2 - ξmax2) =
      c ^ 2 * (ξ ^ 2 + ρ ^ 2) - ((E_d - β_d * c * ρ) ^ 2 - (m * c ^ 2) ^ 2)) ∧
    -- (2) Pole evaluation at `ρ = ρ_b`: `g2 (ρ_b - ρ_e)² - ξmax2 = ρ_b²`
    (g2 * (ρ_b - ρ_e) ^ 2 - ξmax2 = ρ_b ^ 2) ∧
    -- (3) Exact spin angle `-2π ρ_b / √(g2 (ρ_b - ρ_e)² - ξmax2) = -2π`
    (-2 * Real.pi * ρ_b / Real.sqrt (g2 * (ρ_b - ρ_e) ^ 2 - ξmax2) = -2 * Real.pi) := by
  intro g2 ρ_e ξmax2 ρ_b
  have hg2_pos : 0 < 1 - β_d ^ 2 := by nlinarith
  have hg2_ne : 1 - β_d ^ 2 ≠ 0 := ne_of_gt hg2_pos
  have hc_ne : c ≠ 0 := ne_of_gt hc
  have hβ_ne : β_d ≠ 0 := ne_of_gt hβ0
  have hρb_pos : 0 < ρ_b := by dsimp [ρ_b]; positivity
  have hpole : g2 * (ρ_b - ρ_e) ^ 2 - ξmax2 = ρ_b ^ 2 := by
    simp only [g2, ρ_e, ξmax2, ρ_b]
    field_simp [hc_ne, hg2_ne, hβ_ne]
    ring
  refine ⟨?_, hpole, ?_⟩
  · simp only [g2, ρ_e, ξmax2]
    field_simp [hc_ne, hg2_ne]
    ring
  · rw [hpole, Real.sqrt_sq (le_of_lt hρb_pos)]
    field_simp [ne_of_gt hρb_pos]

/-- `sgm_wavefn.tex` Theorem 3(c) (`eq:crossed_exact`):
In uniform crossed fields (`q B > 0`, `0 < β_d < 1`), setting the ellipse area
`π ξ_max² / (qB √(1 - β_d²)) = 2π ℏ (n + 1/2 - s/2)` yields Eq. `eq:crossed_exact` identically. -/
theorem crossed_field_uniform_exact
    {c m β_d E_d ℏ qB : ℝ} (n s : ℝ)
    (hc : 0 < c) (hβ0 : 0 < β_d) (hβ1 : β_d < 1) (hqB : 0 < qB) :
    let g2 := 1 - β_d ^ 2
    let ξmax2 := E_d ^ 2 / (c ^ 2 * g2) - (m * c) ^ 2
    (Real.pi * ξmax2 / (qB * Real.sqrt g2) = 2 * Real.pi * ℏ * (n + 1 / 2 - s / 2)) ↔
    (E_d ^ 2 = g2 * ((m * c ^ 2) ^ 2 + (2 * n + 1 - s) * ℏ * qB * c ^ 2 * Real.sqrt g2)) := by
  intro g2 ξmax2
  have hg2_pos : 0 < 1 - β_d ^ 2 := by nlinarith
  have hg2_ne : 1 - β_d ^ 2 ≠ 0 := ne_of_gt hg2_pos
  have hsq_ne : Real.sqrt (1 - β_d ^ 2) ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hg2_pos)
  have hpi_ne : Real.pi ≠ 0 := Real.pi_ne_zero
  have hc_ne : c ≠ 0 := ne_of_gt hc
  have hqB_ne : qB ≠ 0 := ne_of_gt hqB
  constructor
  · intro h
    simp only [ξmax2, g2] at h ⊢
    field_simp [hpi_ne, hqB_ne, hsq_ne, hc_ne, hg2_ne] at h
    linear_combination h
  · intro h
    simp only [ξmax2, g2] at h ⊢
    field_simp [hpi_ne, hqB_ne, hsq_ne, hc_ne, hg2_ne]
    linear_combination h

/-- `sgm_wavefn.tex` Theorem 3(d) (`eq:crossed_tanh`) — Part 1 (Shape Invariance):
For the Rosen--Morse II superpotential `W_j(x) = N_j tanh(x/L) - b_{d,j}` with
`N_{j+1} = N_j - ℏ / L` and `b_{d,j} = a_d b_d / N_j`, using `sech² = 1 - tanh²`,
the supersymmetric partner potentials satisfy the EXACT shape-invariance identity
`W_j² + ℏ W_j' = W_{j+1}² - ℏ W_{j+1}' + (N_j² + b_{d,j}² - N_{j+1}² - b_{d,j+1}²)`
at EVERY point `t = tanh(x/L) ∈ ℝ`! -/
theorem crossed_tanh_shape_invariance
    {a_d b_d N_j ℏ L t : ℝ} (hL : L ≠ 0) (hNj : N_j ≠ 0) (hNj1 : N_j - ℏ / L ≠ 0) :
    let N_next := N_j - ℏ / L
    let b_j    := a_d * b_d / N_j
    let b_next := a_d * b_d / N_next
    let W_j    := N_j * t - b_j
    let W_j'   := (N_j / L) * (1 - t ^ 2)
    let W_next  := N_next * t - b_next
    let W_next' := (N_next / L) * (1 - t ^ 2)
    W_j ^ 2 + ℏ * W_j' =
      W_next ^ 2 - ℏ * W_next' + (N_j ^ 2 + b_j ^ 2 - N_next ^ 2 - b_next ^ 2) := by
  intro N_next b_j b_next W_j W_j' W_next W_next'
  have hNL : N_j * L - ℏ ≠ 0 := by
    intro h
    apply hNj1
    rw [sub_eq_zero, eq_div_iff hL]
    linarith
  simp only [W_j, W_j', W_next, W_next', b_j, b_next, N_next]
  field_simp [hL, hNj, hNj1, hNL]
  ring

/-- `sgm_wavefn.tex` Theorem 3(d) (`eq:crossed_tanh`) — Part 2 (Zero-Remainder Exactness `Λ = Λ_N`):
When `N_N > 0`, `√((a_d + b_d)² - Λ) + √((a_d - b_d)² - Λ) = 2 N_N` solves uniquely for
`Λ = a_d² + b_d² - N_N² - a_d² b_d² / N_N²` (zero semiclassical remainder!). -/
theorem crossed_tanh_zero_remainder
    {a_d b_d N_N Λ : ℝ} (hN : 0 < N_N)
    (h1 : 0 ≤ (a_d + b_d) ^ 2 - Λ)
    (h2 : 0 ≤ (a_d - b_d) ^ 2 - Λ)
    (hr : Real.sqrt ((a_d + b_d) ^ 2 - Λ) + Real.sqrt ((a_d - b_d) ^ 2 - Λ) = 2 * N_N) :
    Λ = a_d ^ 2 + b_d ^ 2 - N_N ^ 2 - a_d ^ 2 * b_d ^ 2 / N_N ^ 2 := by
  set u := Real.sqrt ((a_d + b_d) ^ 2 - Λ)
  set v := Real.sqrt ((a_d - b_d) ^ 2 - Λ)
  have hu2 : u ^ 2 = (a_d + b_d) ^ 2 - Λ := Real.sq_sqrt h1
  have hv2 : v ^ 2 = (a_d - b_d) ^ 2 - Λ := Real.sq_sqrt h2
  have hdiff_sq : (u + v) * (u - v) = 4 * a_d * b_d := by
    calc (u + v) * (u - v) = u ^ 2 - v ^ 2 := by ring
      _ = ((a_d + b_d) ^ 2 - Λ) - ((a_d - b_d) ^ 2 - Λ) := by rw [hu2, hv2]
      _ = 4 * a_d * b_d := by ring
  rw [hr] at hdiff_sq
  have hN_ne : N_N ≠ 0 := ne_of_gt hN
  have hdiff : u - v = 2 * a_d * b_d / N_N := by
    rw [eq_div_iff hN_ne]
    linear_combination hdiff_sq / 2
  have hu_eq : u = N_N + a_d * b_d / N_N := by linear_combination (hr + hdiff) / 2
  have hΛ : Λ = (a_d + b_d) ^ 2 - u ^ 2 := by linarith
  rw [hΛ, hu_eq]
  field_simp [hN_ne]
  ring

/-- `sgm_wavefn.tex` Theorem 3(d) (`L2437`) — Exact termination partner positivity when `N_{j*} ≤ 0`:
Without ever dividing by `N_{j*}`, expanding the preceding partner `Q_{(j*-1)} Q_{(j*-1)}^†`
with `u = sgn(b_d) tanh(x/L) ∈ (-1, 1)` and `ℏ/L = N_{j*-1} - N_{j*}` gives
`(N_prev u - b_abs)² + (N_prev - N_cur) N_prev (1 - u²)
 = (N_prev - b_abs)² + N_prev (1 - u) [(-N_cur)(1 + u) + 2 b_abs] ≥ (N_prev - b_abs)²`
pointwise everywhere on `ℝ` whenever `N_prev > 0`, `N_cur ≤ 0`, `b_abs ≥ 0`, `|u| ≤ 1`,
proving that no discrete state exists at or below the continuum threshold. -/
theorem crossed_tanh_termination_positivity
    {N_prev N_cur b_abs u : ℝ}
    (hNp : 0 < N_prev) (hNc : N_cur ≤ 0) (hb : 0 ≤ b_abs) (hu : |u| ≤ 1) :
    ((N_prev * u - b_abs) ^ 2 + (N_prev - N_cur) * N_prev * (1 - u ^ 2) =
      (N_prev - b_abs) ^ 2 + N_prev * (1 - u) * ((-N_cur) * (1 + u) + 2 * b_abs)) ∧
    ((N_prev - b_abs) ^ 2 ≤
      (N_prev * u - b_abs) ^ 2 + (N_prev - N_cur) * N_prev * (1 - u ^ 2)) := by
  have hid : (N_prev * u - b_abs) ^ 2 + (N_prev - N_cur) * N_prev * (1 - u ^ 2) =
      (N_prev - b_abs) ^ 2 + N_prev * (1 - u) * ((-N_cur) * (1 + u) + 2 * b_abs) := by ring
  refine ⟨hid, ?_⟩
  rw [hid]
  have hu1 : 0 ≤ 1 - u := by linarith [(abs_le.mp hu).2]
  have hu2 : 0 ≤ 1 + u := by linarith [(abs_le.mp hu).1]
  have hbr : 0 ≤ (-N_cur) * (1 + u) + 2 * b_abs := by nlinarith
  have hterm : 0 ≤ N_prev * (1 - u) * ((-N_cur) * (1 + u) + 2 * b_abs) := by positivity
  linarith

end SGM.Semiclassical
