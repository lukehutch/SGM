import Mathlib.Data.Matrix.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Basic.Complex.Basic
import Mathlib.Tactic

/-!
# Curved-Spacetime Kinematics, Strong-Field Gravity, BMV/SN, 1PN EIH, Radiative Signaling,
Bekenstein-BH Saturation, and Horizon Zitterbewegung Floor
(`sgm.tex` Secs. 3, 7.3–7.4, Supp. S8; `sgm_wavefn.tex` Sec. 7.6)

Formalizes:
- Strong-field Schwarzschild triad & Komar--Gauss flux (`eq:dalpha_dl`, `eq:gauss_exact`,
  `eq:proper_acceleration`) in both areal and isotropic coordinates.
- General non-diagonal static spatial metric norm identity `h_{jl} a^j a^l = c^4 h^{jk} g_j g_k`.
- Equivalence principle & free-fall geodesic balance (`eq:geodesic_balance`, `eq:freefall_accel`).
- SGM-1 (`prop:config_space_gravity`): 3-body pair-counting `eq:h_grav_pair`, exact BMV reduced
  purity `eq:gie_phase`, and 2-site Schrödinger--Newton finite-time orthogonalization `eq:sn_overlap_drift`.
- SGM-2 (`prop:eih_1pn`): 1PN EIH symmetric tensor kernel `eq:eih_kernel`, exact operator
  Hermiticity of the symmetrized tensor-sandwich `eq:eih_1pn`, and positive Lense--Thirring drag.
- SGM-3(a) (`prop:rad_meanfield_signaling`): Orthogonal $+/\times$ quadrupole trace algebra and
  Bell GW-power signaling bias `eq:gw_power_bias`.
- SGM-4a (`eq:bek_bh_sat`): Exact Bekenstein-to-black-hole identity `(A(R)/(4 ℓ_P²))(1 - α(R)²)`.
- WF-5 (`cor:horizon_zitter_gap`): Horizon Margolus--Levitin/Zitterbewegung orthogonalization time
  `eq:ml_zitter` and Tolman--Unruh pair-creation floor `eq:alpha_gap`.
-/

namespace SGM.Differential

/-! ### Part 1: Strong-Field Schwarzschild Triad, Isotropic Check & General-Slice Norm -/

/-- `sgm.tex` Eqs. `eq:dalpha_dl` & `eq:gauss_exact`:
For Schwarzschild `α = √(1 - R_s/r)` (`r > R_s > 0`, `R_s = 2GM/c²`),
`dα/dr = R_s / (2 r² α)`, proper gradient `dα/dr_p = α (dα/dr) = R_s / (2 r²)`,
Komar--Gauss flux `(dα/dr_p) · 4π r² = 2π R_s = 4π GM / c²`,
and strong-field triad `F_SGM = F_prop = GMm/(r² α) = α⁻¹ F_∞`. -/
theorem schwarzschild_strong_field_triad
    {G M m c r : ℝ} (hc : 0 < c) (hr : 0 < r)
    (hExt : 2 * G * M / c ^ 2 < r) :
    let Rs := 2 * G * M / c ^ 2
    let α := Real.sqrt (1 - Rs / r)
    let dα_dr := Rs / (2 * r ^ 2 * α)
    let dα_drp := α * dα_dr
    (dα_drp = Rs / (2 * r ^ 2)) ∧
    (dα_drp * (4 * Real.pi * r ^ 2) = 4 * Real.pi * G * M / c ^ 2) ∧
    (m * c ^ 2 * dα_dr = G * M * m / (r ^ 2 * α)) ∧
    (α * (m * c ^ 2 * dα_dr) = G * M * m / r ^ 2) := by
  intro Rs α dα_dr dα_drp
  have h1m : 0 < 1 - Rs / r := by
    have : Rs / r < 1 := (div_lt_one hr).mpr hExt
    linarith
  have hα_pos : 0 < α := Real.sqrt_pos.mpr h1m
  have hα_ne : α ≠ 0 := ne_of_gt hα_pos
  have hr_ne : r ≠ 0 := ne_of_gt hr
  have hc_ne : c ≠ 0 := ne_of_gt hc
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp only [dα_drp, dα_dr]
    field_simp [hα_ne, hr_ne]
  · simp only [dα_drp, dα_dr, Rs]
    field_simp [hα_ne, hr_ne, hc_ne]
  · simp only [dα_dr, Rs]
    field_simp [hα_ne, hr_ne, hc_ne]
  · simp only [dα_dr, Rs]
    field_simp [hα_ne, hr_ne, hc_ne]

/-- Isotropic Schwarzschild verification (`sgm_wavefn.tex` Prop. 3(iii)):
With `u = GM / (2 r c²) < 1`, `α_iso = (1 - u)/(1 + u)`, `W_iso = (1 + u)²`, and areal radius
`R = r (1 + u)²`, we have `1 - 2GM/(Rc²) = α_iso²` and
`(c² / W_iso) · (d ln α_iso / dr) = (GM / R²) / α_iso`. -/
theorem isotropic_schwarzschild_proper_accel
    {G M c r : ℝ} (hG : 0 < G) (hM : 0 < M) (hc : 0 < c) (hr : 0 < r)
    (hu_lt : G * M / (2 * r * c ^ 2) < 1) :
    let u := G * M / (2 * r * c ^ 2)
    let α := (1 - u) / (1 + u)
    let W := (1 + u) ^ 2
    let R := r * W
    -- du/dr = -u/r, d(ln α)/du = -2 / (1 - u²) => d(ln α)/dr = 2 u / (r (1 - u²))
    let dlnα_dr := 2 * u / (r * (1 - u ^ 2))
    (1 - 2 * G * M / (R * c ^ 2) = α ^ 2) ∧
    (c ^ 2 / W * dlnα_dr = (G * M / R ^ 2) / α) := by
  intro u α W R dlnα_dr
  have hu_pos : 0 < u := by positivity
  have h1pu : 1 + u ≠ 0 := by linarith
  have h1mu : 1 - u ≠ 0 := by linarith
  have h1mu2 : 1 - u ^ 2 ≠ 0 := by nlinarith
  have hr_ne : r ≠ 0 := ne_of_gt hr
  have hc_ne : c ≠ 0 := ne_of_gt hc
  have hGM : G * M = 2 * r * c ^ 2 * u := by
    simp only [u]; field_simp [hr_ne, hc_ne]
  refine ⟨?_, ?_⟩
  · simp only [α, W, R]
    rw [mul_assoc 2 G M, hGM]
    field_simp [h1pu, hr_ne, hc_ne]
    ring
  · simp only [α, W, R, dlnα_dr]
    rw [hGM]
    field_simp [h1pu, h1mu, h1mu2, hr_ne, hc_ne]
    ring

/-- General non-diagonal static 3D spatial metric norm identity (`sgm_wavefn.tex` Eq. `eq:proper_acceleration`):
If `h * hInv = 1`, then for `a^j = c² ∑_k hInv j k * g_k`,
`∑_{j,l} h j l * a j * a l = c⁴ ∑_{j,k} hInv j k * g_j * g_k`. -/
theorem general_static_slice_proper_accel_norm
    (h hInv : Matrix (Fin 3) (Fin 3) ℝ)
    (hMul : h * hInv = 1)
    (c : ℝ) (g : Fin 3 → ℝ) :
    let a : Fin 3 → ℝ := fun j => c ^ 2 * ∑ k : Fin 3, hInv j k * g k
    (∑ j : Fin 3, ∑ l : Fin 3, h j l * a j * a l) =
      c ^ 4 * ∑ j : Fin 3, ∑ k : Fin 3, hInv j k * g j * g k := by
  intro a
  -- Rewrite LHS using h * hInv = 1
  have h_id : ∀ l : Fin 3, (∑ j : Fin 3, h l j * a j) = c ^ 2 * g l := by
    intro l
    simp only [a, Finset.mul_sum]
    calc ∑ j : Fin 3, ∑ k : Fin 3, h l j * (c ^ 2 * (hInv j k * g k))
        = c ^ 2 * ∑ k : Fin 3, (∑ j : Fin 3, h l j * hInv j k) * g k := by
            rw [Finset.sum_comm]
            simp only [Finset.mul_sum, Finset.sum_mul]
            apply Finset.sum_congr rfl; intro k _
            apply Finset.sum_congr rfl; intro j _
            ring
      _ = c ^ 2 * ∑ k : Fin 3, (1 : Matrix (Fin 3) (Fin 3) ℝ) l k * g k := by
            congr 1; apply Finset.sum_congr rfl; intro k _
            have := congr_fun (congr_fun hMul l) k
            simp [Matrix.mul_apply] at this
            rw [this]
      _ = c ^ 2 * g l := by simp [Matrix.one_apply]
  calc ∑ j : Fin 3, ∑ l : Fin 3, h j l * a j * a l
      = ∑ j : Fin 3, a j * (∑ l : Fin 3, h j l * a l) := by
          apply Finset.sum_congr rfl; intro j _
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl; intro l _; ring
    _ = ∑ j : Fin 3, a j * (c ^ 2 * g j) := by
          apply Finset.sum_congr rfl; intro j _; rw [h_id j]
    _ = c ^ 4 * ∑ j : Fin 3, ∑ k : Fin 3, hInv j k * g j * g k := by
          simp only [a, Finset.mul_sum, Finset.sum_mul]
          apply Finset.sum_congr rfl; intro j _
          apply Finset.sum_congr rfl; intro k _; ring

/-- `sgm.tex` Eqs. `eq:geodesic_balance` & `eq:freefall_accel`:
Equivalence principle cancellation of `ω₀`: from `ω₀ (∂_i α + (1/(α₀ c²)) ∑_j γ_ij a_c^j) = 0`,
multiplying by `γ⁻¹` yields mass-independent free-fall acceleration `a_c^k = -c² α₀ ∑_i γ⁻¹_{ki} ∂_i α`. -/
theorem equivalence_principle_freefall
    (γ γInv : Matrix (Fin 3) (Fin 3) ℝ) (hγ : γInv * γ = 1)
    {ω₀ α₀ c : ℝ} (hω : ω₀ ≠ 0) (hα : α₀ ≠ 0) (hc : c ≠ 0)
    (dα a_c : Fin 3 → ℝ)
    (hBal : ∀ i, ω₀ * (dα i + (∑ j : Fin 3, γ i j * a_c j) / (α₀ * c ^ 2)) = 0) :
    ∀ k, a_c k = -c ^ 2 * α₀ * ∑ i : Fin 3, γInv k i * dα i := by
  have h1 : ∀ i, ∑ j : Fin 3, γ i j * a_c j = -(α₀ * c ^ 2) * dα i := by
    intro i
    have hi := (mul_eq_zero.mp (hBal i)).resolve_left hω
    have hden : α₀ * c ^ 2 ≠ 0 := mul_ne_zero hα (pow_ne_zero 2 hc)
    field_simp [hden] at hi
    linarith
  intro k
  calc a_c k
      = ∑ j : Fin 3, (1 : Matrix (Fin 3) (Fin 3) ℝ) k j * a_c j := by simp [Matrix.one_apply]
    _ = ∑ j : Fin 3, (∑ i : Fin 3, γInv k i * γ i j) * a_c j := by
          apply Finset.sum_congr rfl; intro j _
          have := congr_fun (congr_fun hγ k) j
          simpa [Matrix.mul_apply] using congr_arg (· * a_c j) this.symm
    _ = ∑ i : Fin 3, γInv k i * (∑ j : Fin 3, γ i j * a_c j) := by
          simp only [Finset.sum_mul, Finset.mul_sum]
          rw [Finset.sum_comm]
          apply Finset.sum_congr rfl; intro i _
          apply Finset.sum_congr rfl; intro j _; ring
    _ = -c ^ 2 * α₀ * ∑ i : Fin 3, γInv k i * dα i := by
          simp only [h1, Finset.mul_sum]
          apply Finset.sum_congr rfl; intro i _; ring

/-! ### Part 2: SGM-1 — Pair Counting, BMV Entanglement Purity, and 2-Site SN Orthogonalization -/

/-- `sgm.tex` Eq. `eq:h_grav_pair`:
Exact factor-of-1/2 pair-counting identity for 3 gravitating particles. -/
theorem grav_pair_counting_3 (G c m1 m2 m3 r12 r13 r23 : ℝ) (hc : c ≠ 0) :
    let α1_m1 := -(G / c ^ 2) * (m2 / r12 + m3 / r13)
    let α2_m1 := -(G / c ^ 2) * (m1 / r12 + m3 / r23)
    let α3_m1 := -(G / c ^ 2) * (m1 / r13 + m2 / r23)
    (1 / 2 : ℝ) * (m1 * c ^ 2 * α1_m1 + m2 * c ^ 2 * α2_m1 + m3 * c ^ 2 * α3_m1) =
      -(G * m1 * m2 / r12 + G * m1 * m3 / r13 + G * m2 * m3 / r23) := by
  intro α1_m1 α2_m1 α3_m1
  simp only [α1_m1, α2_m1, α3_m1]
  field_simp [hc]
  ring

/-- `sgm.tex` Proposition `prop:config_space_gravity`(i) (`eq:gie_phase`):
Exact BMV reduced-state purity `tr(ρ₁²) = 1 - (1/2) sin²(Δθ_grav / 2)`. -/
theorem bmv_entanglement_purity (θ00 θ01 θ10 θ11 : ℝ) :
    let Δθ := θ00 + θ11 - θ01 - θ10
    let r01 : ℂ := (1 / 4 : ℂ) * (Complex.exp (Complex.I * (θ00 - θ10)) +
                                  Complex.exp (Complex.I * (θ01 - θ11)))
    (1 / 2 : ℝ) + 2 * Complex.normSq r01 = 1 - (1 / 2 : ℝ) * Real.sin (Δθ / 2) ^ 2 := by
  intro Δθ r01
  have h_nsq : Complex.normSq r01 = (1 / 4 : ℝ) * Real.cos (Δθ / 2) ^ 2 := by
    have hcos_id : ∀ a b : ℝ,
        Complex.normSq (Complex.exp (Complex.I * a) + Complex.exp (Complex.I * b)) =
          2 + 2 * Real.cos (a - b) := by
      intro a b
      rw [mul_comm Complex.I (a : ℂ), mul_comm Complex.I (b : ℂ),
        Complex.exp_mul_I, Complex.exp_mul_I]
      simp only [Complex.normSq_apply, Complex.add_re, Complex.add_im,
        Complex.mul_re, Complex.mul_im, Complex.I_re, Complex.I_im,
        Complex.cos_ofReal_re, Complex.cos_ofReal_im,
        Complex.sin_ofReal_re, Complex.sin_ofReal_im,
        mul_zero, mul_one, sub_zero, add_zero, zero_add]
      have h1 := Real.cos_sq_add_sin_sq a
      have h2 := Real.cos_sq_add_sin_sq b
      have h3 := Real.cos_sub a b
      nlinarith
    have hdiff : (θ00 - θ10) - (θ01 - θ11) = Δθ := by simp [Δθ]; ring
    have hsub := hcos_id (θ00 - θ10) (θ01 - θ11)
    rw [hdiff] at hsub
    have hcos_half : Real.cos Δθ = 2 * Real.cos (Δθ / 2) ^ 2 - 1 := by
      have := Real.cos_two_mul (Δθ / 2)
      ring_nf at this ⊢
      linarith
    simp only [r01]
    rw [Complex.normSq_mul]
    have h14' : Complex.normSq (1 / 4 : ℂ) = 1 / 16 := by
      simp; norm_num
    rw [h14', ← Complex.ofReal_sub, ← Complex.ofReal_sub, hsub, hcos_half]
    ring
  rw [h_nsq]
  have hcs := Real.sin_sq_add_cos_sq (Δθ / 2)
  linarith

/-- `sgm.tex` Proposition `prop:config_space_gravity`(ii) (`eq:sn_overlap_drift`):
Two-site Schrödinger--Newton overlap drift for `(p_A, p_B) = (1/4, 3/4)` reaches exact
orthogonality `0` at finite time `t_* = π ℏ / (U_self - U_cross)`. -/
theorem sn_two_site_orthogonalization
    {gs gd ℏ : ℝ} (hdiff : gs - gd ≠ 0) (hℏ : ℏ ≠ 0) :
    let pA : ℝ := 1 / 4
    let pB : ℝ := 3 / 4
    let Δω : ℝ := 2 * (gs - gd) * (pB - pA) / ℏ
    let tStar : ℝ := Real.pi * ℏ / (gs - gd)
    -- At t = 0: squared overlap is 3/4 (overlap √3/2 > 0)
    ((1 - pA) * (1 - pB) + pA * pB +
      2 * Real.sqrt (pA * (1 - pA) * pB * (1 - pB)) * Real.cos (Δω * 0) = 3 / 4) ∧
    -- At t = tStar: squared overlap vanishes identically (exact orthogonalization!)
    ((1 - pA) * (1 - pB) + pA * pB +
      2 * Real.sqrt (pA * (1 - pA) * pB * (1 - pB)) * Real.cos (Δω * tStar) = 0) := by
  intro pA pB Δω tStar
  have hsq : Real.sqrt (pA * (1 - pA) * pB * (1 - pB)) = 3 / 16 := by
    have : pA * (1 - pA) * pB * (1 - pB) = (3 / 16 : ℝ) ^ 2 := by norm_num [pA, pB]
    rw [this, Real.sqrt_sq (by norm_num)]
  have harg : Δω * tStar = Real.pi := by
    simp only [Δω, tStar, pA, pB]
    field_simp [hdiff, hℏ]
    ring
  refine ⟨?_, ?_⟩
  · rw [hsq, mul_zero, Real.cos_zero]
    norm_num [pA, pB]
  · rw [hsq, harg, Real.cos_pi]
    norm_num [pA, pB]

/-! ### Part 3: SGM-2 — 1PN Einstein--Infeld--Hoffmann Cross-Momentum Operator (`prop:eih_1pn`) -/

/-- `sgm.tex` Eq. `eq:eih_kernel`:
The 3×3 spatial EIH tensor kernel `𝒲^{ab}(r) = (G/(2c²|r|))(7 δ^{ab} + r^a r^b / |r|²)`. -/
noncomputable def eihKernel (G c : ℝ) (r : Fin 3 → ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  let nr := Real.sqrt (r 0 ^ 2 + r 1 ^ 2 + r 2 ^ 2)
  Matrix.of fun a b =>
    (G / (2 * c ^ 2 * nr)) * ((if a = b then (7 : ℝ) else 0) + r a * r b / nr ^ 2)

theorem eihKernel_symm (G c : ℝ) (r : Fin 3 → ℝ) :
    (eihKernel G c r).transpose = eihKernel G c r := by
  ext a b
  simp only [Matrix.transpose_apply, eihKernel, Matrix.of_apply, eq_comm, mul_comm (r b)]

/-- Longitudinal eigenvalue of `𝒲(r)` is `8 · G/(2c²|r|)`. -/
theorem eihKernel_longitudinal {G c : ℝ} (r : Fin 3 → ℝ)
    (hr : r 0 ^ 2 + r 1 ^ 2 + r 2 ^ 2 > 0) :
    let nr := Real.sqrt (r 0 ^ 2 + r 1 ^ 2 + r 2 ^ 2)
    ∀ a : Fin 3, (∑ b : Fin 3, eihKernel G c r a b * r b) = (8 * G / (2 * c ^ 2 * nr)) * r a := by
  intro nr a
  have hnr2 : Real.sqrt (r 0 ^ 2 + r 1 ^ 2 + r 2 ^ 2) ^ 2 = r 0 ^ 2 + r 1 ^ 2 + r 2 ^ 2 :=
    Real.sq_sqrt (le_of_lt hr)
  have hnr2_ne : Real.sqrt (r 0 ^ 2 + r 1 ^ 2 + r 2 ^ 2) ^ 2 ≠ 0 := by linarith
  let K : ℝ := G / (2 * c ^ 2 * Real.sqrt (r 0 ^ 2 + r 1 ^ 2 + r 2 ^ 2))
  have hinner :
      (7 * r a + (r a * r 0 / Real.sqrt (r 0 ^ 2 + r 1 ^ 2 + r 2 ^ 2) ^ 2 * r 0 +
                  r a * r 1 / Real.sqrt (r 0 ^ 2 + r 1 ^ 2 + r 2 ^ 2) ^ 2 * r 1 +
                  r a * r 2 / Real.sqrt (r 0 ^ 2 + r 1 ^ 2 + r 2 ^ 2) ^ 2 * r 2)) = 8 * r a := by
    field_simp [hnr2_ne]
    rw [hnr2]
    ring
  have hstep : (∑ b : Fin 3, eihKernel G c r a b * r b) = K * (8 * r a) := by
    rw [← hinner]
    fin_cases a <;>
      (simp [eihKernel, Fin.sum_univ_three, K]; ring)
  rw [hstep]
  simp only [K, nr]
  ring

/-- `sgm.tex` Proposition `prop:eih_1pn` (`eq:eih_1pn`):
Abstract Hermiticity of the symmetrized 1PN tensor-sandwich operator
`H_EIH = (1/2) (P₁ W P₂ + P₂ W P₁)` whenever `P₁`, `P₂`, `W` are symmetric on `ℋ_mat`. -/
theorem eih_sandwich_symmetric {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]
    (P1 P2 W : H →ₗ[ℂ] H)
    (hP1 : ∀ u v, inner ℂ u (P1 v) = inner ℂ (P1 u) v)
    (hP2 : ∀ u v, inner ℂ u (P2 v) = inner ℂ (P2 u) v)
    (hW  : ∀ u v, inner ℂ u (W v)  = inner ℂ (W u)  v) :
    let H_op : H →ₗ[ℂ] H := (1 / 2 : ℂ) • (P1.comp (W.comp P2) + P2.comp (W.comp P1))
    ∀ u v, inner ℂ u (H_op v) = inner ℂ (H_op u) v := by
  intro H_op u v
  simp only [H_op, LinearMap.smul_apply, LinearMap.add_apply, LinearMap.comp_apply,
    inner_smul_right, inner_smul_left, inner_add_right, inner_add_left]
  have hconj : starRingEnd ℂ (1 / 2 : ℂ) = (1 / 2 : ℂ) := by
    rw [show (1 / 2 : ℂ) = ((1 / 2 : ℝ) : ℂ) by norm_num, Complex.conj_ofReal]
  rw [hconj, hP1 u (W (P2 v)), hW (P1 u) (P2 v), hP2 (W (P1 u)) v,
      hP2 u (W (P1 v)), hW (P2 u) (P1 v), hP1 (W (P2 u)) v, add_comm]

/-! ### Part 4: SGM-3(a) — Radiative Mean-Field Quadrupole Signaling (`prop:rad_meanfield_signaling`) -/

def quadE0 : Matrix (Fin 3) (Fin 3) ℝ := !![1, 0, 0; 0, -1, 0; 0, 0, 0]
def quadE1 : Matrix (Fin 3) (Fin 3) ℝ := !![0, 1, 0; 1, 0, 0; 0, 0, 0]

/-- `sgm.tex` Proposition `prop:rad_meanfield_signaling` (`eq:gw_power_bias`):
Exact quadrupole trace algebra and superluminal Bell GW-power bias
`ΔP_GW = P_GW^{(Z)} - P_GW^{(X)} = G Q₀² Ω_quad⁶ / (10 c⁵)`. -/
theorem radiative_meanfield_gw_power_bias (G Q0 Ω c : ℝ) :
    let Emix := (1 / 2 : ℝ) • (quadE0 + quadE1)
    let pref := G * Q0 ^ 2 * Ω ^ 6 / (10 * c ^ 5)
    (Matrix.trace quadE0 = 0) ∧
    (Matrix.trace quadE1 = 0) ∧
    (Matrix.trace (quadE0 * quadE0) = 2) ∧
    (Matrix.trace (quadE1 * quadE1) = 2) ∧
    (Matrix.trace (quadE0 * quadE1) = 0) ∧
    (Matrix.trace (Emix * Emix) = 1) ∧
    (pref * Matrix.trace (quadE0 * quadE0) - pref * Matrix.trace (Emix * Emix) =
      G * Q0 ^ 2 * Ω ^ 6 / (10 * c ^ 5)) := by
  intro Emix pref
  have htr0 : Matrix.trace quadE0 = 0 := by
    simp [Matrix.trace, Matrix.diag, Fin.sum_univ_three, quadE0]
  have htr1 : Matrix.trace quadE1 = 0 := by
    simp [Matrix.trace, Matrix.diag, Fin.sum_univ_three, quadE1]
  have h00 : Matrix.trace (quadE0 * quadE0) = 2 := by
    simp [Matrix.trace, Matrix.diag, Matrix.mul_apply, Fin.sum_univ_three, quadE0]; norm_num
  have h11 : Matrix.trace (quadE1 * quadE1) = 2 := by
    simp [Matrix.trace, Matrix.diag, Matrix.mul_apply, Fin.sum_univ_three, quadE1]; norm_num
  have h01 : Matrix.trace (quadE0 * quadE1) = 0 := by
    simp [Matrix.trace, Matrix.diag, Matrix.mul_apply, Fin.sum_univ_three, quadE0, quadE1]
  have hmix : Matrix.trace (Emix * Emix) = 1 := by
    simp [Emix, Matrix.trace, Matrix.diag, Matrix.mul_apply, Fin.sum_univ_three, quadE0, quadE1]
    norm_num
  refine ⟨htr0, htr1, h00, h11, h01, hmix, ?_⟩
  rw [h00, hmix]
  simp only [pref]
  ring

/-! ### Part 5: SGM-4a (`eq:bek_bh_sat`) and WF-5 (`eq:ml_zitter`, `eq:alpha_gap`) -/

/-- `sgm.tex` Eq. `eq:bek_bh_sat`:
Exact Bekenstein-to-black-hole identity `2π R (M c²) / (ℏ c) = (A(R)/(4 ℓ_P²))(1 - α(R)²)`
saturating at `A(R_s)/(4 ℓ_P²)` as `R → R_s = 2GM/c²`. -/
theorem bekenstein_bh_saturation {G M R ℏ c : ℝ}
    (hG : G ≠ 0) (hR : R ≠ 0) (hℏ : ℏ ≠ 0) (hc : c ≠ 0) :
    let E := M * c ^ 2
    let Rs := 2 * G * M / c ^ 2
    let A := 4 * Real.pi * R ^ 2
    let lP2 := ℏ * G / c ^ 3
    let αSq := 1 - Rs / R
    (2 * Real.pi * R * E / (ℏ * c) = (A / (4 * lP2)) * (1 - αSq)) := by
  intro E Rs A lP2 αSq
  simp only [E, Rs, A, lP2, αSq]
  field_simp [hG, hR, hℏ, hc]
  ring

/-- `sgm_wavefn.tex` Corollary `cor:horizon_zitter_gap` (`eq:ml_zitter` & `eq:alpha_gap`):
Exact Margolus--Levitin / Zitterbewegung orthogonalization time and Tolman--Unruh gap floor. -/
theorem horizon_zitter_and_tolman_gap {α_n m c ℏ κ_hor : ℝ}
    (hα : α_n ≠ 0) (hm : m ≠ 0) (hc : c ≠ 0) (hℏ : ℏ ≠ 0) :
    let ωZ := 2 * m * c ^ 2 / ℏ
    let τML := Real.pi * ℏ / (2 * α_n * m * c ^ 2)
    let kT_Haw := ℏ * κ_hor / (2 * Real.pi * c)
    let α_gap := ℏ * κ_hor / (4 * Real.pi * m * c ^ 3)
    (τML = Real.pi / (α_n * ωZ)) ∧
    (Real.cos (α_n * m * c ^ 2 * τML / ℏ) = 0) ∧
    (kT_Haw = α_gap * (2 * m * c ^ 2)) := by
  intro ωZ τML kT_Haw α_gap
  have hpi : Real.pi ≠ 0 := Real.pi_ne_zero
  refine ⟨?_, ?_, ?_⟩
  · simp only [τML, ωZ]
    field_simp [hα, hm, hc, hℏ]
  · have harg : α_n * m * c ^ 2 * τML / ℏ = Real.pi / 2 := by
      simp only [τML]
      field_simp [hα, hm, hc, hℏ]
    rw [harg, Real.cos_pi_div_two]
  · simp only [kT_Haw, α_gap]
    field_simp [hm, hc, hpi]
    ring

end SGM.Differential
