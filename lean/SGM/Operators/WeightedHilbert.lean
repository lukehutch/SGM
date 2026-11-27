import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.FDeriv.Linear
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Tactic

/-!
# Weighted vs. Unweighted Hilbert Spaces, Bounded-Lapse Norm Equivalence, and Ehrenfest Theorem
(`sgm.tex` Sec. 2.2 `thm:universal`, `eq:ehrenfest_commutator`; `sgm_wavefn.tex` Prop. 2 & Prop. 3(ii))

Formalizes:
- `eq:weighted_norm_equiv` (`prop:causal_lapse_gap`(ii)):
  For any measure space `(X, μ)` and bounded lapse `0 < α_min ≤ α_n(x) ≤ α_max`,
  the weighted `L²` norm `∫ α_n(x)⁻¹ ‖φ(x)‖² dμ` is uniformly equivalent to the unweighted norm:
  `α_max⁻¹ ∫ ‖φ‖² dμ ≤ ∫ α_n⁻¹ ‖φ‖² dμ ≤ α_min⁻¹ ∫ ‖φ‖² dμ`.
- Pointwise unitary rescaling isometry `χ̃(x) = α_n(x)^{-1/2} χ(x)` (`prop:unitary_norm`),
  mapping weighted inner products `α_n⁻¹ ⟨φ, ψ⟩` identically to unweighted inner products `⟨φ̃, ψ̃⟩`.
- `eq:ehrenfest_commutator` (`sgm.tex` Route IV & Theorem 1):
  Exact Ehrenfest expectation-value derivative `d/dt ⟨ψ(t), A ψ(t)⟩ = (1/(i ℏ)) ⟨ψ(t), [A, H] ψ(t)⟩`
  with explicit symmetry hypotheses on `H` and `A` (and `d/dt ⟨ψ(t), ψ(t)⟩ = 0` as the `A = I` case).
- `eq:sgm_expectation`: Whenever `[P, H] = (-i ℏ) • ∇H`, Ehrenfest reduces identically to
  `d/dt ⟨ψ(t), P ψ(t)⟩ = -⟨ψ(t), (∇H) ψ(t)⟩`.
-/

namespace SGM.Operators

/-! ### Part 1: Bounded-Lapse Norm Equivalence & Unitary Rescaling (`eq:weighted_norm_equiv`) -/

/-- `sgm_wavefn.tex` Proposition `prop:causal_lapse_gap`(ii) (`eq:weighted_norm_equiv`):
On any measure space `(X, μ)`, if `0 < α_min ≤ α_n(x) ≤ α_max` pointwise (with `0 < α_max`),
then the weighted `L²` squared norm satisfies
`α_max⁻¹ ∫ ‖φ‖² dμ ≤ ∫ α_n⁻¹ ‖φ‖² dμ ≤ α_min⁻¹ ∫ ‖φ‖² dμ`. -/
theorem weighted_norm_equiv_integral
    {X : Type*} [MeasurableSpace X] {μ : MeasureTheory.Measure X}
    {α_n : X → ℝ} {α_min α_max : ℝ}
    (hmin : 0 < α_min)
    (hlo : ∀ x, α_min ≤ α_n x)
    (hhi : ∀ x, α_n x ≤ α_max)
    {H : Type*} [NormedAddCommGroup H]
    (φ : X → H)
    (hInt0 : MeasureTheory.Integrable (fun x => ‖φ x‖ ^ 2) μ)
    (hIntW : MeasureTheory.Integrable (fun x => (α_n x)⁻¹ * ‖φ x‖ ^ 2) μ) :
    α_max⁻¹ * (∫ x, ‖φ x‖ ^ 2 ∂μ) ≤ (∫ x, (α_n x)⁻¹ * ‖φ x‖ ^ 2 ∂μ) ∧
    (∫ x, (α_n x)⁻¹ * ‖φ x‖ ^ 2 ∂μ) ≤ α_min⁻¹ * (∫ x, ‖φ x‖ ^ 2 ∂μ) := by
  have hα_pos : ∀ x, 0 < α_n x := fun x => lt_of_lt_of_le hmin (hlo x)
  constructor
  · rw [← MeasureTheory.integral_const_mul]
    apply MeasureTheory.integral_mono (hInt0.const_mul α_max⁻¹) hIntW
    intro x
    have hinv : α_max⁻¹ ≤ (α_n x)⁻¹ := inv_anti₀ (hα_pos x) (hhi x)
    exact mul_le_mul_of_nonneg_right hinv (sq_nonneg _)
  · rw [← MeasureTheory.integral_const_mul]
    apply MeasureTheory.integral_mono hIntW (hInt0.const_mul α_min⁻¹)
    intro x
    have hinv : (α_n x)⁻¹ ≤ α_min⁻¹ := inv_anti₀ hmin (hlo x)
    exact mul_le_mul_of_nonneg_right hinv (sq_nonneg _)

/-- Pointwise unitary rescaling isometry (`prop:unitary_norm`, `χ̃ = α_n^{-1/2} χ`):
for any `α > 0` and vectors `φ, ψ` in a complex inner product space,
`⟨(α^{-1/2}) • φ, (α^{-1/2}) • ψ⟩ = α⁻¹ * ⟨φ, ψ⟩`. -/
theorem lapse_rescaling_isometry
    {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]
    {α : ℝ} (hα : 0 < α) (φ ψ : H) :
    inner ℂ ((((Real.sqrt α)⁻¹ : ℝ) : ℂ) • φ) ((((Real.sqrt α)⁻¹ : ℝ) : ℂ) • ψ) =
      ((α⁻¹ : ℝ) : ℂ) * inner ℂ φ ψ := by
  rw [inner_smul_left, inner_smul_right]
  simp only [Complex.conj_ofReal]
  rw [← mul_assoc, ← Complex.ofReal_mul]
  have hsq : (Real.sqrt α)⁻¹ * (Real.sqrt α)⁻¹ = α⁻¹ := by
    rw [← sq, inv_pow, Real.sq_sqrt (le_of_lt hα)]
  rw [hsq]

/-- Operator similarity under lapse rescaling: if `H_w` is symmetric in the `α⁻¹`-weighted
inner product (abstractly, `⟨R u, R (H_w v)⟩ = ⟨R (H_w u), R v⟩`),
then the rescaled operator `H̃ = R H_w R⁻¹` is symmetric in the unweighted inner product! -/
theorem weighted_to_unweighted_symmetric
    {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]
    (H_w R R_inv : H →ₗ[ℂ] H)
    (hR_inv : ∀ u : H, R (R_inv u) = u)
    (hHw_wsym : ∀ u v : H, inner ℂ (R u) (R (H_w v)) = inner ℂ (R (H_w u)) (R v)) :
    let H_tilde := R.comp (H_w.comp R_inv)
    ∀ u v : H, inner ℂ u (H_tilde v) = inner ℂ (H_tilde u) v := by
  intro H_tilde u v
  simp only [H_tilde, LinearMap.comp_apply]
  have hu : R (R_inv u) = u := hR_inv u
  have hv : R (R_inv v) = v := hR_inv v
  calc inner ℂ u (R (H_w (R_inv v)))
      = inner ℂ (R (R_inv u)) (R (H_w (R_inv v))) := by rw [hu]
    _ = inner ℂ (R (H_w (R_inv u))) (R (R_inv v)) := hHw_wsym (R_inv u) (R_inv v)
    _ = inner ℂ (R (H_w (R_inv u))) v := by rw [hv]

/-! ### Part 2: Route IV & Theorem 1 — Ehrenfest Commutator & Force Expectation (`sgm.tex` Sec. 2.2) -/

/-- `sgm.tex` Eq. `eq:ehrenfest_commutator` (Route IV & Theorem 1):
For a differentiable trajectory `ψ : ℝ → H` solving the Schrödinger equation
`∂_t ψ(t) = (-i / ℏ) • H_op (ψ t)` with a symmetric Hamiltonian `H_op` and a continuous
linear observable `A_op`, the time derivative of the expectation value `⟨ψ(t), A_op (ψ t)⟩`
equals `(1 / (i ℏ)) ⟨ψ(t), [A_op, H_op] ψ(t)⟩`. -/
theorem ehrenfest_commutator_derivative
    {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]
    (H_op : H →ₗ[ℂ] H) (A_op : H →L[ℂ] H)
    (hH_sym : ∀ u v : H, inner ℂ u (H_op v) = inner ℂ (H_op u) v)
    {ℏ : ℝ} (hℏ : ℏ ≠ 0)
    (ψ : ℝ → H) (t : ℝ)
    (hSchr : HasDerivAt ψ (((-Complex.I / (ℏ : ℂ)) : ℂ) • H_op (ψ t)) t) :
    HasDerivAt (fun s => inner ℂ (ψ s) (A_op (ψ s)))
      ((1 / (Complex.I * (ℏ : ℂ))) *
        inner ℂ (ψ t) (A_op (H_op (ψ t)) - H_op (A_op (ψ t)))) t := by
  have hA_deriv : HasDerivAt (fun s => A_op (ψ s))
      (A_op (((-Complex.I / (ℏ : ℂ)) : ℂ) • H_op (ψ t))) t :=
    (A_op.restrictScalars ℝ).hasFDerivAt.comp_hasDerivAt t hSchr
  have hprod := hSchr.inner ℂ hA_deriv
  convert hprod using 1
  rw [map_smul, inner_smul_left, inner_smul_right, inner_sub_right]
  have hconj : starRingEnd ℂ (-Complex.I / (ℏ : ℂ)) = Complex.I / (ℏ : ℂ) := by
    simp
  rw [hconj, hH_sym (ψ t) (A_op (ψ t))]
  have hℏc : (ℏ : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr hℏ
  have hI : Complex.I ≠ 0 := Complex.I_ne_zero
  have hc1 : (1 : ℂ) / (Complex.I * (ℏ : ℂ)) = -Complex.I / (ℏ : ℂ) := by
    field_simp [hI, hℏc]
    simp [Complex.I_sq]
  rw [hc1]
  ring

/-- `sgm.tex` Eq. `eq:sgm_expectation`:
If a momentum observable `P_op` satisfies the canonical commutator identity
`[P_op, H_op] ψ = (-i ℏ) • gradH ψ`, then Ehrenfest's theorem yields the exact
Spectral Gradient Mechanics expectation-value force law `d/dt ⟨P_op⟩ = -⟨gradH⟩`. -/
theorem sgm_ehrenfest_force_law
    {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]
    (H_op : H →ₗ[ℂ] H) (P_op : H →L[ℂ] H) (gradH : H →ₗ[ℂ] H)
    (hH_sym : ∀ u v : H, inner ℂ u (H_op v) = inner ℂ (H_op u) v)
    {ℏ : ℝ} (hℏ : ℏ ≠ 0)
    (ψ : ℝ → H) (t : ℝ)
    (hSchr : HasDerivAt ψ (((-Complex.I / (ℏ : ℂ)) : ℂ) • H_op (ψ t)) t)
    (hComm : P_op (H_op (ψ t)) - H_op (P_op (ψ t)) = ((-Complex.I * (ℏ : ℂ)) : ℂ) • gradH (ψ t)) :
    HasDerivAt (fun s => inner ℂ (ψ s) (P_op (ψ s))) (-inner ℂ (ψ t) (gradH (ψ t))) t := by
  have h := ehrenfest_commutator_derivative H_op P_op hH_sym hℏ ψ t hSchr
  rw [hComm, inner_smul_right] at h
  convert h using 1
  have hℏc : (ℏ : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr hℏ
  have hI : Complex.I ≠ 0 := Complex.I_ne_zero
  field_simp [hI, hℏc]

end SGM.Operators
