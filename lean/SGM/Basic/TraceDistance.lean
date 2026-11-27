import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.Adjoint
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic

/-!
# Pure-State Tail Perturbation and Box-Truncation Trace-Distance Bounds (`sgm.tex` Sec. 2)

Formalizes:
- `eq:tail_distance`: For normalized `ψ = ψ_core + ψ_tail` with `ψ_core ⊥ ψ_tail`,
  `ψ' = ψ_core + ψ_tail'` with `ψ_core ⊥ ψ_tail'`, `‖ψ_tail'‖ = ‖ψ_tail‖`, and
  `P_tail = ‖ψ_tail‖^2 ≤ 1/2`, the pure-state trace distance obeys
  `√(1 - ‖⟨ψ, ψ'⟩‖^2) ≤ 2 √P_tail`.
- Box-truncation bound: Truncating to `ψ_box = (√(1 - P_tail))⁻¹ • ψ_core` yields exact
  trace distance `√(1 - ‖⟨ψ, ψ_box⟩‖^2) = √P_tail`.
- Pure-state measurement probability difference bound: for any orthogonal projection / effect
  `0 ≤ ⟨u, Π u⟩ ≤ ‖u‖^2`, the outcome probability shift between unit states `ψ` and `ψ'` is
  bounded by the trace distance `√(1 - ‖⟨ψ, ψ'⟩‖^2)`.
-/

namespace SGM.Basic

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]

/-- `sgm.tex` Eq. `eq:tail_distance`:
Modifying the phase of a normalized state solely in its far tail (of probability `P_tail ≤ 1/2`)
changes the state by trace distance at most `2 √P_tail`. -/
theorem tail_phase_trace_distance
    (ψ_core ψ_tail ψ_tail' : H)
    (h_orth1 : inner ℂ ψ_core ψ_tail = 0)
    (h_orth2 : inner ℂ ψ_core ψ_tail' = 0)
    (h_norm_sum : ‖ψ_core‖ ^ 2 + ‖ψ_tail‖ ^ 2 = 1)
    (h_tail_eq : ‖ψ_tail'‖ = ‖ψ_tail‖)
    (h_half : ‖ψ_tail‖ ^ 2 ≤ 1 / 2) :
    Real.sqrt (1 - ‖inner ℂ (ψ_core + ψ_tail) (ψ_core + ψ_tail')‖ ^ 2) ≤
      2 * Real.sqrt (‖ψ_tail‖ ^ 2) := by
  set P : ℝ := ‖ψ_tail‖ ^ 2
  have hP_nonneg : 0 ≤ P := sq_nonneg _
  have h_core_sq : ‖ψ_core‖ ^ 2 = 1 - P := by linarith
  have h_orth1' : inner ℂ ψ_tail ψ_core = 0 := by
    rw [← inner_conj_symm, h_orth1, map_zero]
  have h_ip : inner ℂ (ψ_core + ψ_tail) (ψ_core + ψ_tail') =
      ((1 - P : ℝ) : ℂ) + inner ℂ ψ_tail ψ_tail' := by
    rw [inner_add_left, inner_add_right, inner_add_right, h_orth1', h_orth2,
      add_zero, zero_add, inner_self_eq_norm_sq_to_K]
    change ((‖ψ_core‖ : ℂ) ^ 2) + inner ℂ ψ_tail ψ_tail' = ((1 - P : ℝ) : ℂ) + inner ℂ ψ_tail ψ_tail'
    rw [← Complex.ofReal_pow, h_core_sq]
  have h_cs : ‖inner ℂ ψ_tail ψ_tail'‖ ≤ P := by
    calc ‖inner ℂ ψ_tail ψ_tail'‖
        ≤ ‖ψ_tail‖ * ‖ψ_tail'‖ := norm_inner_le_norm ψ_tail ψ_tail'
      _ = ‖ψ_tail‖ * ‖ψ_tail‖ := by rw [h_tail_eq]
      _ = P := by ring
  have h_tri : (1 - 2 * P) ≤ ‖inner ℂ (ψ_core + ψ_tail) (ψ_core + ψ_tail')‖ := by
    have h_norm_core : ‖((1 - P : ℝ) : ℂ)‖ = 1 - P := by
      rw [Complex.norm_real, Real.norm_of_nonneg (by linarith)]
    have h_rev := norm_sub_norm_le ((1 - P : ℝ) : ℂ) (-inner ℂ ψ_tail ψ_tail')
    rw [sub_neg_eq_add, norm_neg, h_norm_core, ← h_ip] at h_rev
    linarith
  have h_nonneg_lb : 0 ≤ 1 - 2 * P := by linarith
  have h_sq_lb : (1 - 2 * P) ^ 2 ≤ ‖inner ℂ (ψ_core + ψ_tail) (ψ_core + ψ_tail')‖ ^ 2 := by
    nlinarith
  have h_sub_ub : 1 - ‖inner ℂ (ψ_core + ψ_tail) (ψ_core + ψ_tail')‖ ^ 2 ≤ 4 * P := by
    nlinarith
  calc Real.sqrt (1 - ‖inner ℂ (ψ_core + ψ_tail) (ψ_core + ψ_tail')‖ ^ 2)
      ≤ Real.sqrt (4 * P) := Real.sqrt_le_sqrt h_sub_ub
    _ = 2 * Real.sqrt P := by
        rw [Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 4)]
        norm_num

/-- Exact box-truncation trace distance (`sgm.tex` Sec. 2):
Truncating `ψ = ψ_core + ψ_tail` to its renormalized core `ψ_box = (√(1 - P_tail))⁻¹ • ψ_core`
yields exact trace distance `√P_tail` (half the phase-perturbation bound). -/
theorem box_truncation_trace_distance
    (ψ_core ψ_tail : H)
    (h_orth : inner ℂ ψ_core ψ_tail = 0)
    (h_norm_sum : ‖ψ_core‖ ^ 2 + ‖ψ_tail‖ ^ 2 = 1)
    (h_lt : ‖ψ_tail‖ ^ 2 < 1) :
    let P := ‖ψ_tail‖ ^ 2
    let ψ_box : H := ((Real.sqrt (1 - P))⁻¹ : ℂ) • ψ_core
    Real.sqrt (1 - ‖inner ℂ (ψ_core + ψ_tail) ψ_box‖ ^ 2) = Real.sqrt P := by
  intro P ψ_box
  have h1mP_pos : 0 < 1 - P := by linarith
  have h_core_sq : ‖ψ_core‖ ^ 2 = 1 - P := by linarith
  have h_orth' : inner ℂ ψ_tail ψ_core = 0 := by
    rw [← inner_conj_symm, h_orth, map_zero]
  have h_ip : inner ℂ (ψ_core + ψ_tail) ψ_box = ((Real.sqrt (1 - P) : ℝ) : ℂ) := by
    simp only [ψ_box, inner_smul_right, inner_add_left, h_orth', add_zero,
      inner_self_eq_norm_sq_to_K]
    have hsq : (Real.sqrt (1 - P) : ℂ) ≠ 0 := by
      exact Complex.ofReal_ne_zero.mpr (Real.sqrt_ne_zero'.mpr h1mP_pos)
    have hnum : ((‖ψ_core‖ : ℂ) ^ 2) = (Real.sqrt (1 - P) : ℂ) ^ 2 := by
      rw [← Complex.ofReal_pow, ← Complex.ofReal_pow, h_core_sq, Real.sq_sqrt (le_of_lt h1mP_pos)]
    change (Real.sqrt (1 - P) : ℂ)⁻¹ * ((‖ψ_core‖ : ℂ) ^ 2) = (Real.sqrt (1 - P) : ℂ)
    rw [hnum]
    field_simp [hsq]
  rw [h_ip, Complex.norm_real, Real.norm_of_nonneg (Real.sqrt_nonneg _),
    Real.sq_sqrt (le_of_lt h1mP_pos)]
  ring_nf

/-- Helper (Cauchy--Schwarz for a symmetric positive semidefinite linear operator):
If `T` is symmetric and `Re ⟨u, T u⟩ ≥ 0` for all `u`, then
`‖⟨ψ, T φ⟩‖² ≤ Re ⟨ψ, T ψ⟩ * Re ⟨φ, T φ⟩`. -/
private lemma symm_pos_offdiag_sq_le
    (T : H →ₗ[ℂ] H)
    (hT_sym : ∀ u v : H, inner ℂ u (T v) = inner ℂ (T u) v)
    (hT_pos : ∀ u : H, 0 ≤ (inner ℂ u (T u)).re)
    (ψ φ : H) :
    ‖inner ℂ ψ (T φ)‖ ^ 2 ≤ (inner ℂ ψ (T ψ)).re * (inner ℂ φ (T φ)).re := by
  set a : ℝ := (inner ℂ ψ (T ψ)).re
  set d : ℝ := (inner ℂ φ (T φ)).re
  set z : ℂ := inner ℂ ψ (T φ)
  have ha : 0 ≤ a := hT_pos ψ
  have hd : 0 ≤ d := hT_pos φ
  have hquad : ∀ t : ℝ, 0 ≤ t ^ 2 * a - 2 * t * ‖z‖ ^ 2 + ‖z‖ ^ 2 * d := by
    intro t
    have h0 := hT_pos ((t : ℂ) • ψ - (starRingEnd ℂ z) • φ)
    have hφψ : inner ℂ φ (T ψ) = starRingEnd ℂ z := by
      rw [hT_sym φ ψ, inner_conj_symm]
    have hzz : starRingEnd ℂ z * z = ((‖z‖ ^ 2 : ℝ) : ℂ) := by
      rw [mul_comm, Complex.mul_conj, Complex.normSq_eq_norm_sq]
    have hzz' : z * starRingEnd ℂ z = ((‖z‖ ^ 2 : ℝ) : ℂ) := by
      rw [Complex.mul_conj, Complex.normSq_eq_norm_sq]
    have hexp :
        inner ℂ ((t : ℂ) • ψ - (starRingEnd ℂ z) • φ)
          (T ((t : ℂ) • ψ - (starRingEnd ℂ z) • φ)) =
        ((t ^ 2 : ℝ) : ℂ) * inner ℂ ψ (T ψ) -
          ((2 * t * ‖z‖ ^ 2 : ℝ) : ℂ) +
          ((‖z‖ ^ 2 : ℝ) : ℂ) * inner ℂ φ (T φ) := by
      rw [map_sub, map_smul, map_smul, inner_sub_left, inner_sub_right, inner_sub_right,
        inner_smul_left, inner_smul_left, inner_smul_left, inner_smul_left,
        inner_smul_right, inner_smul_right, inner_smul_right, inner_smul_right,
        hφψ, Complex.conj_ofReal, RingHomCompTriple.comp_apply, RingHom.id_apply]
      change (t : ℂ) * ((t : ℂ) * inner ℂ ψ (T ψ)) -
          (t : ℂ) * (starRingEnd ℂ z * z) -
          (z * ((t : ℂ) * starRingEnd ℂ z) - z * (starRingEnd ℂ z * inner ℂ φ (T φ))) = _
      have h1 : (t : ℂ) * (starRingEnd ℂ z * z) = ((t * ‖z‖ ^ 2 : ℝ) : ℂ) := by
        rw [hzz]; push_cast; ring
      have h2 : z * ((t : ℂ) * starRingEnd ℂ z) = ((t * ‖z‖ ^ 2 : ℝ) : ℂ) := by
        calc z * ((t : ℂ) * starRingEnd ℂ z) = (t : ℂ) * (z * starRingEnd ℂ z) := by ring
          _ = (t : ℂ) * ((‖z‖ ^ 2 : ℝ) : ℂ) := by rw [hzz']
          _ = ((t * ‖z‖ ^ 2 : ℝ) : ℂ) := by push_cast; ring
      have h3 : z * (starRingEnd ℂ z * inner ℂ φ (T φ)) = ((‖z‖ ^ 2 : ℝ) : ℂ) * inner ℂ φ (T φ) := by
        rw [← mul_assoc, hzz']
      rw [h1, h2, h3]
      push_cast
      ring
    rw [hexp] at h0
    simp only [Complex.add_re, Complex.sub_re, Complex.mul_re, Complex.ofReal_re,
      Complex.ofReal_im, zero_mul, sub_zero] at h0
    simpa [a, d] using h0
  by_cases ha0 : a = 0
  · have h1 := hquad (d + 1)
    rw [ha0] at h1 ⊢
    nlinarith [sq_nonneg ‖z‖]
  · have ha_pos : 0 < a := lt_of_le_of_ne ha (Ne.symm ha0)
    have h1 := hquad (‖z‖ ^ 2 / a)
    have hmul : 0 ≤ a * ((‖z‖ ^ 2 / a) ^ 2 * a - 2 * (‖z‖ ^ 2 / a) * ‖z‖ ^ 2 + ‖z‖ ^ 2 * d) :=
      mul_nonneg (le_of_lt ha_pos) h1
    have hsimp : a * ((‖z‖ ^ 2 / a) ^ 2 * a - 2 * (‖z‖ ^ 2 / a) * ‖z‖ ^ 2 + ‖z‖ ^ 2 * d) =
        ‖z‖ ^ 2 * (a * d - ‖z‖ ^ 2) := by
      field_simp [ha0]; ring
    rw [hsimp] at hmul
    by_cases hz : ‖z‖ = 0
    · rw [hz, zero_pow (by norm_num : 2 ≠ 0)]
      exact mul_nonneg ha hd
    · have hz2 : 0 < ‖z‖ ^ 2 := sq_pos_of_ne_zero hz
      nlinarith

/-- **Pure-State Measurement Probability Difference Bound** (`sgm.tex` Sec. 2):
For any symmetric quantum effect `P : H →ₗ[ℂ] H` satisfying `0 ≤ Re ⟨u, P u⟩ ≤ ‖u‖²` for all `u`
and any two unit states `ψ, ψ'`, the difference in outcome probabilities is bounded by the
pure-state trace distance `√(1 - ‖⟨ψ, ψ'⟩‖²)`. -/
theorem pure_state_measurement_prob_diff_le_trace_distance
    (P : H →ₗ[ℂ] H)
    (hP_sym : ∀ u v : H, inner ℂ u (P v) = inner ℂ (P u) v)
    (hP_eff : ∀ u : H, 0 ≤ (inner ℂ u (P u)).re ∧ (inner ℂ u (P u)).re ≤ ‖u‖ ^ 2)
    (ψ ψ' : H) (hψ : ‖ψ‖ = 1) (hψ' : ‖ψ'‖ = 1) :
    |(inner ℂ ψ (P ψ)).re - (inner ℂ ψ' (P ψ')).re| ≤
      Real.sqrt (1 - ‖inner ℂ ψ ψ'‖ ^ 2) := by
  set c : ℂ := inner ℂ ψ ψ'
  have hc_le : ‖c‖ ≤ 1 := by
    calc ‖c‖ ≤ ‖ψ‖ * ‖ψ'‖ := norm_inner_le_norm ψ ψ'
      _ = 1 := by rw [hψ, hψ', mul_one]
  have h1mc2_nonneg : 0 ≤ 1 - ‖c‖ ^ 2 := by nlinarith [norm_nonneg c]
  have hdiff_norm : ‖ψ' - c • ψ‖ ^ 2 = 1 - ‖c‖ ^ 2 := by
    have hre : (inner ℂ (ψ' - c • ψ) (ψ' - c • ψ)).re = ‖ψ' - c • ψ‖ ^ 2 := by
      rw [inner_self_eq_norm_sq_to_K]; simp [pow_two]
    rw [← hre, inner_sub_left, inner_sub_right, inner_sub_right, inner_smul_left, inner_smul_left,
      inner_smul_right, inner_smul_right, inner_self_eq_norm_sq_to_K, inner_self_eq_norm_sq_to_K,
      hψ, hψ']
    have hψ'ψ : inner ℂ ψ' ψ = starRingEnd ℂ c := by
      rw [← inner_conj_symm]
    rw [hψ'ψ]
    have hcc : starRingEnd ℂ c * c = ((‖c‖ ^ 2 : ℝ) : ℂ) := by
      rw [mul_comm, Complex.mul_conj, Complex.normSq_eq_norm_sq]
    have hcc' : c * starRingEnd ℂ c = ((‖c‖ ^ 2 : ℝ) : ℂ) := by
      rw [Complex.mul_conj, Complex.normSq_eq_norm_sq]
    change (((1 : ℂ) ^ 2 - c * starRingEnd ℂ c) -
      (starRingEnd ℂ c * c - starRingEnd ℂ c * (c * (1 : ℂ) ^ 2))).re = 1 - ‖c‖ ^ 2
    rw [one_pow, mul_one, hcc, hcc']
    simp [pow_two]
  by_cases hc1 : ‖c‖ = 1
  · have hzero : ‖ψ' - c • ψ‖ = 0 := by
      have hsq : ‖ψ' - c • ψ‖ ^ 2 = 0 := by rw [hdiff_norm, hc1]; norm_num
      exact sq_eq_zero_iff.mp hsq
    have hψ'_eq : ψ' = c • ψ := sub_eq_zero.mp (norm_eq_zero.mp hzero)
    have hEq : (inner ℂ ψ' (P ψ')).re = (inner ℂ ψ (P ψ)).re := by
      rw [hψ'_eq, map_smul, inner_smul_left, inner_smul_right, ← mul_assoc]
      have hcc : starRingEnd ℂ c * c = ((‖c‖ ^ 2 : ℝ) : ℂ) := by
        rw [mul_comm, Complex.mul_conj, Complex.normSq_eq_norm_sq]
      rw [hcc, hc1]
      simp
    rw [hEq, sub_self, abs_zero]
    exact Real.sqrt_nonneg _
  · have hc_lt : ‖c‖ < 1 := lt_of_le_of_ne hc_le hc1
    set s : ℝ := Real.sqrt (1 - ‖c‖ ^ 2)
    have hs_pos : 0 < s := Real.sqrt_pos.mpr (by nlinarith [norm_nonneg c])
    have hs_sq : s ^ 2 = 1 - ‖c‖ ^ 2 := Real.sq_sqrt h1mc2_nonneg
    set φ : H := ((s⁻¹ : ℝ) : ℂ) • (ψ' - c • ψ)
    have hψ_φ : inner ℂ ψ φ = 0 := by
      simp only [φ, inner_smul_right, inner_sub_right, inner_smul_right, inner_self_eq_norm_sq_to_K, hψ]
      change ((s⁻¹ : ℝ) : ℂ) * (c - c * (1 : ℂ) ^ 2) = 0
      ring
    have hφ_norm : ‖φ‖ = 1 := by
      rw [norm_smul, Complex.norm_real, Real.norm_of_nonneg (inv_nonneg.mpr (le_of_lt hs_pos))]
      have h1 : ‖ψ' - c • ψ‖ = s := by
        rw [← Real.sqrt_sq (norm_nonneg (ψ' - c • ψ)), hdiff_norm]
      rw [h1, inv_mul_cancel₀ (ne_of_gt hs_pos)]
    have hdecomp : ψ' = c • ψ + (s : ℂ) • φ := by
      simp only [φ, smul_smul]
      have hs_mul : (s : ℂ) * ((s⁻¹ : ℝ) : ℂ) = 1 := by
        rw [← Complex.ofReal_mul, mul_inv_cancel₀ (ne_of_gt hs_pos), Complex.ofReal_one]
      rw [hs_mul, one_smul]
      abel
    set a : ℝ := (inner ℂ ψ (P ψ)).re
    set d : ℝ := (inner ℂ φ (P φ)).re
    set z : ℂ := inner ℂ ψ (P φ)
    have ha0 : 0 ≤ a := (hP_eff ψ).1
    have ha1 : a ≤ 1 := by simpa [hψ] using (hP_eff ψ).2
    have hd0 : 0 ≤ d := (hP_eff φ).1
    have hd1 : d ≤ 1 := by simpa [hφ_norm] using (hP_eff φ).2
    -- Cauchy--Schwarz for `P ≥ 0`: `‖z‖² ≤ a * d`
    have hz_ad : ‖z‖ ^ 2 ≤ a * d :=
      symm_pos_offdiag_sq_le P hP_sym (fun u => (hP_eff u).1) ψ φ
    -- Cauchy--Schwarz for `I - P ≥ 0`: `‖z‖² ≤ (1 - a) * (1 - d)`
    let Q : H →ₗ[ℂ] H := LinearMap.id - P
    have hQ_sym : ∀ u v : H, inner ℂ u (Q v) = inner ℂ (Q u) v := by
      intro u v
      simp only [Q, LinearMap.sub_apply, LinearMap.id_apply, inner_sub_right, inner_sub_left, hP_sym]
    have hQ_pos : ∀ u : H, 0 ≤ (inner ℂ u (Q u)).re := by
      intro u
      simp only [Q, LinearMap.sub_apply, LinearMap.id_apply, inner_sub_right, Complex.sub_re,
        inner_self_eq_norm_sq_to_K]
      have hre : (((‖u‖ : ℝ) : ℂ) ^ 2).re = ‖u‖ ^ 2 := by rw [← Complex.ofReal_pow, Complex.ofReal_re]
      have h2 := (hP_eff u).2
      change (((‖u‖ : ℝ) : ℂ) ^ 2).re - (inner ℂ u (P u)).re ≥ 0
      linarith
    have hz_1ad : ‖z‖ ^ 2 ≤ (1 - a) * (1 - d) := by
      have hq := symm_pos_offdiag_sq_le Q hQ_sym hQ_pos ψ φ
      have hQψφ : inner ℂ ψ (Q φ) = -z := by
        simp only [Q, LinearMap.sub_apply, LinearMap.id_apply, inner_sub_right, hψ_φ, zero_sub, z]
      have hQψ : (inner ℂ ψ (Q ψ)).re = 1 - a := by
        simp only [Q, LinearMap.sub_apply, LinearMap.id_apply, inner_sub_right, Complex.sub_re,
          inner_self_eq_norm_sq_to_K, hψ, a]
        simp
      have hQφ : (inner ℂ φ (Q φ)).re = 1 - d := by
        simp only [Q, LinearMap.sub_apply, LinearMap.id_apply, inner_sub_right, Complex.sub_re,
          inner_self_eq_norm_sq_to_K, hφ_norm, d]
        simp
      rw [hQψφ, norm_neg, hQψ, hQφ] at hq
      exact hq
    -- Combine both bounds: `(a - d)² + 4 ‖z‖² ≤ 1`
    have hdisc_le : (a - d) ^ 2 + 4 * ‖z‖ ^ 2 ≤ 1 := by
      by_cases had : a + d ≤ 1
      · nlinarith
      · nlinarith
    -- Expand `a - Re ⟨ψ', P ψ'⟩ = s * (s * (a - d) - 2 * Re(c̄ z))`
    have hψ'_exp :
        (inner ℂ ψ' (P ψ')).re = ‖c‖ ^ 2 * a + s ^ 2 * d + 2 * s * (starRingEnd ℂ c * z).re := by
      have hbilin : ∀ x y : H,
          inner ℂ (c • x + (s : ℂ) • y) (P (c • x + (s : ℂ) • y)) =
            ((‖c‖ ^ 2 : ℝ) : ℂ) * inner ℂ x (P x) +
            ((s ^ 2 : ℝ) : ℂ) * inner ℂ y (P y) +
            (s : ℂ) * (starRingEnd ℂ c * inner ℂ x (P y)) +
            (s : ℂ) * (c * inner ℂ y (P x)) := by
        intro x y
        have hcc : starRingEnd ℂ c * c = ((‖c‖ ^ 2 : ℝ) : ℂ) := by
          rw [mul_comm, Complex.mul_conj, Complex.normSq_eq_norm_sq]
        rw [map_add, map_smul, map_smul, inner_add_left, inner_add_right, inner_add_right,
          inner_smul_left, inner_smul_left, inner_smul_left, inner_smul_left,
          inner_smul_right, inner_smul_right, inner_smul_right, inner_smul_right,
          Complex.conj_ofReal]
        have h1 : starRingEnd ℂ c * (c * inner ℂ x (P x)) = ((‖c‖ ^ 2 : ℝ) : ℂ) * inner ℂ x (P x) := by
          rw [← mul_assoc, hcc]
        rw [h1]
        push_cast
        ring
      have hφψ : inner ℂ φ (P ψ) = starRingEnd ℂ z := by
        rw [hP_sym φ ψ, inner_conj_symm]
      conv_lhs => rw [hdecomp]
      rw [hbilin ψ φ, hφψ]
      have h4 : (s : ℂ) * (c * starRingEnd ℂ z) = (s : ℂ) * starRingEnd ℂ (starRingEnd ℂ c * z) := by
        simp [mul_comm]
      rw [h4]
      simp only [Complex.add_re, Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im,
        Complex.conj_re, zero_mul, sub_zero, a, d, z]
      ring
    have hdiff_fac :
        a - (inner ℂ ψ' (P ψ')).re = s * (s * (a - d) - 2 * (starRingEnd ℂ c * z).re) := by
      rw [hψ'_exp]
      have h1mc : 1 - ‖c‖ ^ 2 = s ^ 2 := hs_sq.symm
      nlinarith
    have hcz_le : |(starRingEnd ℂ c * z).re| ≤ ‖c‖ * ‖z‖ := by
      calc |(starRingEnd ℂ c * z).re| ≤ ‖starRingEnd ℂ c * z‖ := Complex.abs_re_le_norm _
        _ = ‖starRingEnd ℂ c‖ * ‖z‖ := norm_mul _ _
        _ = ‖c‖ * ‖z‖ := by rw [Complex.norm_conj]
    have hinner_le : |s * (a - d) - 2 * (starRingEnd ℂ c * z).re| ≤ 1 := by
      have htri : |s * (a - d) - 2 * (starRingEnd ℂ c * z).re| ≤ s * |a - d| + 2 * ‖c‖ * ‖z‖ := by
        calc |s * (a - d) - 2 * (starRingEnd ℂ c * z).re|
            ≤ |s * (a - d)| + |2 * (starRingEnd ℂ c * z).re| := abs_sub _ _
          _ = s * |a - d| + 2 * |(starRingEnd ℂ c * z).re| := by
                rw [abs_mul, abs_of_pos hs_pos, abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 2)]
          _ ≤ s * |a - d| + 2 * (‖c‖ * ‖z‖) := by linarith
          _ = s * |a - d| + 2 * ‖c‖ * ‖z‖ := by ring
      have hcs : (s * |a - d| + 2 * ‖c‖ * ‖z‖) ^ 2 ≤
          (s ^ 2 + ‖c‖ ^ 2) * ((a - d) ^ 2 + 4 * ‖z‖ ^ 2) := by
        have hsq : 0 ≤ (s * (2 * ‖z‖) - ‖c‖ * |a - d|) ^ 2 := sq_nonneg _
        have habs2 : |a - d| ^ 2 = (a - d) ^ 2 := sq_abs _
        nlinarith
      have hsum1 : s ^ 2 + ‖c‖ ^ 2 = 1 := by linarith
      rw [hsum1, one_mul] at hcs
      have hsq_le : (s * |a - d| + 2 * ‖c‖ * ‖z‖) ^ 2 ≤ 1 := le_trans hcs hdisc_le
      have hnn : 0 ≤ s * |a - d| + 2 * ‖c‖ * ‖z‖ := by positivity
      have hle1 : s * |a - d| + 2 * ‖c‖ * ‖z‖ ≤ 1 := by nlinarith
      exact le_trans htri hle1
    rw [hdiff_fac, abs_mul, abs_of_pos hs_pos]
    nlinarith

end SGM.Basic
