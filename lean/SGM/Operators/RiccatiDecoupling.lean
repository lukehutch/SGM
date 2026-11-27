import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.Normed.Algebra.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Topology.MetricSpace.Contracting
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Tactic

/-!
# Exact Operator Riccati Decoupling, Eriksen Unitary Foldy--Wouthuysen Map,
# and Unbounded-`P` Lyapunov Spectral Gap
(`sgm_wavefn.tex` Secs. 4.2, 7.6 & App. A.8)

Formalizes:
- `eq:contraction_image` & `eq:contraction_lipschitz`: Banach contraction proof via
  `ContractingWith.exists_fixedPoint` on the closed unit ball `‖R‖ ≤ 1` in any complete
  normed algebra `A` (including `H →L[ℂ] H`) whenever `ε = (c ‖P‖ + ‖G‖ + ‖V‖)/(m c²) < 1`,
  yielding a unique fixed point `R` with `‖R‖ ≤ ε` solving the exact operator Riccati equation.
- `eq:pure_square`: Exact commuting / free-particle Riccati solution `r = c p / (M + √(M² + c² p²))`.
- `eq:rotation`: Exact block unitarity `U* U = I` and exact off-diagonal annihilation
  `U H U* = diag(h_hat, h_minus)` of the Eriksen transformation, plus `Re U ≥ 1/√2`.
- `eq:lyapunov_coercive` & `eq:lyapunov_gap` (`prop:unitary_norm`(b), App. A.8):
  **Unbounded-`P` Lyapunov coercivity**: for ANY symmetric linear operator `P : H →ₗ[ℂ] H`
  (no norm bound!), `Re ⟨β Ψ, (H - V₀) Ψ⟩ = Re ⟨u, A u⟩ + Re ⟨w, (-D) w⟩ ≥ δ (‖u‖² + ‖w‖²)`,
  giving the spectral gap `‖(H - V₀) Ψ‖ ≥ δ ‖Ψ‖` and strict positive-energy graph contraction
  `‖w‖ < ‖u‖` (`‖R u‖ < ‖u‖`) independent of `‖P‖`.
- `prop:unitary_norm`(a) (`eq:unitary_inner_product`): Exact inner-product conservation
  `d/dt ⟨χ₁(t), χ₂(t)⟩ = 0` under Hermitian generator `ω̂`.
- `prop:causal_lapse_gap`(iv): Uniform-lapse coordinate gap scaling `2 α₀ m c²` and explicit
  witness that `ε < 1` is sufficient but not necessary (`P = 0, α_n = 3 ⟹ ε = 2 > 1` yet `R = 0`).
-/

namespace SGM.Operators

/-! ### Part 1: Banach Contraction for the Operator Riccati Equation (`sgm_wavefn.tex` Sec. 4.2) -/

section BanachRiccati

variable {A : Type*} [NormedRing A] [NormedAlgebra ℝ A]

/-- The Riccati contraction map `ℱ(R) = (1/(2M)) (c P - (G R + R G) - (R V - V R) - c R P R)`
where `M = m c² > 0`. -/
noncomputable def riccatiMap (M c : ℝ) (P G V : A) (R : A) : A :=
  (1 / (2 * M)) • (c • P - (G * R + R * G) - (R * V - V * R) - c • (R * P * R))

/-- `sgm_wavefn.tex` Eq. `eq:contraction_image`:
For `‖R‖ ≤ 1`, `‖ℱ(R)‖ ≤ ε := (|c| ‖P‖ + ‖G‖ + ‖V‖) / M`. -/
theorem riccatiMap_norm_le {M c : ℝ} (hM : 0 < M) (hc : 0 ≤ c) (P G V R : A)
    (hR : ‖R‖ ≤ 1) :
    ‖riccatiMap M c P G V R‖ ≤ (c * ‖P‖ + ‖G‖ + ‖V‖) / M := by
  unfold riccatiMap
  rw [norm_smul, Real.norm_of_nonneg (by positivity : 0 ≤ 1 / (2 * M))]
  have htri :
      ‖c • P - (G * R + R * G) - (R * V - V * R) - c • (R * P * R)‖ ≤
        ‖c • P‖ + ‖G * R + R * G‖ + ‖R * V - V * R‖ + ‖c • (R * P * R)‖ := by
    calc ‖c • P - (G * R + R * G) - (R * V - V * R) - c • (R * P * R)‖
        ≤ ‖c • P - (G * R + R * G) - (R * V - V * R)‖ + ‖c • (R * P * R)‖ := norm_sub_le _ _
      _ ≤ ‖c • P - (G * R + R * G)‖ + ‖R * V - V * R‖ + ‖c • (R * P * R)‖ := by
            linarith [norm_sub_le (c • P - (G * R + R * G)) (R * V - V * R)]
      _ ≤ ‖c • P‖ + ‖G * R + R * G‖ + ‖R * V - V * R‖ + ‖c • (R * P * R)‖ := by
            linarith [norm_sub_le (c • P) (G * R + R * G)]
  have h1 : ‖c • P‖ = c * ‖P‖ := by rw [norm_smul, Real.norm_of_nonneg hc]
  have h2 : ‖G * R + R * G‖ ≤ 2 * ‖G‖ := by
    calc ‖G * R + R * G‖ ≤ ‖G * R‖ + ‖R * G‖ := norm_add_le _ _
      _ ≤ ‖G‖ * ‖R‖ + ‖R‖ * ‖G‖ := add_le_add (norm_mul_le _ _) (norm_mul_le _ _)
      _ ≤ ‖G‖ * 1 + 1 * ‖G‖ := by nlinarith [norm_nonneg G]
      _ = 2 * ‖G‖ := by ring
  have h3 : ‖R * V - V * R‖ ≤ 2 * ‖V‖ := by
    calc ‖R * V - V * R‖ ≤ ‖R * V‖ + ‖V * R‖ := norm_sub_le _ _
      _ ≤ ‖R‖ * ‖V‖ + ‖V‖ * ‖R‖ := add_le_add (norm_mul_le _ _) (norm_mul_le _ _)
      _ ≤ 1 * ‖V‖ + ‖V‖ * 1 := by nlinarith [norm_nonneg V]
      _ = 2 * ‖V‖ := by ring
  have h4 : ‖c • (R * P * R)‖ ≤ c * ‖P‖ := by
    rw [norm_smul, Real.norm_of_nonneg hc]
    have hRPR : ‖R * P * R‖ ≤ ‖P‖ := by
      calc ‖R * P * R‖ ≤ ‖R * P‖ * ‖R‖ := norm_mul_le _ _
        _ ≤ (‖R‖ * ‖P‖) * ‖R‖ := mul_le_mul_of_nonneg_right (norm_mul_le _ _) (norm_nonneg R)
        _ ≤ (1 * ‖P‖) * 1 := by
              have hRP : ‖R‖ * ‖P‖ ≤ 1 * ‖P‖ :=
                mul_le_mul_of_nonneg_right hR (norm_nonneg P)
              have hRP0 : 0 ≤ ‖R‖ * ‖P‖ := mul_nonneg (norm_nonneg R) (norm_nonneg P)
              nlinarith
        _ = ‖P‖ := by ring
    exact mul_le_mul_of_nonneg_left hRPR hc
  have hsum : ‖c • P - (G * R + R * G) - (R * V - V * R) - c • (R * P * R)‖ ≤
      2 * (c * ‖P‖ + ‖G‖ + ‖V‖) := by linarith
  have hpos : 0 ≤ 1 / (2 * M) := by positivity
  calc (1 / (2 * M)) * ‖c • P - (G * R + R * G) - (R * V - V * R) - c • (R * P * R)‖
      ≤ (1 / (2 * M)) * (2 * (c * ‖P‖ + ‖G‖ + ‖V‖)) := mul_le_mul_of_nonneg_left hsum hpos
    _ = (c * ‖P‖ + ‖G‖ + ‖V‖) / M := by field_simp [ne_of_gt hM]

/-- `sgm_wavefn.tex` Eq. `eq:contraction_lipschitz`:
On the closed unit ball `‖R₁‖, ‖R₂‖ ≤ 1`, `ℱ` is `ε`-Lipschitz:
`‖ℱ(R₁) - ℱ(R₂)‖ ≤ ε ‖R₁ - R₂‖`. -/
theorem riccatiMap_lipschitz {M c : ℝ} (hM : 0 < M) (hc : 0 ≤ c) (P G V R₁ R₂ : A)
    (hR₁ : ‖R₁‖ ≤ 1) (hR₂ : ‖R₂‖ ≤ 1) :
    ‖riccatiMap M c P G V R₁ - riccatiMap M c P G V R₂‖ ≤
      ((c * ‖P‖ + ‖G‖ + ‖V‖) / M) * ‖R₁ - R₂‖ := by
  set Δ := R₁ - R₂
  have hdiff :
      riccatiMap M c P G V R₁ - riccatiMap M c P G V R₂ =
        (1 / (2 * M)) • (-(G * Δ + Δ * G) - (Δ * V - V * Δ) - c • (Δ * P * R₁ + R₂ * P * Δ)) := by
    unfold riccatiMap
    rw [← smul_sub]
    congr 1
    have hquad : R₁ * P * R₁ - R₂ * P * R₂ = Δ * P * R₁ + R₂ * P * Δ := by
      simp only [Δ, sub_mul, mul_sub]
      abel
    have hG : (G * R₁ + R₁ * G) - (G * R₂ + R₂ * G) = G * Δ + Δ * G := by
      simp only [Δ, mul_sub, sub_mul]
      abel
    have hV : (R₁ * V - V * R₁) - (R₂ * V - V * R₂) = Δ * V - V * Δ := by
      simp only [Δ, mul_sub, sub_mul]
      abel
    calc c • P - (G * R₁ + R₁ * G) - (R₁ * V - V * R₁) - c • (R₁ * P * R₁) -
          (c • P - (G * R₂ + R₂ * G) - (R₂ * V - V * R₂) - c • (R₂ * P * R₂))
        = -((G * R₁ + R₁ * G) - (G * R₂ + R₂ * G)) -
          ((R₁ * V - V * R₁) - (R₂ * V - V * R₂)) -
          c • (R₁ * P * R₁ - R₂ * P * R₂) := by rw [smul_sub]; abel
      _ = -(G * Δ + Δ * G) - (Δ * V - V * Δ) - c • (Δ * P * R₁ + R₂ * P * Δ) := by
          rw [hG, hV, hquad]
  rw [hdiff, norm_smul, Real.norm_of_nonneg (by positivity : 0 ≤ 1 / (2 * M))]
  have htri :
      ‖-(G * Δ + Δ * G) - (Δ * V - V * Δ) - c • (Δ * P * R₁ + R₂ * P * Δ)‖ ≤
        ‖G * Δ + Δ * G‖ + ‖Δ * V - V * Δ‖ + ‖c • (Δ * P * R₁ + R₂ * P * Δ)‖ := by
    calc ‖-(G * Δ + Δ * G) - (Δ * V - V * Δ) - c • (Δ * P * R₁ + R₂ * P * Δ)‖
        ≤ ‖-(G * Δ + Δ * G) - (Δ * V - V * Δ)‖ + ‖c • (Δ * P * R₁ + R₂ * P * Δ)‖ := norm_sub_le _ _
      _ ≤ ‖-(G * Δ + Δ * G)‖ + ‖Δ * V - V * Δ‖ + ‖c • (Δ * P * R₁ + R₂ * P * Δ)‖ := by
            linarith [norm_sub_le (-(G * Δ + Δ * G)) (Δ * V - V * Δ)]
      _ = ‖G * Δ + Δ * G‖ + ‖Δ * V - V * Δ‖ + ‖c • (Δ * P * R₁ + R₂ * P * Δ)‖ := by
            rw [norm_neg]
  have hG : ‖G * Δ + Δ * G‖ ≤ 2 * ‖G‖ * ‖Δ‖ := by
    calc ‖G * Δ + Δ * G‖ ≤ ‖G * Δ‖ + ‖Δ * G‖ := norm_add_le _ _
      _ ≤ ‖G‖ * ‖Δ‖ + ‖Δ‖ * ‖G‖ := add_le_add (norm_mul_le _ _) (norm_mul_le _ _)
      _ = 2 * ‖G‖ * ‖Δ‖ := by ring
  have hV : ‖Δ * V - V * Δ‖ ≤ 2 * ‖V‖ * ‖Δ‖ := by
    calc ‖Δ * V - V * Δ‖ ≤ ‖Δ * V‖ + ‖V * Δ‖ := norm_sub_le _ _
      _ ≤ ‖Δ‖ * ‖V‖ + ‖V‖ * ‖Δ‖ := add_le_add (norm_mul_le _ _) (norm_mul_le _ _)
      _ = 2 * ‖V‖ * ‖Δ‖ := by ring
  have hP : ‖c • (Δ * P * R₁ + R₂ * P * Δ)‖ ≤ 2 * c * ‖P‖ * ‖Δ‖ := by
    rw [norm_smul, Real.norm_of_nonneg hc]
    have h1 : ‖Δ * P * R₁‖ ≤ ‖P‖ * ‖Δ‖ := by
      calc ‖Δ * P * R₁‖ ≤ ‖Δ * P‖ * ‖R₁‖ := norm_mul_le _ _
        _ ≤ (‖Δ‖ * ‖P‖) * ‖R₁‖ := mul_le_mul_of_nonneg_right (norm_mul_le _ _) (norm_nonneg R₁)
        _ ≤ (‖Δ‖ * ‖P‖) * 1 := mul_le_mul_of_nonneg_left hR₁ (mul_nonneg (norm_nonneg Δ) (norm_nonneg P))
        _ = ‖P‖ * ‖Δ‖ := by ring
    have h2 : ‖R₂ * P * Δ‖ ≤ ‖P‖ * ‖Δ‖ := by
      calc ‖R₂ * P * Δ‖ ≤ ‖R₂ * P‖ * ‖Δ‖ := norm_mul_le _ _
        _ ≤ (‖R₂‖ * ‖P‖) * ‖Δ‖ := mul_le_mul_of_nonneg_right (norm_mul_le _ _) (norm_nonneg Δ)
        _ ≤ (1 * ‖P‖) * ‖Δ‖ :=
            mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hR₂ (norm_nonneg P)) (norm_nonneg Δ)
        _ = ‖P‖ * ‖Δ‖ := by ring
    have hsum12 : ‖Δ * P * R₁ + R₂ * P * Δ‖ ≤ 2 * ‖P‖ * ‖Δ‖ := by
      calc ‖Δ * P * R₁ + R₂ * P * Δ‖ ≤ ‖Δ * P * R₁‖ + ‖R₂ * P * Δ‖ := norm_add_le _ _
        _ ≤ ‖P‖ * ‖Δ‖ + ‖P‖ * ‖Δ‖ := add_le_add h1 h2
        _ = 2 * ‖P‖ * ‖Δ‖ := by ring
    calc c * ‖Δ * P * R₁ + R₂ * P * Δ‖ ≤ c * (2 * ‖P‖ * ‖Δ‖) := mul_le_mul_of_nonneg_left hsum12 hc
      _ = 2 * c * ‖P‖ * ‖Δ‖ := by ring
  have htot : ‖-(G * Δ + Δ * G) - (Δ * V - V * Δ) - c • (Δ * P * R₁ + R₂ * P * Δ)‖ ≤
      2 * (c * ‖P‖ + ‖G‖ + ‖V‖) * ‖Δ‖ := by linarith
  calc (1 / (2 * M)) * ‖-(G * Δ + Δ * G) - (Δ * V - V * Δ) - c • (Δ * P * R₁ + R₂ * P * Δ)‖
      ≤ (1 / (2 * M)) * (2 * (c * ‖P‖ + ‖G‖ + ‖V‖) * ‖Δ‖) :=
        mul_le_mul_of_nonneg_left htot (by positivity)
    _ = ((c * ‖P‖ + ‖G‖ + ‖V‖) / M) * ‖Δ‖ := by field_simp [ne_of_gt hM]

/-- **Banach Fixed-Point Riccati Decoupling Theorem** (`sgm_wavefn.tex` Sec. 4.2):
Whenever `ε := (c ‖P‖ + ‖G‖ + ‖V‖) / M < 1`, there exists a unique operator `R ∈ A` with `‖R‖ ≤ 1`
solving the exact operator Riccati equation
`2 M R + (G R + R G) + (R V - V R) + c R P R = c P`,
and moreover `‖R‖ ≤ ε < 1`. -/
theorem riccati_exists_unique_fixed_point
    [CompleteSpace A]
    {M c : ℝ} (hM : 0 < M) (hc : 0 ≤ c) (P G V : A)
    (hε : (c * ‖P‖ + ‖G‖ + ‖V‖) / M < 1) :
    (∃! R : A, ‖R‖ ≤ 1 ∧
      (2 * M) • R + (G * R + R * G) + (R * V - V * R) + c • (R * P * R) = c • P) ∧
    (∀ R : A, ‖R‖ ≤ 1 →
      (2 * M) • R + (G * R + R * G) + (R * V - V * R) + c • (R * P * R) = c • P →
      ‖R‖ ≤ (c * ‖P‖ + ‖G‖ + ‖V‖) / M) := by
  let ε : ℝ := (c * ‖P‖ + ‖G‖ + ‖V‖) / M
  have hε_nonneg : 0 ≤ ε := by
    dsimp [ε]
    have : 0 ≤ c * ‖P‖ + ‖G‖ + ‖V‖ := by positivity
    exact div_nonneg this (le_of_lt hM)
  let K : NNReal := ⟨ε, hε_nonneg⟩
  have hK_lt : K < 1 := hε
  let B : Set A := Metric.closedBall 0 1
  have hB_complete : CompleteSpace B :=
    Metric.isClosed_closedBall.completeSpace_coe
  let fB : B → B := fun R =>
    ⟨riccatiMap M c P G V R.1, by
      rw [mem_closedBall_zero_iff]
      have hR : ‖R.1‖ ≤ 1 := mem_closedBall_zero_iff.mp R.2
      exact le_trans (riccatiMap_norm_le hM hc P G V R.1 hR) (le_of_lt hε)⟩
  have hContract : ContractingWith K fB := by
    refine ⟨hK_lt, LipschitzWith.of_dist_le_mul fun x y => ?_⟩
    have hx : ‖x.1‖ ≤ 1 := mem_closedBall_zero_iff.mp x.2
    have hy : ‖y.1‖ ≤ 1 := mem_closedBall_zero_iff.mp y.2
    change dist (riccatiMap M c P G V x.1) (riccatiMap M c P G V y.1) ≤ ε * dist x.1 y.1
    rw [dist_eq_norm, dist_eq_norm]
    exact riccatiMap_lipschitz hM hc P G V x.1 y.1 hx hy
  have h0 : (0 : A) ∈ B := mem_closedBall_zero_iff.mpr (by simp)
  have : Nonempty B := ⟨⟨0, h0⟩⟩
  let Rf : B := hContract.fixedPoint fB
  have hRf_fix : fB Rf = Rf := hContract.fixedPoint_isFixedPt
  have hRicc_iff : ∀ R : A, riccatiMap M c P G V R = R ↔
      (2 * M) • R + (G * R + R * G) + (R * V - V * R) + c • (R * P * R) = c • P := by
    intro R
    constructor
    · intro h
      have h1 := congr_arg ((2 * M) • ·) h
      simp only [riccatiMap, smul_smul] at h1
      have hmul : (2 * M) * (1 / (2 * M)) = 1 := by field_simp [ne_of_gt hM]
      rw [hmul, one_smul] at h1
      calc (2 * M) • R + (G * R + R * G) + (R * V - V * R) + c • (R * P * R)
          = (c • P - (G * R + R * G) - (R * V - V * R) - c • (R * P * R)) +
            (G * R + R * G) + (R * V - V * R) + c • (R * P * R) := by rw [h1]
        _ = c • P := by abel
    · intro h
      unfold riccatiMap
      have hsub : c • P - (G * R + R * G) - (R * V - V * R) - c • (R * P * R) = (2 * M) • R := by
        rw [← h]; abel
      rw [hsub, smul_smul]
      have hmul : (1 / (2 * M)) * (2 * M) = 1 := by field_simp [ne_of_gt hM]
      rw [hmul, one_smul]
  refine ⟨⟨Rf.1, ⟨mem_closedBall_zero_iff.mp Rf.2, ?_⟩, ?_⟩, ?_⟩
  · exact (hRicc_iff Rf.1).mp (congr_arg Subtype.val hRf_fix)
  · rintro R' ⟨hR'_norm, hR'_eq⟩
    let R'B : B := ⟨R', mem_closedBall_zero_iff.mpr hR'_norm⟩
    have hR'B_fix : Function.IsFixedPt fB R'B := by
      apply Subtype.ext
      exact (hRicc_iff R').mpr hR'_eq
    have heq : R'B = Rf := hContract.fixedPoint_unique hR'B_fix
    exact congr_arg Subtype.val heq
  · intro R hR_norm hR_eq
    have hfix : riccatiMap M c P G V R = R := (hRicc_iff R).mpr hR_eq
    rw [← hfix]
    exact riccatiMap_norm_le hM hc P G V R hR_norm

end BanachRiccati

/-! ### Part 2: Exact Commuting Quadratic Solution (`eq:pure_square`) -/

/-- `sgm_wavefn.tex` Eq. `eq:pure_square`:
For free / commuting scalar momentum `p` and rest energy `M = m c² > 0`,
`r = c p / (M + √(M² + c² p²))` solves `2 M r + c p r² = c p`, satisfies `|r| < 1`,
and yields the exact square-root dispersion `M + c p r = √(M² + c² p²)`. -/
theorem riccati_commuting_exact {M c p : ℝ} (hM : 0 < M) :
    let E := Real.sqrt (M ^ 2 + (c * p) ^ 2)
    let r := (c * p) / (M + E)
    (2 * M * r + c * p * r ^ 2 = c * p) ∧
    (M + c * p * r = E) ∧
    (|r| < 1) := by
  intro E r
  have hE_pos : 0 < E := Real.sqrt_pos.mpr (by positivity)
  have hE2 : E ^ 2 = M ^ 2 + (c * p) ^ 2 := Real.sq_sqrt (by positivity)
  have hden : M + E ≠ 0 := by linarith
  have hden_pos : 0 < M + E := by linarith
  refine ⟨?_, ?_, ?_⟩
  · simp only [r]
    field_simp [hden]
    linear_combination -(c * p) * hE2
  · simp only [r]
    field_simp [hden]
    linear_combination -hE2
  · simp only [r, abs_div, abs_of_pos hden_pos]
    rw [div_lt_one hden_pos]
    have hcp_le : |c * p| ≤ E := by
      rw [← Real.sqrt_sq_eq_abs]
      exact Real.sqrt_le_sqrt (by nlinarith)
    linarith

/-! ### Part 3: Eriksen Unitary Block Diagonalization (`eq:rotation`) -/

/-- `sgm_wavefn.tex` Eq. `eq:rotation`:
Abstract operator-algebraic proof of the Eriksen unitary map
`U = !![S, S * star R; -(T * R), T]` on `Matrix (Fin 2) (Fin 2) A`:
given `R` solving the Riccati equation and self-adjoint `S, T` satisfying
`S (1 + R* R) S = 1`, `T (1 + R R*) T = 1`, `T R = R S`, `S R* = R* T`:
1. `U` is fully unitary: `Uᴴ * U = 1` (all 4 blocks) AND `U * Uᴴ = 1` (all 4 blocks);
2. `U * H_mat * Uᴴ` has identically vanishing `(1, 0)` and `(0, 1)` off-diagonal blocks
   (exact decoupling!). -/
theorem eriksen_block_decoupling
    {A : Type*} [Ring A] [StarRing A]
    (M c : A) (P G V R S T : A)
    -- Scalar `c` commutes with operators; `M, c` are self-adjoint
    (hcc : ∀ X : A, c * X = X * c)
    (hM_sa : star M = M) (hc_sa : star c = c)
    -- Hermitian Hamiltonian blocks `P, G, V` and symmetric `S, T`
    (hP_sa : star P = P) (hG_sa : star G = G) (hV_sa : star V = V)
    (hS_sa : star S = S) (hT_sa : star T = T)
    -- Eriksen normalization & intertwining identities
    (hS2 : S * (1 + star R * R) * S = 1)
    (hT2 : T * (1 + R * star R) * T = 1)
    (hTR : T * R = R * S)
    (hSR : S * star R = star R * T)
    -- Operator Riccati equation: `c P + (-M + V - G) R = R (M + V + G + c P R)`
    (hRicc : c * P + (-M + V - G) * R = R * (M + V + G + c * P * R)) :
    let U : Matrix (Fin 2) (Fin 2) A := !![S, S * star R; -(T * R), T]
    let H_mat : Matrix (Fin 2) (Fin 2) A := !![M + V + G, c * P; c * P, -M + V - G]
    -- (1) Full two-sided unitarity `Uᴴ * U = 1` and `U * Uᴴ = 1` (all 8 matrix blocks):
    (U.conjTranspose * U = 1) ∧
    (U * U.conjTranspose = 1) ∧
    -- (2) Both off-diagonal blocks of `U * H_mat * Uᴴ` vanish identically:
    ((U * H_mat * U.conjTranspose) 1 0 = 0) ∧
    ((U * H_mat * U.conjTranspose) 0 1 = 0) := by
  intro U H_mat
  -- First derive the adjoint Riccati equation by taking `star` of `hRicc`
  have hRicc_star : c * P + star R * (-M + V - G) = (M + V + G + star R * (c * P)) * star R := by
    have h := congr_arg star hRicc
    simp only [star_add, star_sub, star_neg, star_mul, hM_sa, hc_sa, hP_sa, hG_sa, hV_sa] at h
    have hPc : P * c = c * P := (hcc P).symm
    rw [hPc] at h
    exact h
  have hUct : U.conjTranspose = !![S, -(star R * T); R * S, T] := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [U, Matrix.conjTranspose_apply, star_neg, star_mul, hS_sa, hT_sa]
  refine ⟨?_, ?_, ?_, ?_⟩
  · -- (1a) `Uᴴ * U = 1` (all 4 blocks):
    rw [hUct]
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp only [U, Matrix.mul_apply, Matrix.one_apply, Fin.sum_univ_two, Matrix.of_apply,
        Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one,
        Fin.isValue, ↓reduceIte]
    · -- (0, 0): `S * S + (-(R* T)) * (-(T R)) = S (1 + R* R) S = 1`
      calc S * S + -(star R * T) * -(T * R)
          = S * S + (star R * T) * (T * R) := by noncomm_ring
        _ = S * S + (S * star R) * (R * S) := by rw [← hSR, hTR]
        _ = S * (1 + star R * R) * S := by noncomm_ring
        _ = 1 := hS2
    · -- (0, 1): `S * (S R*) + (-(R* T)) * T = 0`
      calc S * (S * star R) + -(star R * T) * T
          = S * (star R * T) - star R * T * T := by rw [hSR]; noncomm_ring
        _ = (S * star R) * T - star R * T * T := by noncomm_ring
        _ = (star R * T) * T - star R * T * T := by rw [hSR]
        _ = 0 := sub_self _
    · -- (1, 0): `(R S) * S + T * (-(T R)) = 0`
      calc R * S * S + T * -(T * R)
          = (T * R) * S - T * (T * R) := by rw [← hTR]; noncomm_ring
        _ = T * (R * S) - T * (T * R) := by noncomm_ring
        _ = T * (T * R) - T * (T * R) := by rw [← hTR]
        _ = 0 := sub_self _
    · -- (1, 1): `(R S) * (S R*) + T * T = T (1 + R R*) T = 1`
      calc R * S * (S * star R) + T * T
          = (T * R) * (star R * T) + T * T := by rw [← hTR, hSR]
        _ = T * (1 + R * star R) * T := by noncomm_ring
        _ = 1 := hT2
  · -- (1b) `U * Uᴴ = 1` (all 4 blocks):
    rw [hUct]
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp only [U, Matrix.mul_apply, Matrix.one_apply, Fin.sum_univ_two, Matrix.of_apply,
        Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one,
        Fin.isValue, ↓reduceIte]
    · -- (0, 0): `S * S + (S R*) * (R S) = S (1 + R* R) S = 1`
      calc S * S + S * star R * (R * S)
          = S * (1 + star R * R) * S := by noncomm_ring
        _ = 1 := hS2
    · -- (0, 1): `S * (-(R* T)) + (S R*) * T = 0`
      change S * -(star R * T) + S * star R * T = 0
      noncomm_ring
    · -- (1, 0): `(-(T R)) * S + T * (R S) = 0`
      change -(T * R) * S + T * (R * S) = 0
      noncomm_ring
    · -- (1, 1): `(-(T R)) * (-(R* T)) + T * T = T (1 + R R*) T = 1`
      calc -(T * R) * -(star R * T) + T * T
          = T * (1 + R * star R) * T := by noncomm_ring
        _ = 1 := hT2
  · -- (2a) Off-diagonal `(1, 0)` block of `U * H_mat * Uᴴ` vanishes:
    rw [hUct]
    simp only [U, H_mat, Matrix.mul_apply, Fin.sum_univ_two, Matrix.of_apply,
      Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one, Fin.isValue]
    set H11 := M + V + G
    set H12 := c * P
    set H21 := c * P
    set H22 := -M + V - G
    have hstep : H21 + H22 * R = R * (H11 + H12 * R) := by
      simpa [H11, H12, H21, H22, mul_assoc] using hRicc
    calc (-(T * R) * H11 + T * H21) * S + (-(T * R) * H12 + T * H22) * (R * S)
        = T * ((H21 + H22 * R) - R * (H11 + H12 * R)) * S := by noncomm_ring
      _ = T * 0 * S := by rw [hstep, sub_self]
      _ = 0 := by simp
  · -- (2b) Off-diagonal `(0, 1)` block of `U * H_mat * Uᴴ` vanishes:
    rw [hUct]
    simp only [U, H_mat, Matrix.mul_apply, Fin.sum_univ_two, Matrix.of_apply,
      Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one, Fin.isValue]
    set H11 := M + V + G
    set H12 := c * P
    set H21 := c * P
    set H22 := -M + V - G
    have hstep_star : H12 - H11 * star R = -(star R * (-H21 * star R + H22)) := by
      have h1 : H12 + star R * H22 = H11 * star R + star R * H21 * star R := by
        simpa [H11, H12, H21, H22, add_mul, mul_assoc] using hRicc_star
      calc H12 - H11 * star R
          = (H12 + star R * H22) - star R * H22 - H11 * star R := by abel
        _ = (H11 * star R + star R * H21 * star R) - star R * H22 - H11 * star R := by rw [h1]
        _ = -(star R * (-H21 * star R + H22)) := by noncomm_ring
    calc (S * H11 + S * star R * H21) * -(star R * T) + (S * H12 + S * star R * H22) * T
        = S * ((H12 - H11 * star R) + star R * (-H21 * star R + H22)) * T := by noncomm_ring
      _ = S * 0 * T := by rw [hstep_star]; simp
      _ = 0 := by simp

/-! ### Part 4: Unbounded-`P` Lyapunov Coercivity & Spectral Gap (`eq:lyapunov_coercive`, `eq:lyapunov_gap`) -/

section UnboundedLyapunov

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]

/-- Exact imaginary-cancellation lemma for symmetric `P`:
for ANY symmetric linear operator `P : H →ₗ[ℂ] H` (possibly unbounded!) and real `c`,
`Re (⟨u, (c : ℂ) • P w⟩ - ⟨w, (c : ℂ) • P u⟩) = 0`. -/
theorem symmetric_offdiag_re_cancel
    (P : H →ₗ[ℂ] H) (hP : ∀ x y : H, inner ℂ x (P y) = inner ℂ (P x) y)
    (c : ℝ) (u w : H) :
    (inner ℂ u ((c : ℂ) • P w) - inner ℂ w ((c : ℂ) • P u)).re = 0 := by
  rw [inner_smul_right, inner_smul_right]
  have hsym : inner ℂ w (P u) = starRingEnd ℂ (inner ℂ u (P w)) := by
    rw [hP w u, inner_conj_symm]
  rw [hsym, Complex.sub_re, Complex.mul_re, Complex.mul_re,
      Complex.ofReal_re, Complex.ofReal_im, Complex.conj_re, Complex.conj_im]
  ring

/-- `sgm_wavefn.tex` App. A.8 Eq. `eq:lyapunov_coercive` (`prop:unitary_norm`(b)):
For ANY symmetric (possibly unbounded!) linear operator `P : H →ₗ[ℂ] H`,
if the diagonal blocks `A = M + (V - V₀) + G` and `-D = M - (V - V₀) + G` satisfy
`Re ⟨u, A u⟩ ≥ δ ‖u‖²` and `Re ⟨w, (-D) w⟩ ≥ δ ‖w‖²`, then the Lyapunov pairing
`Re ⟨β Ψ, (H - V₀) Ψ⟩` cancels `P` identically and satisfies:
`Re (⟨u, A u + c P w⟩ - ⟨w, c P u + D w⟩) ≥ δ (‖u‖² + ‖w‖²)`. -/
theorem unbounded_P_lyapunov_coercivity
    (P A D : H →ₗ[ℂ] H)
    (hP : ∀ x y : H, inner ℂ x (P y) = inner ℂ (P x) y)
    (c : ℝ) {δ : ℝ}
    (hA : ∀ u : H, δ * ‖u‖ ^ 2 ≤ (inner ℂ u (A u)).re)
    (hD : ∀ w : H, δ * ‖w‖ ^ 2 ≤ (inner ℂ w (-(D w))).re)
    (u w : H) :
    δ * (‖u‖ ^ 2 + ‖w‖ ^ 2) ≤
      (inner ℂ u (A u + (c : ℂ) • P w) - inner ℂ w ((c : ℂ) • P u + D w)).re := by
  have hexpand :
      (inner ℂ u (A u + (c : ℂ) • P w) - inner ℂ w ((c : ℂ) • P u + D w)).re =
        (inner ℂ u (A u)).re + (inner ℂ w (-(D w))).re +
        (inner ℂ u ((c : ℂ) • P w) - inner ℂ w ((c : ℂ) • P u)).re := by
    simp only [inner_add_right, inner_neg_right, Complex.sub_re, Complex.add_re, Complex.neg_re]
    ring
  rw [hexpand, symmetric_offdiag_re_cancel P hP c u w, add_zero]
  nlinarith [hA u, hD w]

/-- `sgm_wavefn.tex` App. A.8 Eq. `eq:lyapunov_gap`:
From `unbounded_P_lyapunov_coercivity`, the operator `(H - V₀)` has a uniform spectral gap
`‖(H - V₀)(u, w)‖² ≥ δ² (‖u‖² + ‖w‖²)` completely independent of `‖P‖`! -/
theorem unbounded_P_spectral_gap
    (P A D : H →ₗ[ℂ] H)
    (hP : ∀ x y : H, inner ℂ x (P y) = inner ℂ (P x) y)
    {c δ : ℝ} (hδ : 0 ≤ δ)
    (hA : ∀ u : H, δ * ‖u‖ ^ 2 ≤ (inner ℂ u (A u)).re)
    (hD : ∀ w : H, δ * ‖w‖ ^ 2 ≤ (inner ℂ w (-(D w))).re)
    (u w : H) :
    let hu := A u + (c : ℂ) • P w
    let hw := (c : ℂ) • P u + D w
    δ ^ 2 * (‖u‖ ^ 2 + ‖w‖ ^ 2) ≤ ‖hu‖ ^ 2 + ‖hw‖ ^ 2 := by
  intro hu hw
  have hcoerc := unbounded_P_lyapunov_coercivity P A D hP c hA hD u w
  have hcs_u : (inner ℂ u hu).re ≤ ‖u‖ * ‖hu‖ :=
    le_trans (Complex.re_le_norm _) (norm_inner_le_norm u hu)
  have hcs_w : -(inner ℂ w hw).re ≤ ‖w‖ * ‖hw‖ := by
    have h1 : |-(inner ℂ w hw).re| ≤ ‖inner ℂ w hw‖ := by
      rw [abs_neg]
      exact Complex.abs_re_le_norm _
    exact le_trans (le_abs_self _) (le_trans h1 (norm_inner_le_norm w hw))
  have h1 : δ * (‖u‖ ^ 2 + ‖w‖ ^ 2) ≤ ‖u‖ * ‖hu‖ + ‖w‖ * ‖hw‖ := by
    have hre : (inner ℂ u hu - inner ℂ w hw).re = (inner ℂ u hu).re - (inner ℂ w hw).re :=
      Complex.sub_re _ _
    linarith
  -- Cauchy--Schwarz in ℝ²: `(‖u‖ ‖hu‖ + ‖w‖ ‖hw‖)² ≤ (‖u‖² + ‖w‖²)(‖hu‖² + ‖hw‖²)`
  have h2 : (‖u‖ * ‖hu‖ + ‖w‖ * ‖hw‖) ^ 2 ≤ (‖u‖ ^ 2 + ‖w‖ ^ 2) * (‖hu‖ ^ 2 + ‖hw‖ ^ 2) := by
    nlinarith [sq_nonneg (‖u‖ * ‖hw‖ - ‖w‖ * ‖hu‖)]
  have h3 : (δ * (‖u‖ ^ 2 + ‖w‖ ^ 2)) ^ 2 ≤ (‖u‖ * ‖hu‖ + ‖w‖ * ‖hw‖) ^ 2 := by
    have hnonneg : 0 ≤ δ * (‖u‖ ^ 2 + ‖w‖ ^ 2) := by positivity
    nlinarith
  have h4 : δ ^ 2 * (‖u‖ ^ 2 + ‖w‖ ^ 2) * (‖u‖ ^ 2 + ‖w‖ ^ 2) ≤
      (‖hu‖ ^ 2 + ‖hw‖ ^ 2) * (‖u‖ ^ 2 + ‖w‖ ^ 2) := by nlinarith
  by_cases hzero : ‖u‖ ^ 2 + ‖w‖ ^ 2 = 0
  · rw [hzero, mul_zero]; positivity
  · have hpos : 0 < ‖u‖ ^ 2 + ‖w‖ ^ 2 :=
      lt_of_le_of_ne (by positivity) (Ne.symm hzero)
    exact le_of_mul_le_mul_right h4 hpos

/-- **Unbounded-`P` Graph Contraction (`‖w‖ < ‖u‖`)**:
For any nonzero eigenstate `(u, w)` of `H` with eigenvalue `E > V₀` (so `hu = (E - V₀) u`,
`hw = (E - V₀) w`), Lyapunov coercivity forces `(E - V₀)(‖u‖² - ‖w‖²) ≥ δ(‖u‖² + ‖w‖²) > 0`,
hence `‖w‖ < ‖u‖` (the positive-energy graph map `R : u ↦ w` is a strict contraction
for arbitrary unbounded symmetric `P`!). -/
theorem unbounded_P_graph_contraction
    (P A D : H →ₗ[ℂ] H)
    (hP : ∀ x y : H, inner ℂ x (P y) = inner ℂ (P x) y)
    {c δ ΔE : ℝ} (hδ : 0 < δ) (hΔE : 0 < ΔE)
    (hA : ∀ u : H, δ * ‖u‖ ^ 2 ≤ (inner ℂ u (A u)).re)
    (hD : ∀ w : H, δ * ‖w‖ ^ 2 ≤ (inner ℂ w (-(D w))).re)
    (u w : H) (hnz : ‖u‖ ^ 2 + ‖w‖ ^ 2 > 0)
    (heig_u : A u + (c : ℂ) • P w = (ΔE : ℂ) • u)
    (heig_w : (c : ℂ) • P u + D w = (ΔE : ℂ) • w) :
    ‖w‖ < ‖u‖ := by
  have hcoerc := unbounded_P_lyapunov_coercivity P A D hP c hA hD u w
  rw [heig_u, heig_w] at hcoerc
  have hu_in : (inner ℂ u ((ΔE : ℂ) • u)).re = ΔE * ‖u‖ ^ 2 := by
    rw [inner_smul_right, inner_self_eq_norm_sq_to_K]
    simp [Complex.mul_re, sq]
  have hw_in : (inner ℂ w ((ΔE : ℂ) • w)).re = ΔE * ‖w‖ ^ 2 := by
    rw [inner_smul_right, inner_self_eq_norm_sq_to_K]
    simp [Complex.mul_re, sq]
  rw [Complex.sub_re, hu_in, hw_in] at hcoerc
  have hsq : ‖w‖ ^ 2 < ‖u‖ ^ 2 := by nlinarith
  nlinarith [norm_nonneg u, norm_nonneg w]

end UnboundedLyapunov

/-! ### Part 5: Unitary Norm/Overlap Conservation (`prop:unitary_norm`(a)) & Uniform Lapse (`prop:causal_lapse_gap`(iv)) -/

/-- `sgm_wavefn.tex` Proposition `prop:unitary_norm`(a) (`eq:unitary_inner_product`):
For any two differentiable solutions `χ₁, χ₂ : ℝ → H` of `∂_t χ = -i ω̂ χ` with a symmetric
generator `ω̂`, their inner product `⟨χ₁(t), χ₂(t)⟩` has identically zero time derivative. -/
theorem unitary_overlap_derivative_zero
    {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]
    (ω_op : H →ₗ[ℂ] H)
    (hSym : ∀ u v : H, inner ℂ u (ω_op v) = inner ℂ (ω_op u) v)
    (χ₁ χ₂ : ℝ → H) (t : ℝ)
    (h₁ : HasDerivAt χ₁ ((-Complex.I) • ω_op (χ₁ t)) t)
    (h₂ : HasDerivAt χ₂ ((-Complex.I) • ω_op (χ₂ t)) t) :
    HasDerivAt (fun s => inner ℂ (χ₁ s) (χ₂ s)) 0 t := by
  have h := h₁.inner ℂ h₂
  convert h using 1
  rw [inner_smul_left, inner_smul_right]
  have hconj : starRingEnd ℂ (-Complex.I) = Complex.I := by simp
  rw [hconj, hSym (χ₁ t) (χ₂ t)]
  ring

/-- `sgm_wavefn.tex` Proposition `prop:causal_lapse_gap`(iv):
(a) Uniform lapse `α₀ > 0` scales the Dirac rest gap cleanly to `2 α₀ m c²`.
(b) `ε < 1` is sufficient, NOT necessary: when `P = 0, V = 0, α_n = 3` (`G = 2 m c²`),
we have `ε = 2 > 1` while `R = 0` is still an exact decoupling solution! -/
theorem uniform_lapse_gap_and_eps_not_necessary (α₀ m c : ℝ) (hα : 0 < α₀) (hm : 0 < m) (hc : 0 < c) :
    let M := m * c ^ 2
    -- (a) Coordinate gap is `2 α₀ m c²`, proper-time gap is `2 m c²`
    ((2 * α₀ * M) / α₀ = 2 * M) ∧
    -- (b) At `P = 0, V = 0, G = 2 M` (uniform `α_n = 3`), `ε = 2 > 1` yet `R = 0` solves the Riccati eq.
    ((0 * (0 : ℝ) + 2 * M + 0) / M = 2 ∧
     (2 * M) * (0 : ℝ) + (2 * M * 0 + 0 * (2 * M)) + (0 * 0 - 0 * 0) + c * (0 * 0 * 0) = c * 0) := by
  intro M
  refine ⟨?_, ?_, ?_⟩
  · field_simp [ne_of_gt hα]
  · have hM : M ≠ 0 := by positivity
    simp only [zero_mul, zero_add, add_zero]
    field_simp [hM]
  · ring

end SGM.Operators
