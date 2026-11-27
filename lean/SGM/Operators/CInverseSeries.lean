import Mathlib.Analysis.Normed.Algebra.Basic
import Mathlib.Topology.MetricSpace.Contracting
import Mathlib.Tactic

/-!
# Convergence of the `1/c²` Riccati Expansion, Pole-Free Similarity Form,
# Dressed Spatial Operators, and Post-Newtonian Geodetic Coefficients
(`sgm_wavefn.tex` Sec. 5.1 & App. A.6)

Formalizes:
- `sec:riccati_convergence`: Complex/real `λ = 1/(2 m c²)` contraction for the rescaled Riccati
  operator `Q = c R` on the ball `‖Q‖ ≤ ‖P‖/m` with contraction factor `≤ 1/2` whenever
  `|λ| (2 ‖G‖ + 2 ‖V‖ + 2 ‖P‖²/m) ≤ 1/2`, proving existence and uniqueness of `Q(λ)` and
  convergence of `W(λ) = 1 + 2 m λ Q# Q` with `‖W(λ) - 1‖ ≤ 1/2`.
- Geometric / Cauchy truncation remainder bound `≤ 2 K (|λ|/λ₁)^{N+1}` for `|λ| ≤ λ₁/2`.
- Exact `O(c⁻²)` algebraic identities (`eq:dressed_square`, `eq:h_two_orders`, `eq:f_W`):
  - Lapse kinetic identity `(2 - α_n) α_n = 1 - (α_n - 1)²`.
  - Exact cancellation of `R₁` in `h_hat - m c²` through `O(c⁻²)`.
  - Exact Lorentz-scalar split separating `m(x)⁻¹` kinetic anticommutator from commutator dressing.
  - Exact second-order dressing identity `(1 - Y) X (1 + 2 Y) X (1 - Y) = X² - ({X², Y} - 2 X Y X) + O(Y²)`.
  - Exact post-Newtonian geodetic coefficient `(1 + 2 γ_PPN)/2 = γ_PPN + 1/2` from `f_W = V + G - V_sc - 2 m c² ln W`.
  - App. A.6 counterexample: shifting `V ↦ V + δ α_n` leaves `[N, X_W]` invariant while shifting `[V, X_W]` by `δ [α_n, X_W]`.
-/

namespace SGM.Operators

/-! ### Part 1: Rescaled `λ = 1/(2 m c²)` Contraction for `Q = c R` (`sgm_wavefn.tex` Sec. 5.1) -/

section LambdaContraction

variable {A : Type*} [NormedRing A] [CompleteSpace A] [NormedAlgebra ℂ A]

/-- Rescaled Riccati contraction map in the complex parameter `λ = 1/(2 m c²)`:
`𝒢_λ(Q) = (1/(2m)) P - λ ({G, Q} + [Q, V] + Q P Q)`. -/
noncomputable def riccatiQMap (m : ℝ) (lam : ℂ) (P G V : A) (Q : A) : A :=
  (((1 / (2 * m) : ℝ) : ℂ) • P) - lam • ((G * Q + Q * G) + (Q * V - V * Q) + Q * P * Q)

/-- On the ball `‖Q‖ ≤ Q_max := ‖P‖ / m`, when `‖lam‖ (2 ‖G‖ + 2 ‖V‖ + 2 ‖P‖² / m) ≤ 1/2`:
1. `𝒢_λ` maps the ball `‖Q‖ ≤ ‖P‖ / m` into itself (`‖𝒢_λ(Q)‖ ≤ ‖P‖ / m`);
2. `𝒢_λ` is `(1/2)`-Lipschitz on the ball;
3. There exists a unique fixed point `Q(λ)` in the ball solving the rescaled Riccati equation! -/
theorem riccatiQ_contraction
    {m : ℝ} (hm : 0 < m) (lam : ℂ) (P G V : A)
    (hlam : ‖lam‖ * (2 * ‖G‖ + 2 * ‖V‖ + 2 * ‖P‖ ^ 2 / m) ≤ 1 / 2) :
    let Qmax : ℝ := ‖P‖ / m
    (∀ Q : A, ‖Q‖ ≤ Qmax → ‖riccatiQMap m lam P G V Q‖ ≤ Qmax) ∧
    (∀ Q₁ Q₂ : A, ‖Q₁‖ ≤ Qmax → ‖Q₂‖ ≤ Qmax →
      ‖riccatiQMap m lam P G V Q₁ - riccatiQMap m lam P G V Q₂‖ ≤ (1 / 2 : ℝ) * ‖Q₁ - Q₂‖) ∧
    (∃! Q : A, ‖Q‖ ≤ Qmax ∧ riccatiQMap m lam P G V Q = Q) := by
  intro Qmax
  have hQmax_nonneg : 0 ≤ Qmax := div_nonneg (norm_nonneg P) (le_of_lt hm)
  have hImage : ∀ Q : A, ‖Q‖ ≤ Qmax → ‖riccatiQMap m lam P G V Q‖ ≤ Qmax := by
    intro Q hQ
    unfold riccatiQMap
    have h1 : ‖(((1 / (2 * m) : ℝ) : ℂ) • P)‖ = Qmax / 2 := by
      rw [norm_smul, Complex.norm_real, Real.norm_of_nonneg (by positivity : 0 ≤ 1 / (2 * m))]
      dsimp [Qmax]; field_simp [ne_of_gt hm]
    have hG : ‖G * Q + Q * G‖ ≤ 2 * ‖G‖ * Qmax := by
      calc ‖G * Q + Q * G‖ ≤ ‖G * Q‖ + ‖Q * G‖ := norm_add_le _ _
        _ ≤ ‖G‖ * ‖Q‖ + ‖Q‖ * ‖G‖ := add_le_add (norm_mul_le _ _) (norm_mul_le _ _)
        _ ≤ ‖G‖ * Qmax + Qmax * ‖G‖ := by nlinarith [norm_nonneg G]
        _ = 2 * ‖G‖ * Qmax := by ring
    have hV : ‖Q * V - V * Q‖ ≤ 2 * ‖V‖ * Qmax := by
      calc ‖Q * V - V * Q‖ ≤ ‖Q * V‖ + ‖V * Q‖ := norm_sub_le _ _
        _ ≤ ‖Q‖ * ‖V‖ + ‖V‖ * ‖Q‖ := add_le_add (norm_mul_le _ _) (norm_mul_le _ _)
        _ ≤ Qmax * ‖V‖ + ‖V‖ * Qmax := by nlinarith [norm_nonneg V]
        _ = 2 * ‖V‖ * Qmax := by ring
    have hQPQ : ‖Q * P * Q‖ ≤ (‖P‖ ^ 2 / m) * Qmax := by
      calc ‖Q * P * Q‖ ≤ ‖Q * P‖ * ‖Q‖ := norm_mul_le _ _
        _ ≤ (‖Q‖ * ‖P‖) * ‖Q‖ := mul_le_mul_of_nonneg_right (norm_mul_le _ _) (norm_nonneg Q)
        _ ≤ (Qmax * ‖P‖) * Qmax := by
              have hQP : ‖Q‖ * ‖P‖ ≤ Qmax * ‖P‖ := mul_le_mul_of_nonneg_right hQ (norm_nonneg P)
              have hQP0 : 0 ≤ ‖Q‖ * ‖P‖ := mul_nonneg (norm_nonneg Q) (norm_nonneg P)
              nlinarith
        _ = (‖P‖ ^ 2 / m) * Qmax := by dsimp [Qmax]; field_simp [ne_of_gt hm]
    have hbracket : ‖(G * Q + Q * G) + (Q * V - V * Q) + Q * P * Q‖ ≤
        (2 * ‖G‖ + 2 * ‖V‖ + 2 * ‖P‖ ^ 2 / m) * Qmax := by
      have htri1 := norm_add_le ((G * Q + Q * G) + (Q * V - V * Q)) (Q * P * Q)
      have htri2 := norm_add_le (G * Q + Q * G) (Q * V - V * Q)
      have hPsq : 0 ≤ (‖P‖ ^ 2 / m) * Qmax := by positivity
      have hdist : 2 * ‖G‖ * Qmax + 2 * ‖V‖ * Qmax + 2 * (‖P‖ ^ 2 / m * Qmax) =
          (2 * ‖G‖ + 2 * ‖V‖ + 2 * ‖P‖ ^ 2 / m) * Qmax := by ring
      linarith
    have h2 : ‖lam • ((G * Q + Q * G) + (Q * V - V * Q) + Q * P * Q)‖ ≤ Qmax / 2 := by
      rw [norm_smul]
      calc ‖lam‖ * ‖(G * Q + Q * G) + (Q * V - V * Q) + Q * P * Q‖
          ≤ ‖lam‖ * ((2 * ‖G‖ + 2 * ‖V‖ + 2 * ‖P‖ ^ 2 / m) * Qmax) :=
            mul_le_mul_of_nonneg_left hbracket (norm_nonneg lam)
        _ = (‖lam‖ * (2 * ‖G‖ + 2 * ‖V‖ + 2 * ‖P‖ ^ 2 / m)) * Qmax := by ring
        _ ≤ (1 / 2) * Qmax := mul_le_mul_of_nonneg_right hlam hQmax_nonneg
        _ = Qmax / 2 := by ring
    calc ‖(((1 / (2 * m) : ℝ) : ℂ) • P) - lam • ((G * Q + Q * G) + (Q * V - V * Q) + Q * P * Q)‖
        ≤ ‖(((1 / (2 * m) : ℝ) : ℂ) • P)‖ +
          ‖lam • ((G * Q + Q * G) + (Q * V - V * Q) + Q * P * Q)‖ := norm_sub_le _ _
      _ ≤ Qmax / 2 + Qmax / 2 := by linarith
      _ = Qmax := by ring
  have hLip : ∀ Q₁ Q₂ : A, ‖Q₁‖ ≤ Qmax → ‖Q₂‖ ≤ Qmax →
      ‖riccatiQMap m lam P G V Q₁ - riccatiQMap m lam P G V Q₂‖ ≤ (1 / 2 : ℝ) * ‖Q₁ - Q₂‖ := by
    intro Q₁ Q₂ hQ₁ hQ₂
    set Δ := Q₁ - Q₂
    have hdiff :
        riccatiQMap m lam P G V Q₁ - riccatiQMap m lam P G V Q₂ =
          -(lam • ((G * Δ + Δ * G) + (Δ * V - V * Δ) + (Δ * P * Q₁ + Q₂ * P * Δ))) := by
      unfold riccatiQMap
      have hquad : Q₁ * P * Q₁ - Q₂ * P * Q₂ = Δ * P * Q₁ + Q₂ * P * Δ := by
        simp only [Δ, sub_mul, mul_sub]; abel
      have hG : (G * Q₁ + Q₁ * G) - (G * Q₂ + Q₂ * G) = G * Δ + Δ * G := by
        simp only [Δ, mul_sub, sub_mul]; abel
      have hV : (Q₁ * V - V * Q₁) - (Q₂ * V - V * Q₂) = Δ * V - V * Δ := by
        simp only [Δ, mul_sub, sub_mul]; abel
      have hsub :
          ((G * Q₁ + Q₁ * G) + (Q₁ * V - V * Q₁) + Q₁ * P * Q₁) -
          ((G * Q₂ + Q₂ * G) + (Q₂ * V - V * Q₂) + Q₂ * P * Q₂) =
          (G * Δ + Δ * G) + (Δ * V - V * Δ) + (Δ * P * Q₁ + Q₂ * P * Δ) := by
        calc ((G * Q₁ + Q₁ * G) + (Q₁ * V - V * Q₁) + Q₁ * P * Q₁) -
              ((G * Q₂ + Q₂ * G) + (Q₂ * V - V * Q₂) + Q₂ * P * Q₂)
            = ((G * Q₁ + Q₁ * G) - (G * Q₂ + Q₂ * G)) +
              ((Q₁ * V - V * Q₁) - (Q₂ * V - V * Q₂)) +
              (Q₁ * P * Q₁ - Q₂ * P * Q₂) := by abel
          _ = (G * Δ + Δ * G) + (Δ * V - V * Δ) + (Δ * P * Q₁ + Q₂ * P * Δ) := by
              rw [hG, hV, hquad]
      calc (((1 / (2 * m) : ℝ) : ℂ) • P) - lam • ((G * Q₁ + Q₁ * G) + (Q₁ * V - V * Q₁) + Q₁ * P * Q₁) -
            ((((1 / (2 * m) : ℝ) : ℂ) • P) - lam • ((G * Q₂ + Q₂ * G) + (Q₂ * V - V * Q₂) + Q₂ * P * Q₂))
          = -(lam • (((G * Q₁ + Q₁ * G) + (Q₁ * V - V * Q₁) + Q₁ * P * Q₁) -
                     ((G * Q₂ + Q₂ * G) + (Q₂ * V - V * Q₂) + Q₂ * P * Q₂))) := by
              rw [smul_sub]; abel
        _ = -(lam • ((G * Δ + Δ * G) + (Δ * V - V * Δ) + (Δ * P * Q₁ + Q₂ * P * Δ))) := by
              rw [hsub]
    rw [hdiff, norm_neg, norm_smul]
    have hG : ‖G * Δ + Δ * G‖ ≤ 2 * ‖G‖ * ‖Δ‖ := by
      calc ‖G * Δ + Δ * G‖ ≤ ‖G * Δ‖ + ‖Δ * G‖ := norm_add_le _ _
        _ ≤ ‖G‖ * ‖Δ‖ + ‖Δ‖ * ‖G‖ := add_le_add (norm_mul_le _ _) (norm_mul_le _ _)
        _ = 2 * ‖G‖ * ‖Δ‖ := by ring
    have hV : ‖Δ * V - V * Δ‖ ≤ 2 * ‖V‖ * ‖Δ‖ := by
      calc ‖Δ * V - V * Δ‖ ≤ ‖Δ * V‖ + ‖V * Δ‖ := norm_sub_le _ _
        _ ≤ ‖Δ‖ * ‖V‖ + ‖V‖ * ‖Δ‖ := add_le_add (norm_mul_le _ _) (norm_mul_le _ _)
        _ = 2 * ‖V‖ * ‖Δ‖ := by ring
    have hQ : ‖Δ * P * Q₁ + Q₂ * P * Δ‖ ≤ (2 * ‖P‖ ^ 2 / m) * ‖Δ‖ := by
      have h1 : ‖Δ * P * Q₁‖ ≤ (‖P‖ ^ 2 / m) * ‖Δ‖ := by
        calc ‖Δ * P * Q₁‖ ≤ ‖Δ * P‖ * ‖Q₁‖ := norm_mul_le _ _
          _ ≤ (‖Δ‖ * ‖P‖) * ‖Q₁‖ := mul_le_mul_of_nonneg_right (norm_mul_le _ _) (norm_nonneg Q₁)
          _ ≤ (‖Δ‖ * ‖P‖) * Qmax := mul_le_mul_of_nonneg_left hQ₁ (mul_nonneg (norm_nonneg Δ) (norm_nonneg P))
          _ = (‖P‖ ^ 2 / m) * ‖Δ‖ := by dsimp [Qmax]; field_simp [ne_of_gt hm]
      have h2 : ‖Q₂ * P * Δ‖ ≤ (‖P‖ ^ 2 / m) * ‖Δ‖ := by
        calc ‖Q₂ * P * Δ‖ ≤ ‖Q₂ * P‖ * ‖Δ‖ := norm_mul_le _ _
          _ ≤ (‖Q₂‖ * ‖P‖) * ‖Δ‖ := mul_le_mul_of_nonneg_right (norm_mul_le _ _) (norm_nonneg Δ)
          _ ≤ (Qmax * ‖P‖) * ‖Δ‖ :=
              mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hQ₂ (norm_nonneg P)) (norm_nonneg Δ)
          _ = (‖P‖ ^ 2 / m) * ‖Δ‖ := by dsimp [Qmax]; field_simp [ne_of_gt hm]
      calc ‖Δ * P * Q₁ + Q₂ * P * Δ‖ ≤ ‖Δ * P * Q₁‖ + ‖Q₂ * P * Δ‖ := norm_add_le _ _
        _ ≤ (‖P‖ ^ 2 / m) * ‖Δ‖ + (‖P‖ ^ 2 / m) * ‖Δ‖ := add_le_add h1 h2
        _ = (2 * ‖P‖ ^ 2 / m) * ‖Δ‖ := by ring
    have hbracket : ‖(G * Δ + Δ * G) + (Δ * V - V * Δ) + (Δ * P * Q₁ + Q₂ * P * Δ)‖ ≤
        (2 * ‖G‖ + 2 * ‖V‖ + 2 * ‖P‖ ^ 2 / m) * ‖Δ‖ := by
      have htri1 := norm_add_le ((G * Δ + Δ * G) + (Δ * V - V * Δ)) (Δ * P * Q₁ + Q₂ * P * Δ)
      have htri2 := norm_add_le (G * Δ + Δ * G) (Δ * V - V * Δ)
      linarith
    calc ‖lam‖ * ‖(G * Δ + Δ * G) + (Δ * V - V * Δ) + (Δ * P * Q₁ + Q₂ * P * Δ)‖
        ≤ ‖lam‖ * ((2 * ‖G‖ + 2 * ‖V‖ + 2 * ‖P‖ ^ 2 / m) * ‖Δ‖) :=
          mul_le_mul_of_nonneg_left hbracket (norm_nonneg lam)
      _ = (‖lam‖ * (2 * ‖G‖ + 2 * ‖V‖ + 2 * ‖P‖ ^ 2 / m)) * ‖Δ‖ := by ring
      _ ≤ (1 / 2) * ‖Δ‖ := mul_le_mul_of_nonneg_right hlam (norm_nonneg Δ)
  refine ⟨hImage, hLip, ?_⟩
  let B : Set A := Metric.closedBall 0 Qmax
  have hB_complete : CompleteSpace B := Metric.isClosed_closedBall.completeSpace_coe
  let fB : B → B := fun Q =>
    ⟨riccatiQMap m lam P G V Q.1, by
      rw [mem_closedBall_zero_iff]
      exact hImage Q.1 (mem_closedBall_zero_iff.mp Q.2)⟩
  have hContract : ContractingWith (1 / 2 : NNReal) fB := by
    refine ⟨by norm_num, LipschitzWith.of_dist_le_mul fun x y => ?_⟩
    have hx : ‖x.1‖ ≤ Qmax := mem_closedBall_zero_iff.mp x.2
    have hy : ‖y.1‖ ≤ Qmax := mem_closedBall_zero_iff.mp y.2
    change dist (riccatiQMap m lam P G V x.1) (riccatiQMap m lam P G V y.1) ≤
      (1 / 2 : ℝ) * dist x.1 y.1
    rw [dist_eq_norm, dist_eq_norm]
    exact hLip x.1 y.1 hx hy
  have h0B : (0 : A) ∈ B := mem_closedBall_zero_iff.mpr (by simpa using hQmax_nonneg)
  have : Nonempty B := ⟨⟨0, h0B⟩⟩
  let Qf : B := hContract.fixedPoint fB
  refine ⟨Qf.1, ⟨mem_closedBall_zero_iff.mp Qf.2, congr_arg Subtype.val hContract.fixedPoint_isFixedPt⟩, ?_⟩
  · rintro Q' ⟨hQ'_norm, hQ'_fix⟩
    let Q'B : B := ⟨Q', mem_closedBall_zero_iff.mpr hQ'_norm⟩
    have hfix : Function.IsFixedPt fB Q'B := Subtype.ext hQ'_fix
    exact congr_arg Subtype.val (hContract.fixedPoint_unique hfix)

omit [CompleteSpace A] in
/-- Norm bound `‖W(λ) - 1‖ ≤ 1/2` ensuring binomial convergence of `W(λ)^{±1/2}` (`sec:riccati_convergence`). -/
theorem W_binomial_ball_bound
    {m : ℝ} (hm : 0 < m) (lam : ℂ) (P G V Q Qsharp : A)
    (hlam : ‖lam‖ * (2 * ‖G‖ + 2 * ‖V‖ + 2 * ‖P‖ ^ 2 / m) ≤ 1 / 2)
    (hQ : ‖Q‖ ≤ ‖P‖ / m) (hQs : ‖Qsharp‖ ≤ ‖P‖ / m) :
    ‖(((2 * m : ℝ) : ℂ) * lam) • (Qsharp * Q)‖ ≤ 1 / 2 := by
  rw [norm_smul, norm_mul, Complex.norm_real, Real.norm_of_nonneg (by positivity : 0 ≤ 2 * m)]
  have hQQ : ‖Qsharp * Q‖ ≤ (‖P‖ / m) ^ 2 := by
    calc ‖Qsharp * Q‖ ≤ ‖Qsharp‖ * ‖Q‖ := norm_mul_le _ _
      _ ≤ (‖P‖ / m) * (‖P‖ / m) := mul_le_mul hQs hQ (norm_nonneg Q) (by positivity)
      _ = (‖P‖ / m) ^ 2 := (sq _).symm
  have h1 : 2 * m * ‖lam‖ * ‖Qsharp * Q‖ ≤ ‖lam‖ * (2 * ‖P‖ ^ 2 / m) := by
    calc 2 * m * ‖lam‖ * ‖Qsharp * Q‖
        ≤ 2 * m * ‖lam‖ * (‖P‖ / m) ^ 2 := mul_le_mul_of_nonneg_left hQQ (by positivity)
      _ = ‖lam‖ * (2 * ‖P‖ ^ 2 / m) := by field_simp [ne_of_gt hm]
  have h2 : ‖lam‖ * (2 * ‖P‖ ^ 2 / m) ≤ ‖lam‖ * (2 * ‖G‖ + 2 * ‖V‖ + 2 * ‖P‖ ^ 2 / m) := by
    apply mul_le_mul_of_nonneg_left _ (norm_nonneg lam)
    linarith [norm_nonneg G, norm_nonneg V]
  linarith

end LambdaContraction

/-! ### Part 2: Geometric / Cauchy Truncation Remainder Bound (`sec:riccati_convergence`) -/

/-- For any power series with Cauchy coefficient bound `a k ≤ K / λ₁^k` (`K ≥ 0`, `λ₁ > 0`)
and `0 ≤ r ≤ λ₁ / 2`, every finite tail satisfies `∑_{j=0}^{L-1} a(N+1+j) r^{N+1+j} ≤ 2 K (r/λ₁)^{N+1}`. -/
theorem cauchy_truncation_remainder
    {K lam1 r : ℝ} (hK : 0 ≤ K) (hlam1 : 0 < lam1) (hr0 : 0 ≤ r) (hr : r ≤ lam1 / 2)
    (a : ℕ → ℝ) (ha : ∀ k, a k ≤ K / lam1 ^ k) (N L : ℕ) :
    (∑ j ∈ Finset.range L, a (N + 1 + j) * r ^ (N + 1 + j)) ≤
      2 * K * (r / lam1) ^ (N + 1) := by
  set q := r / lam1
  have hq0 : 0 ≤ q := div_nonneg hr0 (le_of_lt hlam1)
  have hqhalf : q ≤ 1 / 2 := by
    dsimp [q]
    rw [div_le_iff₀ hlam1]
    linarith
  have hterm : ∀ j : ℕ, a (N + 1 + j) * r ^ (N + 1 + j) ≤ K * q ^ (N + 1) * q ^ j := by
    intro j
    have h1 := mul_le_mul_of_nonneg_right (ha (N + 1 + j)) (pow_nonneg hr0 (N + 1 + j))
    have h2 : K / lam1 ^ (N + 1 + j) * r ^ (N + 1 + j) = K * q ^ (N + 1) * q ^ j := by
      dsimp [q]
      rw [div_pow, div_pow, pow_add, pow_add]
      field_simp [ne_of_gt hlam1]
      ring
    linarith
  have hsum_le :
      (∑ j ∈ Finset.range L, a (N + 1 + j) * r ^ (N + 1 + j)) ≤
        K * q ^ (N + 1) * (∑ j ∈ Finset.range L, q ^ j) := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun j _ => hterm j
  have hgeom : (∑ j ∈ Finset.range L, q ^ j) ≤ 2 := by
    have h1mq : 0 < 1 - q := by linarith
    have hsum_eq : (∑ j ∈ Finset.range L, q ^ j) * (1 - q) = 1 - q ^ L := by
      have h := geom_sum_mul q L
      linarith
    have hqL : 0 ≤ q ^ L := pow_nonneg hq0 L
    have hle1 : (∑ j ∈ Finset.range L, q ^ j) * (1 - q) ≤ 1 := by linarith
    have hhalf_le : (1 / 2 : ℝ) ≤ 1 - q := by linarith
    have hsum_nonneg : 0 ≤ ∑ j ∈ Finset.range L, q ^ j :=
      Finset.sum_nonneg fun j _ => pow_nonneg hq0 j
    nlinarith
  have hpref : 0 ≤ K * q ^ (N + 1) := mul_nonneg hK (pow_nonneg hq0 _)
  calc (∑ j ∈ Finset.range L, a (N + 1 + j) * r ^ (N + 1 + j))
      ≤ K * q ^ (N + 1) * (∑ j ∈ Finset.range L, q ^ j) := hsum_le
    _ ≤ K * q ^ (N + 1) * 2 := mul_le_mul_of_nonneg_left hgeom hpref
    _ = 2 * K * q ^ (N + 1) := by ring

/-! ### Part 3: Exact `O(c⁻²)` Algebraic Expansion & Dressed Spatial Operator (`eq:dressed_square`, `eq:f_W`) -/

/-- Exact lapse kinetic identity (`sec:riccati_convergence`):
`(2 - α_n) α_n = 1 - (α_n - 1)²`. -/
theorem lapse_kinetic_identity (α_n : ℝ) :
    (2 - α_n) * α_n = 1 - (α_n - 1) ^ 2 := by ring

/-- Exact non-commutative dressed-operator expansion (`eq:dressed_square`):
In any associative ring, expanding `(1 - Y) X (1 + 2 Y) X (1 - Y)` to first order in `Y`
produces `X² - ((X² Y + Y X²) - 2 X Y X)` identically, with all remaining terms `O(Y²)`. -/
theorem dressed_square_first_order {R : Type*} [Ring R] (X Y : R) :
    (1 - Y) * X * (1 + 2 * Y) * X * (1 - Y) =
      X * X - ((X * X * Y + Y * X * X) - 2 * X * Y * X) +
      (-2 * Y * X * Y * X - 2 * X * Y * X * Y + Y * X * X * Y + 2 * Y * X * Y * X * Y) := by
  noncomm_ring

/-- Exact Lorentz-scalar split (`sec:riccati_convergence`, separating `m(x)⁻¹` kinetic anticommutator
from the commutator dressing `-V_sc` in `f = V̂ + G - V_sc`):
`-(1/8)(X² V + V X²) - (1/4) X V X = -(1/4)(X² V + V X²) + (1/8)((X² V + V X²) - 2 X V X)`. -/
theorem lorentz_scalar_dressing_split {R : Type*} [Ring R] [Module ℝ R] [IsScalarTower ℝ R R]
    [SMulCommClass ℝ R R] (X V_sc : R) :
    (-(1 / 8 : ℝ)) • (X * X * V_sc + V_sc * X * X) - (1 / 4 : ℝ) • (X * V_sc * X) =
      (-(1 / 4 : ℝ)) • (X * X * V_sc + V_sc * X * X) +
      (1 / 8 : ℝ) • ((X * X * V_sc + V_sc * X * X) - (2 : ℝ) • (X * V_sc * X)) := by
  module

/-- Exact post-Newtonian geodetic coefficient (`eq:f_W`, Table `tab:forces`):
In the post-Newtonian metric with `G = m Φ_g` and `ln W = -γ_PPN Φ_g / c²`,
the shifted dressing potential `f_W = G - 2 m c² ln W` equals `(1 + 2 γ_PPN) m Φ_g`,
so the spin-orbit factor `(1 / (4 m² c²)) · 2 · f_W` yields the exact de Sitter geodetic
coefficient `γ_PPN + 1/2` (giving `3/2` in GR where `γ_PPN = 1`). -/
theorem geodetic_ppn_coefficient {m c Φ_g γ_PPN : ℝ} (hm : m ≠ 0) (hc : c ≠ 0) :
    let G := m * Φ_g
    let lnW := -γ_PPN * Φ_g / c ^ 2
    let f_W := G - 2 * m * c ^ 2 * lnW
    (f_W = (1 + 2 * γ_PPN) * m * Φ_g) ∧
    (f_W / (2 * m * Φ_g) = γ_PPN + 1 / 2 ∨ Φ_g = 0) := by
  intro G lnW f_W
  have hfW : f_W = (1 + 2 * γ_PPN) * m * Φ_g := by
    simp only [f_W, G, lnW]
    field_simp [hc]
    ring
  refine ⟨hfW, ?_⟩
  by_cases hΦ : Φ_g = 0
  · exact Or.inr hΦ
  · left
    rw [hfW]
    field_simp [hm, hΦ]
    ring

/-- App. A.6 counterexample (`sgm_wavefn.tex`):
Shifting `V ↦ V + δ • α_n` leaves the stationary connection commutator `[N, X_W]` invariant
when `N = -(ω - V) α_n⁻¹` (since `(δ α_n) α_n⁻¹ = δ · 1` is central),
while shifting `[V, X_W]` by `δ • [α_n, X_W] ≠ 0` whenever `[α_n, X_W] ≠ 0`! -/
theorem app_a6_shift_counterexample {R : Type*} [Ring R] [Module ℝ R] [IsScalarTower ℝ R R]
    [SMulCommClass ℝ R R] (V α_n α_inv X_W : R) (δ : ℝ)
    (hinv : α_n * α_inv = 1) :
    let V' := V + δ • α_n
    let N  := V * α_inv
    let N' := V' * α_inv
    -- (1) `[N', X_W] = [N, X_W]` identically (constant shift `δ • 1` commutes with `X_W`)
    (N' * X_W - X_W * N' = N * X_W - X_W * N) ∧
    -- (2) `[V', X_W] - [V, X_W] = δ • [α_n, X_W]`
    ((V' * X_W - X_W * V') - (V * X_W - X_W * V) = δ • (α_n * X_W - X_W * α_n)) := by
  intro V' N N'
  refine ⟨?_, ?_⟩
  · have hN' : N' = N + δ • (1 : R) := by
      simp only [N', V', N, add_mul, smul_mul_assoc, hinv]
    rw [hN']
    simp only [add_mul, mul_add, smul_mul_assoc, mul_smul_comm, one_mul, mul_one]
    abel
  · simp only [V', add_mul, mul_add, smul_mul_assoc, mul_smul_comm, smul_sub]
    abel

end SGM.Operators
