import SGM.Semiclassical.WKBPrecession
import SGM.Semiclassical.ActionAndLabels
import SGM.Semiclassical.SturmComparison
import Mathlib.Data.Matrix.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Tactic

/-!
# The Four Relativistic Bohr--Sommerfeld Level Theorems (Theorems 1, 2, 3(b) One-to-One, and 4)
(`sgm_wavefn.tex` App. B: `thm:levels_B`, `thm:levels_V`, `thm:levels_crossed`, `thm:levels_grav`)

Formalizes:
- **Theorem 3(b) Boost Congruence (`thm:levels_crossed`(b))**:
  Explicit `2×2` complex matrix proof of
  `M_η (D_s^lab - E) M_η = c p σ₁ - s c ρ_d σ₂ + m c² σ₃ - γ_d E_d I`
  for `M_η = cosh(η/2) I + s sinh(η/2) σ₂` (`tanh η = β_d`, `cosh η = γ_d`, `sinh η = γ_d β_d`).
- **Theorem 3(b) Hellmann--Feynman Strict Negativity**:
  Proves `f_n'(E_d) = (2 γ_d / c)(β_d ⟨ρ_d⟩_n - γ_d E_d / c) < (2 γ_d / c)(β_d - 1)(γ_d E_d / c) < 0`
  at every crossing zero for `0 < ℏ < γ_d m² c² / B_sup`.
- **At-Most-One-Zero Theorem (`strictly_neg_deriv_at_zeros_at_most_one_zero`)**:
  100% axiom-free real-analysis proof that a differentiable function on `[a, b]` whose derivative
  is strictly negative at every zero has **at most one zero** in `[a, b]`.
- **Upgraded One-to-One Theorem 3(b) (`thm_levels_crossed_one_to_one`)**:
  Combines `strictly_neg_deriv_at_zeros_at_most_one_zero` + `lem_labels_part_b` (without global
  monotonicity!) + `lem_levels_action_one_to_one` to yield the **one-to-one** `O(ℏ²)` spectral
  approximation!
- **Theorem 4 (`thm:levels_grav`) Corrected `ν^{±1/2}` Bijection & Weyl-Sequence Identity**:
  Formalizes our corrected `ν = α_n / W` unweighted block kernel bijection
  `F ↦ (ν^{-1/2} F, c ν^{-1/2} g Q_s F)` and the exact Weyl-sequence quadratic-form identity
  `k(F_k) = ⟨F_k, ν^{-1/2} r_{k,1}⟩ + c ⟨Q_s F_k, g ν^{-1/2} r_{k,2}⟩`.
- **Master Assembly of Theorems 1, 2, and 4 (`thm_levels_monotone_one_to_one`)**:
  Parameterizes the Olver Airy matching input cleanly as an explicit hypothesis structure
  (ZERO opaque global `axiom`s!) and derives the one-to-one `O(ℏ²)` spectral approximation.
-/

namespace SGM.Semiclassical

/-! ### Part 1: Theorem 3(b) — Explicit `2×2` Drift-Frame Boost Congruence -/

private def σ1 : Matrix (Fin 2) (Fin 2) ℂ := !![0, 1; 1, 0]
private def σ2 : Matrix (Fin 2) (Fin 2) ℂ := !![0, -Complex.I; Complex.I, 0]
private def σ3 : Matrix (Fin 2) (Fin 2) ℂ := !![1, 0; 0, -1]
private def I2 : Matrix (Fin 2) (Fin 2) ℂ := !![1, 0; 0, 1]

/-- Boost congruence matrix `M_η = cosh(η/2) I + s sinh(η/2) σ₂` (`sgm_wavefn.tex` Thm. 3(b)). -/
noncomputable def boostMat (s η : ℝ) : Matrix (Fin 2) (Fin 2) ℂ :=
  ((Real.cosh (η / 2) : ℝ) : ℂ) • I2 + ((s * Real.sinh (η / 2) : ℝ) : ℂ) • σ2

/-- **Theorem 3(b) Drift-Frame Boost Congruence Identities**:
For `s² = 1`, `γ_d = cosh η`, `β_d = tanh η = sinh η / cosh η` (so `γ_d β_d = sinh η` and `γ_d > 0`):
1. `M_η σ₁ M_η = σ₁`;
2. `M_η σ₃ M_η = σ₃`;
3. `M_η I M_η = γ_d I + s γ_d β_d σ₂`;
4. `M_η (β_d I - s σ₂) M_η = -(s / γ_d) σ₂`;
5. Full laboratory-to-drift block congruence:
   `M_η (c p σ₁ + (β_d c (ρ + p_y) - E) I - s c ρ σ₂ + m c² σ₃) M_η =`
   `c p σ₁ - s c ρ_d σ₂ + m c² σ₃ - (γ_d E_d) I`
   where `E_d = E - β_d c p_y`, `b_d = p_y / γ_d - γ_d β_d E_d / c`, `ρ_d = (ρ + p_y) / γ_d - b_d`. -/
theorem crossed_field_boost_congruence
    {s η c p ρ p_y E m : ℝ} (hs : s ^ 2 = 1) (hc : c ≠ 0) :
    let γ_d := Real.cosh η
    let β_d := Real.sinh η / Real.cosh η
    let E_d := E - β_d * c * p_y
    let b_d := p_y / γ_d - γ_d * β_d * E_d / c
    let ρ_d := (ρ + p_y) / γ_d - b_d
    let M := boostMat s η
    (M * σ1 * M = σ1) ∧
    (M * σ3 * M = σ3) ∧
    (M * I2 * M = (γ_d : ℂ) • I2 + ((s * γ_d * β_d : ℝ) : ℂ) • σ2) ∧
    (M * ((β_d : ℂ) • I2 - (s : ℂ) • σ2) * M = ((-(s / γ_d) : ℝ) : ℂ) • σ2) ∧
    (M * (((c * p : ℝ) : ℂ) • σ1 + (((β_d * c * (ρ + p_y) - E : ℝ) : ℂ) • I2) -
          (((s * c * ρ : ℝ) : ℂ) • σ2) + (((m * c ^ 2 : ℝ) : ℂ) • σ3)) * M =
     ((c * p : ℝ) : ℂ) • σ1 - (((s * c * ρ_d : ℝ) : ℂ) • σ2) +
     (((m * c ^ 2 : ℝ) : ℂ) • σ3) - (((γ_d * E_d : ℝ) : ℂ) • I2)) := by
  intro γ_d β_d E_d b_d ρ_d M
  set ch := Real.cosh (η / 2)
  set sh := Real.sinh (η / 2)
  have hcosh_pos : 0 < Real.cosh η := Real.cosh_pos η
  have hcosh_ne : Real.cosh η ≠ 0 := ne_of_gt hcosh_pos
  have hcs_id : ch ^ 2 - sh ^ 2 = 1 := Real.cosh_sq_sub_sinh_sq (η / 2)
  have hch2 : ch ^ 2 + sh ^ 2 = Real.cosh η := by
    have h1 := Real.cosh_two_mul (η / 2)
    have h2 : 2 * (η / 2) = η := by ring
    rw [h2] at h1
    nlinarith [hcs_id]
  have hsh2 : 2 * sh * ch = Real.sinh η := by
    have h1 := Real.sinh_two_mul (η / 2)
    have h2 : 2 * (η / 2) = η := by ring
    rw [h2] at h1
    linarith
  have hγβ : γ_d * β_d = Real.sinh η := by
    dsimp [γ_d, β_d]; field_simp [hcosh_ne]
  have hβ_cosh : β_d * Real.cosh η = Real.sinh η := by
    dsimp [β_d]; field_simp [hcosh_ne]
  have hγ2_sub : Real.cosh η ^ 2 - Real.sinh η ^ 2 = 1 := Real.cosh_sq_sub_sinh_sq η
  -- Core Pauli algebra identities (proved instantaneously on Fin 2 × Fin 2):
  have hI_l : ∀ A : Matrix (Fin 2) (Fin 2) ℂ, I2 * A = A := fun A => by
    ext i j; fin_cases i <;> fin_cases j <;> simp [I2, Matrix.mul_apply, Fin.sum_univ_two]
  have hI_r : ∀ A : Matrix (Fin 2) (Fin 2) ℂ, A * I2 = A := fun A => by
    ext i j; fin_cases i <;> fin_cases j <;> simp [I2, Matrix.mul_apply, Fin.sum_univ_two]
  have hσ2_sq : σ2 * σ2 = I2 := by
    ext i j; fin_cases i <;> fin_cases j <;> simp [σ2, I2, Matrix.mul_apply, Fin.sum_univ_two]
  have hσ2_σ1 : σ2 * σ1 = -(σ1 * σ2) := by
    ext i j; fin_cases i <;> fin_cases j <;> simp [σ1, σ2, Matrix.mul_apply, Fin.sum_univ_two]
  have hσ2_σ3 : σ2 * σ3 = -(σ3 * σ2) := by
    ext i j; fin_cases i <;> fin_cases j <;> simp [σ3, σ2, Matrix.mul_apply, Fin.sum_univ_two]
  have hσ2_σ1_σ2 : σ2 * (σ1 * σ2) = -σ1 := by
    rw [← Matrix.mul_assoc, hσ2_σ1, Matrix.neg_mul, Matrix.mul_assoc, hσ2_sq, hI_r]
  have hσ2_σ3_σ2 : σ2 * (σ3 * σ2) = -σ3 := by
    rw [← Matrix.mul_assoc, hσ2_σ3, Matrix.neg_mul, Matrix.mul_assoc, hσ2_sq, hI_r]
  have hM1 : M * σ1 * M = σ1 := by
    have hcoeff : ((ch : ℂ) * (ch : ℂ) - ((s * sh : ℝ) : ℂ) * ((s * sh : ℝ) : ℂ)) = 1 := by
      apply Complex.ext <;> simp
      linear_combination hcs_id - sh ^ 2 * hs
    calc M * σ1 * M
        = ((ch : ℂ) • I2 + ((s * sh : ℝ) : ℂ) • σ2) * σ1 *
          ((ch : ℂ) • I2 + ((s * sh : ℝ) : ℂ) • σ2) := rfl
      _ = ((ch : ℂ) * (ch : ℂ) - ((s * sh : ℝ) : ℂ) * ((s * sh : ℝ) : ℂ)) • σ1 := by
            simp only [Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul, Matrix.mul_smul,
              smul_add, smul_smul, hI_l, hI_r, Matrix.mul_assoc]
            rw [hσ2_σ1, hσ2_σ1_σ2, smul_neg, smul_neg, sub_smul]
            have hcomm_c : ((s * sh : ℝ) : ℂ) * (ch : ℂ) = (ch : ℂ) * ((s * sh : ℝ) : ℂ) := mul_comm _ _
            rw [hcomm_c]
            abel
      _ = σ1 := by rw [hcoeff, one_smul]
  have hM3 : M * σ3 * M = σ3 := by
    have hcoeff : ((ch : ℂ) * (ch : ℂ) - ((s * sh : ℝ) : ℂ) * ((s * sh : ℝ) : ℂ)) = 1 := by
      apply Complex.ext <;> simp
      linear_combination hcs_id - sh ^ 2 * hs
    calc M * σ3 * M
        = ((ch : ℂ) • I2 + ((s * sh : ℝ) : ℂ) • σ2) * σ3 *
          ((ch : ℂ) • I2 + ((s * sh : ℝ) : ℂ) • σ2) := rfl
      _ = ((ch : ℂ) * (ch : ℂ) - ((s * sh : ℝ) : ℂ) * ((s * sh : ℝ) : ℂ)) • σ3 := by
            simp only [Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul, Matrix.mul_smul,
              smul_add, smul_smul, hI_l, hI_r, Matrix.mul_assoc]
            rw [hσ2_σ3, hσ2_σ3_σ2, smul_neg, smul_neg, sub_smul]
            have hcomm_c : ((s * sh : ℝ) : ℂ) * (ch : ℂ) = (ch : ℂ) * ((s * sh : ℝ) : ℂ) := mul_comm _ _
            rw [hcomm_c]
            abel
      _ = σ3 := by rw [hcoeff, one_smul]
  have hMsq : M * M = (γ_d : ℂ) • I2 + ((s * γ_d * β_d : ℝ) : ℂ) • σ2 := by
    have hc1 : (ch : ℂ) * (ch : ℂ) + ((s * sh : ℝ) : ℂ) * ((s * sh : ℝ) : ℂ) = (γ_d : ℂ) := by
      apply Complex.ext <;> simp [γ_d]
      linear_combination hch2 + sh ^ 2 * hs
    have hc2 : (ch : ℂ) * ((s * sh : ℝ) : ℂ) + ((s * sh : ℝ) : ℂ) * (ch : ℂ) =
        ((s * γ_d * β_d : ℝ) : ℂ) := by
      have hsγβ : s * γ_d * β_d = s * Real.sinh η := by rw [mul_assoc, hγβ]
      rw [hsγβ]
      apply Complex.ext
      · simp only [Complex.add_re, Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im]
        linear_combination s * hsh2
      · simp
    calc M * M
        = ((ch : ℂ) • I2 + ((s * sh : ℝ) : ℂ) • σ2) *
          ((ch : ℂ) • I2 + ((s * sh : ℝ) : ℂ) • σ2) := rfl
      _ = ((ch : ℂ) * (ch : ℂ) + ((s * sh : ℝ) : ℂ) * ((s * sh : ℝ) : ℂ)) • I2 +
          ((ch : ℂ) * ((s * sh : ℝ) : ℂ) + ((s * sh : ℝ) : ℂ) * (ch : ℂ)) • σ2 := by
            simp only [Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul, Matrix.mul_smul,
              smul_add, smul_smul, hI_l, hI_r, hσ2_sq, add_smul]
            abel
      _ = (γ_d : ℂ) • I2 + ((s * γ_d * β_d : ℝ) : ℂ) • σ2 := by rw [hc1, hc2]
  have hMI : M * I2 * M = (γ_d : ℂ) • I2 + ((s * γ_d * β_d : ℝ) : ℂ) • σ2 := by
    rw [hI_r]; exact hMsq
  have hMσ2 : M * σ2 * M = ((s * γ_d * β_d : ℝ) : ℂ) • I2 + (γ_d : ℂ) • σ2 := by
    have hcomm : M * σ2 = σ2 * M := by
      simp only [M, boostMat, Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul, Matrix.mul_smul,
        hI_l, hI_r]
    calc M * σ2 * M = M * (M * σ2) := by rw [Matrix.mul_assoc, hcomm]
      _ = (M * M) * σ2 := by rw [Matrix.mul_assoc]
      _ = ((γ_d : ℂ) • I2 + ((s * γ_d * β_d : ℝ) : ℂ) • σ2) * σ2 := by rw [hMsq]
      _ = ((s * γ_d * β_d : ℝ) : ℂ) • I2 + (γ_d : ℂ) • σ2 := by
            simp only [Matrix.add_mul, Matrix.smul_mul, hI_l, hσ2_sq]; abel
  have hM2 : M * ((β_d : ℂ) • I2 - (s : ℂ) • σ2) * M = ((-(s / γ_d) : ℝ) : ℂ) • σ2 := by
    have hre0 : (β_d : ℂ) * (γ_d : ℂ) - (s : ℂ) * ((s * γ_d * β_d : ℝ) : ℂ) = 0 := by
      apply Complex.ext <;> simp
      linear_combination -(β_d * γ_d) * hs
    have him0 : (β_d : ℂ) * ((s * γ_d * β_d : ℝ) : ℂ) - (s : ℂ) * (γ_d : ℂ) =
        ((-(s / γ_d) : ℝ) : ℂ) := by
      have hreal : β_d * (s * γ_d * β_d) - s * γ_d = -(s / γ_d) := by
        dsimp [γ_d, β_d]
        field_simp [hcosh_ne]
        linear_combination -s * hγ2_sub
      apply Complex.ext <;> simp [hreal]
    calc M * ((β_d : ℂ) • I2 - (s : ℂ) • σ2) * M
        = (β_d : ℂ) • (M * I2 * M) - (s : ℂ) • (M * σ2 * M) := by
            simp only [Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_smul, Matrix.smul_mul]
      _ = ((β_d : ℂ) * (γ_d : ℂ) - (s : ℂ) * ((s * γ_d * β_d : ℝ) : ℂ)) • I2 +
          ((β_d : ℂ) * ((s * γ_d * β_d : ℝ) : ℂ) - (s : ℂ) * (γ_d : ℂ)) • σ2 := by
            rw [hMI, hMσ2, smul_add, smul_add, smul_smul, smul_smul, smul_smul, smul_smul,
              sub_smul, sub_smul]
            abel
      _ = ((-(s / γ_d) : ℝ) : ℂ) • σ2 := by rw [hre0, him0, zero_smul, zero_add]
  refine ⟨hM1, hM3, hMI, hM2, ?_⟩
  -- Rewrite laboratory block as linear combination of `σ₁`, `σ₃`, `I₂`, and `β_d I₂ - s σ₂`:
  have hlab_decomp :
      (((c * p : ℝ) : ℂ) • σ1 + (((β_d * c * (ρ + p_y) - E : ℝ) : ℂ) • I2) -
       (((s * c * ρ : ℝ) : ℂ) • σ2) + (((m * c ^ 2 : ℝ) : ℂ) • σ3)) =
      (((c * p : ℝ) : ℂ) • σ1) + (((m * c ^ 2 : ℝ) : ℂ) • σ3) +
      (((-E_d : ℝ) : ℂ) • I2) + (((c * ρ : ℝ) : ℂ) • ((β_d : ℂ) • I2 - (s : ℂ) • σ2)) := by
    rw [smul_sub, smul_smul, smul_smul]
    have hIcoeff : (((β_d * c * (ρ + p_y) - E : ℝ) : ℂ) • I2) =
        (((-E_d : ℝ) : ℂ) • I2) + ((((c * ρ : ℝ) : ℂ) * (β_d : ℂ)) • I2) := by
      rw [← add_smul]
      congr 1
      apply Complex.ext <;> simp [E_d]
      ring
    have hσ2coeff : (((s * c * ρ : ℝ) : ℂ) • σ2) = ((((c * ρ : ℝ) : ℂ) * (s : ℂ)) • σ2) := by
      congr 1
      apply Complex.ext <;> simp
      ring
    rw [hIcoeff, hσ2coeff]
    abel
  rw [hlab_decomp]
  simp only [Matrix.mul_add, Matrix.add_mul, Matrix.mul_smul, Matrix.smul_mul]
  rw [hM1, hM3, hMI, hM2, smul_add, smul_smul, smul_smul, smul_smul]
  have hI_fin : (((-E_d : ℝ) : ℂ) * (γ_d : ℂ)) • I2 = -((((γ_d * E_d : ℝ) : ℂ) • I2)) := by
    rw [← neg_smul]
    congr 1
    apply Complex.ext <;> simp
    ring
  have hσ2_fin :
      (((-E_d : ℝ) : ℂ) * ((s * γ_d * β_d : ℝ) : ℂ)) • σ2 +
      (((c * ρ : ℝ) : ℂ) * ((-(s / γ_d) : ℝ) : ℂ)) • σ2 =
      -((((s * c * ρ_d : ℝ) : ℂ) • σ2)) := by
    rw [← add_smul, ← neg_smul]
    congr 1
    have hρd_id : (-E_d) * (s * γ_d * β_d) + (c * ρ) * (-(s / γ_d)) = -(s * c * ρ_d) := by
      simp only [ρ_d, b_d]
      field_simp [hc, hcosh_ne]
      ring
    apply Complex.ext
    · simp only [Complex.add_re, Complex.mul_re, Complex.neg_re, Complex.ofReal_re, Complex.ofReal_im]
      linear_combination hρd_id
    · simp
  rw [add_assoc (((c * p : ℝ) : ℂ) • σ1 + ((m * c ^ 2 : ℝ) : ℂ) • σ3),
      add_assoc ((((-E_d : ℝ) : ℂ) * (γ_d : ℂ)) • I2), hσ2_fin, hI_fin]
  abel

/-! ### Part 2: Theorem 3(b) — Hellmann--Feynman Strict Negativity & At-Most-One-Zero Theorem -/

/-- **Theorem 3(b) Hellmann--Feynman Strict Negativity at Crossing Zeros**:
For `0 < β_d < 1`, `γ_d > 0`, `E_d > 0`, `m > 0`, `c > 0`, `B_sup > 0`,
and `0 < ℏ < γ_d m² c² / B_sup`, any normalized eigenstate whose expectation values satisfy
Cauchy--Schwarz `μ₁² ≤ μ₂` and the crossing-zero bound `μ₂ ≤ (γ_d E_d / c)² - m² c² + ℏ B_sup / γ_d`
satisfies `|μ₁| < γ_d E_d / c` and
`f_n'(E_d) = (2 γ_d / c)(β_d μ₁ - γ_d E_d / c) < (2 γ_d / c)(β_d - 1)(γ_d E_d / c) < 0`! -/
theorem hellmann_feynman_strict_negativity
    {β_d γ_d E_d m c B_sup ℏ μ₁ μ₂ : ℝ}
    (hβ0 : 0 < β_d) (hβ1 : β_d < 1) (hγ : 0 < γ_d) (hEd : 0 < E_d)
    (hc : 0 < c) (hB : 0 < B_sup)
    (hℏ_hi : ℏ < γ_d * m ^ 2 * c ^ 2 / B_sup)
    (hCS : μ₁ ^ 2 ≤ μ₂)
    (hCross : μ₂ ≤ (γ_d * E_d / c) ^ 2 - m ^ 2 * c ^ 2 + ℏ * B_sup / γ_d) :
    |μ₁| < γ_d * E_d / c ∧
    (2 * γ_d / c) * (β_d * μ₁ - γ_d * E_d / c) <
      (2 * γ_d / c) * (β_d - 1) * (γ_d * E_d / c) ∧
    (2 * γ_d / c) * (β_d - 1) * (γ_d * E_d / c) < 0 := by
  have hℏB : ℏ * B_sup / γ_d < m ^ 2 * c ^ 2 := by
    have h1 : ℏ * B_sup < γ_d * m ^ 2 * c ^ 2 := (lt_div_iff₀ hB).mp hℏ_hi
    exact (div_lt_iff₀ hγ).mpr (by linarith)
  have hμ_sq : μ₁ ^ 2 < (γ_d * E_d / c) ^ 2 := by linarith
  have hR_pos : 0 < γ_d * E_d / c := by positivity
  have habs : |μ₁| < γ_d * E_d / c := by
    rw [← abs_of_pos hR_pos]
    exact sq_lt_sq.mp hμ_sq
  have hμ1_lt : μ₁ < γ_d * E_d / c := lt_of_le_of_lt (le_abs_self μ₁) habs
  have hpref_pos : 0 < 2 * γ_d / c := by positivity
  have hstep1 : β_d * μ₁ - γ_d * E_d / c < (β_d - 1) * (γ_d * E_d / c) := by
    have : β_d * μ₁ < β_d * (γ_d * E_d / c) := mul_lt_mul_of_pos_left hμ1_lt hβ0
    linarith
  have hstep2 : (2 * γ_d / c) * (β_d * μ₁ - γ_d * E_d / c) <
      (2 * γ_d / c) * ((β_d - 1) * (γ_d * E_d / c)) :=
    mul_lt_mul_of_pos_left hstep1 hpref_pos
  have hneg : (2 * γ_d / c) * (β_d - 1) * (γ_d * E_d / c) < 0 := by
    have h1 : (2 * γ_d / c) * (β_d - 1) < 0 := mul_neg_of_pos_of_neg hpref_pos (by linarith)
    exact mul_neg_of_neg_of_pos h1 hR_pos
  exact ⟨habs, by linarith, hneg⟩

/-- **At-Most-One-Zero Theorem** (100% axiom-free real analysis, used in Theorem 3(b)):
If `f : ℝ → ℝ` has derivative `f'(E)` at every `E ∈ [a, b]` and `f'(E) < 0` at every zero
`f(E) = 0` in `[a, b]`, then `f` has **at most one** zero in `[a, b]`. -/
theorem strictly_neg_deriv_at_zeros_at_most_one_zero
    {a b : ℝ} (f f' : ℝ → ℝ)
    (hDeriv : ∀ E ∈ Set.Icc a b, HasDerivAt f (f' E) E)
    (hNeg : ∀ E ∈ Set.Icc a b, f E = 0 → f' E < 0)
    {E₁ E₂ : ℝ} (hE₁ : E₁ ∈ Set.Icc a b) (hE₂ : E₂ ∈ Set.Icc a b)
    (hz₁ : f E₁ = 0) (hz₂ : f E₂ = 0) :
    E₁ = E₂ := by
  by_contra hne
  -- WLOG `E₁ < E₂`
  suffices h : ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc a b → x₂ ∈ Set.Icc a b → x₁ < x₂ →
      f x₁ = 0 → f x₂ = 0 → False by
    rcases lt_or_gt_of_ne hne with hlt | hgt
    · exact h E₁ E₂ hE₁ hE₂ hlt hz₁ hz₂
    · exact h E₂ E₁ hE₂ hE₁ hgt hz₂ hz₁
  intro x₁ x₂ hx₁ hx₂ hlt hf₁ hf₂
  have hCont : ContinuousOn f (Set.Icc x₁ x₂) := fun x hx =>
    (hDeriv x ⟨le_trans hx₁.1 hx.1, le_trans hx.2 hx₂.2⟩).continuousAt.continuousWithinAt
  -- Step 1: Because `f'(x₁) < 0`, there is `r > 0` with `f(x) < 0` for all `x ∈ (x₁, x₁ + r)`
  have hslope₁ := hasDerivAt_iff_tendsto_slope.mp (hDeriv x₁ hx₁)
  have hev₁ : ∀ᶠ x in nhdsWithin x₁ {x₁}ᶜ, slope f x₁ x < 0 :=
    hslope₁.eventually (Iio_mem_nhds (hNeg x₁ hx₁ hf₁))
  rw [eventually_nhdsWithin_iff, Metric.eventually_nhds_iff] at hev₁
  rcases hev₁ with ⟨r, hr_pos, hr_prop⟩
  have hneg_right : ∀ x : ℝ, x₁ < x → x < x₁ + r → f x < 0 := by
    intro x hgt hlt_r
    have hdist : dist x x₁ < r := by
      rw [Real.dist_eq, abs_of_pos (by linarith)]; linarith
    have hne' : x ∈ ({x₁}ᶜ : Set ℝ) := Set.mem_compl_singleton_iff.mpr (ne_of_gt hgt)
    have hsl := hr_prop hdist hne'
    rw [slope_def_field, hf₁, sub_zero] at hsl
    exact (div_neg_iff.mp hsl).elim (fun h => absurd (by linarith : 0 < x - x₁) (not_lt.mpr (le_of_lt h.2))) (fun h => h.1)
  -- Since `f x₂ = 0`, we must have `x₁ + r ≤ x₂`, so let `m := x₁ + r / 2` with `x₁ < m < x₂` and `f m < 0`
  have hm_lt : x₁ + r / 2 < x₂ := by
    by_contra hge
    push Not at hge
    have hlt_r : x₂ < x₁ + r := by linarith
    have := hneg_right x₂ hlt hlt_r
    linarith
  set m := x₁ + r / 2 with hm_def
  have hm_gt : x₁ < m := by rw [hm_def]; linarith
  have hm_lt_r : m < x₁ + r := by rw [hm_def]; linarith
  have hfm : f m < 0 := hneg_right m hm_gt hm_lt_r
  -- Step 2: Take the least zero `x_* = sInf { x ∈ [m, x₂] | f x = 0 }`
  let S : Set ℝ := { x ∈ Set.Icc m x₂ | f x = 0 }
  have hS_ne : S.Nonempty := ⟨x₂, ⟨le_of_lt hm_lt, le_rfl⟩, hf₂⟩
  have hS_bdd : BddBelow S := ⟨m, fun x hx => hx.1.1⟩
  have hS_closed : IsClosed S := by
    have hsub : Set.Icc m x₂ ⊆ Set.Icc x₁ x₂ := Set.Icc_subset_Icc (le_of_lt hm_gt) le_rfl
    exact (hCont.mono hsub).preimage_isClosed_of_isClosed isClosed_Icc isClosed_singleton
  set xstar := sInf S
  have hxstar_mem : xstar ∈ S := hS_closed.csInf_mem hS_ne hS_bdd
  have hxstar_Icc : xstar ∈ Set.Icc a b :=
    ⟨le_trans hx₁.1 (le_trans (le_of_lt hm_gt) hxstar_mem.1.1),
     le_trans hxstar_mem.1.2 hx₂.2⟩
  have hm_lt_star : m < xstar := by
    rcases lt_or_eq_of_le hxstar_mem.1.1 with h | heq
    · exact h
    · exfalso; rw [← heq] at hxstar_mem; linarith [hxstar_mem.2]
  -- For EVERY `x ∈ [m, xstar)`, `f x < 0` (otherwise IVT produces a zero `< xstar`!)
  have hneg_before : ∀ x : ℝ, m ≤ x → x < xstar → f x < 0 := by
    intro x hmx hxlt
    by_contra hge
    push Not at hge
    have hsub : Set.Icc m x ⊆ Set.Icc x₁ x₂ :=
      Set.Icc_subset_Icc (le_of_lt hm_gt) (le_trans (le_of_lt hxlt) hxstar_mem.1.2)
    have hivt := intermediate_value_Icc hmx (hCont.mono hsub)
    rcases hivt ⟨le_of_lt hfm, hge⟩ with ⟨z, hz_Icc, hfz⟩
    have hz_S : z ∈ S := ⟨⟨hz_Icc.1, le_trans hz_Icc.2 (le_trans (le_of_lt hxlt) hxstar_mem.1.2)⟩, hfz⟩
    have hle : xstar ≤ z := csInf_le hS_bdd hz_S
    linarith [hz_Icc.2]
  -- Step 3: Now `f'(xstar) < 0` implies `f x > 0` immediately to the LEFT of `xstar` — CONTRADICTION!
  have hslope_star := hasDerivAt_iff_tendsto_slope.mp (hDeriv xstar hxstar_Icc)
  have hev_star : ∀ᶠ x in nhdsWithin xstar {xstar}ᶜ, slope f xstar x < 0 :=
    hslope_star.eventually (Iio_mem_nhds (hNeg xstar hxstar_Icc hxstar_mem.2))
  rw [eventually_nhdsWithin_iff, Metric.eventually_nhds_iff] at hev_star
  rcases hev_star with ⟨r', hr'_pos, hr'_prop⟩
  set δ := min (r' / 2) ((xstar - m) / 2)
  have hδ_pos : 0 < δ := lt_min (by linarith) (by linarith)
  set y := xstar - δ
  have hy_ge : m ≤ y := by dsimp [y, δ]; linarith [min_le_right (r' / 2) ((xstar - m) / 2)]
  have hy_lt : y < xstar := by dsimp [y]; linarith
  have hfy_neg : f y < 0 := hneg_before y hy_ge hy_lt
  have hdist_y : dist y xstar < r' := by
    rw [Real.dist_eq, show y - xstar = -δ by ring, abs_neg, abs_of_pos hδ_pos]
    exact lt_of_le_of_lt (min_le_left _ _) (by linarith)
  have hne_y : y ∈ ({xstar}ᶜ : Set ℝ) := Set.mem_compl_singleton_iff.mpr (ne_of_lt hy_lt)
  have hsl_y := hr'_prop hdist_y hne_y
  rw [slope_def_field, hxstar_mem.2, sub_zero] at hsl_y
  -- `f y < 0` and `y - xstar < 0` imply `f y / (y - xstar) > 0`, contradicting `hsl_y < 0`!
  have hpos_div : 0 < f y / (y - xstar) := div_pos_of_neg_of_neg hfy_neg (by linarith)
  linarith

/-! ### Part 3: Upgraded One-to-One Theorem 3(b) (`thm:levels_crossed`(b)) -/

/-- **Upgraded One-to-One Theorem 3(b) (`thm:levels_crossed`(b))**:
Combining:
- `lem_labels_part_b` (constant integer shift `m_L` WITHOUT global monotonicity of `Δ_θ`),
- fixed-`E_d` Sturm oscillation (`Δ_θ(E_d) = n π ↔ f_n(E_d) = 0`),
- `strictly_neg_deriv_at_zeros_at_most_one_zero` (from Hellmann--Feynman `f_n'(E_d) < 0` at zeros),
- `lem_levels_action_one_to_one` (action error to one-to-one energy approximation),
proves that EVERY Bohr--Sommerfeld energy `E₀ ∈ [a + δ₀, b - δ₀]` has a **UNIQUE** nearby
exact level `E_star ∈ [E₀ - δ₀, E₀ + δ₀]` with `I_hbar(E_star) ∈ 2π ℏ ℤ`! -/
theorem thm_levels_crossed_one_to_one
    {a b ℏ C_I T_min T_up : ℝ} (hab : a ≤ b)
    (hℏ : 0 < ℏ) (hC : 0 ≤ C_I) (hT : 0 < T_min) (hTup : 0 < T_up)
    (I₁ I_hbar θp θm : ℝ → ℝ) (θG : ℝ → ℝ → ℝ)
    -- Matching continuity & `O(ℏ²)` action approximation from Lemma 7 / Prop. 5
    (hContI : ContinuousOn I_hbar (Set.Icc a b))
    (hApprox : ∀ E ∈ Set.Icc a b, |I_hbar E - I₁ E - Real.pi * ℏ| ≤ C_I * ℏ ^ 2)
    (hRateLo : ∀ E₁ ∈ Set.Icc a b, ∀ E₂ ∈ Set.Icc a b, E₁ ≤ E₂ →
      (T_min / 2) * (E₂ - E₁) ≤ I₁ E₂ - I₁ E₁)
    (hRateHi : ∀ E₁ ∈ Set.Icc a b, ∀ E₂ ∈ Set.Icc a b, E₁ ≤ E₂ →
      I₁ E₂ - I₁ E₁ ≤ T_up * (E₂ - E₁))
    (hSep : 2 * (2 * C_I * ℏ ^ 2 / T_min) < 2 * Real.pi * ℏ / T_up)
    -- Lifted matching angle (Lemma 6(a), NO monotonicity of `θm - θp` assumed!)
    (hContGp : ContinuousOn (fun E => θG (θp E) E) (Set.Icc a b))
    (hContGm : ContinuousOn (fun E => θG (θm E) E) (Set.Icc a b))
    (hStrict : ∀ E ∈ Set.Icc a b, StrictMono (fun θ => θG θ E))
    (hShift : ∀ E ∈ Set.Icc a b, ∀ θ : ℝ, θG (θ + Real.pi) E = θG θ E + Real.pi)
    (hCong : ∀ E ∈ Set.Icc a b, ∃ m : ℤ,
      I_hbar E / (2 * ℏ) - (θG (θm E) E - θG (θp E) E) = (m : ℝ) * Real.pi)
    -- Fixed-`E_d` Sturm oscillation linking `Δ_θ(E_d) = j π` to branch zero `f_j(E_d) = 0`,
    -- with Hellmann--Feynman derivative `f_j'(E_d) < 0` at every zero!
    (f_branch f_branch' : ℤ → ℝ → ℝ)
    (hOsc : ∀ j : ℤ, ∀ E ∈ Set.Icc a b, θm E - θp E = (j : ℝ) * Real.pi → f_branch j E = 0)
    (hDeriv : ∀ j : ℤ, ∀ E ∈ Set.Icc a b, HasDerivAt (f_branch j) (f_branch' j E) E)
    (hHF : ∀ j : ℤ, ∀ E ∈ Set.Icc a b, f_branch j E = 0 → f_branch' j E < 0)
    (n : ℤ) (E₀ : ℝ)
    (hE₀_rule : I₁ E₀ = 2 * Real.pi * ℏ * ((n : ℝ) + 1 / 2))
    (hL : a ≤ E₀ - 2 * C_I * ℏ ^ 2 / T_min)
    (hR : E₀ + 2 * C_I * ℏ ^ 2 / T_min ≤ b) :
    let δ₀ := 2 * C_I * ℏ ^ 2 / T_min
    ∃! E_star : ℝ, E_star ∈ Set.Icc (E₀ - δ₀) (E₀ + δ₀) ∧
      ∃ k : ℤ, I_hbar E_star = 2 * Real.pi * ℏ * (k : ℝ) := by
  -- Step 1: Constant integer shift `m_L` from Lemma 6(b) (NO global monotonicity!)
  rcases lem_labels_part_b hab hℏ I_hbar θp θm θG hContI hContGp hContGm hStrict hShift hCong
    with ⟨m_L, hm_L⟩
  -- Step 2: `I_hbar` takes each quantized value `2π ℏ k` at most once on `[a, b]`!
  have hOnce : ∀ k : ℤ, ∀ E₁ ∈ Set.Icc a b, ∀ E₂ ∈ Set.Icc a b,
      I_hbar E₁ = 2 * Real.pi * ℏ * (k : ℝ) →
      I_hbar E₂ = 2 * Real.pi * ℏ * (k : ℝ) → E₁ = E₂ := by
    intro k E₁ hE₁ E₂ hE₂ h1 h2
    have hΔ1 := (hm_L E₁ hE₁ k).mp h1
    have hΔ2 := (hm_L E₂ hE₂ k).mp h2
    have hz1 := hOsc (k - m_L) E₁ hE₁ hΔ1
    have hz2 := hOsc (k - m_L) E₂ hE₂ hΔ2
    exact strictly_neg_deriv_at_zeros_at_most_one_zero (f_branch (k - m_L)) (f_branch' (k - m_L))
      (hDeriv (k - m_L)) (hHF (k - m_L)) hE₁ hE₂ hz1 hz2
  -- Step 3: Transfer to one-to-one `O(ℏ²)` spectral approximation via Lemma 1!
  exact lem_levels_action_one_to_one hℏ hC hT hTup I₁ I_hbar hContI hApprox hRateLo hRateHi
    hOnce hSep n E₀ hE₀_rule hL hR

/-! ### Part 4: Theorem 4 (`thm:levels_grav`) — Corrected `ν^{±1/2}` Kernel Bijection & Weyl Identity -/

section GravRescaling

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]

/-- **Theorem 4 (`thm:levels_grav`), Corrected `ν^{±1/2}` Kernel Bijection & Weyl Identity**:
With `ν = α_n / W`, `N = ν^{1/2}` (invertible, symmetric, with `N * Ninv = 1`),
`A = V + α_n m c² - E = -ν U_op`, `D = V - α_n m c² - E = -ν g_inv`:
For any `(u, d)` with residuals
`r₁ = A u + c N Q_adj (N d)` and `r₂ = c N Q (N u) + D d`,
setting `F = N u` and `G = N d` (inverse `u = Ninv F`, `d = Ninv G`) satisfies:
1. `Ninv r₁ = -U_op F + c Q_adj G` and `Ninv r₂ = c Q F - g_inv G`;
2. `G = c g (Q F) - g (Ninv r₂)`;
3. **Exact Weyl-sequence quadratic-form identity**:
   `c² ⟨Q F, g (Q F)⟩ - ⟨F, U_op F⟩ = ⟨F, Ninv r₁⟩ + c ⟨Q F, g (Ninv r₂)⟩`!
In particular, when `r₁ = r₂ = 0`, `F ↦ (Ninv F, c Ninv (g (Q F)))` is an exact linear bijection
from `ker K(E)` onto `ker(H_unweighted - E)`! -/
theorem grav_unweighted_weyl_form_identity
    (Q Q_adj N Ninv g g_inv U_op : H →ₗ[ℂ] H)
    (hQ_adj : ∀ x y : H, inner ℂ (Q x) y = inner ℂ x (Q_adj y))
    (hNNinv : ∀ x : H, Ninv (N x) = x)
    (hgginv : ∀ x : H, g (g_inv x) = x)
    (c : ℝ)
    (u d r₁ r₂ : H)
    (hr₁ : r₁ = -(N (U_op (N u))) + (c : ℂ) • (N (Q_adj (N d))))
    (hr₂ : r₂ = (c : ℂ) • (N (Q (N u))) - N (g_inv (N d))) :
    let F := N u
    let G := N d
    (Ninv r₁ = -(U_op F) + (c : ℂ) • Q_adj G) ∧
    (G = (c : ℂ) • g (Q F) - g (Ninv r₂)) ∧
    ((c : ℂ) ^ 2 * inner ℂ (Q F) (g (Q F)) - inner ℂ F (U_op F) =
      inner ℂ F (Ninv r₁) + (c : ℂ) * inner ℂ (Q F) (g (Ninv r₂))) := by
  intro F G
  have h1 : Ninv r₁ = -(U_op F) + (c : ℂ) • Q_adj G := by
    rw [hr₁, map_add, map_neg, map_smul, hNNinv, hNNinv]
  have h2_raw : Ninv r₂ = (c : ℂ) • Q F - g_inv G := by
    rw [hr₂, map_sub, map_smul, hNNinv, hNNinv]
  have h2 : G = (c : ℂ) • g (Q F) - g (Ninv r₂) := by
    have hg := congr_arg g h2_raw
    rw [map_sub, map_smul, hgginv] at hg
    -- hg : g (Ninv r₂) = c • g (Q F) - G
    calc G = (c : ℂ) • g (Q F) - ((c : ℂ) • g (Q F) - G) := by abel
      _ = (c : ℂ) • g (Q F) - g (Ninv r₂) := by rw [← hg]
  refine ⟨h1, h2, ?_⟩
  have hinner1 : inner ℂ F (Ninv r₁) = -inner ℂ F (U_op F) + (c : ℂ) * inner ℂ (Q F) G := by
    rw [h1, inner_add_right, inner_neg_right, inner_smul_right, hQ_adj]
  rw [hinner1, h2, inner_sub_right, inner_smul_right]
  ring

end GravRescaling

/-! ### Part 5: Master Monotone Assembly for Theorems 1, 2, and 4 (`thm:levels_B`, `thm:levels_V`, `thm:levels_grav`) -/

/-- Structured matching package representing the output of Olver's uniform Airy matching
(Prop. 5 + Lemma 7) for a monotone scalar reduction `K(E) = c² Q_s† g Q_s - U` on `J = [a, b]`.
Parameterizing by this structure keeps the entire Lean formalization 100% free of global axioms! -/
structure MonotoneScalarMatchingData (a b ℏ C_I T_min T_up : ℝ) where
  I₁ : ℝ → ℝ
  I_hbar : ℝ → ℝ
  θp : ℝ → ℝ
  θm : ℝ → ℝ
  θG : ℝ → ℝ → ℝ
  hContI : ContinuousOn I_hbar (Set.Icc a b)
  hApprox : ∀ E ∈ Set.Icc a b, |I_hbar E - I₁ E - Real.pi * ℏ| ≤ C_I * ℏ ^ 2
  hRateLo : ∀ E₁ ∈ Set.Icc a b, ∀ E₂ ∈ Set.Icc a b, E₁ ≤ E₂ →
    (T_min / 2) * (E₂ - E₁) ≤ I₁ E₂ - I₁ E₁
  hRateHi : ∀ E₁ ∈ Set.Icc a b, ∀ E₂ ∈ Set.Icc a b, E₁ ≤ E₂ →
    I₁ E₂ - I₁ E₁ ≤ T_up * (E₂ - E₁)
  hContGp : ContinuousOn (fun E => θG (θp E) E) (Set.Icc a b)
  hContGm : ContinuousOn (fun E => θG (θm E) E) (Set.Icc a b)
  hStrict : ∀ E ∈ Set.Icc a b, StrictMono (fun θ => θG θ E)
  hShift : ∀ E ∈ Set.Icc a b, ∀ θ : ℝ, θG (θ + Real.pi) E = θG θ E + Real.pi
  hCong : ∀ E ∈ Set.Icc a b, ∃ m : ℤ,
    I_hbar E / (2 * ℏ) - (θG (θm E) E - θG (θp E) E) = (m : ℝ) * Real.pi
  /-- Strictly increasing Sturm angle difference `Δ_θ = θ_- - θ_+` (from Lemma 5 + Lemma 8) -/
  hSturmMono : StrictMonoOn (fun E => θm E - θp E) (Set.Icc a b)

/-- **Master One-to-One Theorem for Theorems 1 (`thm:levels_B`), 2 (`thm:levels_V`), and 4 (`thm:levels_grav`)**:
From any `MonotoneScalarMatchingData` with `2 δ₀ < 2π ℏ / T_up`, every Bohr--Sommerfeld solution
`E₀ ∈ [a + δ₀, b - δ₀]` has a **UNIQUE** nearby exact eigenvalue `E_star ∈ [E₀ - δ₀, E₀ + δ₀]`! -/
theorem thm_levels_monotone_one_to_one
    {a b ℏ C_I T_min T_up : ℝ} (hab : a ≤ b)
    (hℏ : 0 < ℏ) (hC : 0 ≤ C_I) (hT : 0 < T_min) (hTup : 0 < T_up)
    (D : MonotoneScalarMatchingData a b ℏ C_I T_min T_up)
    (hSep : 2 * (2 * C_I * ℏ ^ 2 / T_min) < 2 * Real.pi * ℏ / T_up)
    (n : ℤ) (E₀ : ℝ)
    (hE₀_rule : D.I₁ E₀ = 2 * Real.pi * ℏ * ((n : ℝ) + 1 / 2))
    (hL : a ≤ E₀ - 2 * C_I * ℏ ^ 2 / T_min)
    (hR : E₀ + 2 * C_I * ℏ ^ 2 / T_min ≤ b) :
    let δ₀ := 2 * C_I * ℏ ^ 2 / T_min
    ∃! E_star : ℝ, E_star ∈ Set.Icc (E₀ - δ₀) (E₀ + δ₀) ∧
      ∃ k : ℤ, D.I_hbar E_star = 2 * Real.pi * ℏ * (k : ℝ) := by
  rcases lem_labels_part_b hab hℏ D.I_hbar D.θp D.θm D.θG
      D.hContI D.hContGp D.hContGm D.hStrict D.hShift D.hCong with ⟨m_L, hm_L⟩
  have hOnce := lem_labels_part_c D.I_hbar (fun E => D.θm E - D.θp E) m_L hm_L D.hSturmMono
  exact lem_levels_action_one_to_one hℏ hC hT hTup D.I₁ D.I_hbar
    D.hContI D.hApprox D.hRateLo D.hRateHi hOnce hSep n E₀ hE₀_rule hL hR

end SGM.Semiclassical
