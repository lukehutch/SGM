import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic

/-!
# Dimensional Analysis of the Gravitational Sector (`sgm.tex` Prop. 2 & Supp. S3.1)

Formalizes:
- `prop:no_grav_action`: No product `c^{n_c} G^{n_G}` has the dimension of action `M L^2 T^{-1}`.
- `eq:hbar_cancels_identity`: Planck's constant cancels identically from `-ℏ ∇((mc^2/ℏ) α)`.
- `eq:dimension_matrix`: The 3×3 dimensional matrix `D` of `(ℏ, c, G)` has `det D = 2 ≠ 0`
  and explicit rational inverse `D⁻¹`.
- `eq:force_unit` & `eq:force_unit_planck`: The Planck force `F_P = ℏ / (ℓ_P t_P) = c^4 / G`
  is independent of `ℏ`.
- `eq:radians_per_crossing`: The gravitational phase accumulated per light-crossing time
  `(GMm / r^2) / (ℏc / r^2) = GMm / (ℏc)` is scale-invariant.
-/

namespace SGM.Basic

/-- Dimensional exponent vector `(M, L, T)` over `ℚ`. -/
abbrev DimVec := Fin 3 → ℚ

/-- Dimension of action (`ℏ`): `M^1 L^2 T^{-1}`. -/
def dimHbar : DimVec := ![1, 2, -1]

/-- Dimension of speed (`c`): `M^0 L^1 T^{-1}`. -/
def dimC : DimVec := ![0, 1, -1]

/-- Dimension of Newton's constant (`G`): `M^{-1} L^3 T^{-2}`. -/
def dimG : DimVec := ![-1, 3, -2]

/-- Dimension of force (`F`): `M^1 L^1 T^{-2}`. -/
def dimForce : DimVec := ![1, 1, -2]

/-- Dimension of mass (`M`): `M^1 L^0 T^0`. -/
def dimMass : DimVec := ![1, 0, 0]

/-- Dimension of length (`L`): `M^0 L^1 T^0`. -/
def dimLength : DimVec := ![0, 1, 0]

/-- Dimension of time (`T`): `M^0 L^0 T^1`. -/
def dimTime : DimVec := ![0, 0, 1]

/-- `sgm.tex` Proposition 2 (`prop:no_grav_action`):
No combination `c^{n_c} G^{n_G}` has the dimension of action `ℏ`. -/
theorem no_grav_action_constant :
    ¬ ∃ (nc nG : ℚ), nc • dimC + nG • dimG = dimHbar := by
  rintro ⟨nc, nG, h⟩
  have h0 := congr_fun h 0
  have h1 := congr_fun h 1
  have h2 := congr_fun h 2
  simp [dimC, dimG, dimHbar] at h0 h1 h2
  linarith

/-- The 3×3 dimensional matrix `D` whose rows are the `(M, L, T)` exponents of `(ℏ, c, G)`
(`sgm.tex` Supp. S3.1). -/
def dimMatrix : Matrix (Fin 3) (Fin 3) ℚ :=
  !![1, 2, -1;
     0, 1, -1;
    -1, 3, -2]

/-- Explicit inverse `D⁻¹` (`sgm.tex` Eq. `eq:dimension_matrix`). -/
def dimMatrixInv : Matrix (Fin 3) (Fin 3) ℚ :=
  !![1/2,  1/2, -1/2;
     1/2, -3/2,  1/2;
     1/2, -5/2,  1/2]

/-- `det(D) = 2 ≠ 0`, so `(ℏ, c, G)` form a basis of mechanical dimensions. -/
theorem dimMatrix_det : dimMatrix.det = 2 := by
  rw [Matrix.det_fin_three]
  norm_num [dimMatrix]

/-- `D * D⁻¹ = I` (`eq:dimension_matrix`). -/
theorem dimMatrix_mul_inv : dimMatrix * dimMatrixInv = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [dimMatrix, dimMatrixInv, Matrix.mul_apply, Fin.sum_univ_three] <;>
    norm_num

/-- `D⁻¹ * D = I`. -/
theorem dimMatrix_inv_mul : dimMatrixInv * dimMatrix = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [dimMatrix, dimMatrixInv, Matrix.mul_apply, Fin.sum_univ_three] <;>
    norm_num

/-- Planck force dimensions `c^4 G^{-1}` equal `M L T^{-2}` with zero power of `ℏ`. -/
theorem planck_force_dim :
    (0 : ℚ) • dimHbar + (4 : ℚ) • dimC + (-1 : ℚ) • dimG = dimForce := by
  ext i
  fin_cases i <;> simp [dimHbar, dimC, dimG, dimForce] <;> norm_num

/-- Planck length dimensions `ℏ^{1/2} G^{1/2} c^{-3/2} = L` (`eq:planck_length`). -/
theorem planck_length_dim :
    (1/2 : ℚ) • dimHbar + (-3/2 : ℚ) • dimC + (1/2 : ℚ) • dimG = dimLength := by
  ext i
  fin_cases i <;> simp [dimHbar, dimC, dimG, dimLength] <;> norm_num

/-- Planck time dimensions `ℏ^{1/2} G^{1/2} c^{-5/2} = T` (`eq:planck_time`). -/
theorem planck_time_dim :
    (1/2 : ℚ) • dimHbar + (-5/2 : ℚ) • dimC + (1/2 : ℚ) • dimG = dimTime := by
  ext i
  fin_cases i <;> simp [dimHbar, dimC, dimG, dimTime] <;> norm_num

/-- Planck mass dimensions `ℏ^{1/2} c^{1/2} G^{-1/2} = M` (`eq:planck_mass`). -/
theorem planck_mass_dim :
    (1/2 : ℚ) • dimHbar + (1/2 : ℚ) • dimC + (-1/2 : ℚ) • dimG = dimMass := by
  ext i
  fin_cases i <;> simp [dimHbar, dimC, dimG, dimMass] <;> norm_num

/-- `sgm.tex` Eq. `eq:force_unit_planck` & `eq:force_unit`:
`F_P = ℏ / (ℓ_P t_P) = c^4 / G`, completely independent of `ℏ`. -/
theorem planck_force_cancel_hbar {ℏ c G : ℝ} (hℏ : 0 < ℏ) (hc : 0 < c) (hG : 0 < G) :
    ℏ / (Real.sqrt (ℏ * G / c ^ 3) * Real.sqrt (ℏ * G / c ^ 5)) = c ^ 4 / G := by
  have h1 : 0 ≤ ℏ * G / c ^ 3 := by positivity
  rw [← Real.sqrt_mul h1]
  have hsq : ℏ * G / c ^ 3 * (ℏ * G / c ^ 5) = (ℏ * G / c ^ 4) ^ 2 := by
    field_simp
  rw [hsq, Real.sqrt_sq (by positivity)]
  field_simp

/-- `sgm.tex` Eq. `eq:hbar_cancels_identity`:
Planck's constant cancels identically from `-ℏ ∇((mc^2/ℏ) α) = -mc^2 ∇α`. -/
theorem hbar_cancels_grav_force {ℏ m c : ℝ} (hℏ : ℏ ≠ 0) (gradAlpha : Fin 3 → ℝ) :
    (fun i => -ℏ * ((m * c ^ 2 / ℏ) * gradAlpha i)) = (fun i => -(m * c ^ 2) * gradAlpha i) := by
  ext i
  field_simp

/-- `sgm.tex` Eq. `eq:radians_per_crossing`:
The ratio of Newtonian force `GMm/r^2` to `F_P(r) = ℏc/r^2` is `GMm/(ℏc)`, independent of `r`. -/
theorem radians_per_crossing_scale_invariant {G M m r ℏ c : ℝ}
    (hr : r ≠ 0) (hℏ : ℏ ≠ 0) (hc : c ≠ 0) :
    (G * M * m / r ^ 2) / (ℏ * c / r ^ 2) = G * M * m / (ℏ * c) := by
  field_simp

end SGM.Basic
