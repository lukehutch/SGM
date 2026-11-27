import Mathlib.Algebra.Quaternion
import Mathlib.Basic.Complex.Basic
import Mathlib.Tactic

/-!
# Quaternionic Spinor Isomorphism and Pauli Algebra (`sgm_wavefn.tex` Sec. 2.3)

Formalizes:
- `eq:spinor_quaternion`: The exact `ℝ`-linear isomorphism `ℂ² ≃ₗ[ℝ] ℍ[ℝ]`,
  `(ψ₁, ψ₂) ↦ Re ψ₁ - (Im ψ₂) e₁ + (Re ψ₂) e₂ - (Im ψ₁) e₃`.
- `eq:i_right_e3`: Complex multiplication `i • ψ ↔ -ψ * e₃`.
- `eq:pauli_quaternion` & `eq:minus_i_sigma`: `σ_k ψ ↔ -e_k * ψ * e₃` and `-i σ_k ψ ↔ e_k * ψ`.
- `eq:vector_quaternion_product`: `(u · e)(v · e) = -(u · v) + (u × v) · e`.
- `eq:spin_vector`: Spin density vector `s · e = ψ * e₃ * star ψ`, invariant under right `U(1)`
  phases `ψ ↦ ψ * (cos φ + sin φ e₃)` and rotated by left `SU(2)` rotors `ψ ↦ R * ψ`.
-/

namespace SGM.Algebra

abbrev Quat := Quaternion ℝ

/-- Basis quaternion unit `e₁ = i`. -/
def e1 : Quat := ⟨0, 1, 0, 0⟩
/-- Basis quaternion unit `e₂ = j`. -/
def e2 : Quat := ⟨0, 0, 1, 0⟩
/-- Basis quaternion unit `e₃ = k`. -/
def e3 : Quat := ⟨0, 0, 0, 1⟩

/-- The three imaginary units indexed by `Fin 3`. -/
def eUnit : Fin 3 → Quat
  | 0 => e1
  | 1 => e2
  | 2 => e3

/-- Embed a real 3-vector `v` as the pure imaginary quaternion `v · e`. -/
def vecQuat (v : Fin 3 → ℝ) : Quat := ⟨0, v 0, v 1, v 2⟩

/-- `sgm_wavefn.tex` Eq. `eq:unit_table`: Quaternion multiplication rules. -/
theorem e1_sq : e1 * e1 = -1 := by ext <;> simp [e1]
theorem e2_sq : e2 * e2 = -1 := by ext <;> simp [e2]
theorem e3_sq : e3 * e3 = -1 := by ext <;> simp [e3]
theorem e1_mul_e2 : e1 * e2 = e3 := by ext <;> simp [e1, e2, e3]
theorem e2_mul_e3 : e2 * e3 = e1 := by ext <;> simp [e1, e2, e3]
theorem e3_mul_e1 : e3 * e1 = e2 := by ext <;> simp [e1, e2, e3]
theorem e2_mul_e1 : e2 * e1 = -e3 := by ext <;> simp [e1, e2, e3]
theorem e3_mul_e2 : e3 * e2 = -e1 := by ext <;> simp [e1, e2, e3]
theorem e1_mul_e3 : e1 * e3 = -e2 := by ext <;> simp [e1, e2, e3]

/-- `sgm_wavefn.tex` Eq. `eq:vector_quaternion_product`:
`(u · e)(v · e) = -(u · v) + (u × v) · e`. -/
theorem vector_quaternion_product (u v : Fin 3 → ℝ) :
    vecQuat u * vecQuat v =
      ⟨-(u 0 * v 0 + u 1 * v 1 + u 2 * v 2),
        u 1 * v 2 - u 2 * v 1,
        u 2 * v 0 - u 0 * v 2,
        u 0 * v 1 - u 1 * v 0⟩ := by
  ext <;> simp [vecQuat] <;> ring

/-- Two-component complex spinors `ℂ × ℂ`. -/
abbrev Spinor2 := ℂ × ℂ

/-- `sgm_wavefn.tex` Eq. `eq:spinor_quaternion`:
Map `(ψ₁, ψ₂) ∈ ℂ²` to `Re ψ₁ - (Im ψ₂) e₁ + (Re ψ₂) e₂ - (Im ψ₁) e₃ ∈ ℍ`. -/
def spinorToQuat (ψ : Spinor2) : Quat :=
  ⟨ψ.1.re, -ψ.2.im, ψ.2.re, -ψ.1.im⟩

/-- Inverse map `ℍ → ℂ²`. -/
def quatToSpinor (q : Quat) : Spinor2 :=
  (⟨q.re, -q.imK⟩, ⟨q.imJ, -q.imI⟩)

theorem quatToSpinor_spinorToQuat (ψ : Spinor2) : quatToSpinor (spinorToQuat ψ) = ψ := by
  rcases ψ with ⟨⟨a, b⟩, ⟨c, d⟩⟩
  simp [spinorToQuat, quatToSpinor]

theorem spinorToQuat_quatToSpinor (q : Quat) : spinorToQuat (quatToSpinor q) = q := by
  rcases q with ⟨a, b, c, d⟩
  simp [spinorToQuat, quatToSpinor]

/-- The spinor-to-quaternion correspondence as an explicit `ℝ`-linear equivalence `ℂ² ≃ₗ[ℝ] ℍ`. -/
def spinorEquivQuat : Spinor2 ≃ₗ[ℝ] Quat where
  toFun := spinorToQuat
  invFun := quatToSpinor
  left_inv := quatToSpinor_spinorToQuat
  right_inv := spinorToQuat_quatToSpinor
  map_add' x y := by ext <;> simp [spinorToQuat] <;> ring
  map_smul' r x := by ext <;> simp [spinorToQuat]

/-- `sgm_wavefn.tex` Eq. `eq:i_right_e3`:
Multiplying a complex spinor by `i` corresponds to right multiplication by `-e₃`. -/
theorem i_mul_eq_right_neg_e3 (ψ : Spinor2) :
    spinorToQuat (Complex.I * ψ.1, Complex.I * ψ.2) = -(spinorToQuat ψ * e3) := by
  ext <;> simp [spinorToQuat, e3]

/-- Pauli matrix actions on `Spinor2`. -/
def sigma1 (ψ : Spinor2) : Spinor2 := (ψ.2, ψ.1)
def sigma2 (ψ : Spinor2) : Spinor2 := (-Complex.I * ψ.2, Complex.I * ψ.1)
def sigma3 (ψ : Spinor2) : Spinor2 := (ψ.1, -ψ.2)

/-- `sgm_wavefn.tex` Eq. `eq:pauli_quaternion`: `σ_k ψ ↔ -e_k * ψ * e₃`. -/
theorem sigma1_eq_quat (ψ : Spinor2) :
    spinorToQuat (sigma1 ψ) = -(e1 * spinorToQuat ψ * e3) := by
  ext <;> simp [spinorToQuat, sigma1, e1, e3]

theorem sigma2_eq_quat (ψ : Spinor2) :
    spinorToQuat (sigma2 ψ) = -(e2 * spinorToQuat ψ * e3) := by
  ext <;> simp [spinorToQuat, sigma2, e2, e3]

theorem sigma3_eq_quat (ψ : Spinor2) :
    spinorToQuat (sigma3 ψ) = -(e3 * spinorToQuat ψ * e3) := by
  ext <;> simp [spinorToQuat, sigma3, e3]

/-- `sgm_wavefn.tex` Eq. `eq:minus_i_sigma`: `-i σ_k ψ ↔ e_k * ψ`. -/
theorem neg_i_sigma1_eq_left_e1 (ψ : Spinor2) :
    spinorToQuat (-Complex.I * (sigma1 ψ).1, -Complex.I * (sigma1 ψ).2) = e1 * spinorToQuat ψ := by
  ext <;> simp [spinorToQuat, sigma1, e1]

theorem neg_i_sigma2_eq_left_e2 (ψ : Spinor2) :
    spinorToQuat (-Complex.I * (sigma2 ψ).1, -Complex.I * (sigma2 ψ).2) = e2 * spinorToQuat ψ := by
  ext <;> simp [spinorToQuat, sigma2, e2]

theorem neg_i_sigma3_eq_left_e3 (ψ : Spinor2) :
    spinorToQuat (-Complex.I * (sigma3 ψ).1, -Complex.I * (sigma3 ψ).2) = e3 * spinorToQuat ψ := by
  ext <;> simp [spinorToQuat, sigma3, e3]

/-- Squared Euclidean norm `|ψ|² = (ψ * star ψ).re` of a quaternion. -/
def qNormSq (q : Quat) : ℝ := q.re ^ 2 + q.imI ^ 2 + q.imJ ^ 2 + q.imK ^ 2

theorem qNormSq_eq_mul_star (q : Quat) : qNormSq q = (q * star q).re := by
  simp [qNormSq]
  ring

/-- Probability density equality `|ψ₁|² + |ψ₂|² = qNormSq(ψ)` (`sgm_wavefn.tex` Sec. 2.3). -/
theorem normSq_spinorToQuat (ψ : Spinor2) :
    qNormSq (spinorToQuat ψ) =
      Complex.normSq ψ.1 + Complex.normSq ψ.2 := by
  simp [spinorToQuat, qNormSq, Complex.normSq]
  ring

/-- `sgm_wavefn.tex` Eq. `eq:spin_vector`:
The spin density vector `s · e = ψ * e₃ * star ψ` is pure imaginary and matches `ψ† σ ψ`. -/
def spinQuat (ψ : Quat) : Quat := ψ * e3 * star ψ

theorem spinQuat_re_zero (ψ : Quat) : (spinQuat ψ).re = 0 := by
  simp [spinQuat, e3]
  ring

/-- Components of `ψ * e₃ * star ψ` match `Re(ψ† σ_k ψ)` for `k = 1, 2, 3`. -/
theorem spinQuat_matches_pauli (ψ : Spinor2) :
    (spinQuat (spinorToQuat ψ)).imI = (star ψ.1 * (sigma1 ψ).1 + star ψ.2 * (sigma1 ψ).2).re ∧
    (spinQuat (spinorToQuat ψ)).imJ = (star ψ.1 * (sigma2 ψ).1 + star ψ.2 * (sigma2 ψ).2).re ∧
    (spinQuat (spinorToQuat ψ)).imK = (star ψ.1 * (sigma3 ψ).1 + star ψ.2 * (sigma3 ψ).2).re := by
  refine ⟨?_, ?_, ?_⟩ <;>
    simp [spinQuat, spinorToQuat, sigma1, sigma2, sigma3, e3] <;> ring

/-- Right `U(1)` phase factor `exp(e₃ φ) = cos φ + sin φ e₃`. -/
noncomputable def rightPhase (φ : ℝ) : Quat := ⟨Real.cos φ, 0, 0, Real.sin φ⟩

/-- `sgm_wavefn.tex` Eq. `eq:right_phase`:
Right `U(1)` phase rotation leaves the spin density vector `ψ e₃ star ψ` identically invariant. -/
theorem spinQuat_right_phase_invariant (ψ : Quat) (φ : ℝ) :
    spinQuat (ψ * rightPhase φ) = spinQuat ψ := by
  have hcs : Real.cos φ ^ 2 + Real.sin φ ^ 2 = 1 := Real.cos_sq_add_sin_sq φ
  have hcomm : rightPhase φ * e3 * star (rightPhase φ) = e3 := by
    ext <;> simp [rightPhase, e3] <;> nlinarith [hcs]
  calc spinQuat (ψ * rightPhase φ)
      = ψ * rightPhase φ * e3 * (star (rightPhase φ) * star ψ) := by
          simp [spinQuat, star_mul]
    _ = ψ * (rightPhase φ * e3 * star (rightPhase φ)) * star ψ := by
          simp [mul_assoc]
    _ = ψ * e3 * star ψ := by rw [hcomm]
    _ = spinQuat ψ := rfl

/-- `sgm_wavefn.tex` Eq. `eq:left_rotation`:
Left rotor action rotates the spin density vector by conjugation `R (s · e) star R`. -/
theorem spinQuat_left_rotor (R ψ : Quat) :
    spinQuat (R * ψ) = R * spinQuat ψ * star R := by
  simp [spinQuat, star_mul, mul_assoc]

end SGM.Algebra
