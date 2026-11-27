import SGM.Algebra.QuaternionSpinor
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Tactic

/-!
# Abstract Covariant Derivations and the 4×4 Quaternion Operator Table
(`sgm_wavefn.tex` Secs. 3.2 & 7.6)

Formalizes:
- `eq:jacobi`: The operator Jacobi identity `[D_α, [D_β, D_γ]] + cyc = 0` for arbitrary
  linear operators on any module.
- `prop:bimodule_associativity`: Derivative commutator `[∂_x, R_{f e₃}] ψ = (∂_x f) ψ e₃`
  via the real product rule (`HasDerivAt`).
- `eq:general_table` & `eq:conjugate_unit_table`: Exact expansion of the second-order
  quaternionic wave operator `(N⁻₀ - ∑_j e_j N⁻_j)(N⁺₀ + ∑_k e_k N⁺_k)` into:
  - diagonal Klein--Gordon part `N⁻₀ N⁺₀ + ∑_k N⁻_k N⁺_k`,
  - border electric/gravitational commutator part `∑_k e_k (N⁻₀ N⁺_k - N⁻_k N⁺₀)`,
  - interior magnetic/curvature spin part `∑_{j<k} e_{j×k} (N⁻_j N⁺_k - N⁻_k N⁺_j)`.
-/

namespace SGM.Differential

open SGM.Algebra

/-- Operator commutator `[A, B] = A ∘ B - B ∘ A`. -/
def opComm {V : Type*} [AddCommGroup V] [Module ℝ V] (A B : V →ₗ[ℝ] V) : V →ₗ[ℝ] V :=
  A.comp B - B.comp A

/-- `sgm_wavefn.tex` Eq. `eq:jacobi`:
The operator Jacobi identity holds identically for any three linear operators `D_α, D_β, D_γ`. -/
theorem operator_jacobi {V : Type*} [AddCommGroup V] [Module ℝ V]
    (A B C : V →ₗ[ℝ] V) :
    opComm A (opComm B C) + opComm B (opComm C A) + opComm C (opComm A B) = 0 := by
  ext v
  simp only [opComm, LinearMap.add_apply, LinearMap.sub_apply, LinearMap.comp_apply,
    LinearMap.zero_apply, map_sub]
  abel

/-- `sgm_wavefn.tex` Proposition `prop:bimodule_associativity`:
For any differentiable real function `f` and differentiable quaternion component `ψ`,
the commutator `[∂_x, R_{f e₃}] ψ` equals `(∂_x f) ψ e₃` (generating `E` and `B` when `f` varies). -/
theorem deriv_right_fe3_commutator
    (f : ℝ → ℝ) (ψ : ℝ → Quat) (x df : ℝ) (dψ : Quat)
    (hf : HasDerivAt f df x)
    (hre : HasDerivAt (fun s => (ψ s).re) dψ.re x)
    (hi : HasDerivAt (fun s => (ψ s).imI) dψ.imI x)
    (hj : HasDerivAt (fun s => (ψ s).imJ) dψ.imJ x)
    (hk : HasDerivAt (fun s => (ψ s).imK) dψ.imK x) :
    let prod := fun s => f s • (ψ s * e3)
    let commVal := df • (ψ x * e3)
    HasDerivAt (fun s => (prod s).re) ((f x • (dψ * e3) + commVal).re) x ∧
    HasDerivAt (fun s => (prod s).imI) ((f x • (dψ * e3) + commVal).imI) x ∧
    HasDerivAt (fun s => (prod s).imJ) ((f x • (dψ * e3) + commVal).imJ) x ∧
    HasDerivAt (fun s => (prod s).imK) ((f x • (dψ * e3) + commVal).imK) x := by
  intro prod commVal
  refine ⟨?_, ?_, ?_, ?_⟩
  · have h := hf.mul (hk.const_mul (-1 : ℝ))
    convert h using 1
    · ext s; simp [prod, e3]
    · simp [commVal, e3]; ring
  · have h := hf.mul hj
    convert h using 1
    · ext s; simp [prod, e3]
    · simp [commVal, e3]; ring
  · have h := hf.mul (hi.const_mul (-1 : ℝ))
    convert h using 1
    · ext s; simp [prod, e3]
    · simp [commVal, e3]; ring
  · have h := hf.mul hre
    convert h using 1
    · ext s; simp [prod, e3]
    · simp [commVal, e3]; ring

/-- Helper: left multiplication of a quaternion operator output by a constant unit `q`. -/
def leftMulOp (q : Quat) (T : Quat →ₗ[ℝ] Quat) : Quat →ₗ[ℝ] Quat where
  toFun ψ := q * T ψ
  map_add' x y := by simp [mul_add]
  map_smul' r x := by simp

/-- `sgm_wavefn.tex` Eqs. `eq:general_table` & `eq:conjugate_unit_table`:
When each component operator `N⁻_μ` commutes with left multiplication by constant units `e_k`,
the product `(N⁻₀ - e₁ N⁻₁ - e₂ N⁻₂ - e₃ N⁻₃) ∘ (N⁺₀ + e₁ N⁺₁ + e₂ N⁺₂ + e₃ N⁺₃)`
decomposes exactly over the 4×4 quaternion table into:
- Diagonal: `N⁻₀ N⁺₀ + N⁻₁ N⁺₁ + N⁻₂ N⁺₂ + N⁻₃ N⁺₃` (Klein--Gordon),
- Border: `e₁(N⁻₀ N⁺₁ - N⁻₁ N⁺₀) + e₂(N⁻₀ N⁺₂ - N⁻₂ N⁺₀) + e₃(N⁻₀ N⁺₃ - N⁻₃ N⁺₀)` (E & gravity),
- Interior: `e₁(N⁻₃ N⁺₂ - N⁻₂ N⁺₃) + e₂(N⁻₁ N⁺₃ - N⁻₃ N⁺₁) + e₃(N⁻₂ N⁺₁ - N⁻₁ N⁺₂)` (B & curvature). -/
theorem four_by_four_operator_table
    (Nm0 Nm1 Nm2 Nm3 Np0 Np1 Np2 Np3 : Quat →ₗ[ℝ] Quat)
    (h0 : ∀ q ψ : Quat, Nm0 (q * ψ) = q * Nm0 ψ)
    (h1 : ∀ q ψ : Quat, Nm1 (q * ψ) = q * Nm1 ψ)
    (h2 : ∀ q ψ : Quat, Nm2 (q * ψ) = q * Nm2 ψ)
    (h3 : ∀ q ψ : Quat, Nm3 (q * ψ) = q * Nm3 ψ) :
    let Lminus := Nm0 - leftMulOp e1 Nm1 - leftMulOp e2 Nm2 - leftMulOp e3 Nm3
    let Lplus  := Np0 + leftMulOp e1 Np1 + leftMulOp e2 Np2 + leftMulOp e3 Np3
    let diagPart :=
      Nm0.comp Np0 + Nm1.comp Np1 + Nm2.comp Np2 + Nm3.comp Np3
    let borderPart :=
      leftMulOp e1 (Nm0.comp Np1 - Nm1.comp Np0) +
      leftMulOp e2 (Nm0.comp Np2 - Nm2.comp Np0) +
      leftMulOp e3 (Nm0.comp Np3 - Nm3.comp Np0)
    let interiorPart :=
      leftMulOp e1 (Nm3.comp Np2 - Nm2.comp Np3) +
      leftMulOp e2 (Nm1.comp Np3 - Nm3.comp Np1) +
      leftMulOp e3 (Nm2.comp Np1 - Nm1.comp Np2)
    Lminus.comp Lplus = diagPart + borderPart + interiorPart := by
  intro Lminus Lplus diagPart borderPart interiorPart
  apply LinearMap.ext
  intro ψ
  ext <;>
    simp only [Lminus, Lplus, diagPart, borderPart, interiorPart, leftMulOp,
      LinearMap.comp_apply, LinearMap.add_apply, LinearMap.sub_apply,
      LinearMap.coe_mk, AddHom.coe_mk, map_add, h0, h1, h2, h3] <;>
    simp [e1, e2, e3] <;> ring

end SGM.Differential
