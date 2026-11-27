import Mathlib.Basic.Real.Basic
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.Tactic

/-!
# Phase-Space Derivations of the Force-Frequency Identity (`sgm.tex` Secs. 2–3)

Formalizes:
- Route I (`eq:sgm`, `eq:sgm_as_dot_p`): Planck--Einstein + Hamilton's equation.
- Route II (`eq:route2_components`): Exact cancellation of convective Hessian terms for any `C^2`
  phase field `θ` with symmetric spatial Hessian `∂_i ∂_j θ = ∂_j ∂_i θ`.
- Route III (`eq:legendre_chain`, `eq:legendre_bridge`): Exact Legendre-transform bridge
  `∂H/∂x_i|_p = -∂L/∂x_i|_v` under `p_j = ∂L/∂v_j`.
- Noether pair-momentum conservation (`sec:canonical`): Translation-invariant pair frequency
  `ω(x₁, x₂) = f(x₁ - x₂)` yields equal and opposite forces `F₁ + F₂ = 0`.
- Isotropic light dispersion (`eq:isotropic_dispersion`): Weak-field optical refractive index
  `n_r = W/α = (1+ε)/(1-ε) = 1 + 2ε + 2ε²/(1-ε)`, doubling the Newtonian deflection.
- Lorentz force algebra (`eq:lorentz`): `q(E + v × B)` does zero magnetic work `(v × B) · v = 0`.
-/

namespace SGM.Basic

/-- Route I (`sgm.tex` Eq. `eq:sgm_as_dot_p`):
From `H = ℏ ω` and Hamilton's equation `p_dot_i = -∂H/∂x_i|_p`, we obtain `p_dot_i = -ℏ ∂ω/∂x_i|_p`. -/
theorem route_I_force_frequency {ℏ : ℝ} {gradH gradOmega pDot : Fin 3 → ℝ}
    (hPlanck : ∀ i, gradH i = ℏ * gradOmega i)
    (hHamilton : ∀ i, pDot i = -gradH i) :
    ∀ i, pDot i = -ℏ * gradOmega i := by
  intro i
  rw [hHamilton i, hPlanck i]
  ring

/-- Route II (`sgm.tex` Eq. `eq:route2_components`):
On a `C^2` phase field with symmetric spatial Hessian `H_θ i j = H_θ j i`, the convective chain-rule
term and the material derivative term cancel identically at every velocity `v`, leaving
`p_dot i = -ℏ * gradOmega_p i`. -/
theorem route_II_convective_cancellation
    (ℏ : ℝ) (gradOmega_p v : Fin 3 → ℝ) (H_θ : Fin 3 → Fin 3 → ℝ)
    (hSym : ∀ i j, H_θ i j = H_θ j i) :
    ∀ i, (-ℏ * gradOmega_p i - ℏ * ∑ j : Fin 3, v j * H_θ i j +
          ℏ * ∑ j : Fin 3, v j * H_θ j i) = -ℏ * gradOmega_p i := by
  intro i
  have hsum : (∑ j : Fin 3, v j * H_θ i j) = ∑ j : Fin 3, v j * H_θ j i := by
    apply Finset.sum_congr rfl
    intro j _
    rw [hSym i j]
  rw [hsum]
  ring

/-- Route III (`sgm.tex` Eqs. `eq:legendre_chain` & `eq:legendre_bridge`):
Differentiating `H(x, p) = ∑_j p_j v_j(x, p) - L(x, v(x, p))` at fixed `p` using `p_j = ∂L/∂v_j`
cancels the implicit `∂v_j/∂x_i|_p` terms and gives `∂H/∂x_i|_p = -∂L/∂x_i|_v`. -/
theorem legendre_bridge
    (p dL_dv dL_dx_v : Fin 3 → ℝ) (dv_dx_p : Fin 3 → Fin 3 → ℝ)
    (hCan : ∀ j, p j = dL_dv j) :
    ∀ i, ((∑ j : Fin 3, p j * dv_dx_p j i) - dL_dx_v i -
          ∑ j : Fin 3, dL_dv j * dv_dx_p j i) = -dL_dx_v i := by
  intro i
  have hsum : (∑ j : Fin 3, p j * dv_dx_p j i) = ∑ j : Fin 3, dL_dv j * dv_dx_p j i := by
    apply Finset.sum_congr rfl
    intro j _
    rw [hCan j]
  rw [hsum]
  ring

/-- Weak-field isotropic gravitational refractive index (`sgm.tex` Sec. 3.5):
For `α = 1 - ε` and `W = 1 + ε` (`ε = GM/(rc^2) < 1`), the optical index `n_r = W/α`
equals `1 + 2ε + 2ε²/(1-ε)`, doubling the `1 + ε` Newtonian lapse contribution. -/
theorem isotropic_refractive_index_expansion {ε : ℝ} (hε : ε ≠ 1) :
    (1 + ε) / (1 - ε) = 1 + 2 * ε + 2 * ε ^ 2 / (1 - ε) := by
  have hne : 1 - ε ≠ 0 := sub_ne_zero.mpr (Ne.symm hε)
  field_simp [hne]
  ring

/-- Cross product on `ℝ³`. -/
def cross3 (u v : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![u 1 * v 2 - u 2 * v 1,
    u 2 * v 0 - u 0 * v 2,
    u 0 * v 1 - u 1 * v 0]

/-- Dot product on `ℝ³`. -/
def dot3 (u v : Fin 3 → ℝ) : ℝ :=
  u 0 * v 0 + u 1 * v 1 + u 2 * v 2

/-- Magnetic Lorentz force `q (v × B)` does identically zero mechanical work. -/
theorem magnetic_lorentz_zero_work (q : ℝ) (v B : Fin 3 → ℝ) :
    dot3 (fun i => q * cross3 v B i) v = 0 := by
  simp [dot3, cross3]
  ring

/-- Translation-invariant pair potential (`sgm.tex` Supp. S4):
If `∇_{x_2} ω = -∇_{x_1} ω`, then `F_1 + F_2 = 0` (Newton's third law from spectral gradients). -/
theorem spectral_third_law (ℏ : ℝ) (grad1 grad2 : Fin 3 → ℝ)
    (hTrans : ∀ i, grad2 i = -grad1 i) :
    ∀ i, (-ℏ * grad1 i) + (-ℏ * grad2 i) = 0 := by
  intro i
  rw [hTrans i]
  ring

end SGM.Basic
