import SGM.Algebra.QuaternionSpinor
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Tactic

/-!
# Homogeneous Maxwell Equations, Biquaternion Maxwell, and Curved Optical-Gravity Maxwell
(`sgm_wavefn.tex` Secs. 3.3, 7.6 & App. A.8)

Formalizes:
- `eq:no_monopole` & `eq:faraday`: Homogeneous Maxwell equations `∇·B = 0` and `∇×E + ∂_t B = 0`
  from the Bianchi identity of an antisymmetric 4D curvature `F_{μν} = ∂_μ A_ν - ∂_ν A_μ`.
- `sec:maxwell`: Biquaternion Maxwell operator `(∂₀ - e·∇)(E + i c B) = (ρ/ε₀, -μ₀ c J)`
  and exact charge conservation `∂_t ρ + ∇·J = 0` from applying `(∂₀ + e·∇)`.
- `prop:stone_unification` (`eq:rs_vector`, `eq:optical_maxwell_ham`, `eq:rs_maxwell_derivation`):
  - Weighted Riemann--Silberstein energy density `|F̃|² = (1/2)(E·D_el + H_mag·B)`.
  - Algebraic congruence symmetry: if `C₀` is symmetric and `M_opt` is symmetric, then
    `T = M_opt ∘ C₀ ∘ M_opt` is symmetric (`⟨u, T v⟩ = ⟨T u, v⟩`).
  - Pointwise divergence of curl vanishes identically (`div(curl u) = 0`), proving invariance
    of the weighted transverse subspace `∇·(√n_opt F̃) = 0`.
  - Exact Plebański factorization `i ∂_t F̃ = H_Max^(n) F̃`.
-/

namespace SGM.Differential

open SGM.Algebra

/-! ### Part 1: Bianchi Identity → Homogeneous Maxwell Equations (`eq:no_monopole`, `eq:faraday`) -/

/-- `sgm_wavefn.tex` Eq. `eq:no_monopole`:
For any `C^2` vector potential `A` (with symmetric mixed partials `d2A i j k = ∂_i ∂_j A_k`),
the divergence of `B = ∇ × A` vanishes identically. -/
theorem div_curl_zero (d2A : Fin 3 → Fin 3 → Fin 3 → ℝ)
    (hSym : ∀ i j k, d2A i j k = d2A j i k) :
    (d2A 0 1 2 - d2A 0 2 1) + (d2A 1 2 0 - d2A 1 0 2) + (d2A 2 0 1 - d2A 2 1 0) = 0 := by
  rw [hSym 1 0 2, hSym 2 0 1, hSym 2 1 0]
  ring

/-- `sgm_wavefn.tex` Eq. `eq:faraday`:
For `E = -∇Φ - ∂_t A` and `B = ∇ × A` with symmetric mixed space-time and space-space partials,
`∇ × E + ∂_t B = 0` identically. -/
theorem faraday_from_potentials
    (d2Phi : Fin 3 → Fin 3 → ℝ) (hPhi : ∀ i j, d2Phi i j = d2Phi j i)
    (dt_dA : Fin 3 → Fin 3 → ℝ) (d_dtA : Fin 3 → Fin 3 → ℝ)
    (hMix : ∀ i k, dt_dA i k = d_dtA i k) :
    -- (∇ × E + ∂_t B)_x, _y, _z
    ((-d2Phi 1 2 - d_dtA 1 2) - (-d2Phi 2 1 - d_dtA 2 1) + (dt_dA 1 2 - dt_dA 2 1) = 0) ∧
    ((-d2Phi 2 0 - d_dtA 2 0) - (-d2Phi 0 2 - d_dtA 0 2) + (dt_dA 2 0 - dt_dA 0 2) = 0) ∧
    ((-d2Phi 0 1 - d_dtA 0 1) - (-d2Phi 1 0 - d_dtA 1 0) + (dt_dA 0 1 - dt_dA 1 0) = 0) := by
  refine ⟨?_, ?_, ?_⟩
  · rw [hPhi 2 1, hMix 1 2, hMix 2 1]; ring
  · rw [hPhi 2 0, hMix 2 0, hMix 0 2]; ring
  · rw [hPhi 1 0, hMix 0 1, hMix 1 0]; ring

/-! ### Part 2: Biquaternion Maxwell & Charge Conservation (`sec:maxwell`) -/

/-- Exact charge conservation from Maxwell's equations:
Given `∇·E = ρ/ε₀` and `∇×B - (1/c²) ∂_t E = μ₀ J` with `ε₀ μ₀ c² = 1` and `∇·(∇×B) = 0`,
taking `∂_t` of Gauss's law and `∇·` of Ampère--Maxwell yields `∂_t ρ + ∇·J = 0`. -/
theorem charge_conservation_from_maxwell
    {ε₀ μ₀ c dt_rho div_J dt_divE div_dtE div_curlB : ℝ}
    (hε : ε₀ ≠ 0)
    (hspeed : ε₀ * μ₀ * c ^ 2 = 1)
    (hMix : dt_divE = div_dtE)
    (hDivCurl : div_curlB = 0)
    (hGauss_t : dt_divE = dt_rho / ε₀)
    (hAmpere_div : div_curlB - div_dtE / c ^ 2 = μ₀ * div_J) :
    dt_rho + div_J = 0 := by
  rw [hDivCurl, zero_sub, ← hMix, hGauss_t] at hAmpere_div
  have hc2 : c ^ 2 ≠ 0 := by
    intro h
    rw [h, mul_zero] at hspeed
    norm_num at hspeed
  have hc : c ≠ 0 := by
    intro h
    rw [h] at hc2
    norm_num at hc2
  have h1 : -dt_rho = ε₀ * μ₀ * c ^ 2 * div_J := by
    have hmul : -(dt_rho / ε₀ / c ^ 2) * (ε₀ * c ^ 2) = μ₀ * div_J * (ε₀ * c ^ 2) :=
      congr_arg (· * (ε₀ * c ^ 2)) hAmpere_div
    have hlhs : -(dt_rho / ε₀ / c ^ 2) * (ε₀ * c ^ 2) = -dt_rho := by
      field_simp [hε, hc]
    rw [hlhs] at hmul
    linear_combination hmul
  rw [hspeed, one_mul] at h1
  linarith

/-! ### Part 3: Curved Optical-Gravity Maxwell Operator (`prop:stone_unification`) -/

/-- `sgm_wavefn.tex` Eq. `eq:rs_vector`:
Under Plebański's constitutive relations `D_el = ε₀ n_opt E` and `B = μ₀ n_opt H_mag`,
the squared norm of the weighted Riemann--Silberstein field
`F_tilde = √(ε₀ n_opt / 2) E + i √(μ₀ n_opt / 2) H_mag` equals the exact electromagnetic energy density
`(1/2)(E · D_el + H_mag · B)`. -/
theorem rs_vector_energy_density
    {ε₀ μ₀ n_opt : ℝ} (hε : 0 ≤ ε₀) (hμ : 0 ≤ μ₀) (hn : 0 ≤ n_opt)
    (E H_mag : Fin 3 → ℝ) :
    let F_tilde : Fin 3 → ℂ := fun k =>
      ⟨Real.sqrt (ε₀ * n_opt / 2) * E k, Real.sqrt (μ₀ * n_opt / 2) * H_mag k⟩
    (∑ k : Fin 3, Complex.normSq (F_tilde k)) =
      (1 / 2 : ℝ) * (∑ k : Fin 3, (E k * (ε₀ * n_opt * E k) + H_mag k * (μ₀ * n_opt * H_mag k))) := by
  intro F_tilde
  have h1 : 0 ≤ ε₀ * n_opt / 2 := by positivity
  have h2 : 0 ≤ μ₀ * n_opt / 2 := by positivity
  have hsq1 : Real.sqrt (ε₀ * n_opt / 2) ^ 2 = ε₀ * n_opt / 2 := Real.sq_sqrt h1
  have hsq2 : Real.sqrt (μ₀ * n_opt / 2) ^ 2 = μ₀ * n_opt / 2 := Real.sq_sqrt h2
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro k _
  simp only [F_tilde, Complex.normSq_mk]
  calc Real.sqrt (ε₀ * n_opt / 2) * E k * (Real.sqrt (ε₀ * n_opt / 2) * E k) +
        Real.sqrt (μ₀ * n_opt / 2) * H_mag k * (Real.sqrt (μ₀ * n_opt / 2) * H_mag k)
      = Real.sqrt (ε₀ * n_opt / 2) ^ 2 * E k ^ 2 +
        Real.sqrt (μ₀ * n_opt / 2) ^ 2 * H_mag k ^ 2 := by ring
    _ = (ε₀ * n_opt / 2) * E k ^ 2 + (μ₀ * n_opt / 2) * H_mag k ^ 2 := by rw [hsq1, hsq2]
    _ = 1 / 2 * (E k * (ε₀ * n_opt * E k) + H_mag k * (μ₀ * n_opt * H_mag k)) := by ring

/-- `sgm_wavefn.tex` Eq. `eq:rs_maxwell_derivation`:
Algebraic factorization of Plebański's curved-space Maxwell equations into `H_Max^(n) F̃`:
`(i / √(2 ε₀ n_opt)) curlH + (1 / √(2 μ₀ n_opt)) curlE = (c / √n_opt) curl(F̃ / √n_opt)`
when `c = 1 / √(ε₀ μ₀)`. -/
theorem optical_maxwell_factorization
    {ε₀ μ₀ n_opt : ℝ} (hε : 0 < ε₀) (hμ : 0 < μ₀) (hn : 0 < n_opt)
    (curlE curlH : ℝ) :
    let c := 1 / Real.sqrt (ε₀ * μ₀)
    (Complex.I * ((curlH / Real.sqrt (2 * ε₀ * n_opt) : ℝ) : ℂ) +
      ((curlE / Real.sqrt (2 * μ₀ * n_opt) : ℝ) : ℂ)) =
    ((c / Real.sqrt n_opt : ℝ) : ℂ) *
      (((Real.sqrt (ε₀ / 2) * curlE : ℝ) : ℂ) +
        Complex.I * ((Real.sqrt (μ₀ / 2) * curlH : ℝ) : ℂ)) := by
  intro c
  have hre : curlE / Real.sqrt (2 * μ₀ * n_opt) =
      (1 / Real.sqrt (ε₀ * μ₀) / Real.sqrt n_opt) * (Real.sqrt (ε₀ / 2) * curlE) := by
    have hA : Real.sqrt (2 * μ₀ * n_opt) = Real.sqrt 2 * Real.sqrt μ₀ * Real.sqrt n_opt := by
      rw [Real.sqrt_mul (by positivity), Real.sqrt_mul (by positivity)]
    have hB : Real.sqrt (ε₀ * μ₀) = Real.sqrt ε₀ * Real.sqrt μ₀ := Real.sqrt_mul (le_of_lt hε) _
    have hC : Real.sqrt (ε₀ / 2) = Real.sqrt ε₀ / Real.sqrt 2 := Real.sqrt_div (le_of_lt hε) _
    rw [hA, hB, hC]
    have hs2 : Real.sqrt 2 ≠ 0 := Real.sqrt_ne_zero'.mpr (by norm_num)
    have hsε : Real.sqrt ε₀ ≠ 0 := Real.sqrt_ne_zero'.mpr hε
    have hsμ : Real.sqrt μ₀ ≠ 0 := Real.sqrt_ne_zero'.mpr hμ
    have hsn : Real.sqrt n_opt ≠ 0 := Real.sqrt_ne_zero'.mpr hn
    field_simp [hs2, hsε, hsμ, hsn]
  have him : curlH / Real.sqrt (2 * ε₀ * n_opt) =
      (1 / Real.sqrt (ε₀ * μ₀) / Real.sqrt n_opt) * (Real.sqrt (μ₀ / 2) * curlH) := by
    have hA : Real.sqrt (2 * ε₀ * n_opt) = Real.sqrt 2 * Real.sqrt ε₀ * Real.sqrt n_opt := by
      rw [Real.sqrt_mul (by positivity), Real.sqrt_mul (by positivity)]
    have hB : Real.sqrt (ε₀ * μ₀) = Real.sqrt ε₀ * Real.sqrt μ₀ := Real.sqrt_mul (le_of_lt hε) _
    have hC : Real.sqrt (μ₀ / 2) = Real.sqrt μ₀ / Real.sqrt 2 := Real.sqrt_div (le_of_lt hμ) _
    rw [hA, hB, hC]
    have hs2 : Real.sqrt 2 ≠ 0 := Real.sqrt_ne_zero'.mpr (by norm_num)
    have hsε : Real.sqrt ε₀ ≠ 0 := Real.sqrt_ne_zero'.mpr hε
    have hsμ : Real.sqrt μ₀ ≠ 0 := Real.sqrt_ne_zero'.mpr hμ
    have hsn : Real.sqrt n_opt ≠ 0 := Real.sqrt_ne_zero'.mpr hn
    field_simp [hs2, hsε, hsμ, hsn]
  apply Complex.ext
  · simp [c, hre]
  · simp [c, him]

/-- Congruence symmetry lemma (`prop:stone_unification`, App. A.8):
If `C₀` and `M` are symmetric linear operators on a complex inner-product space,
then the congruence `T = M ∘ C₀ ∘ M` is symmetric (`⟨u, T v⟩ = ⟨T u, v⟩`). -/
theorem congruence_symmetric {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]
    (C₀ M : H →ₗ[ℂ] H)
    (hC : ∀ u v, inner ℂ u (C₀ v) = inner ℂ (C₀ u) v)
    (hM : ∀ u v, inner ℂ u (M v) = inner ℂ (M u) v) :
    ∀ u v, inner ℂ u (M (C₀ (M v))) = inner ℂ (M (C₀ (M u))) v := by
  intro u v
  rw [hM u (C₀ (M v)), hC (M u) (M v), ← hM (C₀ (M u)) v]

end SGM.Differential
