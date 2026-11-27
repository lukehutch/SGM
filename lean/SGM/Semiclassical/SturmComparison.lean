import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Tactic

/-!
# Prüfer Angle ODE, Sturm Comparison Inequality, Riccati Tail Band Invariance,
# and Center-Point Wronskian Phase Criterion
(`sgm_wavefn.tex` App. B: Proposition 5(a), Lemma 5 `lem:sturm`, Lemma 7 Step (5))

Formalizes:
- **Lemma 5 (`lem:sturm`) Prüfer angle ODE**:
  From `F = r_P sin θ`, `ℏ g F' = r_P cos θ` (`r_P > 0`) solving `(ℏ g F')' = (k_S / ℏ) F`,
  derives `θ' = cos² θ / (ℏ g) - (k_S / ℏ) sin² θ` globally (zero division by `sin θ` or `cos θ`!).
- **Lemma 5 (`lem:sturm`) strict comparison inequality**:
  Under (S1) `0 < g(E') ≤ g(E)` and `k_S(E') < k_S(E)`, proves
  `b_S ≥ ((k_S - k_S') / ℏ) sin² θ ≥ 0`, strict positivity whenever `sin θ ≠ 0`, and
  `θ' = 1 / (ℏ g) > 0` at every zero of `sin θ` (isolation of zeros).
- **Proposition 5(a) Riccati tail-band invariance**:
  At `z = ±b_T` (`b_T = ℏ C_T ≤ (1/2) √q`, `|(√q)'| ≤ (1/2) C_T √q`), proves
  `ℏ z'|_{z=+b_T} ≥ b_T √q > 0` and `ℏ z'|_{z=-b_T} ≤ -b_T √q < 0` (backward trapping!).
- **Lemma 7 Step (5) Wronskian phase criterion**:
  `W = u_- u_+' - u_+ u_-' = (a_m / ℏ) r_- r_+ sin(ϑ_- - ϑ_+)`, so `W = 0 ↔ ∃ k : ℤ, ϑ_- - ϑ_+ = k π`.
-/

namespace SGM.Semiclassical

/-! ### Part 1: Lemma 5 (`lem:sturm`) — Prüfer Angle ODE & Comparison Source Positivity -/

/-- **Lemma 5 (`lem:sturm`), Prüfer angle ODE**:
If `F(x) = r(x) sin(θ(x))` and `G(x) = r(x) cos(θ(x))` (`r(x) > 0`) satisfy the first-order
Sturm--Liouville system `F'(x) = G(x) / (ℏ g)` and `G'(x) = (k_S / ℏ) F(x)` at `x`,
then `θ'(x) = cos²(θ(x)) / (ℏ g) - (k_S / ℏ) sin²(θ(x))`. -/
theorem prufer_angle_ode
    {r θ : ℝ → ℝ} {x r' θ' ℏ g k_S : ℝ}
    (hr_pos : 0 < r x)
    (hr : HasDerivAt r r' x)
    (hθ : HasDerivAt θ θ' x)
    (hF_ode : HasDerivAt (fun t => r t * Real.sin (θ t)) ((r x * Real.cos (θ x)) / (ℏ * g)) x)
    (hG_ode : HasDerivAt (fun t => r t * Real.cos (θ t)) ((k_S / ℏ) * (r x * Real.sin (θ x))) x) :
    θ' = Real.cos (θ x) ^ 2 / (ℏ * g) - (k_S / ℏ) * Real.sin (θ x) ^ 2 := by
  have hsin : HasDerivAt (fun t => Real.sin (θ t)) (Real.cos (θ x) * θ') x :=
    (Real.hasDerivAt_sin (θ x)).comp x hθ
  have hcos : HasDerivAt (fun t => Real.cos (θ t)) (-(Real.sin (θ x)) * θ') x :=
    (Real.hasDerivAt_cos (θ x)).comp x hθ
  have hF_prod : HasDerivAt (fun t => r t * Real.sin (θ t))
      (r' * Real.sin (θ x) + r x * (Real.cos (θ x) * θ')) x := hr.mul hsin
  have hG_prod : HasDerivAt (fun t => r t * Real.cos (θ t))
      (r' * Real.cos (θ x) + r x * (-(Real.sin (θ x)) * θ')) x := hr.mul hcos
  have hF_eq : r' * Real.sin (θ x) + r x * (Real.cos (θ x) * θ') =
      (r x * Real.cos (θ x)) / (ℏ * g) := hF_prod.unique hF_ode
  have hG_eq : r' * Real.cos (θ x) + r x * (-(Real.sin (θ x)) * θ') =
      (k_S / ℏ) * (r x * Real.sin (θ x)) := hG_prod.unique hG_ode
  -- Eliminate `r'` via `cos θ * hF_eq - sin θ * hG_eq`:
  have helim : r x * (Real.cos (θ x) ^ 2 + Real.sin (θ x) ^ 2) * θ' =
      r x * (Real.cos (θ x) ^ 2 / (ℏ * g) - (k_S / ℏ) * Real.sin (θ x) ^ 2) := by
    have h1 := congr_arg (Real.cos (θ x) * ·) hF_eq
    have h2 := congr_arg (Real.sin (θ x) * ·) hG_eq
    calc r x * (Real.cos (θ x) ^ 2 + Real.sin (θ x) ^ 2) * θ'
        = Real.cos (θ x) * (r' * Real.sin (θ x) + r x * (Real.cos (θ x) * θ')) -
          Real.sin (θ x) * (r' * Real.cos (θ x) + r x * (-Real.sin (θ x) * θ')) := by ring
      _ = Real.cos (θ x) * ((r x * Real.cos (θ x)) / (ℏ * g)) -
          Real.sin (θ x) * ((k_S / ℏ) * (r x * Real.sin (θ x))) := by rw [h1, h2]
      _ = r x * (Real.cos (θ x) ^ 2 / (ℏ * g) - (k_S / ℏ) * Real.sin (θ x) ^ 2) := by ring
  rw [Real.cos_sq_add_sin_sq, mul_one] at helim
  exact mul_left_cancel₀ (ne_of_gt hr_pos) helim

/-- **Lemma 5 (`lem:sturm`), strict comparison source inequality**:
Under (S1) `0 < g' ≤ g` and `k_S' < k_S` with `ℏ > 0`:
1. `b_S ≥ ((k_S - k_S') / ℏ) sin² θ ≥ 0`;
2. `sin θ ≠ 0 → 0 < b_S`;
3. `sin θ = 0 → cos² θ / (ℏ g) - (k_S / ℏ) sin² θ = 1 / (ℏ g) > 0` (zeros of `sin θ` are isolated). -/
theorem sturm_comparison_source_positivity
    {ℏ g g' k_S k_S' θ : ℝ}
    (hℏ : 0 < ℏ) (hg' : 0 < g') (hgg : g' ≤ g) (hk : k_S' < k_S) :
    let b_S := (Real.cos θ ^ 2 / (ℏ * g') - (k_S' / ℏ) * Real.sin θ ^ 2) -
               (Real.cos θ ^ 2 / (ℏ * g) - (k_S / ℏ) * Real.sin θ ^ 2)
    (((k_S - k_S') / ℏ) * Real.sin θ ^ 2 ≤ b_S) ∧
    (0 ≤ b_S) ∧
    (Real.sin θ ≠ 0 → 0 < b_S) ∧
    (Real.sin θ = 0 → Real.cos θ ^ 2 / (ℏ * g) - (k_S / ℏ) * Real.sin θ ^ 2 = 1 / (ℏ * g) ∧
      0 < 1 / (ℏ * g)) := by
  intro b_S
  have hg : 0 < g := lt_of_lt_of_le hg' hgg
  have hinv : 1 / (ℏ * g) ≤ 1 / (ℏ * g') := by
    apply one_div_le_one_div_of_le (by positivity)
    exact mul_le_mul_of_nonneg_left hgg (le_of_lt hℏ)
  have hcos_term : 0 ≤ Real.cos θ ^ 2 / (ℏ * g') - Real.cos θ ^ 2 / (ℏ * g) := by
    have : Real.cos θ ^ 2 / (ℏ * g') - Real.cos θ ^ 2 / (ℏ * g) =
        Real.cos θ ^ 2 * (1 / (ℏ * g') - 1 / (ℏ * g)) := by ring
    rw [this]
    exact mul_nonneg (sq_nonneg _) (by linarith)
  have hb_split : b_S = (Real.cos θ ^ 2 / (ℏ * g') - Real.cos θ ^ 2 / (ℏ * g)) +
      ((k_S - k_S') / ℏ) * Real.sin θ ^ 2 := by
    simp only [b_S]; ring
  have hsin_nonneg : 0 ≤ ((k_S - k_S') / ℏ) * Real.sin θ ^ 2 := by
    have : 0 < (k_S - k_S') / ℏ := div_pos (by linarith) hℏ
    exact mul_nonneg (le_of_lt this) (sq_nonneg _)
  refine ⟨by linarith, by linarith, ?_, ?_⟩
  · intro hsin_ne
    have hsin2 : 0 < Real.sin θ ^ 2 := sq_pos_of_ne_zero hsin_ne
    have hcoeff : 0 < (k_S - k_S') / ℏ := div_pos (by linarith) hℏ
    have hpos : 0 < ((k_S - k_S') / ℏ) * Real.sin θ ^ 2 := mul_pos hcoeff hsin2
    linarith
  · intro hsin0
    have hcos2 : Real.cos θ ^ 2 = 1 := by
      have := Real.sin_sq_add_cos_sq θ
      rw [hsin0] at this
      linarith
    rw [hcos2, hsin0]
    refine ⟨by ring, by positivity⟩

/-! ### Part 2: Proposition 5(a) — Riccati Tail-Band Invariance -/

/-- **Proposition 5(a) (`prop:matching`, Riccati tail-band invariance)**:
For the logarithmic-derivative Riccati remainder `ℏ z' = z (2 √q - z) + ℏ s_deriv` on the
forbidden tail `q > 0` where `|s_deriv| ≤ (1/2) C_T √q` and `b_T = ℏ C_T ≤ (1/2) √q`:
1. At the upper boundary `z = +b_T`, `ℏ z' ≥ b_T √q > 0`;
2. At the lower boundary `z = -b_T`, `ℏ z' ≤ -b_T √q < 0`!
Hence following solutions backward from `+∞` traps `|z| ≤ b_T` identically. -/
theorem riccati_tail_band_invariance
    {q ℏ C_T s_deriv : ℝ} (hq : 0 < q) (hℏ : 0 < ℏ) (hCT : 0 < C_T)
    (hs : |s_deriv| ≤ (1 / 2 : ℝ) * C_T * Real.sqrt q)
    (hb_small : ℏ * C_T ≤ (1 / 2 : ℝ) * Real.sqrt q) :
    let b_T := ℏ * C_T
    let sq := Real.sqrt q
    (b_T * sq ≤ b_T * (2 * sq - b_T) + ℏ * s_deriv ∧ 0 < b_T * sq) ∧
    ((-b_T) * (2 * sq - (-b_T)) + ℏ * s_deriv ≤ -(b_T * sq) ∧ -(b_T * sq) < 0) := by
  intro b_T sq
  have hsq : 0 < sq := Real.sqrt_pos.mpr hq
  have hb : 0 < b_T := mul_pos hℏ hCT
  have hprod : 0 < b_T * sq := mul_pos hb hsq
  have hs_lo : -((1 / 2 : ℝ) * C_T * sq) ≤ s_deriv := (abs_le.mp hs).1
  have hs_hi : s_deriv ≤ (1 / 2 : ℝ) * C_T * sq := (abs_le.mp hs).2
  have hℏs_lo : -((1 / 2 : ℝ) * b_T * sq) ≤ ℏ * s_deriv := by
    have := mul_le_mul_of_nonneg_left hs_lo (le_of_lt hℏ)
    dsimp [b_T]; linarith
  have hℏs_hi : ℏ * s_deriv ≤ (1 / 2 : ℝ) * b_T * sq := by
    have := mul_le_mul_of_nonneg_left hs_hi (le_of_lt hℏ)
    dsimp [b_T]; linarith
  refine ⟨⟨?_, hprod⟩, ⟨?_, by linarith⟩⟩
  · -- At `z = +b_T`: `b_T (2 sq - b_T) + ℏ s_deriv ≥ b_T (3/2 sq) - (1/2) b_T sq = b_T sq`
    have : b_T * ((3 / 2 : ℝ) * sq) ≤ b_T * (2 * sq - b_T) :=
      mul_le_mul_of_nonneg_left (by linarith) (le_of_lt hb)
    linarith
  · -- At `z = -b_T`: `(-b_T)(2 sq + b_T) + ℏ s_deriv ≤ -2 b_T sq + (1/2) b_T sq = -(3/2) b_T sq ≤ -b_T sq`
    have : 0 ≤ b_T * b_T := mul_nonneg (le_of_lt hb) (le_of_lt hb)
    linarith

/-! ### Part 3: Lemma 7 Step (5) — Wronskian Vanishing iff Angle Difference in `π ℤ` -/

/-- **Lemma 7 Step (5) (`lem:scalar_rule`)**:
At the center point `τ₀`, for `u_± = r_± sin ϑ_±` and `u_±' = (a_m / ℏ) r_± cos ϑ_±`
(`a_m, ℏ, r_-, r_+ > 0`), the Wronskian `W = u_- u_+' - u_+ u_-'` equals
`(a_m / ℏ) r_- r_+ sin(ϑ_- - ϑ_+)`, so `W = 0 ↔ ∃ k : ℤ, ϑ_- - ϑ_+ = k π`. -/
theorem wronskian_vanish_iff_angle_diff
    {a_m ℏ r_m r_p ϑ_m ϑ_p : ℝ}
    (ha : 0 < a_m) (hℏ : 0 < ℏ) (hrm : 0 < r_m) (hrp : 0 < r_p) :
    let u_m  := r_m * Real.sin ϑ_m
    let u_m' := (a_m / ℏ) * (r_m * Real.cos ϑ_m)
    let u_p  := r_p * Real.sin ϑ_p
    let u_p' := (a_m / ℏ) * (r_p * Real.cos ϑ_p)
    (u_m * u_p' - u_p * u_m' = (a_m / ℏ) * r_m * r_p * Real.sin (ϑ_m - ϑ_p)) ∧
    (u_m * u_p' - u_p * u_m' = 0 ↔ ∃ k : ℤ, ϑ_m - ϑ_p = (k : ℝ) * Real.pi) := by
  intro u_m u_m' u_p u_p'
  have hW : u_m * u_p' - u_p * u_m' = (a_m / ℏ) * r_m * r_p * Real.sin (ϑ_m - ϑ_p) := by
    simp only [u_m, u_m', u_p, u_p']
    rw [Real.sin_sub]
    ring
  refine ⟨hW, ?_⟩
  rw [hW]
  have hpref : (a_m / ℏ) * r_m * r_p ≠ 0 := by positivity
  constructor
  · intro h0
    have hsin : Real.sin (ϑ_m - ϑ_p) = 0 := (mul_eq_zero.mp h0).resolve_left hpref
    rcases Real.sin_eq_zero_iff.mp hsin with ⟨k, hk⟩
    exact ⟨k, hk.symm⟩
  · rintro ⟨k, hk⟩
    rw [hk, Real.sin_int_mul_pi, mul_zero]

end SGM.Semiclassical
