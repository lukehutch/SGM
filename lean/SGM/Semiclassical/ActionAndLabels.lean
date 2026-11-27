import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Topology.Connected.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Arctan
import Mathlib.Tactic

/-!
# Semiclassical Action-to-Energy Transfer, Phase-to-Action Construction,
# Integer Labels, and Monotonicity Lemma
(`sgm_wavefn.tex` App. B: Lemmas 1, 4, 6, 8 — 100% self-contained, zero external axioms)

Formalizes:
- **Definition 1 (`def:two_way`) & Lemma 1 (`lem:levels_action`)**:
  Complete real-analysis proof via the Intermediate Value Theorem (`intermediate_value_Icc`)
  that `|I_hbar(E) - I₁(E) - π ℏ| ≤ C_I ℏ²` together with `T_min/2 ≤ ΔI₁/ΔE ≤ T_max + ℏ C₁`
  implies two-way `O(ℏ²)` spectral approximation with radius `δ₀ = 2 C_I ℏ² / T_min`,
  unique quantum label `n` when `2 δ₀ < 2π ℏ / (T_max + ℏ C₁)`, AND **one-to-one**
  approximation whenever `I_hbar` takes each value in `2π ℏ ℤ` at most once on `J`!
- **Lemma 4 (`lem:phase_action`)**:
  Constructs the unique continuous representative `δ_ϑ(E) ∈ (-π/2, π/2)` and continuous action
  `I_hbar = 2 Φ_hbar + π ℏ + 2 ℏ δ_ϑ` with `|I_hbar - 2 Φ_hbar - π ℏ| ≤ 4 C ℏ²`.
- **Lemma 6 (`lem:labels`(b) & (c))**:
  Proves part (b) WITHOUT global monotonicity of `Δ_θ` (constant integer shift `m_L` on the
  connected interval `[a, b]` transferring `I_hbar(E) = 2π ℏ k ↔ Δ_θ(E) = (k - m_L) π`),
  and part (c) (at-most-once crossing when `Δ_θ` is strictly increasing).
- **Lemma 8 (`lem:monotone`)**:
  Exact quadratic-completion lower bound `b_mon + a_mon ρ² - s ℏ (a_mon' ρ + a_mon ρ') ≥ b_* / 2`
  for all `ρ ∈ ℝ` whenever `ℏ ≤ min(b_* / (4 M_mon), √(a_* b_*) / M_mon)`.
-/

namespace SGM.Semiclassical

/-! ### Part 1: Lemma 1 (`lem:levels_action`) — Action Error to Two-Way & One-to-One Energy Bounds -/

/-- **Lemma 1 (`lem:levels_action`), Step (1) (Forward existence via IVT)**:
If `I_hbar` is continuous on `[a, b]`, `|I_hbar E - I₁ E - π ℏ| ≤ C_I ℏ²`, `I₁` grows at rate
at least `T_min / 2`, and `E₀ ∈ [a + δ₀, b - δ₀]` solves `I₁(E₀) = 2π ℏ (n + 1/2)`
where `δ₀ = 2 C_I ℏ² / T_min`, then there exists `E_star ∈ [E₀ - δ₀, E₀ + δ₀]` with
`I_hbar(E_star) = 2π ℏ (n + 1)`. -/
theorem lem_levels_action_forward
    {a b ℏ C_I T_min : ℝ} (hℏ : 0 < ℏ) (hC : 0 ≤ C_I) (hT : 0 < T_min)
    (I₁ I_hbar : ℝ → ℝ)
    (hCont : ContinuousOn I_hbar (Set.Icc a b))
    (hApprox : ∀ E ∈ Set.Icc a b, |I_hbar E - I₁ E - Real.pi * ℏ| ≤ C_I * ℏ ^ 2)
    (hRate : ∀ E₁ ∈ Set.Icc a b, ∀ E₂ ∈ Set.Icc a b, E₁ ≤ E₂ →
      (T_min / 2) * (E₂ - E₁) ≤ I₁ E₂ - I₁ E₁)
    (n : ℤ) (E₀ : ℝ)
    (hE₀_rule : I₁ E₀ = 2 * Real.pi * ℏ * ((n : ℝ) + 1 / 2))
    (hL : a ≤ E₀ - 2 * C_I * ℏ ^ 2 / T_min)
    (hR : E₀ + 2 * C_I * ℏ ^ 2 / T_min ≤ b) :
    let δ₀ := 2 * C_I * ℏ ^ 2 / T_min
    ∃ E_star ∈ Set.Icc (E₀ - δ₀) (E₀ + δ₀),
      I_hbar E_star = 2 * Real.pi * ℏ * ((n : ℝ) + 1) := by
  intro δ₀
  have hδ₀ : 0 ≤ δ₀ := by dsimp [δ₀]; positivity
  have hsub : Set.Icc (E₀ - δ₀) (E₀ + δ₀) ⊆ Set.Icc a b :=
    Set.Icc_subset_Icc hL hR
  have hE₀_mem : E₀ ∈ Set.Icc a b := hsub ⟨by linarith, by linarith⟩
  have hLo_mem : E₀ - δ₀ ∈ Set.Icc a b := hsub ⟨le_rfl, by linarith⟩
  have hHi_mem : E₀ + δ₀ ∈ Set.Icc a b := hsub ⟨by linarith, le_rfl⟩
  -- Check signs of `I_hbar(E₀ ± δ₀) - 2π ℏ (n + 1)`:
  have hrate_lo := hRate (E₀ - δ₀) hLo_mem E₀ hE₀_mem (by linarith)
  have hrate_hi := hRate E₀ hE₀_mem (E₀ + δ₀) hHi_mem (by linarith)
  have hTδ : (T_min / 2) * δ₀ = C_I * ℏ ^ 2 := by
    dsimp [δ₀]; field_simp [ne_of_gt hT]
  have hAppLo := abs_le.mp (hApprox (E₀ - δ₀) hLo_mem)
  have hAppHi := abs_le.mp (hApprox (E₀ + δ₀) hHi_mem)
  have hval_lo : I_hbar (E₀ - δ₀) ≤ 2 * Real.pi * ℏ * ((n : ℝ) + 1) := by
    have h1 : I₁ (E₀ - δ₀) ≤ I₁ E₀ - C_I * ℏ ^ 2 := by linarith [hrate_lo, hTδ]
    linarith [hAppLo.2, hE₀_rule]
  have hval_hi : 2 * Real.pi * ℏ * ((n : ℝ) + 1) ≤ I_hbar (E₀ + δ₀) := by
    have h1 : I₁ E₀ + C_I * ℏ ^ 2 ≤ I₁ (E₀ + δ₀) := by linarith [hrate_hi, hTδ]
    linarith [hAppHi.1, hE₀_rule]
  have hcont_sub : ContinuousOn I_hbar (Set.Icc (E₀ - δ₀) (E₀ + δ₀)) := hCont.mono hsub
  have hivt := intermediate_value_Icc (show E₀ - δ₀ ≤ E₀ + δ₀ by linarith) hcont_sub
  exact hivt ⟨hval_lo, hval_hi⟩

/-- **Lemma 1 (`lem:levels_action`), Step (2) (Backward existence via IVT)**:
If `I₁` is continuous and grows at rate at least `T_min / 2`, and `E' ∈ [a + δ₀, b - δ₀]`
satisfies `I_hbar(E') = 2π ℏ (n + 1)`, then there exists `E₀ ∈ [E' - δ₀, E' + δ₀]` with
`I₁(E₀) = 2π ℏ (n + 1/2)`. -/
theorem lem_levels_action_backward
    {a b ℏ C_I T_min : ℝ} (hℏ : 0 < ℏ) (hC : 0 ≤ C_I) (hT : 0 < T_min)
    (I₁ I_hbar : ℝ → ℝ)
    (hCont₁ : ContinuousOn I₁ (Set.Icc a b))
    (hApprox : ∀ E ∈ Set.Icc a b, |I_hbar E - I₁ E - Real.pi * ℏ| ≤ C_I * ℏ ^ 2)
    (hRate : ∀ E₁ ∈ Set.Icc a b, ∀ E₂ ∈ Set.Icc a b, E₁ ≤ E₂ →
      (T_min / 2) * (E₂ - E₁) ≤ I₁ E₂ - I₁ E₁)
    (n : ℤ) (E' : ℝ)
    (hE'_quant : I_hbar E' = 2 * Real.pi * ℏ * ((n : ℝ) + 1))
    (hL : a ≤ E' - 2 * C_I * ℏ ^ 2 / T_min)
    (hR : E' + 2 * C_I * ℏ ^ 2 / T_min ≤ b) :
    let δ₀ := 2 * C_I * ℏ ^ 2 / T_min
    ∃ E₀ ∈ Set.Icc (E' - δ₀) (E' + δ₀),
      I₁ E₀ = 2 * Real.pi * ℏ * ((n : ℝ) + 1 / 2) := by
  intro δ₀
  have hδ₀ : 0 ≤ δ₀ := by dsimp [δ₀]; positivity
  have hsub : Set.Icc (E' - δ₀) (E' + δ₀) ⊆ Set.Icc a b := Set.Icc_subset_Icc hL hR
  have hE'_mem : E' ∈ Set.Icc a b := hsub ⟨by linarith, by linarith⟩
  have hLo_mem : E' - δ₀ ∈ Set.Icc a b := hsub ⟨le_rfl, by linarith⟩
  have hHi_mem : E' + δ₀ ∈ Set.Icc a b := hsub ⟨by linarith, le_rfl⟩
  have hTδ : (T_min / 2) * δ₀ = C_I * ℏ ^ 2 := by
    dsimp [δ₀]; field_simp [ne_of_gt hT]
  have hrate_lo := hRate (E' - δ₀) hLo_mem E' hE'_mem (by linarith)
  have hrate_hi := hRate E' hE'_mem (E' + δ₀) hHi_mem (by linarith)
  have hApp := abs_le.mp (hApprox E' hE'_mem)
  have hval_lo : I₁ (E' - δ₀) ≤ 2 * Real.pi * ℏ * ((n : ℝ) + 1 / 2) := by
    linarith [hApp.1, hE'_quant, hrate_lo, hTδ]
  have hval_hi : 2 * Real.pi * ℏ * ((n : ℝ) + 1 / 2) ≤ I₁ (E' + δ₀) := by
    linarith [hApp.2, hE'_quant, hrate_hi, hTδ]
  exact intermediate_value_Icc (show E' - δ₀ ≤ E' + δ₀ by linarith) (hCont₁.mono hsub)
    ⟨hval_lo, hval_hi⟩

/-- **Lemma 1 (`lem:levels_action`), Steps (3) & (4) (Separation & One-to-One Transfer)**:
If `I₁` has upper slope `≤ T_up` (`T_up > 0`), `2 δ₀ < 2π ℏ / T_up`, and `I_hbar` takes each
value `2π ℏ k` (`k : ℤ`) at most once on `[a, b]`, then for every `E₀ ∈ [a + δ₀, b - δ₀]`
solving `I₁(E₀) = 2π ℏ (n + 1/2)`, there is a **UNIQUE** `E_star ∈ [E₀ - δ₀, E₀ + δ₀]`
with `I_hbar(E_star) ∈ 2π ℏ ℤ`! -/
theorem lem_levels_action_one_to_one
    {a b ℏ C_I T_min T_up : ℝ} (hℏ : 0 < ℏ) (hC : 0 ≤ C_I) (hT : 0 < T_min) (hTup : 0 < T_up)
    (I₁ I_hbar : ℝ → ℝ)
    (hCont : ContinuousOn I_hbar (Set.Icc a b))
    (hApprox : ∀ E ∈ Set.Icc a b, |I_hbar E - I₁ E - Real.pi * ℏ| ≤ C_I * ℏ ^ 2)
    (hRateLo : ∀ E₁ ∈ Set.Icc a b, ∀ E₂ ∈ Set.Icc a b, E₁ ≤ E₂ →
      (T_min / 2) * (E₂ - E₁) ≤ I₁ E₂ - I₁ E₁)
    (hRateHi : ∀ E₁ ∈ Set.Icc a b, ∀ E₂ ∈ Set.Icc a b, E₁ ≤ E₂ →
      I₁ E₂ - I₁ E₁ ≤ T_up * (E₂ - E₁))
    (hOnce : ∀ k : ℤ, ∀ E₁ ∈ Set.Icc a b, ∀ E₂ ∈ Set.Icc a b,
      I_hbar E₁ = 2 * Real.pi * ℏ * (k : ℝ) →
      I_hbar E₂ = 2 * Real.pi * ℏ * (k : ℝ) → E₁ = E₂)
    (hSep : 2 * (2 * C_I * ℏ ^ 2 / T_min) < 2 * Real.pi * ℏ / T_up)
    (n : ℤ) (E₀ : ℝ)
    (hE₀_rule : I₁ E₀ = 2 * Real.pi * ℏ * ((n : ℝ) + 1 / 2))
    (hL : a ≤ E₀ - 2 * C_I * ℏ ^ 2 / T_min)
    (hR : E₀ + 2 * C_I * ℏ ^ 2 / T_min ≤ b) :
    let δ₀ := 2 * C_I * ℏ ^ 2 / T_min
    ∃! E_star : ℝ, E_star ∈ Set.Icc (E₀ - δ₀) (E₀ + δ₀) ∧
      ∃ k : ℤ, I_hbar E_star = 2 * Real.pi * ℏ * (k : ℝ) := by
  intro δ₀
  rcases lem_levels_action_forward hℏ hC hT I₁ I_hbar hCont hApprox hRateLo n E₀ hE₀_rule hL hR
    with ⟨E_star, hE_mem, hE_eq⟩
  have hsub : Set.Icc (E₀ - δ₀) (E₀ + δ₀) ⊆ Set.Icc a b := Set.Icc_subset_Icc hL hR
  have hδ₀ : 0 ≤ δ₀ := by dsimp [δ₀]; positivity
  have hE₀_mem : E₀ ∈ Set.Icc a b := hsub ⟨by linarith, by linarith⟩
  refine ⟨E_star, ⟨hE_mem, ⟨n + 1, by push_cast; exact hE_eq⟩⟩, ?_⟩
  rintro E' ⟨hE'_mem, ⟨k, hk⟩⟩
  -- Show `k = n + 1` using the upper slope bound on `|I₁(E') - I₁(E₀)|`:
  have hI₁_diff : |I₁ E' - I₁ E₀| ≤ T_up * δ₀ := by
    rcases le_total E' E₀ with hle | hle
    · have h1 := hRateHi E' (hsub hE'_mem) E₀ hE₀_mem hle
      have h2 := hRateLo E' (hsub hE'_mem) E₀ hE₀_mem hle
      have h3 : 0 ≤ I₁ E₀ - I₁ E' := by nlinarith
      rw [abs_le]
      constructor <;> nlinarith [hE'_mem.1, hE'_mem.2]
    · have h1 := hRateHi E₀ hE₀_mem E' (hsub hE'_mem) hle
      have h2 := hRateLo E₀ hE₀_mem E' (hsub hE'_mem) hle
      rw [abs_le]
      constructor <;> nlinarith [hE'_mem.1, hE'_mem.2]
  -- Also `|I_hbar E' - I₁ E' - π ℏ| ≤ C_I ℏ²` and `I_hbar E' = 2π ℏ k`, `I₁ E₀ + π ℏ = 2π ℏ (n + 1)`
  have hApp := abs_le.mp (hApprox E' (hsub hE'_mem))
  have hTup_δ : T_up * δ₀ < Real.pi * ℏ := by
    have h1 : 2 * δ₀ * T_up < 2 * Real.pi * ℏ := by
      have := (lt_div_iff₀ hTup).mp hSep
      linarith
    linarith
  have hC_bound : C_I * ℏ ^ 2 ≤ T_up * δ₀ := by
    have hTδ : (T_min / 2) * δ₀ = C_I * ℏ ^ 2 := by dsimp [δ₀]; field_simp [ne_of_gt hT]
    -- Wait: we only need `C_I ℏ² < π ℏ` (or `T_up δ₀ + C_I ℏ² < 2 π ℏ`)!
    -- Actually check: `|2π ℏ (k - (n + 1))| = |(I_hbar E' - I₁ E' - π ℏ) + (I₁ E' - I₁ E₀)| ≤ C_I ℏ² + T_up δ₀`.
    -- Does `T_min / 2 ≤ T_up`? YES, apply `hRateLo` and `hRateHi` on `[E₀ - δ₀, E₀ + δ₀]` (or directly if `δ₀ > 0`)!
    by_cases hδ0_zero : δ₀ = 0
    · have hC0 : C_I * ℏ ^ 2 = 0 := by
        have hTδ : (T_min / 2) * δ₀ = C_I * ℏ ^ 2 := by dsimp [δ₀]; field_simp [ne_of_gt hT]
        rw [hδ0_zero, mul_zero] at hTδ; linarith
      rw [hC0, hδ0_zero, mul_zero]
    · have hδ_pos : 0 < δ₀ := lt_of_le_of_ne hδ₀ (Ne.symm hδ0_zero)
      have hlo := hRateLo (E₀ - δ₀) (hsub ⟨le_rfl, by linarith⟩) (E₀ + δ₀) (hsub ⟨by linarith, le_rfl⟩) (by linarith)
      have hhi := hRateHi (E₀ - δ₀) (hsub ⟨le_rfl, by linarith⟩) (E₀ + δ₀) (hsub ⟨by linarith, le_rfl⟩) (by linarith)
      have hT_le : T_min / 2 ≤ T_up := by nlinarith
      nlinarith
  have hdiff_int : |2 * Real.pi * ℏ * ((k : ℝ) - ((n : ℝ) + 1))| < 2 * Real.pi * ℏ := by
    have heq : 2 * Real.pi * ℏ * ((k : ℝ) - ((n : ℝ) + 1)) =
        (I_hbar E' - I₁ E' - Real.pi * ℏ) + (I₁ E' - I₁ E₀) := by
      rw [hk, hE₀_rule]; ring
    rw [heq]
    have htri := abs_add_le (I_hbar E' - I₁ E' - Real.pi * ℏ) (I₁ E' - I₁ E₀)
    have hApp_abs := hApprox E' (hsub hE'_mem)
    linarith
  have h2piℏ : 0 < 2 * Real.pi * ℏ := by positivity
  have hkn : |(k : ℝ) - ((n : ℝ) + 1)| < 1 := by
    have h1 : 2 * Real.pi * ℏ * |(k : ℝ) - ((n : ℝ) + 1)| < 2 * Real.pi * ℏ * 1 := by
      rw [← abs_of_pos h2piℏ, ← abs_mul]
      simpa [abs_of_pos hℏ] using hdiff_int
    exact lt_of_mul_lt_mul_left h1 (le_of_lt h2piℏ)
  have hk_eq : k = n + 1 := by
    have hcast : ((k - (n + 1) : ℤ) : ℝ) = (k : ℝ) - ((n : ℝ) + 1) := by push_cast; ring
    rw [← hcast] at hkn
    have habs_lt : |k - (n + 1)| < 1 := by exact_mod_cast hkn
    rw [abs_lt] at habs_lt
    omega
  rw [hk_eq] at hk
  exact hOnce (n + 1) E' (hsub hE'_mem) E_star (hsub hE_mem) hk (by push_cast; exact hE_eq)

/-! ### Part 2: Lemma 4 (`lem:phase_action`) — Modulo-`π` Unique Lifting & Continuous Action -/

/-- **Unique Continuous Modulo-`π` Representative** (`sgm_wavefn.tex` Lemma 4):
If `f : ℝ → ℝ` is continuous modulo `π` on `[a, b]` (meaning `cos(2 f)` and `sin(2 f)` are
continuous on `[a, b]`) and stays within `M < π/2` of `π ℤ` on `[a, b]`, then
`δ(E) := arctan(sin(2 f E) / (1 + cos(2 f E)))` is:
1. continuous on `[a, b]`;
2. bounded by `|δ(E)| ≤ M < π/2` and congruent to `f(E)` modulo `π`;
3. the **UNIQUE** representative in `(-π/2, π/2)` congruent to `f(E)` modulo `π`. -/
theorem mod_pi_unique_continuous_rep
    {a b M : ℝ} (hM_lt : M < Real.pi / 2)
    (f : ℝ → ℝ)
    (hCos : ContinuousOn (fun E => Real.cos (2 * f E)) (Set.Icc a b))
    (hSin : ContinuousOn (fun E => Real.sin (2 * f E)) (Set.Icc a b))
    (hClose : ∀ E ∈ Set.Icc a b, ∃ k : ℤ, |f E - (k : ℝ) * Real.pi| ≤ M) :
    let δ : ℝ → ℝ := fun E => Real.arctan (Real.sin (2 * f E) / (1 + Real.cos (2 * f E)))
    ContinuousOn δ (Set.Icc a b) ∧
    (∀ E ∈ Set.Icc a b, |δ E| ≤ M) ∧
    (∀ E ∈ Set.Icc a b, ∃ k : ℤ, f E - δ E = (k : ℝ) * Real.pi) ∧
    (∀ E ∈ Set.Icc a b, ∀ r : ℝ, |r| < Real.pi / 2 →
      (∃ k : ℤ, f E - r = (k : ℝ) * Real.pi) → r = δ E) := by
  intro δ
  -- Pointwise identity: at every `E ∈ [a, b]`, for `k` with `|f E - k π| ≤ M < π/2`,
  -- `1 + cos(2 f E) > 0` and `δ E = f E - k π`.
  have hpoint : ∀ E ∈ Set.Icc a b,
      0 < 1 + Real.cos (2 * f E) ∧
      |δ E| ≤ M ∧
      ∃ k : ℤ, f E - δ E = (k : ℝ) * Real.pi := by
    intro E hE
    rcases hClose E hE with ⟨k, hk⟩
    set r : ℝ := f E - (k : ℝ) * Real.pi
    have hr_lt : |r| < Real.pi / 2 := lt_of_le_of_lt hk hM_lt
    have hr_Ioo : r ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2) := abs_lt.mp hr_lt
    have hcos_pos : 0 < Real.cos r := Real.cos_pos_of_mem_Ioo hr_Ioo
    have h2f : 2 * f E = 2 * r + (k : ℝ) * (2 * Real.pi) := by dsimp [r]; ring
    have hcos2 : Real.cos (2 * f E) = Real.cos (2 * r) := by
      rw [h2f, Real.cos_add_int_mul_two_pi]
    have hsin2 : Real.sin (2 * f E) = Real.sin (2 * r) := by
      rw [h2f, Real.sin_add_int_mul_two_pi]
    have hden_eq : 1 + Real.cos (2 * r) = 2 * Real.cos r ^ 2 := by
      rw [Real.cos_two_mul]; ring
    have hden_pos : 0 < 1 + Real.cos (2 * f E) := by
      rw [hcos2, hden_eq]; positivity
    have hratio : Real.sin (2 * f E) / (1 + Real.cos (2 * f E)) = Real.tan r := by
      rw [hsin2, hcos2, Real.sin_two_mul, hden_eq, Real.tan_eq_sin_div_cos]
      have hcne : Real.cos r ≠ 0 := ne_of_gt hcos_pos
      field_simp [hcne]
    have hδ_eq : δ E = r := by
      dsimp [δ]
      rw [hratio, Real.arctan_tan hr_Ioo.1 hr_Ioo.2]
    refine ⟨hden_pos, ?_, k, ?_⟩
    · rw [hδ_eq]; exact hk
    · rw [hδ_eq]; dsimp [r]; ring
  have hCont : ContinuousOn δ (Set.Icc a b) := by
    refine Real.continuous_arctan.comp_continuousOn ?_
    exact hSin.div (continuousOn_const.add hCos) (fun E hE => ne_of_gt (hpoint E hE).1)
  have hUniq : ∀ E ∈ Set.Icc a b, ∀ r : ℝ, |r| < Real.pi / 2 →
      (∃ k : ℤ, f E - r = (k : ℝ) * Real.pi) → r = δ E := by
    intro E hE r hr ⟨k₁, hk₁⟩
    rcases (hpoint E hE).2.2 with ⟨k₂, hk₂⟩
    have hδ_lt : |δ E| < Real.pi / 2 := lt_of_le_of_lt (hpoint E hE).2.1 hM_lt
    have hdiff : ((k₂ - k₁ : ℤ) : ℝ) * Real.pi = r - δ E := by push_cast; linarith
    have habs_lt : |((k₂ - k₁ : ℤ) : ℝ) * Real.pi| < Real.pi := by
      rw [hdiff]
      calc |r - δ E| ≤ |r| + |δ E| := abs_sub _ _
        _ < Real.pi / 2 + Real.pi / 2 := add_lt_add hr hδ_lt
        _ = Real.pi := by ring
    rw [abs_mul, abs_of_pos Real.pi_pos] at habs_lt
    have h1 : |((k₂ - k₁ : ℤ) : ℝ)| < 1 := by
      nlinarith [Real.pi_pos]
    have h2 : |k₂ - k₁| < 1 := by exact_mod_cast h1
    rw [abs_lt] at h2
    have hkeq : k₂ = k₁ := by omega
    rw [hkeq] at hk₂
    linarith
  exact ⟨hCont, fun E hE => (hpoint E hE).2.1, fun E hE => (hpoint E hE).2.2, hUniq⟩

/-- **Lemma 4 (`lem:phase_action`)**:
Given continuous real functions `Φ_+, Φ_-` on `[a, b]`, modulo-`π` continuous phases `ϑ_+, ϑ_-`
(i.e. `cos(2 ϑ_±)` and `sin(2 ϑ_±)` continuous on `[a, b]`), and `0 ≤ C`, `0 < ℏ` with the
topological uniqueness threshold `4 C ℏ < π` such that
`ϑ_+(E) + Φ_+(E)/ℏ + π/4` and `ϑ_-(E) - Φ_-(E)/ℏ - π/4` stay within `C ℏ` of `π ℤ`,
there exists a continuous action `I_hbar` on `[a, b]` (constructed from the unique modulo-`π`
representatives `δ_±(E) ∈ (-π/2, π/2)`) satisfying:
1. `|I_hbar(E) - 2(Φ_-(E) + Φ_+(E)) - π ℏ| ≤ 4 C ℏ²`;
2. `ϑ_-(E) - ϑ_+(E) ∈ π ℤ ↔ I_hbar(E) ∈ 2π ℏ ℤ`! -/
theorem lem_phase_action
    {a b ℏ C : ℝ} (hℏ : 0 < ℏ) (hC : 0 ≤ C) (hSmall : 4 * C * ℏ < Real.pi)
    (Φp Φm ϑp ϑm : ℝ → ℝ)
    (hContΦp : ContinuousOn Φp (Set.Icc a b))
    (hContΦm : ContinuousOn Φm (Set.Icc a b))
    (hCosP : ContinuousOn (fun E => Real.cos (2 * ϑp E)) (Set.Icc a b))
    (hSinP : ContinuousOn (fun E => Real.sin (2 * ϑp E)) (Set.Icc a b))
    (hCosM : ContinuousOn (fun E => Real.cos (2 * ϑm E)) (Set.Icc a b))
    (hSinM : ContinuousOn (fun E => Real.sin (2 * ϑm E)) (Set.Icc a b))
    (hCloseP : ∀ E ∈ Set.Icc a b, ∃ kp : ℤ, |ϑp E + Φp E / ℏ + Real.pi / 4 - (kp : ℝ) * Real.pi| ≤ C * ℏ)
    (hCloseM : ∀ E ∈ Set.Icc a b, ∃ km : ℤ, |ϑm E - Φm E / ℏ - Real.pi / 4 - (km : ℝ) * Real.pi| ≤ C * ℏ) :
    ∃ I_hbar : ℝ → ℝ,
      ContinuousOn I_hbar (Set.Icc a b) ∧
      (∀ E ∈ Set.Icc a b, |I_hbar E - 2 * (Φm E + Φp E) - Real.pi * ℏ| ≤ 4 * C * ℏ ^ 2) ∧
      (∀ E ∈ Set.Icc a b,
        (∃ m : ℤ, ϑm E - ϑp E = (m : ℝ) * Real.pi) ↔
        (∃ k : ℤ, I_hbar E = 2 * Real.pi * ℏ * (k : ℝ))) := by
  have hM_lt : C * ℏ < Real.pi / 2 := by
    have hnonneg : 0 ≤ C * ℏ := mul_nonneg hC (le_of_lt hℏ)
    linarith
  let fp : ℝ → ℝ := fun E => ϑp E + Φp E / ℏ + Real.pi / 4
  let fm : ℝ → ℝ := fun E => ϑm E - Φm E / ℏ - Real.pi / 4
  -- `cos(2 fp)` and `sin(2 fp)` are continuous via angle-addition formulas:
  have hCosFp : ContinuousOn (fun E => Real.cos (2 * fp E)) (Set.Icc a b) := by
    have heq : (fun E => Real.cos (2 * fp E)) =
        (fun E => Real.cos (2 * ϑp E) * Real.cos (2 * (Φp E / ℏ) + Real.pi / 2) -
                  Real.sin (2 * ϑp E) * Real.sin (2 * (Φp E / ℏ) + Real.pi / 2)) := by
      ext E
      have h2 : 2 * fp E = 2 * ϑp E + (2 * (Φp E / ℏ) + Real.pi / 2) := by dsimp [fp]; ring
      rw [h2, Real.cos_add]
    rw [heq]
    have harg : ContinuousOn (fun E => 2 * (Φp E / ℏ) + Real.pi / 2) (Set.Icc a b) :=
      (continuousOn_const.mul (hContΦp.div_const ℏ)).add continuousOn_const
    exact (hCosP.mul (Real.continuous_cos.comp_continuousOn harg)).sub
      (hSinP.mul (Real.continuous_sin.comp_continuousOn harg))
  have hSinFp : ContinuousOn (fun E => Real.sin (2 * fp E)) (Set.Icc a b) := by
    have heq : (fun E => Real.sin (2 * fp E)) =
        (fun E => Real.sin (2 * ϑp E) * Real.cos (2 * (Φp E / ℏ) + Real.pi / 2) +
                  Real.cos (2 * ϑp E) * Real.sin (2 * (Φp E / ℏ) + Real.pi / 2)) := by
      ext E
      have h2 : 2 * fp E = 2 * ϑp E + (2 * (Φp E / ℏ) + Real.pi / 2) := by dsimp [fp]; ring
      rw [h2, Real.sin_add]
    rw [heq]
    have harg : ContinuousOn (fun E => 2 * (Φp E / ℏ) + Real.pi / 2) (Set.Icc a b) :=
      (continuousOn_const.mul (hContΦp.div_const ℏ)).add continuousOn_const
    exact (hSinP.mul (Real.continuous_cos.comp_continuousOn harg)).add
      (hCosP.mul (Real.continuous_sin.comp_continuousOn harg))
  have hCosFm : ContinuousOn (fun E => Real.cos (2 * fm E)) (Set.Icc a b) := by
    have heq : (fun E => Real.cos (2 * fm E)) =
        (fun E => Real.cos (2 * ϑm E) * Real.cos (2 * (Φm E / ℏ) + Real.pi / 2) +
                  Real.sin (2 * ϑm E) * Real.sin (2 * (Φm E / ℏ) + Real.pi / 2)) := by
      ext E
      have h2 : 2 * fm E = 2 * ϑm E - (2 * (Φm E / ℏ) + Real.pi / 2) := by dsimp [fm]; ring
      rw [h2, Real.cos_sub]
    rw [heq]
    have harg : ContinuousOn (fun E => 2 * (Φm E / ℏ) + Real.pi / 2) (Set.Icc a b) :=
      (continuousOn_const.mul (hContΦm.div_const ℏ)).add continuousOn_const
    exact (hCosM.mul (Real.continuous_cos.comp_continuousOn harg)).add
      (hSinM.mul (Real.continuous_sin.comp_continuousOn harg))
  have hSinFm : ContinuousOn (fun E => Real.sin (2 * fm E)) (Set.Icc a b) := by
    have heq : (fun E => Real.sin (2 * fm E)) =
        (fun E => Real.sin (2 * ϑm E) * Real.cos (2 * (Φm E / ℏ) + Real.pi / 2) -
                  Real.cos (2 * ϑm E) * Real.sin (2 * (Φm E / ℏ) + Real.pi / 2)) := by
      ext E
      have h2 : 2 * fm E = 2 * ϑm E - (2 * (Φm E / ℏ) + Real.pi / 2) := by dsimp [fm]; ring
      rw [h2, Real.sin_sub]
    rw [heq]
    have harg : ContinuousOn (fun E => 2 * (Φm E / ℏ) + Real.pi / 2) (Set.Icc a b) :=
      (continuousOn_const.mul (hContΦm.div_const ℏ)).add continuousOn_const
    exact (hSinM.mul (Real.continuous_cos.comp_continuousOn harg)).sub
      (hCosM.mul (Real.continuous_sin.comp_continuousOn harg))
  -- Lift unique continuous modulo-`π` representatives `δp, δm ∈ [-C ℏ, C ℏ] ⊂ (-π/2, π/2)`:
  rcases mod_pi_unique_continuous_rep hM_lt fp hCosFp hSinFp hCloseP
    with ⟨hContδp, hBp, hCongP, _⟩
  rcases mod_pi_unique_continuous_rep hM_lt fm hCosFm hSinFm hCloseM
    with ⟨hContδm, hBm, hCongM, _⟩
  set δp : ℝ → ℝ := fun E => Real.arctan (Real.sin (2 * fp E) / (1 + Real.cos (2 * fp E)))
  set δm : ℝ → ℝ := fun E => Real.arctan (Real.sin (2 * fm E) / (1 + Real.cos (2 * fm E)))
  let I_hbar : ℝ → ℝ := fun E => 2 * (Φm E + Φp E) + Real.pi * ℏ + 2 * ℏ * (δm E - δp E)
  refine ⟨I_hbar, ?_, ?_, ?_⟩
  · unfold I_hbar
    fun_prop (disch := assumption)
  · intro E hE
    have hdiff : I_hbar E - 2 * (Φm E + Φp E) - Real.pi * ℏ = 2 * ℏ * (δm E - δp E) := by
      simp only [I_hbar]; ring
    rw [hdiff, abs_mul, abs_of_pos (by positivity : 0 < 2 * ℏ)]
    have h1 : |δm E - δp E| ≤ 2 * C * ℏ := by
      calc |δm E - δp E| ≤ |δm E| + |δp E| := abs_sub _ _
        _ ≤ C * ℏ + C * ℏ := add_le_add (hBm E hE) (hBp E hE)
        _ = 2 * C * ℏ := by ring
    nlinarith
  · intro E hE
    rcases hCongP E hE with ⟨kp, hkp⟩
    rcases hCongM E hE with ⟨km, hkm⟩
    have hrel : ϑm E - ϑp E - I_hbar E / (2 * ℏ) = ((km - kp : ℤ) : ℝ) * Real.pi := by
      simp only [I_hbar]
      push_cast
      have hℏ_ne : ℏ ≠ 0 := ne_of_gt hℏ
      have hsub : (ϑm E - Φm E / ℏ - Real.pi / 4 - δm E) -
                  (ϑp E + Φp E / ℏ + Real.pi / 4 - δp E) = ((km : ℝ) - (kp : ℝ)) * Real.pi := by
        dsimp [fp, fm] at hkp hkm; linarith
      field_simp [hℏ_ne] at hsub ⊢
      linarith
    have hℏ_ne : ℏ ≠ 0 := ne_of_gt hℏ
    constructor
    · rintro ⟨m, hm⟩
      refine ⟨m - (km - kp), ?_⟩
      push_cast at hrel ⊢
      rw [hm] at hrel
      field_simp [hℏ_ne] at hrel ⊢
      linear_combination -hrel
    · rintro ⟨k, hk⟩
      refine ⟨k + (km - kp), ?_⟩
      push_cast at hrel ⊢
      rw [hk] at hrel
      field_simp [hℏ_ne] at hrel ⊢
      linear_combination hrel

/-! ### Part 3: Lemma 6 (`lem:labels`(b) & (c)) — Constant Integer Shift & At-Most-Once Crossing -/

/-- Helper: Any continuous function `f : ℝ → ℝ` on `[a, b]` (`a ≤ b`) taking values only in `ℤ`
is constant on `[a, b]` (connectedness of `[a, b]`). -/
theorem continuous_int_valued_const_on_Icc {a b : ℝ} (hab : a ≤ b)
    (f : ℝ → ℝ) (hCont : ContinuousOn f (Set.Icc a b))
    (hInt : ∀ E ∈ Set.Icc a b, ∃ m : ℤ, f E = (m : ℝ)) :
    ∃ m₀ : ℤ, ∀ E ∈ Set.Icc a b, f E = (m₀ : ℝ) := by
  have ha : a ∈ Set.Icc a b := ⟨le_rfl, hab⟩
  rcases hInt a ha with ⟨m₀, hm₀⟩
  refine ⟨m₀, fun E hE => ?_⟩
  by_contra hne
  rcases lt_or_gt_of_ne hne with hlt | hgt
  · rcases hInt E hE with ⟨m, hm⟩
    rw [hm] at hlt
    have hm_lt : m < m₀ := by exact_mod_cast hlt
    -- By IVT, `f` takes the non-integer value `(m₀ : ℝ) - 1/2` between `a` and `E`!
    have hmid : (m₀ : ℝ) - 1 / 2 ∈ Set.Icc (f E) (f a) := by
      rw [hm, hm₀]
      have : (m : ℝ) + 1 ≤ (m₀ : ℝ) := by exact_mod_cast Int.add_one_le_iff.mpr hm_lt
      constructor <;> linarith
    have hivt := intermediate_value_Icc' hE.1 (hCont.mono (Set.Icc_subset_Icc le_rfl hE.2))
    rcases hivt hmid with ⟨E', hE'_sub, hE'_val⟩
    rcases hInt E' ⟨hE'_sub.1, le_trans hE'_sub.2 hE.2⟩ with ⟨m', hm'⟩
    rw [hm'] at hE'_val
    have h2 : ((2 * m' : ℤ) : ℝ) = ((2 * m₀ - 1 : ℤ) : ℝ) := by push_cast; linarith
    have h2z : 2 * m' = 2 * m₀ - 1 := by exact_mod_cast h2
    omega
  · rcases hInt E hE with ⟨m, hm⟩
    rw [hm] at hgt
    have hm_gt : m₀ < m := by exact_mod_cast hgt
    have hmid : (m₀ : ℝ) + 1 / 2 ∈ Set.Icc (f a) (f E) := by
      rw [hm, hm₀]
      have : (m₀ : ℝ) + 1 ≤ (m : ℝ) := by exact_mod_cast Int.add_one_le_iff.mpr hm_gt
      constructor <;> linarith
    have hivt := intermediate_value_Icc hE.1 (hCont.mono (Set.Icc_subset_Icc le_rfl hE.2))
    rcases hivt hmid with ⟨E', hE'_sub, hE'_val⟩
    rcases hInt E' ⟨hE'_sub.1, le_trans hE'_sub.2 hE.2⟩ with ⟨m', hm'⟩
    rw [hm'] at hE'_val
    have h2 : ((2 * m' : ℤ) : ℝ) = ((2 * m₀ + 1 : ℤ) : ℝ) := by push_cast; linarith
    have h2z : 2 * m' = 2 * m₀ + 1 := by exact_mod_cast h2
    omega

/-- Helper: If `g : ℝ → ℝ` satisfies `g(θ + π) = g(θ) + π`, then `g(θ + j π) = g(θ) + j π`
for every integer `j : ℤ`. -/
theorem shift_pi_int (g : ℝ → ℝ) (hstep : ∀ θ, g (θ + Real.pi) = g θ + Real.pi)
    (j : ℤ) (θ : ℝ) : g (θ + (j : ℝ) * Real.pi) = g θ + (j : ℝ) * Real.pi := by
  induction j using Int.induction_on with
  | zero => simp
  | succ n ih =>
    push_cast at ih ⊢
    have h1 : θ + ((n : ℝ) + 1) * Real.pi = (θ + (n : ℝ) * Real.pi) + Real.pi := by ring
    rw [h1, hstep, ih]; ring
  | pred n ih =>
    push_cast at ih ⊢
    have h1 : θ + (-(n : ℝ) - 1) * Real.pi + Real.pi = θ + (-(n : ℝ)) * Real.pi := by ring
    have h2 := hstep (θ + (-(n : ℝ) - 1) * Real.pi)
    rw [h1] at h2
    linarith [ih]

/-- **Lemma 6 (`lem:labels`), Part (b) (NO global monotonicity needed!)**:
On a compact interval `[a, b]` (`a ≤ b`), given continuous `I_hbar`, `θ_+`, `θ_-`, and a lifted
matching angle `θ_G(θ, E)` that is strictly increasing in `θ`, satisfies `θ_G(θ + π, E) = θ_G(θ, E) + π`,
and makes `E ↦ θ_G(θ_±(E), E)` continuous with `I_hbar(E)/(2ℏ) - (θ_G(θ_-(E), E) - θ_G(θ_+(E), E)) ∈ π ℤ`:
there exists a SINGLE constant integer `m_L : ℤ` on `[a, b]` such that for ALL `E ∈ [a, b]` and `k : ℤ`,
`I_hbar(E) = 2π ℏ k ↔ θ_-(E) - θ_+(E) = (k - m_L) π`! -/
theorem lem_labels_part_b
    {a b ℏ : ℝ} (hab : a ≤ b) (hℏ : 0 < ℏ)
    (I_hbar θp θm : ℝ → ℝ)
    (θG : ℝ → ℝ → ℝ)
    (hContI : ContinuousOn I_hbar (Set.Icc a b))
    (hContGp : ContinuousOn (fun E => θG (θp E) E) (Set.Icc a b))
    (hContGm : ContinuousOn (fun E => θG (θm E) E) (Set.Icc a b))
    (hStrict : ∀ E ∈ Set.Icc a b, StrictMono (fun θ => θG θ E))
    (hShift : ∀ E ∈ Set.Icc a b, ∀ θ : ℝ, θG (θ + Real.pi) E = θG θ E + Real.pi)
    (hCong : ∀ E ∈ Set.Icc a b, ∃ m : ℤ,
      I_hbar E / (2 * ℏ) - (θG (θm E) E - θG (θp E) E) = (m : ℝ) * Real.pi) :
    ∃ m_L : ℤ, ∀ E ∈ Set.Icc a b, ∀ k : ℤ,
      I_hbar E = 2 * Real.pi * ℏ * (k : ℝ) ↔
        θm E - θp E = ((k - m_L : ℤ) : ℝ) * Real.pi := by
  let f : ℝ → ℝ := fun E => (I_hbar E / (2 * ℏ) - (θG (θm E) E - θG (θp E) E)) / Real.pi
  have hContF : ContinuousOn f (Set.Icc a b) :=
    ((hContI.div_const (2 * ℏ)).sub (hContGm.sub hContGp)).div_const Real.pi
  have hIntF : ∀ E ∈ Set.Icc a b, ∃ m : ℤ, f E = (m : ℝ) := by
    intro E hE
    rcases hCong E hE with ⟨m, hm⟩
    refine ⟨m, ?_⟩
    dsimp [f]
    rw [hm, mul_div_cancel_right₀ _ Real.pi_ne_zero]
  rcases continuous_int_valued_const_on_Icc hab f hContF hIntF with ⟨m_L, hm_L⟩
  refine ⟨m_L, fun E hE k => ?_⟩
  have hconst : I_hbar E / (2 * ℏ) - (θG (θm E) E - θG (θp E) E) = (m_L : ℝ) * Real.pi := by
    have h1 := hm_L E hE
    dsimp [f] at h1
    exact (div_eq_iff Real.pi_ne_zero).mp h1
  have hℏ_ne : ℏ ≠ 0 := ne_of_gt hℏ
  have hGshift := shift_pi_int (fun θ => θG θ E) (hShift E hE) (k - m_L) (θp E)
  have hinj := (hStrict E hE).injective
  constructor
  · intro hI
    have hGeq : θG (θm E) E = θG (θp E + ((k - m_L : ℤ) : ℝ) * Real.pi) E := by
      rw [hGshift]
      rw [hI] at hconst
      push_cast
      field_simp [hℏ_ne] at hconst
      linarith
    have hθ := hinj hGeq
    linarith
  · intro hΔ
    have hθm : θm E = θp E + ((k - m_L : ℤ) : ℝ) * Real.pi := by linarith
    rw [hθm, hGshift] at hconst
    push_cast at hconst
    field_simp [hℏ_ne] at hconst ⊢
    linarith

/-- **Lemma 6 (`lem:labels`), Part (c) (Monotonicity → at-most-once)**:
If `m_L` is a constant shift with `I_hbar E = 2π ℏ k ↔ Δ_θ E = (k - m_L) π` on `[a, b]`
and `Δ_θ` is strictly increasing on `[a, b]`, then `I_hbar` takes each value `2π ℏ k`
at most once on `[a, b]`. -/
theorem lem_labels_part_c
    {a b ℏ : ℝ} (I_hbar Δ_θ : ℝ → ℝ) (m_L : ℤ)
    (hEquiv : ∀ E ∈ Set.Icc a b, ∀ k : ℤ,
      I_hbar E = 2 * Real.pi * ℏ * (k : ℝ) ↔ Δ_θ E = ((k - m_L : ℤ) : ℝ) * Real.pi)
    (hMono : StrictMonoOn Δ_θ (Set.Icc a b)) :
    ∀ k : ℤ, ∀ E₁ ∈ Set.Icc a b, ∀ E₂ ∈ Set.Icc a b,
      I_hbar E₁ = 2 * Real.pi * ℏ * (k : ℝ) →
      I_hbar E₂ = 2 * Real.pi * ℏ * (k : ℝ) → E₁ = E₂ := by
  intro k E₁ hE₁ E₂ hE₂ h1 h2
  have hd1 := (hEquiv E₁ hE₁ k).mp h1
  have hd2 := (hEquiv E₂ hE₂ k).mp h2
  exact hMono.injOn hE₁ hE₂ (hd1.trans hd2.symm)

/-! ### Part 4: Lemma 8 (`lem:monotone`) — Exact Quadratic Completion Bound -/

/-- **Lemma 8 (`lem:monotone`)**:
For `a_mon ≥ a_* > 0`, `b_mon ≥ b_* > 0`, `|a_mon'| + a_mon |ρ'| ≤ M_mon` (`M_mon > 0`),
`s ∈ {-1, 1}`, and `0 < ℏ ≤ min(b_* / (4 M_mon), √(a_* b_*) / M_mon)`:
`b_mon + a_mon ρ² - s ℏ (a_mon' ρ + a_mon ρ') ≥ b_* / 2` for EVERY `ρ ∈ ℝ`! -/
theorem lem_monotone_quadratic_bound
    {a_mon b_mon a_mon' ρ' a_star b_star M_mon ℏ s ρ : ℝ}
    (ha_star : 0 < a_star) (hb_star : 0 < b_star) (hM : 0 < M_mon)
    (ha : a_star ≤ a_mon) (hb : b_star ≤ b_mon)
    (hBound : |a_mon'| + a_mon * |ρ'| ≤ M_mon)
    (hs : |s| ≤ 1)
    (hℏ0 : 0 ≤ ℏ)
    (hℏ1 : ℏ ≤ b_star / (4 * M_mon))
    (hℏ2 : ℏ ≤ Real.sqrt (a_star * b_star) / M_mon) :
    b_star / 2 ≤ b_mon + a_mon * ρ ^ 2 - s * ℏ * (a_mon' * ρ + a_mon * ρ') := by
  have h1 : |a_mon'| ≤ M_mon := by
    have : 0 ≤ a_mon * |ρ'| := mul_nonneg (by linarith) (abs_nonneg ρ')
    linarith
  have h2 : a_mon * |ρ'| ≤ M_mon := by linarith [abs_nonneg a_mon']
  -- Bound `-s ℏ a_mon ρ' ≥ -ℏ M_mon ≥ -b_* / 4`
  have hterm_const : -(b_star / 4) ≤ -s * ℏ * (a_mon * ρ') := by
    have hA : |s * ℏ * (a_mon * ρ')| ≤ ℏ * M_mon := by
      rw [abs_mul, abs_mul, abs_of_nonneg hℏ0, abs_mul, abs_of_pos (by linarith : 0 < a_mon)]
      have h_sℏ : |s| * ℏ ≤ 1 * ℏ := mul_le_mul_of_nonneg_right hs hℏ0
      have h_prod : |s| * ℏ * (a_mon * |ρ'|) ≤ (1 * ℏ) * M_mon :=
        mul_le_mul h_sℏ h2 (mul_nonneg (by linarith) (abs_nonneg _)) (by positivity)
      linarith
    have hB : ℏ * M_mon ≤ b_star / 4 := by
      have := (le_div_iff₀ (by positivity : 0 < 4 * M_mon)).mp hℏ1
      linarith
    have hC := (abs_le.mp hA).2
    linarith
  -- Bound `a_mon ρ² - s ℏ a_mon' ρ ≥ -ℏ² M_mon² / (4 a_*) ≥ -b_* / 4` by completing the square!
  have hℏM_sq : (ℏ * M_mon) ^ 2 ≤ a_star * b_star := by
    have hA : ℏ * M_mon ≤ Real.sqrt (a_star * b_star) := (le_div_iff₀ hM).mp hℏ2
    have hB : 0 ≤ ℏ * M_mon := by positivity
    have hC : (ℏ * M_mon) ^ 2 ≤ Real.sqrt (a_star * b_star) ^ 2 := by nlinarith
    rw [Real.sq_sqrt (by positivity)] at hC
    exact hC
  have hquad : -(b_star / 4) ≤ a_mon * ρ ^ 2 - s * ℏ * (a_mon' * ρ) := by
    have hA : |s * ℏ * a_mon'| ≤ ℏ * M_mon := by
      rw [abs_mul, abs_mul, abs_of_nonneg hℏ0]
      have h_sℏ : |s| * ℏ ≤ 1 * ℏ := mul_le_mul_of_nonneg_right hs hℏ0
      have h_prod : |s| * ℏ * |a_mon'| ≤ (1 * ℏ) * M_mon :=
        mul_le_mul h_sℏ h1 (abs_nonneg _) (by positivity)
      linarith
    have hlin : -(ℏ * M_mon * |ρ|) ≤ -(s * ℏ * (a_mon' * ρ)) := by
      have h1' : |s * ℏ * (a_mon' * ρ)| ≤ ℏ * M_mon * |ρ| := by
        rw [show s * ℏ * (a_mon' * ρ) = (s * ℏ * a_mon') * ρ by ring, abs_mul]
        exact mul_le_mul_of_nonneg_right hA (abs_nonneg ρ)
      linarith [(abs_le.mp h1').2]
    -- `a_* |ρ|² - (ℏ M_mon) |ρ| + b_* / 4 ≥ 0` since discriminant `(ℏ M_mon)² - a_* b_* ≤ 0`!
    have hdisc : 0 ≤ a_star * |ρ| ^ 2 - (ℏ * M_mon) * |ρ| + b_star / 4 := by
      have hsq : 0 ≤ (2 * a_star * |ρ| - ℏ * M_mon) ^ 2 := sq_nonneg _
      nlinarith [hℏM_sq, ha_star]
    rw [sq_abs] at hdisc
    have hmono_a : a_star * ρ ^ 2 ≤ a_mon * ρ ^ 2 :=
      mul_le_mul_of_nonneg_right ha (sq_nonneg ρ)
    linarith
  linarith

end SGM.Semiclassical
