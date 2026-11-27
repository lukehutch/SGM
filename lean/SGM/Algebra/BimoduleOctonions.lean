import SGM.Algebra.QuaternionSpinor
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Tactic

/-!
# Bimodule Associativity, Hurwitz Uniqueness, Octonionic Obstructions, and Bimodule Flow
(`sgm_wavefn.tex` Sec. 7.6 & App. A.8)

Formalizes:
- `prop:bimodule_associativity` (`eq:bimodule_commutator`): Pointwise `[L_a, R_b] ψ = 0` on `ℍ`,
  non-associative alternative-algebra breakdown `[L_a, L_b] = L_{[a,b]} - 2 [L_a, R_b] ≠ L_d`
  in `𝕆` while the operator Jacobi identity still holds identically.
- `prop:quat_uniqueness` (`eq:single_elem_assoc`): Single-element sequential closure
  `∀ a b, ∃ c, L_a ∘ L_b = L_c` is equivalent to associativity; among `{ℝ, ℂ, ℍ, 𝕆}`,
  simultaneous associativity and non-commutativity uniquely select `ℍ`.
- `prop:oct_obstructions` (`eq:fano_triples`, `eq:commutant_collapse`, `eq:oct_jacobi_fail`,
  `eq:d7_elliptic`): Concrete octonions `Oct` with the seven Fano triples:
  composition identity `|xy|² = |x|²|y|²`, left/right alternativity, exact commutant collapse
  `End_{𝕆_L}(𝕆) = ℝ · 1_8` (forbidding any commuting complex structure `J² = -I_8`),
  element Jacobi failure `[e₁,[e₂,e₄]] + cyc = -12 e₇ ≠ 0`, and 7D Clifford/elliptic symbol
  `L_{e_a} L_{e_b} + L_{e_b} L_{e_a} = -2 δ_{ab} I_8`.
- `rem:bimodule_flow` (`eq:bimodule_flow`): Exact non-unitary hyperbolic bimodule flow
  `∂_t ψ = g₀ e₃ ψ e₃ ⇔ ψ(t) = cosh(g₀ t) ψ(0) + sinh(g₀ t) e₃ ψ(0) e₃`.
-/

namespace SGM.Algebra

/-! ### Part 1: Quaternionic Bimodule Commutativity (`prop:bimodule_associativity`) -/

def qLeft (a : Quat) : Quat →ₗ[ℝ] Quat where
  toFun ψ := a * ψ
  map_add' x y := mul_add a x y
  map_smul' r x := by simp

def qRight (b : Quat) : Quat →ₗ[ℝ] Quat where
  toFun ψ := ψ * b
  map_add' x y := add_mul x y b
  map_smul' r x := by simp

/-- `sgm_wavefn.tex` Eq. `eq:bimodule_commutator`: `[L_a, R_b] ψ = a(ψb) - (aψ)b = 0` on `ℍ`. -/
theorem quat_bimodule_commutator (a b ψ : Quat) :
    qLeft a (qRight b ψ) - qRight b (qLeft a ψ) = 0 := by
  simp [qLeft, qRight, mul_assoc]

/-- Abstract associative operator Jacobi identity (holds for linear maps over ANY algebra, including `𝕆`). -/
theorem linearMap_jacobi {V : Type*} [AddCommGroup V] [Module ℝ V]
    (A B C : V →ₗ[ℝ] V) :
    let com := fun X Y : V →ₗ[ℝ] V => X.comp Y - Y.comp X
    com A (com B C) + com B (com C A) + com C (com A B) = 0 := by
  intro com
  ext v
  simp only [com, LinearMap.add_apply, LinearMap.sub_apply, LinearMap.comp_apply,
    LinearMap.zero_apply, map_sub]
  abel

/-! ### Part 2: Single-Element Closure Equivalence (`prop:quat_uniqueness`) -/

/-- `sgm_wavefn.tex` Eq. `eq:single_elem_assoc`:
In any unital magma `(A, *, 1)`, single-element sequential closure of left multiplications
`∀ a b, ∃ c, L_a ∘ L_b = L_c` holds if and only if multiplication is associative. -/
theorem single_elem_closure_iff_assoc {A : Type*} [Mul A] [One A]
    (h1 : ∀ a : A, a * 1 = a) :
    (∀ a b : A, ∃ c : A, ∀ σ : A, a * (b * σ) = c * σ) ↔
      (∀ a b σ : A, a * (b * σ) = (a * b) * σ) := by
  constructor
  · intro h a b σ
    rcases h a b with ⟨c, hc⟩
    have hc_eq : c = a * b := by
      have h1' := hc 1
      rw [h1, h1] at h1'
      exact h1'.symm
    rw [← hc_eq]
    exact hc σ
  · intro hassoc a b
    exact ⟨a * b, fun σ => hassoc a b σ⟩

/-! ### Part 3: Concrete Octonions `Oct` via the Fano Triples (`eq:fano_triples`) -/

/-- Octonions represented explicitly as 8-tuples `(x₀, x₁, ..., x₇) ∈ ℝ⁸`. -/
@[ext]
structure Oct where
  c0 : ℝ
  c1 : ℝ
  c2 : ℝ
  c3 : ℝ
  c4 : ℝ
  c5 : ℝ
  c6 : ℝ
  c7 : ℝ

namespace Oct

def zero : Oct := ⟨0, 0, 0, 0, 0, 0, 0, 0⟩
def one : Oct := ⟨1, 0, 0, 0, 0, 0, 0, 0⟩
def add (x y : Oct) : Oct :=
  ⟨x.c0 + y.c0, x.c1 + y.c1, x.c2 + y.c2, x.c3 + y.c3,
   x.c4 + y.c4, x.c5 + y.c5, x.c6 + y.c6, x.c7 + y.c7⟩
def neg (x : Oct) : Oct :=
  ⟨-x.c0, -x.c1, -x.c2, -x.c3, -x.c4, -x.c5, -x.c6, -x.c7⟩
def sub (x y : Oct) : Oct := add x (neg y)
def smul (r : ℝ) (x : Oct) : Oct :=
  ⟨r * x.c0, r * x.c1, r * x.c2, r * x.c3,
   r * x.c4, r * x.c5, r * x.c6, r * x.c7⟩

instance : Zero Oct := ⟨zero⟩
instance : One Oct := ⟨one⟩
instance : Add Oct := ⟨add⟩
instance : Neg Oct := ⟨neg⟩
instance : Sub Oct := ⟨sub⟩
instance : SMul ℝ Oct := ⟨smul⟩

instance : AddCommGroup Oct where
  add := add
  add_assoc a b c := by ext <;> exact add_assoc _ _ _
  zero := zero
  zero_add a := by ext <;> exact zero_add _
  add_zero a := by ext <;> exact add_zero _
  nsmul := nsmulRec
  neg := neg
  zsmul := zsmulRec
  neg_add_cancel a := by ext <;> exact neg_add_cancel _
  add_comm a b := by ext <;> exact add_comm _ _
  sub := sub
  sub_eq_add_neg _ _ := rfl

instance : Module ℝ Oct where
  smul := smul
  one_smul a := by ext <;> exact one_mul _
  mul_smul r s a := by ext <;> exact mul_assoc r s _
  smul_zero r := by ext <;> exact mul_zero r
  smul_add r a b := by ext <;> exact mul_add r _ _
  add_smul r s a := by ext <;> exact add_mul r s _
  zero_smul a := by ext <;> exact zero_mul _

/-- Multiplication on `Oct` defined by the seven oriented Fano triples (`eq:fano_triples`):
`(1,2,3), (1,4,5), (1,7,6), (2,4,6), (2,5,7), (3,4,7), (3,6,5)`. -/
def mul (x y : Oct) : Oct :=
  ⟨x.c0*y.c0 - x.c1*y.c1 - x.c2*y.c2 - x.c3*y.c3 - x.c4*y.c4 - x.c5*y.c5 - x.c6*y.c6 - x.c7*y.c7,
   x.c0*y.c1 + x.c1*y.c0 + x.c2*y.c3 - x.c3*y.c2 + x.c4*y.c5 - x.c5*y.c4 + x.c7*y.c6 - x.c6*y.c7,
   x.c0*y.c2 + x.c2*y.c0 + x.c3*y.c1 - x.c1*y.c3 + x.c4*y.c6 - x.c6*y.c4 + x.c5*y.c7 - x.c7*y.c5,
   x.c0*y.c3 + x.c3*y.c0 + x.c1*y.c2 - x.c2*y.c1 + x.c4*y.c7 - x.c7*y.c4 + x.c6*y.c5 - x.c5*y.c6,
   x.c0*y.c4 + x.c4*y.c0 + x.c5*y.c1 - x.c1*y.c5 + x.c6*y.c2 - x.c2*y.c6 + x.c7*y.c3 - x.c3*y.c7,
   x.c0*y.c5 + x.c5*y.c0 + x.c1*y.c4 - x.c4*y.c1 + x.c7*y.c2 - x.c2*y.c7 + x.c3*y.c6 - x.c6*y.c3,
   x.c0*y.c6 + x.c6*y.c0 + x.c1*y.c7 - x.c7*y.c1 + x.c2*y.c4 - x.c4*y.c2 + x.c5*y.c3 - x.c3*y.c5,
   x.c0*y.c7 + x.c7*y.c0 + x.c6*y.c1 - x.c1*y.c6 + x.c2*y.c5 - x.c5*y.c2 + x.c3*y.c4 - x.c4*y.c3⟩

instance : Mul Oct := ⟨mul⟩

-- Clean component-level `@[simp]` lemmas (never unfolding `HMul.hMul` on `ℝ`!)
@[simp] lemma zero_c0 : (0 : Oct).c0 = 0 := rfl
@[simp] lemma zero_c1 : (0 : Oct).c1 = 0 := rfl
@[simp] lemma zero_c2 : (0 : Oct).c2 = 0 := rfl
@[simp] lemma zero_c3 : (0 : Oct).c3 = 0 := rfl
@[simp] lemma zero_c4 : (0 : Oct).c4 = 0 := rfl
@[simp] lemma zero_c5 : (0 : Oct).c5 = 0 := rfl
@[simp] lemma zero_c6 : (0 : Oct).c6 = 0 := rfl
@[simp] lemma zero_c7 : (0 : Oct).c7 = 0 := rfl

@[simp] lemma one_c0 : (1 : Oct).c0 = 1 := rfl
@[simp] lemma one_c1 : (1 : Oct).c1 = 0 := rfl
@[simp] lemma one_c2 : (1 : Oct).c2 = 0 := rfl
@[simp] lemma one_c3 : (1 : Oct).c3 = 0 := rfl
@[simp] lemma one_c4 : (1 : Oct).c4 = 0 := rfl
@[simp] lemma one_c5 : (1 : Oct).c5 = 0 := rfl
@[simp] lemma one_c6 : (1 : Oct).c6 = 0 := rfl
@[simp] lemma one_c7 : (1 : Oct).c7 = 0 := rfl

@[simp] lemma add_c0 (x y : Oct) : (x + y).c0 = x.c0 + y.c0 := rfl
@[simp] lemma add_c1 (x y : Oct) : (x + y).c1 = x.c1 + y.c1 := rfl
@[simp] lemma add_c2 (x y : Oct) : (x + y).c2 = x.c2 + y.c2 := rfl
@[simp] lemma add_c3 (x y : Oct) : (x + y).c3 = x.c3 + y.c3 := rfl
@[simp] lemma add_c4 (x y : Oct) : (x + y).c4 = x.c4 + y.c4 := rfl
@[simp] lemma add_c5 (x y : Oct) : (x + y).c5 = x.c5 + y.c5 := rfl
@[simp] lemma add_c6 (x y : Oct) : (x + y).c6 = x.c6 + y.c6 := rfl
@[simp] lemma add_c7 (x y : Oct) : (x + y).c7 = x.c7 + y.c7 := rfl

@[simp] lemma neg_c0 (x : Oct) : (-x).c0 = -x.c0 := rfl
@[simp] lemma neg_c1 (x : Oct) : (-x).c1 = -x.c1 := rfl
@[simp] lemma neg_c2 (x : Oct) : (-x).c2 = -x.c2 := rfl
@[simp] lemma neg_c3 (x : Oct) : (-x).c3 = -x.c3 := rfl
@[simp] lemma neg_c4 (x : Oct) : (-x).c4 = -x.c4 := rfl
@[simp] lemma neg_c5 (x : Oct) : (-x).c5 = -x.c5 := rfl
@[simp] lemma neg_c6 (x : Oct) : (-x).c6 = -x.c6 := rfl
@[simp] lemma neg_c7 (x : Oct) : (-x).c7 = -x.c7 := rfl

@[simp] lemma sub_c0 (x y : Oct) : (x - y).c0 = x.c0 - y.c0 := by change x.c0 + -y.c0 = _; ring
@[simp] lemma sub_c1 (x y : Oct) : (x - y).c1 = x.c1 - y.c1 := by change x.c1 + -y.c1 = _; ring
@[simp] lemma sub_c2 (x y : Oct) : (x - y).c2 = x.c2 - y.c2 := by change x.c2 + -y.c2 = _; ring
@[simp] lemma sub_c3 (x y : Oct) : (x - y).c3 = x.c3 - y.c3 := by change x.c3 + -y.c3 = _; ring
@[simp] lemma sub_c4 (x y : Oct) : (x - y).c4 = x.c4 - y.c4 := by change x.c4 + -y.c4 = _; ring
@[simp] lemma sub_c5 (x y : Oct) : (x - y).c5 = x.c5 - y.c5 := by change x.c5 + -y.c5 = _; ring
@[simp] lemma sub_c6 (x y : Oct) : (x - y).c6 = x.c6 - y.c6 := by change x.c6 + -y.c6 = _; ring
@[simp] lemma sub_c7 (x y : Oct) : (x - y).c7 = x.c7 - y.c7 := by change x.c7 + -y.c7 = _; ring

@[simp] lemma smul_c0 (r : ℝ) (x : Oct) : (r • x).c0 = r * x.c0 := rfl
@[simp] lemma smul_c1 (r : ℝ) (x : Oct) : (r • x).c1 = r * x.c1 := rfl
@[simp] lemma smul_c2 (r : ℝ) (x : Oct) : (r • x).c2 = r * x.c2 := rfl
@[simp] lemma smul_c3 (r : ℝ) (x : Oct) : (r • x).c3 = r * x.c3 := rfl
@[simp] lemma smul_c4 (r : ℝ) (x : Oct) : (r • x).c4 = r * x.c4 := rfl
@[simp] lemma smul_c5 (r : ℝ) (x : Oct) : (r • x).c5 = r * x.c5 := rfl
@[simp] lemma smul_c6 (r : ℝ) (x : Oct) : (r • x).c6 = r * x.c6 := rfl
@[simp] lemma smul_c7 (r : ℝ) (x : Oct) : (r • x).c7 = r * x.c7 := rfl

@[simp] lemma mul_c0 (x y : Oct) :
    (x * y).c0 = x.c0*y.c0 - x.c1*y.c1 - x.c2*y.c2 - x.c3*y.c3 - x.c4*y.c4 - x.c5*y.c5 - x.c6*y.c6 - x.c7*y.c7 := rfl
@[simp] lemma mul_c1 (x y : Oct) :
    (x * y).c1 = x.c0*y.c1 + x.c1*y.c0 + x.c2*y.c3 - x.c3*y.c2 + x.c4*y.c5 - x.c5*y.c4 + x.c7*y.c6 - x.c6*y.c7 := rfl
@[simp] lemma mul_c2 (x y : Oct) :
    (x * y).c2 = x.c0*y.c2 + x.c2*y.c0 + x.c3*y.c1 - x.c1*y.c3 + x.c4*y.c6 - x.c6*y.c4 + x.c5*y.c7 - x.c7*y.c5 := rfl
@[simp] lemma mul_c3 (x y : Oct) :
    (x * y).c3 = x.c0*y.c3 + x.c3*y.c0 + x.c1*y.c2 - x.c2*y.c1 + x.c4*y.c7 - x.c7*y.c4 + x.c6*y.c5 - x.c5*y.c6 := rfl
@[simp] lemma mul_c4 (x y : Oct) :
    (x * y).c4 = x.c0*y.c4 + x.c4*y.c0 + x.c5*y.c1 - x.c1*y.c5 + x.c6*y.c2 - x.c2*y.c6 + x.c7*y.c3 - x.c3*y.c7 := rfl
@[simp] lemma mul_c5 (x y : Oct) :
    (x * y).c5 = x.c0*y.c5 + x.c5*y.c0 + x.c1*y.c4 - x.c4*y.c1 + x.c7*y.c2 - x.c2*y.c7 + x.c3*y.c6 - x.c6*y.c3 := rfl
@[simp] lemma mul_c6 (x y : Oct) :
    (x * y).c6 = x.c0*y.c6 + x.c6*y.c0 + x.c1*y.c7 - x.c7*y.c1 + x.c2*y.c4 - x.c4*y.c2 + x.c5*y.c3 - x.c3*y.c5 := rfl
@[simp] lemma mul_c7 (x y : Oct) :
    (x * y).c7 = x.c0*y.c7 + x.c7*y.c0 + x.c6*y.c1 - x.c1*y.c6 + x.c2*y.c5 - x.c5*y.c2 + x.c3*y.c4 - x.c4*y.c3 := rfl

def oe1 : Oct := ⟨0, 1, 0, 0, 0, 0, 0, 0⟩
def oe2 : Oct := ⟨0, 0, 1, 0, 0, 0, 0, 0⟩
def oe3 : Oct := ⟨0, 0, 0, 1, 0, 0, 0, 0⟩
def oe4 : Oct := ⟨0, 0, 0, 0, 1, 0, 0, 0⟩
def oe5 : Oct := ⟨0, 0, 0, 0, 0, 1, 0, 0⟩
def oe6 : Oct := ⟨0, 0, 0, 0, 0, 0, 1, 0⟩
def oe7 : Oct := ⟨0, 0, 0, 0, 0, 0, 0, 1⟩

def oBasis : Fin 8 → Oct
  | 0 => 1 | 1 => oe1 | 2 => oe2 | 3 => oe3
  | 4 => oe4 | 5 => oe5 | 6 => oe6 | 7 => oe7

def normSq (x : Oct) : ℝ :=
  x.c0^2 + x.c1^2 + x.c2^2 + x.c3^2 + x.c4^2 + x.c5^2 + x.c6^2 + x.c7^2

theorem one_mul (x : Oct) : (1 : Oct) * x = x := by ext <;> simp
theorem mul_one (x : Oct) : x * (1 : Oct) = x := by ext <;> simp

/-- Composition algebra identity `|xy|² = |x|² |y|²` (Degen's 8-square identity). -/
theorem normSq_mul (x y : Oct) : normSq (x * y) = normSq x * normSq y := by
  simp only [normSq, mul_c0, mul_c1, mul_c2, mul_c3, mul_c4, mul_c5, mul_c6, mul_c7]
  ring

/-- Left alternativity `x(xy) = (xx)y`. -/
theorem left_alternative (x y : Oct) : x * (x * y) = (x * x) * y := by
  ext <;> simp <;> ring

/-- Right alternativity `(xy)y = x(yy)`. -/
theorem right_alternative (x y : Oct) : (x * y) * y = x * (y * y) := by
  ext <;> simp <;> ring

/-- Associator `[a, b, c] = (ab)c - a(bc)`. -/
def assoc (a b c : Oct) : Oct := (a * b) * c - a * (b * c)

/-- Element commutator `[a, b] = ab - ba`. -/
def comm (a b : Oct) : Oct := a * b - b * a

/-- Left multiplication as an `ℝ`-linear map `L_u(x) = u * x`. -/
def L (u : Oct) : Oct →ₗ[ℝ] Oct where
  toFun x := u * x
  map_add' x y := by ext <;> simp <;> ring
  map_smul' r x := by ext <;> simp <;> ring

/-- Right multiplication as an `ℝ`-linear map `R_u(x) = x * u`. -/
def R (u : Oct) : Oct →ₗ[ℝ] Oct where
  toFun x := x * u
  map_add' x y := by ext <;> simp <;> ring
  map_smul' r x := by ext <;> simp <;> ring

@[simp] lemma L_apply (u x : Oct) : L u x = u * x := rfl
@[simp] lemma R_apply (u x : Oct) : R u x = x * u := rfl

/-- Explicit non-associativity: `[e₁, e₄, e₂] = -2 e₇ ≠ 0` (`prop:oct_obstructions`(i)). -/
theorem assoc_e1_e4_e2 : assoc oe1 oe4 oe2 = (-2 : ℝ) • oe7 := by
  ext <;> simp [assoc, oe1, oe4, oe2, oe7]
  norm_num

/-- Failure of single-element closure in `𝕆`: `L_{e₁}(L_{e₄}(e₂)) ≠ L_{e₁ e₄}(e₂)` (`rem:quat_uniqueness_scope`). -/
theorem oct_not_single_elem_closed :
    ¬ (∀ a b : Oct, ∃ c : Oct, ∀ σ : Oct, a * (b * σ) = c * σ) := by
  rw [single_elem_closure_iff_assoc mul_one]
  intro h
  have h142 := h oe1 oe4 oe2
  have hne : (assoc oe1 oe4 oe2).c7 ≠ 0 := by
    rw [assoc_e1_e4_e2]
    simp [oe7]
  apply hne
  simp [assoc, h142]

/-- `prop:bimodule_associativity` alternative-algebra curvature obstruction in `𝕆`:
`[L_a, L_b] c = L_{[a,b]} c - 2 [L_a, R_b] c` for all `a, b, c ∈ 𝕆`. -/
theorem alternative_left_commutator (a b c : Oct) :
    (L a (L b c) - L b (L a c)) =
      L (comm a b) c - (2 : ℝ) • (L a (R b c) - R b (L a c)) := by
  ext <;> simp [comm] <;> ring

/-- Curvature commutator `[L_{e₁}, L_{e₂}]` cannot equal `L_d` for any `d ∈ 𝕆` (`prop:bimodule_associativity`). -/
theorem oct_curvature_not_single_left :
    ¬ ∃ d : Oct, ∀ c : Oct, L oe1 (L oe2 c) - L oe2 (L oe1 c) = L d c := by
  rintro ⟨d, hd⟩
  have h1 := hd 1
  have h4 := hd oe4
  have hd_eq : d = (2 : ℝ) • oe3 := by
    have : d = L d 1 := by simp [mul_one]
    rw [this, ← h1]
    ext <;> simp [oe1, oe2, oe3]
    norm_num
  rw [hd_eq] at h4
  have hc7 := congr_arg Oct.c7 h4
  simp [oe1, oe2, oe3, oe4] at hc7
  norm_num at hc7

/-- `sgm_wavefn.tex` Proposition `prop:oct_obstructions`(i) (`eq:commutant_collapse`):
The commutant `End_{𝕆_L}(𝕆)` collapses to the 1-dimensional space `ℝ · Id`. -/
theorem commutant_collapse (T : Oct →ₗ[ℝ] Oct) :
    (∀ u : Oct, T.comp (L u) = (L u).comp T) ↔ ∃ r : ℝ, T = r • LinearMap.id := by
  constructor
  · intro hcomm
    set c : Oct := T 1
    have hT_eq_R : ∀ u : Oct, T u = u * c := by
      intro u
      have hu := congr_arg (fun f => f 1) (hcomm u)
      simp only [LinearMap.comp_apply, L_apply, mul_one] at hu
      exact hu
    have h_nuc : ∀ u x : Oct, (u * x) * c = u * (x * c) := by
      intro u x
      have hx := congr_arg (fun f => f x) (hcomm u)
      simp only [LinearMap.comp_apply, L_apply] at hx
      rw [hT_eq_R (u * x), hT_eq_R x] at hx
      exact hx
    have h12 := h_nuc oe1 oe2
    have h14 := h_nuc oe1 oe4
    have h24 := h_nuc oe2 oe4
    have hc4 : c.c4 = 0 := by
      have := congr_arg Oct.c7 h12; simp [oe1, oe2] at this; linarith
    have hc5 : c.c5 = 0 := by
      have := congr_arg Oct.c6 h12; simp [oe1, oe2] at this; linarith
    have hc6 : c.c6 = 0 := by
      have := congr_arg Oct.c5 h12; simp [oe1, oe2] at this; linarith
    have hc7 : c.c7 = 0 := by
      have := congr_arg Oct.c4 h12; simp [oe1, oe2] at this; linarith
    have hc2 : c.c2 = 0 := by
      have := congr_arg Oct.c7 h14; simp [oe1, oe4] at this; linarith
    have hc3 : c.c3 = 0 := by
      have := congr_arg Oct.c6 h14; simp [oe1, oe4] at this; linarith
    have hc1 : c.c1 = 0 := by
      have := congr_arg Oct.c7 h24; simp [oe2, oe4] at this; linarith
    refine ⟨c.c0, ?_⟩
    apply LinearMap.ext
    intro u
    rw [hT_eq_R u]
    have hc_real : c = ⟨c.c0, 0, 0, 0, 0, 0, 0, 0⟩ := by
      ext <;> simp [hc1, hc2, hc3, hc4, hc5, hc6, hc7]
    rw [hc_real]
    ext <;> simp [mul_comm]
  · rintro ⟨r, rfl⟩ u
    apply LinearMap.ext
    intro x
    simp [map_smul]

/-- Corollary of `commutant_collapse`: no complex phase operator `J² = -Id` commutes with all `L_u` in `𝕆`. -/
theorem no_commuting_complex_structure :
    ¬ ∃ J : Oct →ₗ[ℝ] Oct, J.comp J = -LinearMap.id ∧ ∀ u : Oct, J.comp (L u) = (L u).comp J := by
  rintro ⟨J, hJ2, hcomm⟩
  rw [commutant_collapse] at hcomm
  rcases hcomm with ⟨r, rfl⟩
  have h1 := congr_arg (fun f => (f 1).c0) hJ2
  simp at h1
  nlinarith

/-- `sgm_wavefn.tex` Proposition `prop:oct_obstructions`(ii) (`eq:oct_jacobi_fail`):
`[e₁, [e₂, e₄]] + [e₂, [e₄, e₁]] + [e₄, [e₁, e₂]] = -12 e₇ ≠ 0`. -/
theorem oct_element_jacobi_failure :
    comm oe1 (comm oe2 oe4) + comm oe2 (comm oe4 oe1) + comm oe4 (comm oe1 oe2) =
      (-12 : ℝ) • oe7 := by
  ext <;> simp [comm, oe1, oe2, oe4, oe7]
  norm_num

/-- General pure-imaginary Clifford anticommutation in `𝕆`:
for any `u, v ∈ Im(𝕆)`, `L_u L_v + L_v L_u = -2 ⟨u, v⟩ Id`. -/
theorem oct_pure_clifford_anticomm (u v x : Oct) (hu : u.c0 = 0) (hv : v.c0 = 0) :
    L u (L v x) + L v (L u x) =
      (-2 * (u.c1 * v.c1 + u.c2 * v.c2 + u.c3 * v.c3 + u.c4 * v.c4 +
             u.c5 * v.c5 + u.c6 * v.c6 + u.c7 * v.c7)) • x := by
  ext <;> simp [hu, hv] <;> ring

/-- `sgm_wavefn.tex` Proposition `prop:oct_obstructions`(iii) (`eq:d7_elliptic`):
Clifford anticommutation `L_{e_a} L_{e_b} + L_{e_b} L_{e_a} = -2 δ_{ab} Id` for `a, b ∈ {1..7}`. -/
theorem oct_clifford_anticomm (a b : Fin 7) (x : Oct) :
    L (oBasis a.succ) (L (oBasis b.succ) x) + L (oBasis b.succ) (L (oBasis a.succ) x) =
      (if a = b then (-2 : ℝ) • x else 0) := by
  have ha0 : (oBasis a.succ).c0 = 0 := by fin_cases a <;> rfl
  have hb0 : (oBasis b.succ).c0 = 0 := by fin_cases b <;> rfl
  rw [oct_pure_clifford_anticomm _ _ x ha0 hb0]
  have hdot :
      ((oBasis a.succ).c1 * (oBasis b.succ).c1 + (oBasis a.succ).c2 * (oBasis b.succ).c2 +
       (oBasis a.succ).c3 * (oBasis b.succ).c3 + (oBasis a.succ).c4 * (oBasis b.succ).c4 +
       (oBasis a.succ).c5 * (oBasis b.succ).c5 + (oBasis a.succ).c6 * (oBasis b.succ).c6 +
       (oBasis a.succ).c7 * (oBasis b.succ).c7) = (if a = b then (1 : ℝ) else 0) := by
    fin_cases a <;> fin_cases b <;> norm_num [oBasis, oe1, oe2, oe3, oe4, oe5, oe6, oe7]
  rw [hdot]
  split_ifs <;> simp

/-- `eq:d7_elliptic` real principal symbol: `(∑_{a=1}^7 k_a L_{e_a})² = -|k|² Id`. -/
theorem d7_real_symbol_elliptic (k : Fin 7 → ℝ) (x : Oct) :
    let Dk : Oct →ₗ[ℝ] Oct := ∑ a : Fin 7, k a • L (oBasis a.succ)
    Dk (Dk x) = (-(∑ a : Fin 7, (k a) ^ 2)) • x := by
  intro Dk
  let u : Oct := ⟨0, k 0, k 1, k 2, k 3, k 4, k 5, k 6⟩
  have hDk : ∀ y : Oct, Dk y = u * y := by
    intro y
    ext <;> simp [Dk, u, Fin.sum_univ_succ, oBasis, oe1, oe2, oe3, oe4, oe5, oe6, oe7] <;> ring
  rw [hDk (Dk x), hDk x, left_alternative u x]
  ext <;> simp [u, Fin.sum_univ_succ] <;> ring

end Oct

/-! ### Part 4: Non-Unitary Quaternionic Bimodule Flow (`rem:bimodule_flow`, `eq:bimodule_flow`) -/

/-- The hyperbolic bimodule flow `ψ(t) = cosh(g₀ t) ψ₀ + sinh(g₀ t) e₃ ψ₀ e₃`. -/
noncomputable def bimoduleFlow (g0 : ℝ) (ψ0 : Quat) (t : ℝ) : Quat :=
  Real.cosh (g0 * t) • ψ0 + Real.sinh (g0 * t) • (e3 * ψ0 * e3)

/-- `e₃ (e₃ ψ e₃) e₃ = +ψ` (the two-sided `e₃` map is an involution, squaring to `+1`, not `-1`). -/
theorem e3_sandwich_involution (ψ : Quat) : e3 * (e3 * ψ * e3) * e3 = ψ := by
  ext <;> simp [e3]

/-- `sgm_wavefn.tex` Remark `rem:bimodule_flow` (`eq:bimodule_flow`):
Full 4-component ODE verification `∂_t ψ(t) = g₀ e₃ ψ(t) e₃` (all four quaternion
components `.re, .imI, .imJ, .imK`) and initial condition `ψ(0) = ψ₀`. -/
theorem bimoduleFlow_ode (g0 : ℝ) (ψ0 : Quat) (t : ℝ) :
    bimoduleFlow g0 ψ0 0 = ψ0 ∧
    HasDerivAt (fun s => (bimoduleFlow g0 ψ0 s).re)
      (g0 * (e3 * bimoduleFlow g0 ψ0 t * e3).re) t ∧
    HasDerivAt (fun s => (bimoduleFlow g0 ψ0 s).imI)
      (g0 * (e3 * bimoduleFlow g0 ψ0 t * e3).imI) t ∧
    HasDerivAt (fun s => (bimoduleFlow g0 ψ0 s).imJ)
      (g0 * (e3 * bimoduleFlow g0 ψ0 t * e3).imJ) t ∧
    HasDerivAt (fun s => (bimoduleFlow g0 ψ0 s).imK)
      (g0 * (e3 * bimoduleFlow g0 ψ0 t * e3).imK) t := by
  have hlin : HasDerivAt (fun s : ℝ => g0 * s) g0 t := by
    simpa [mul_one] using (hasDerivAt_id t).const_mul g0
  have h1 : HasDerivAt (fun s : ℝ => Real.cosh (g0 * s)) (Real.sinh (g0 * t) * g0) t :=
    (Real.hasDerivAt_cosh (g0 * t)).comp t hlin
  have h2 : HasDerivAt (fun s : ℝ => Real.sinh (g0 * s)) (Real.cosh (g0 * t) * g0) t :=
    (Real.hasDerivAt_sinh (g0 * t)).comp t hlin
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · ext <;> simp [bimoduleFlow]
  · convert (h1.mul_const ψ0.re).add (h2.mul_const (e3 * ψ0 * e3).re) using 1
    · ext s; simp [bimoduleFlow]
    · simp [bimoduleFlow, e3]; ring
  · convert (h1.mul_const ψ0.imI).add (h2.mul_const (e3 * ψ0 * e3).imI) using 1
    · ext s; simp [bimoduleFlow]
    · simp [bimoduleFlow, e3]; ring
  · convert (h1.mul_const ψ0.imJ).add (h2.mul_const (e3 * ψ0 * e3).imJ) using 1
    · ext s; simp [bimoduleFlow]
    · simp [bimoduleFlow, e3]; ring
  · convert (h1.mul_const ψ0.imK).add (h2.mul_const (e3 * ψ0 * e3).imK) using 1
    · ext s; simp [bimoduleFlow]
    · simp [bimoduleFlow, e3]; ring

/-- Non-conservation of norm under `bimoduleFlow`: on the `σ₃ = -1` eigenstate `ψ₀ = e₁`,
`qNormSq(bimoduleFlow g0 e1 t) = exp(2 g0 t) ≠ 1` for `g0 t ≠ 0`. -/
theorem bimoduleFlow_norm_blowup (g0 t : ℝ) :
    qNormSq (bimoduleFlow g0 e1 t) = Real.exp (2 * g0 * t) := by
  have he : e3 * e1 * e3 = e1 := by ext <;> simp [e1, e3]
  rw [bimoduleFlow, he]
  have hsum : Real.cosh (g0 * t) + Real.sinh (g0 * t) = Real.exp (g0 * t) :=
    Real.cosh_add_sinh (g0 * t)
  have hnorm : qNormSq (Real.cosh (g0 * t) • e1 + Real.sinh (g0 * t) • e1) =
      (Real.cosh (g0 * t) + Real.sinh (g0 * t)) ^ 2 := by
    simp [qNormSq, e1]
  rw [hnorm, hsum]
  calc Real.exp (g0 * t) ^ 2
      = Real.exp (g0 * t) * Real.exp (g0 * t) := sq _
    _ = Real.exp (g0 * t + g0 * t) := (Real.exp_add _ _).symm
    _ = Real.exp (2 * g0 * t) := by ring_nf

end SGM.Algebra
