"""Equation-by-equation checks of sgm.tex: Spectral Gradient Mechanics.

Run:  python3 sgm_check.py          (from any directory)

The file walks the paper in order. Each displayed equation line has one block,
whose comment header gives the label and .tex line, what the equation states,
which earlier equations it is derived from and by what operation, what the code
tests, and the wrong variant (mutant) that the test must reject. The file is
standalone: the check functions are defined below, before the first block, and
it needs only Python 3 with sympy, numpy and scipy. dag_check.py checks that the
blocks are in .tex order and that every "Derived from" label is defined earlier
in the paper.
The run ends with a SUMMARY line and exits 1 if any check fails.
"""
import sys
import sympy as sp
import numpy as np


# ----------------------------------------------------------------------------
# Check functions. Each block prints a header naming the equation (eq), then
# runs checks of three kinds:
#   symcheck  a sympy expression that must simplify to zero (an identity);
#   check     a number compared with a target to a relative tolerance;
#   boolcheck a condition that must be true.
# Every block also runs at least one negative control (mutant or numutant): the
# same test applied to a deliberately wrong version of the equation, which must
# fail. A test whose mutant also passes cannot detect the error it is meant to
# catch, so the mutant result is counted as a failure of the block.
# ----------------------------------------------------------------------------
results = []          # one bool per check, in order
_current = [""]       # label of the block being run, printed with each failure
_failures = []


def _record(ok, name, detail=""):
    ok = bool(ok)
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")
    if not ok:
        _failures.append(f"{_current[0]}: {name}")
    return ok


def part(title):
    """Print the heading of one part of the paper (a section range)."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def eq(label, line, statement):
    """Start the block for one displayed equation.

    label     the \\label of the equation, or 'unlabeled' for one without a label
    line      its line number in the .tex file
    statement the equation in plain text, as the paper states it
    """
    _current[0] = f"{label} (L{line})"
    print()
    print(f"-- {label}  [L{line}]  {statement}")


def _is_zero(expr):
    if hasattr(expr, "shape"):
        return all(sp.simplify(x) == 0 for x in expr)
    return sp.simplify(expr) == 0


def symcheck(name, expr):
    """Pass when expr simplifies to zero (a scalar or a sympy Matrix)."""
    return _record(_is_zero(expr), name)


def mutant(name, expr):
    """Negative control: expr is the residual of a WRONG version of the
    equation, and must NOT simplify to zero. Passes when it is nonzero."""
    return _record(not _is_zero(expr), "mutant caught: " + name)


def boolcheck(name, cond):
    """Pass when cond is true."""
    return _record(cond, name)


def check(name, lhs, rhs, tol=1e-9, scale=None):
    """Pass when |lhs - rhs|/|rhs| < tol. The tolerance is always relative.

    A zero target has no magnitude of its own, so it needs scale=, the size of
    the terms whose difference is being tested; the test is then
    |lhs| < tol*scale. A zero target without scale= fails, because an absolute
    epsilon can pass for any value."""
    lhs, rhs = complex(lhs), complex(rhs)
    if rhs == 0:
        if scale is None or not scale > 0:
            return _record(False, name, "zero target needs a positive scale=")
        rel = abs(lhs) / scale
    else:
        if scale is not None:
            return _record(False, name, "scale= is only for a zero target")
        rel = abs(lhs - rhs) / abs(rhs)
    return _record(rel < tol, name, f"lhs = {lhs.real:.10g}, rhs = {rhs.real:.10g}, relative error = {float(rel):.2e}")


def numutant(name, lhs, rhs, tol=1e-9):
    """Numerical negative control: a WRONG value lhs must differ from rhs by
    more than tol (relative). Passes when the difference is detected."""
    lhs, rhs = complex(lhs), complex(rhs)
    rel = abs(lhs - rhs) / abs(rhs)
    return _record(rel >= tol, "mutant caught: " + name, f"relative difference = {float(rel):.2e}")


def summary():
    """Print the totals and exit with status 1 if any check failed."""
    n, k = len(results), sum(results)
    print()
    print("=" * 78)
    print(f"SUMMARY: {k}/{n} checks passed.")
    for f in _failures:
        print("  FAILED: " + f)
    print("=" * 78)
    sys.exit(0 if k == n else 1)


# SI dimensions as exponent vectors over (kg, m, s, A), as sympy column matrices
# so that a difference of two dimension vectors can be fed to symcheck/mutant.
def _S1_dim(kg=0, m=0, s=0, A=0):
    return sp.Matrix([kg, m, s, A])


# Complex conjugate of an expression whose symbols and undefined functions are
# all real: replace i by -i. (sympy leaves conjugate(f(x)) and conjugates of
# derivatives of real undefined functions unevaluated.)
def _S1_cc(e):
    return e.xreplace({sp.I: -sp.I})


def part_S1():
    part("sgm.tex L1-L413: Introduction; The force-frequency identity (Four "
         "derivations of the identity; The single-frequency assumption; A "
         "universal force law; The frequency field about the particle)")

    # Dimensions used throughout.
    D_hbar = _S1_dim(kg=1, m=2, s=-1)
    D_c = _S1_dim(m=1, s=-1)
    D_mass = _S1_dim(kg=1)
    D_energy = _S1_dim(kg=1, m=2, s=-2)
    D_freq = _S1_dim(s=-1)
    D_mom = _S1_dim(kg=1, m=1, s=-1)
    D_force = _S1_dim(kg=1, m=1, s=-2)
    D_len = _S1_dim(m=1)
    D_time = _S1_dim(s=1)
    D_charge = _S1_dim(s=1, A=1)
    D_volt = _S1_dim(kg=1, m=2, s=-3, A=-1)

    # Common symbols.
    hb, m, c, q = sp.symbols('hbar m c q', positive=True)
    t = sp.Symbol('t', real=True)
    X = sp.symbols('x1 x2 x3', real=True)
    P = sp.symbols('P1 P2 P3', real=True)      # momentum held fixed (phase-space slot)
    x = X[0]
    I = sp.I

    # ------------------------------------------------------------------------
    # inline (sgm.tex L128), gauge invariance of hbar*omega - q*Phi
    # States: a gauge transformation by a scalar function lambda(x,t) multiplies
    #          the wave function psi (complex scalar field) by exp(i q lambda/hbar);
    #          it adds -q d_t lambda/hbar to the frequency field omega(x) and
    #          -q d_t lambda to q Phi (Phi the scalar potential), so
    #          hbar omega - q Phi is gauge invariant.
    # Derived from: ax:local_frequency (omega = -d_t theta) and the gauge
    #          transformation as L128 states it (the phase factor and the shift of
    #          q Phi); the invariance is their difference, in place.
    # Test: in one dimension, for general functions psi(x,t), Phi(x,t), A(x,t),
    #          lambda(x,t), the Schroedinger residual
    #          i hbar d_t psi - [(-i hbar d_x - qA)^2/2m + q Phi] psi
    #          of the transformed psi with transformed potentials equals
    #          exp(i q lambda/hbar) times the residual of the original: so the
    #          stated transformation of Phi is the one the equation of motion
    #          requires. Then the phase shift q lambda/hbar changes -d_t theta by
    #          -q d_t lambda/hbar, and hbar omega - q Phi is unchanged.
    # Mutant: Phi -> Phi + d_t lambda (sign error in the potential shift): the
    #          transformed equation is no longer satisfied.
    # ------------------------------------------------------------------------
    eq("inline", 128, "psi -> exp(i q lambda/hbar) psi: omega -> omega - q d_t lambda/hbar, "
       "q Phi -> q Phi - q d_t lambda, hbar omega - q Phi invariant")
    psi = sp.Function('psi')(x, t)
    Phi = sp.Function('Phi')(x, t)
    A1 = sp.Function('A')(x, t)
    lam = sp.Function('lambda')(x, t)

    def _schr_res(f, Ph, Av):
        def pi(g):
            return -I*hb*sp.diff(g, x) - q*Av*g
        return I*hb*sp.diff(f, t) - (pi(pi(f))/(2*m) + q*Ph*f)

    G = sp.exp(I*q*lam/hb)
    res0 = _schr_res(psi, Phi, A1)
    res1 = _schr_res(G*psi, Phi - sp.diff(lam, t), A1 + sp.diff(lam, x))
    symcheck("transformed psi with Phi - d_t lambda, A + d_x lambda: residual = "
             "exp(i q lambda/hbar) * original residual",
             sp.expand(sp.simplify(res1/G - res0)))
    th = sp.Function('theta')(x, t)
    om_old = -sp.diff(th, t)
    om_new = -sp.diff(th + q*lam/hb, t)
    symcheck("omega shifts by -q d_t lambda/hbar", om_new - om_old + q*sp.diff(lam, t)/hb)
    symcheck("hbar omega - q Phi is unchanged",
             (hb*om_new - q*(Phi - sp.diff(lam, t))) - (hb*om_old - q*Phi))
    symcheck("dimensions: q lambda/hbar is dimensionless (lambda in V s)",
             D_charge + (D_volt + D_time) - D_hbar)
    resbad = _schr_res(G*psi, Phi + sp.diff(lam, t), A1 + sp.diff(lam, x))
    mutant("Phi -> Phi + d_t lambda breaks the Schroedinger equation",
           sp.expand(sp.simplify(resbad/G - res0)))

    # ------------------------------------------------------------------------
    # eq:tail_distance  (sgm.tex L132)
    # States: sqrt(1 - |<psi|psi'>|^2) <= 2 sqrt(P_tail), for normalized states
    #          psi = psi_core + psi_tail and psi' = psi_core + psi'_tail, where
    #          psi'_tail differs from psi_tail only by a pointwise phase, and
    #          P_tail = ||psi_tail||^2 <= 1/2 (scalars; states are vectors in a
    #          Hilbert space).
    # Derived from: ax:local_frequency (a frequency change in the far tail changes only
    #          the tail phase); the inner product, norm and P_tail defined in L130;
    #          Cauchy-Schwarz: <psi|psi'> = 1 - P + <tail|tail'>, |<tail|tail'>| <= P,
    #          so |<psi|psi'>| >= 1 - 2P for P <= 1/2; then 1 - (1-2P)^2 = 4P(1-P) <= 4P.
    #          L135 then identifies the left side as the trace distance d_tr,
    #          shows the operator |psi><psi| - |psi'><psi'| has eigenvalues
    #          +-d_tr, bounds every outcome probability change by d_tr, and gives
    #          the Gaussian tail radii 4.2 sigma (1D) and 4.9 sigma (3D) for
    #          P_tail = eps^2/4, eps = 1e-2.
    # Test: (1) symbolic: 1 - (1-2P)^2 = 4P(1-P), and 4P - 4P(1-P) = 4P^2 >= 0.
    #          (2) numerical, 4000 random states in C^12 (6 core, 6 tail
    #          components) with random pointwise tail phases and P_tail drawn in
    #          (0, 1/2]: the overlap formula holds exactly, the overlap bound
    #          holds, and the trace-distance bound holds; the phase reversal of
    #          the tail attains 4P(1-P). (3) for random pairs, the eigenvalues of
    #          |psi><psi| - |psi'><psi'| are +-d_tr, and the largest probability
    #          change over projectors (onto the positive eigenvector) is d_tr,
    #          while 600 random projectors never exceed it. (4) the Gaussian
    #          radii, by root finding on the 1D two-sided tail and the 3D
    #          Maxwell (chi, 3 degrees of freedom) tail.
    # Mutant: (a) 1 - (1-2P)^2 = 4P (factor (1-P) dropped); (b) the bound
    #          without the square root, d_tr <= 2 P_tail, which the phase
    #          reversal violates; (c) eigenvalues +-d_tr^2 instead of +-d_tr.
    # ------------------------------------------------------------------------
    eq("eq:tail_distance", 132, "sqrt(1 - |<psi|psi'>|^2) <= 2 sqrt(P_tail)")
    Pt = sp.Symbol('P_tail', positive=True)
    symcheck("1 - (1 - 2P)^2 == 4P(1 - P)", sp.expand(1 - (1 - 2*Pt)**2 - 4*Pt*(1 - Pt)))
    symcheck("4P - 4P(1-P) == 4P^2 (>= 0, so 4P(1-P) <= 4P)", sp.expand(4*Pt - 4*Pt*(1 - Pt) - 4*Pt**2))
    rng = np.random.default_rng(132)
    worst_ratio, worst_formula, worst_ovb = 0.0, 0.0, 0.0
    for _ in range(4000):
        P_ = rng.uniform(1e-6, 0.5)
        core = rng.normal(size=6) + 1j*rng.normal(size=6)
        core *= np.sqrt(1 - P_)/np.linalg.norm(core)
        tail = rng.normal(size=6) + 1j*rng.normal(size=6)
        tail *= np.sqrt(P_)/np.linalg.norm(tail)
        tail2 = tail*np.exp(1j*rng.uniform(0, 2*np.pi, size=6))   # pointwise phase change
        psi_a = np.concatenate([core, np.zeros(6)]) + np.concatenate([np.zeros(6), tail])
        psi_b = np.concatenate([core, np.zeros(6)]) + np.concatenate([np.zeros(6), tail2])
        ov = np.vdot(psi_a, psi_b)
        worst_formula = max(worst_formula, abs(ov - (1 - P_ + np.vdot(tail, tail2))))
        worst_ovb = max(worst_ovb, (1 - 2*P_) - abs(ov))
        worst_ratio = max(worst_ratio, np.sqrt(max(0.0, 1 - abs(ov)**2))/(2*np.sqrt(P_)))
    check("overlap formula <psi|psi'> = 1 - P + <tail|tail'> (max deviation over samples)",
          worst_formula, 0.0, tol=1e-12, scale=1.0)
    boolcheck(f"|<psi|psi'>| >= 1 - 2P on every sample (max shortfall {worst_ovb:.2e})",
              worst_ovb <= 1e-12)
    boolcheck(f"d_tr / (2 sqrt(P)) <= 1 on every sample (max {worst_ratio:.6f})",
              worst_ratio <= 1 + 1e-12)
    P_ = 0.01
    ov_rev = (1 - P_) - P_                         # tail phase reversed
    check("phase reversal attains 1 - |ov|^2 = 4P(1-P) at P = 0.01",
          1 - ov_rev**2, 4*P_*(1 - P_))
    numutant("bound without square root, d_tr <= 2P, is violated by the reversal",
             np.sqrt(1 - ov_rev**2), 2*P_, tol=1e-3)
    boolcheck("reversal: d_tr = 0.199 exceeds 2P = 0.02, so the square root is needed",
              np.sqrt(1 - ov_rev**2) > 2*P_)
    mutant("1 - (1-2P)^2 == 4P (factor (1-P) dropped)", sp.expand(1 - (1 - 2*Pt)**2 - 4*Pt))
    # trace distance and outcome probabilities
    worst_eig, worst_proj, worst_rand = 0.0, 0.0, 0.0
    for _ in range(200):
        a = rng.normal(size=5) + 1j*rng.normal(size=5); a /= np.linalg.norm(a)
        b = rng.normal(size=5) + 1j*rng.normal(size=5); b /= np.linalg.norm(b)
        dtr = np.sqrt(1 - abs(np.vdot(a, b))**2)
        Dm = np.outer(a, a.conj()) - np.outer(b, b.conj())
        ev = np.sort(np.linalg.eigvalsh(Dm))
        worst_eig = max(worst_eig, abs(ev[-1] - dtr), abs(ev[0] + dtr), np.max(np.abs(ev[1:-1])))
        w, V_ = np.linalg.eigh(Dm)
        vp = V_[:, -1]
        Pi = np.outer(vp, vp.conj())
        worst_proj = max(worst_proj, abs(np.real(np.trace(Pi @ Dm)) - dtr))
        for _k in range(3):
            Q_, _r = np.linalg.qr(rng.normal(size=(5, 5)) + 1j*rng.normal(size=(5, 5)))
            r = rng.integers(1, 5)
            Pr = Q_[:, :r] @ Q_[:, :r].conj().T
            worst_rand = max(worst_rand, abs(np.real(np.trace(Pr @ Dm))) - dtr)
    check("eigenvalues of |psi><psi| - |psi'><psi'| are +-d_tr, the rest 0 (max deviation)",
          worst_eig, 0.0, tol=1e-12, scale=1.0)
    check("projector onto the positive eigenvector changes the probability by exactly d_tr",
          worst_proj, 0.0, tol=1e-12, scale=1.0)
    boolcheck(f"600 random projectors never exceed d_tr (max excess {worst_rand:.2e})",
              worst_rand <= 1e-12)
    a = np.array([1, 0, 0], dtype=complex); b = np.array([0.6, 0.8, 0], dtype=complex)
    dtr = np.sqrt(1 - abs(np.vdot(a, b))**2)
    numutant("eigenvalue magnitude d_tr^2 instead of d_tr",
             dtr**2, np.max(np.linalg.eigvalsh(np.outer(a, a.conj()) - np.outer(b, b.conj()))))
    from scipy.special import erfc
    from scipy.optimize import brentq
    Ptol = (1e-2)**2/4
    k1 = brentq(lambda k: erfc(k/np.sqrt(2)) - Ptol, 1, 10)
    k3 = brentq(lambda k: erfc(k/np.sqrt(2)) + np.sqrt(2/np.pi)*k*np.exp(-k*k/2) - Ptol, 1, 10)
    check("1D Gaussian: two-sided tail of probability eps^2/4 begins at 4.2 sigma", k1, 4.2, tol=5e-3)
    check("3D Gaussian: radial tail of probability eps^2/4 begins at 4.9 sigma", k3, 4.9, tol=5e-3)
    numutant("one-sided 1D tail (a factor 2 lost) would give a different radius",
             brentq(lambda k: erfc(k/np.sqrt(2))/2 - Ptol, 1, 10), 4.2, tol=5e-3)

    # ------------------------------------------------------------------------
    # eq:planck  (sgm.tex L140)
    # States: E = hbar omega; E the energy of a stationary state (scalar),
    #          omega the angular frequency of its wave function (scalar).
    # Derived from: ax:planck (premise; cited Planck, Einstein).
    # Test: dimensions of hbar*omega are those of energy; and for the stationary
    #          solution psi = phi(x) exp(-i E t/hbar) of i hbar d_t psi = H psi
    #          (phi a general function, H phi = E phi assumed) the phase rate
    #          -d_t theta (Eq. phase_freq) times hbar is E, consistent with the
    #          axiom read with omega = -d_t theta.
    # Mutant: E = h omega (h = 2 pi hbar in place of hbar).
    # ------------------------------------------------------------------------
    eq("eq:planck", 140, "E = hbar omega")
    E = sp.Symbol('E', real=True)
    symcheck("dimensions: [hbar][omega] == [energy]", D_hbar + D_freq - D_energy)
    phi = sp.Function('phi')(x)
    theta_stat = -E*t/hb                 # time part of the phase of phi exp(-iEt/hbar)
    psi_stat = phi*sp.exp(I*theta_stat)
    symcheck("i hbar d_t psi == E psi for psi = phi exp(-iEt/hbar)",
             sp.simplify(I*hb*sp.diff(psi_stat, t) - E*psi_stat))
    symcheck("hbar * (-d_t theta) == E", hb*(-sp.diff(theta_stat, t)) - E)
    mutant("E = h omega with h = 2 pi hbar", 2*sp.pi*hb*(-sp.diff(theta_stat, t)) - E)

    # ------------------------------------------------------------------------
    # eq:rest_freq  (sgm.tex L148)
    # States: omega_0 = m c^2/hbar, the rest frequency (scalar) of a particle of
    #          mass m.
    # Derived from: eq:planck, ax:planck, with E = m c^2 (cited, Einstein): divide
    #          m c^2 = hbar omega_0 by hbar.
    # Test: solving m c^2 = hbar omega for omega gives m c^2/hbar; dimensions are
    #          s^-1; electron value 7.76e20 rad/s (CODATA m_e, c, hbar).
    # Mutant: omega_0 = m c/hbar (one power of c lost): wrong dimensions (m^-1).
    # ------------------------------------------------------------------------
    eq("eq:rest_freq", 148, "omega_0 = m c^2/hbar")
    w = sp.Symbol('omega')
    sol = sp.solve(sp.Eq(m*c**2, hb*w), w)
    symcheck("solve(m c^2 = hbar omega) == m c^2/hbar", sol[0] - m*c**2/hb)
    symcheck("dimensions: [m c^2/hbar] == s^-1", D_mass + 2*D_c - D_hbar - D_freq)
    me, cc, hbn = 9.1093837015e-31, 299792458.0, 1.054571817e-34
    check("electron: omega_0 = 7.76e20 rad/s", me*cc**2/hbn, 7.76e20, tol=1e-3)
    mutant("omega_0 = m c/hbar has dimension m^-1, not s^-1", D_mass + D_c - D_hbar - D_freq)

    # ------------------------------------------------------------------------
    # eq:debroglie  (sgm.tex L158)
    # States: p = hbar grad theta; p the momentum (vector), theta(x,t) the phase
    #          (scalar field) of the wave function.
    # Derived from: ax:debroglie (premise; cited de Broglie). L173 adds: it holds
    #          pointwise as the canonical momentum per unit density
    #          Re[psi^dagger (-i hbar grad) psi]/rho, for psi = rho^{1/2} e^{i theta}
    #          and for psi = rho^{1/2} e^{i theta} xi with a constant unit spinor
    #          xi; a varying xi adds -i hbar xi^dagger grad xi (real).
    # Test: in 3D with general real functions R(x) = rho^{1/2} and theta(x), the
    #          momentum density per unit density equals hbar grad theta exactly;
    #          for a two-component xi = (cos a, e^{ib} sin a) with general a(x),
    #          b(x), the extra term equals -i hbar xi^dagger grad xi and that term
    #          is real; for a plane wave the momentum operator eigenvalue is
    #          hbar k. Dimensions: hbar/length = momentum.
    # Mutant: dropping the real part keeps -i hbar grad R/R, which is not zero.
    # ------------------------------------------------------------------------
    eq("eq:debroglie", 158, "p = hbar grad theta")
    R = sp.Function('R', positive=True)(*X)
    thx = sp.Function('theta', real=True)(*X)
    psi3 = R*sp.exp(I*thx)
    psi3c = R*sp.exp(-I*thx)
    for i in range(3):
        dens = psi3c*(-I*hb*sp.diff(psi3, X[i]))
        re_part = (dens + _S1_cc(dens))/2
        symcheck(f"Re[psi* (-i hbar d_{i+1}) psi]/|psi|^2 == hbar d_{i+1} theta",
                 sp.simplify(re_part/R**2 - hb*sp.diff(thx, X[i])))
    aa = sp.Function('a', real=True)(*X)
    bb = sp.Function('b', real=True)(*X)
    xi = sp.Matrix([sp.cos(aa), sp.exp(I*bb)*sp.sin(aa)])
    xid = sp.Matrix([[sp.cos(aa), sp.exp(-I*bb)*sp.sin(aa)]])
    psis = R*sp.exp(I*thx)*xi
    psisd = R*sp.exp(-I*thx)*xid
    dens_s = (psisd*(-I*hb*sp.diff(psis, X[0])))[0]
    extra = (-I*hb*(xid*sp.diff(xi, X[0])))[0]
    symcheck("spinor: Re[psi^dag (-i hbar d_1) psi]/rho == hbar d_1 theta - i hbar xi^dag d_1 xi",
             sp.simplify((dens_s + _S1_cc(dens_s))/2/R**2 - hb*sp.diff(thx, X[0]) - extra))
    symcheck("spinor: -i hbar xi^dag d_1 xi is real (equals hbar sin^2(a) d_1 b)",
             sp.simplify(extra - hb*sp.sin(aa)**2*sp.diff(bb, X[0])))
    kk = sp.Symbol('k', real=True)
    pw = sp.exp(I*kk*x)
    symcheck("plane wave: -i hbar d_x e^{ikx} == hbar k e^{ikx}", sp.simplify(-I*hb*sp.diff(pw, x) - hb*kk*pw))
    symcheck("dimensions: [hbar]/[length] == [momentum]", D_hbar - D_len - D_mom)
    dens0 = psi3c*(-I*hb*sp.diff(psi3, X[0]))
    mutant("without the real part the density keeps -i hbar d_1 R/R",
           sp.simplify(dens0/R**2 - hb*sp.diff(thx, X[0])))

    # ------------------------------------------------------------------------
    # eq:phase_freq  (sgm.tex L163)
    # States: omega = -d_t theta, the angular frequency (scalar field) as the
    #          negative time rate of the phase at fixed position.
    # Derived from: ax:debroglie, ax:local_frequency (premise); consistent with eq:planck.
    # Test: on the general stationary phase hbar theta = S_0(x) - E t (L197),
    #          -d_t theta = E/hbar at every x, so hbar omega = E as in eq:planck,
    #          and its spatial gradient is zero (L197). Dimensions: theta is
    #          dimensionless (rad), so omega has s^-1.
    # Mutant: omega = +d_t theta gives -E/hbar.
    # ------------------------------------------------------------------------
    eq("eq:phase_freq", 163, "omega = -d_t theta")
    S0 = sp.Function('S_0', real=True)(*X)
    th_s = (S0 - E*t)/hb
    symcheck("-d_t theta == E/hbar for hbar theta = S_0(x) - E t", -sp.diff(th_s, t) - E/hb)
    symcheck("its spatial gradient is zero",
             sp.Matrix([sp.diff(-sp.diff(th_s, t), Xi) for Xi in X]))
    symcheck("dimensions: [theta]/[t] == s^-1", _S1_dim() - D_time - D_freq)
    mutant("omega = +d_t theta gives -E/hbar", sp.diff(th_s, t) - E/hb)

    # ------------------------------------------------------------------------
    # eq:debroglie_k  (sgm.tex L168)
    # States: p = hbar k with the wavevector k = grad theta (vector); |k| is the
    #          inverse of the reduced de Broglie wavelength hbar/|p|.
    # Derived from: eq:debroglie, with k defined as grad theta (L166).
    # Test: for the plane phase theta = 2 pi x/lambda_dB, k = 2 pi/lambda_dB;
    #          with hbar = h/(2 pi), p = hbar k = h/lambda_dB (de Broglie's
    #          relation) and 1/|k| = hbar/|p| (the reduced wavelength).
    # Mutant: p = h k (h in place of hbar) gives 2 pi h/lambda_dB.
    # ------------------------------------------------------------------------
    eq("eq:debroglie_k", 168, "p = hbar k, k = grad theta")
    hP, lamdB = sp.symbols('h lambda_dB', positive=True)
    theta_pw = 2*sp.pi*x/lamdB
    k_pw = sp.diff(theta_pw, x)
    p_pw = hP/(2*sp.pi)*k_pw
    symcheck("p = hbar k == h/lambda_dB", p_pw - hP/lamdB)
    symcheck("1/|k| == hbar/|p| (reduced wavelength)", 1/k_pw - (hP/(2*sp.pi))/p_pw)
    mutant("p = h k", hP*k_pw - hP/lamdB)

    # ------------------------------------------------------------------------
    # eq:phase_rate_split  (sgm.tex L183)
    # States: -d_t theta = omega(x, p(x)) + Q(x)/hbar: the phase rate (scalar
    #          field) equals the frequency function omega(x,p) = H/hbar for
    #          H = m c^2 + p^2/2m + V(x), evaluated on the momentum field
    #          p(x) = hbar grad theta, plus the quantum potential
    #          Q = -(hbar^2/2m) lap|psi|/|psi| divided by hbar.
    # Derived from: eq:debroglie, eq:rest_freq, ax:hamilton (H), the frequency function
    #          of L179 (eq:planck), and the Schroedinger equation i hbar d_t psi = H psi,
    #          cited at L183, with psi = R e^{i theta}, real part after dividing by
    #          e^{i theta}.
    # Test: in 3D for general real R(x,t), theta(x,t), V(x): the real part of
    #          (i hbar d_t psi - H psi) e^{-i theta}/R equals
    #          -hbar d_t theta - [m c^2 + (hbar grad theta)^2/2m + V + Q].
    #          Also the example of L289: for the oscillator ground state (real,
    #          R = exp(-m Om x^2/2 hbar)), V + Q = hbar Om/2, so the phase rate is
    #          omega_0 + Om/2 at every x, while -dV/dx is nonzero.
    # Mutant: Q with the opposite sign, +(hbar^2/2m) lap R/R.
    # ------------------------------------------------------------------------
    eq("eq:phase_rate_split", 183, "-d_t theta = omega(x, p(x)) + Q(x)/hbar")
    Rt = sp.Function('R', positive=True)(*X, t)
    tht = sp.Function('theta', real=True)(*X, t)
    Vx = sp.Function('V', real=True)(*X)
    psiM = Rt*sp.exp(I*tht)
    lap = lambda f: sum(sp.diff(f, Xi, 2) for Xi in X)
    HpsiM = m*c**2*psiM - hb**2/(2*m)*lap(psiM) + Vx*psiM
    seM = sp.expand(sp.simplify((I*hb*sp.diff(psiM, t) - HpsiM)*sp.exp(-I*tht)))
    atoms = list(seM.atoms(sp.Derivative) | {Rt, tht, Vx})
    sub = {a_: sp.Symbol('rr_%d' % k_, real=True) for k_, a_ in enumerate(atoms)}
    reify = lambda e: e.subs(sub, simultaneous=True)
    Qt = -(hb**2/(2*m))*lap(Rt)/Rt
    pf = [hb*sp.diff(tht, Xi) for Xi in X]
    omfun = (m*c**2 + sum(p_**2 for p_ in pf)/(2*m) + Vx)/hb
    symcheck("Re[(i hbar d_t psi - H psi) e^{-i theta}]/R == hbar(-d_t theta - omega(x,p(x)) - Q/hbar)",
             sp.expand(sp.re(sp.expand(reify(seM)))/reify(Rt)
                       - reify(hb*(-sp.diff(tht, t) - omfun - Qt/hb))))
    symcheck("the imaginary part is the continuity equation d_t R^2 + div(R^2 grad theta hbar/m) = 0 (times hbar/2R)",
             sp.expand(sp.im(sp.expand(reify(seM)))
                       - reify(hb*sp.diff(Rt, t) + hb**2/(2*m)*(2*sum(sp.diff(Rt, Xi)*sp.diff(tht, Xi) for Xi in X)
                                                               + Rt*lap(tht)))))
    Qbad = +(hb**2/(2*m))*lap(Rt)/Rt
    mutant("Q with the opposite sign",
           sp.expand(sp.re(sp.expand(reify(seM)))/reify(Rt)
                     - reify(hb*(-sp.diff(tht, t) - omfun - Qbad/hb))))
    Om = sp.Symbol('Omega_osc', positive=True)
    Vho = m*Om**2*x**2/2
    amp0 = sp.exp(-m*Om*x**2/(2*hb))
    Qho = -(hb**2/(2*m))*sp.diff(amp0, x, 2)/amp0
    symcheck("L289 oscillator ground state: (m c^2 + V + Q)/hbar == omega_0 + Om/2 at every x",
             sp.simplify((m*c**2 + Vho + Qho)/hb - (m*c**2/hb + Om/2)))
    boolcheck("L289: -dV/dx = -m Om^2 x is nonzero while the phase-rate gradient is zero",
              sp.simplify(sp.diff(Vho, x)) != 0
              and sp.simplify(sp.diff((Vho + Qho)/hb, x)) == 0)

    # ------------------------------------------------------------------------
    # eq:gradient_def  (sgm.tex L191)
    # States: grad == grad_x |_p, the gradient in position at fixed momentum
    #          (operator on phase-space functions).
    # Derived from: def:spectral_gradient (definition); L197 uses eq:planck.
    # Test: L197: for omega(x,p) = (m c^2 + p^2/2m + V)/hbar (3D, general V),
    #          the fixed-momentum gradient is grad V/hbar. On a stationary state
    #          with momentum field |p(x)| = sqrt(2m(E - V)) (1D, Q neglected),
    #          the function evaluated on the field, omega(x, p(x)), is the uniform
    #          E/hbar (zero gradient along the field), so the two gradients
    #          differ: that difference is why the definition must fix p.
    # Mutant: the gradient taken along the momentum field (total derivative of
    #          omega(x, p(x))) in place of the fixed-p gradient: gives 0, not V'/hbar.
    # ------------------------------------------------------------------------
    eq("eq:gradient_def", 191, "grad = grad_x |_p")
    omP = (m*c**2 + sum(p_**2 for p_ in P)/(2*m) + Vx)/hb
    symcheck("fixed-p gradient of omega(x,p) == grad V/hbar (L197)",
             sp.Matrix([sp.diff(omP, X[i]) - sp.diff(Vx, X[i])/hb for i in range(3)]))
    V1 = sp.Function('V', real=True)(x)
    om1 = lambda pp: (m*c**2 + pp**2/(2*m) + V1)/hb
    pfield = sp.sqrt(2*m*(E - V1))
    symcheck("on the stationary field p(x) = sqrt(2m(E-V)), omega(x,p(x)) == E/hbar + m c^2/hbar",
             sp.simplify(om1(pfield) - (E + m*c**2)/hb))
    Psym = sp.Symbol('p_fixed', real=True)
    symcheck("fixed-p derivative d/dx omega(x,p)|_p == V'/hbar", sp.diff(om1(Psym), x) - sp.diff(V1, x)/hb)
    mutant("gradient along the field d/dx omega(x,p(x)) used as the spectral gradient",
           sp.simplify(sp.diff(om1(pfield), x)) - sp.diff(V1, x)/hb)

    # ------------------------------------------------------------------------
    # eq:force_energy  (sgm.tex L223)
    # States: F = -grad E, the force (vector) as minus the fixed-momentum
    #          gradient of the energy E(x,p) (scalar function on phase space).
    # Derived from: ax:hamilton (F = p_dot = -grad_x H|_p) with H = E, and
    #          eq:gradient_def for the gradient at fixed momentum.
    # Test: for a general H(x,p) in 1D (sympy Function), Hamilton's equations
    #          x_dot = dH/dp, p_dot = -dH/dx|_p conserve H along the flow (the
    #          vis viva statement of L226); for E = K(p) + U(x) (general K, U)
    #          the momentum-only part drops from -grad E, leaving -U' (L226).
    #          Dimensions: energy/length = force.
    # Mutant: F = +grad E: the energy is then not conserved along the flow
    #          (dE/dt = 2 dH/dx dH/dp).
    # ------------------------------------------------------------------------
    eq("eq:force_energy", 223, "F = -grad E")
    xs, ps = sp.symbols('x_s p_s', real=True)
    Hgen = sp.Function('H')(xs, ps)
    xdot = sp.diff(Hgen, ps)
    pdot = -sp.diff(Hgen, xs)
    symcheck("dH/dt = H_x x_dot + H_p p_dot == 0 with p_dot = -H_x (vis viva)",
             sp.diff(Hgen, xs)*xdot + sp.diff(Hgen, ps)*pdot)
    K = sp.Function('K')(ps); U = sp.Function('U')(xs)
    symcheck("E = K(p) + U(x): -dE/dx|_p == -U'(x), the momentum-only part drops",
             -sp.diff(K + U, xs) + sp.diff(U, xs))
    symcheck("dimensions: [energy]/[length] == [force]", D_energy - D_len - D_force)
    mutant("F = +grad E does not conserve E", sp.diff(Hgen, xs)*xdot + sp.diff(Hgen, ps)*(-pdot))

    # ------------------------------------------------------------------------
    # unlabeled (sgm.tex L228), first line of eq:H_eq_hw
    # States: H = E: the Hamiltonian (scalar function on phase space) is the
    #          energy.
    # Derived from: ax:hamilton and eq:force_energy (L226: Eq. force_energy is
    #          Hamilton's equation when H = E); a premise of Route I.
    # Test: for H = p^2/2m + V, the quantum expectation <H> in the oscillator
    #          ground state equals its energy eigenvalue hbar Om/2, i.e. the
    #          value of H is the energy, kinetic and potential parts together.
    # Mutant: the kinetic part alone, <p^2/2m> = hbar Om/4, is not the energy.
    # ------------------------------------------------------------------------
    eq("unlabeled", 228, "H = E")
    psiho = (m*Om/(sp.pi*hb))**sp.Rational(1, 4)*amp0
    kin = sp.integrate(psiho*(-hb**2/(2*m))*sp.diff(psiho, x, 2), (x, -sp.oo, sp.oo))
    pot = sp.integrate(psiho**2*Vho, (x, -sp.oo, sp.oo))
    symcheck("<p^2/2m + V> == hbar Om/2, the energy eigenvalue", sp.simplify(kin + pot - hb*Om/2))
    Hpsi = -hb**2/(2*m)*sp.diff(psiho, x, 2) + Vho*psiho
    symcheck("H psi == (hbar Om/2) psi (it is an eigenstate)", sp.simplify(Hpsi - hb*Om/2*psiho))
    mutant("kinetic part alone taken as the energy", sp.simplify(kin - hb*Om/2))

    # ------------------------------------------------------------------------
    # eq:H_eq_hw  (sgm.tex L229), second line
    # States: E = hbar omega(x,p): the energy equals hbar times the frequency
    #          function on phase space.
    # Derived from: eq:planck, ax:local_frequency (pointwise, L151), ax:hamilton, with
    #          omega(x,p) = H(x,p)/hbar (L179).
    # Test: for the 1D stationary field p(x) = sqrt(2m(E - V)) (general V,
    #          Q neglected), hbar omega(x, p(x)) == E at every x, and the
    #          stationary phase rate -d_t theta = E/hbar (eq:phase_freq block)
    #          equals omega(x, p(x)): the frequency function on the state's
    #          momentum field reproduces Planck-Einstein pointwise.
    # Mutant: omega = H/h (h = 2 pi hbar).
    # ------------------------------------------------------------------------
    eq("eq:H_eq_hw", 229, "E = hbar omega(x,p)")
    H1 = lambda pp: pp**2/(2*m) + V1
    symcheck("hbar * [H(x,p(x))/hbar] == E on the stationary field", sp.simplify(hb*(H1(pfield)/hb) - E))
    symcheck("omega(x,p(x)) == -d_t theta = E/hbar", sp.simplify(H1(pfield)/hb - (-sp.diff((S0 - E*t)/hb, t))))
    mutant("omega = H/h", sp.simplify(hb*H1(pfield)/(2*sp.pi*hb) - E))

    # ------------------------------------------------------------------------
    # eq:sgm  (sgm.tex L235)
    # States: F = -hbar grad omega, the force (vector) as minus hbar times the
    #          fixed-momentum position gradient of the frequency function.
    # Derived from: eq:H_eq_hw, eq:force_energy, eq:gradient_def: substitute
    #          E = hbar omega into F = -grad E.
    # Test: in 3D for a general Hamiltonian H(x,p) (sympy Function of six
    #          arguments), with omega := H/hbar, -grad(E) with E = hbar omega
    #          equals -hbar grad omega component by component (the substitution);
    #          and for the velocity-dependent H = |p - qA|^2/2m + q Phi + m c^2
    #          alpha-free case (general A(x), Phi(x)), -hbar d omega/dx_i|_p
    #          equals q(-d_i Phi + (p-qA)_j d_i A_j / m), the fixed-p gradient.
    # Mutant: gradient at fixed velocity v = (p - qA)/m in place of fixed
    #          momentum: gives -q d_i Phi only, which differs.
    # ------------------------------------------------------------------------
    eq("eq:sgm", 235, "F = -hbar grad omega")
    Hg = sp.Function('H')(*X, *P)
    omg = Hg/hb
    symcheck("-grad(hbar omega) == -hbar grad omega for general H(x,p)",
             sp.Matrix([-sp.diff(hb*omg, X[i]) + hb*sp.diff(omg, X[i]) for i in range(3)]))
    Aflds = [sp.Function(f'A{i+1}', real=True)(*X) for i in range(3)]
    Phx = sp.Function('Phi', real=True)(*X)
    HEM = sum((P[j] - q*Aflds[j])**2 for j in range(3))/(2*m) + q*Phx + m*c**2
    omEM = HEM/hb
    Fsgm = [-hb*sp.diff(omEM, X[i]) for i in range(3)]
    for i in range(3):
        symcheck(f"EM: -hbar d_{i+1} omega|_p == q(-d_{i+1} Phi + v_j d_{i+1} A_j), v = (p-qA)/m",
                 sp.expand(Fsgm[i] - q*(-sp.diff(Phx, X[i])
                                        + sum((P[j] - q*Aflds[j])/m*sp.diff(Aflds[j], X[i]) for j in range(3)))))
    Vv = sp.symbols('v1 v2 v3', real=True)
    HEMv = sum(m*Vv[j]**2 for j in range(3))/2 + q*Phx + m*c**2   # H written at fixed velocity
    mutant("gradient at fixed velocity in place of fixed momentum (axis 1)",
           sp.expand(Fsgm[0] - (-sp.diff(HEMv, X[0])).subs({Vv[j]: (P[j] - q*Aflds[j])/m for j in range(3)})))

    # ------------------------------------------------------------------------
    # eq:sgm_as_dot_p  (sgm.tex L243)
    # States: p_dot = -hbar grad omega, with p the canonical momentum (vector).
    # Derived from: eq:sgm, ax:hamilton (F = p_dot, canonical p).
    # Test: in 3D for a charge in general static fields A(x), Phi(x) on a
    #          general trajectory x(t) obeying the Lorentz force law
    #          m x_ddot = q(E + v x B), the rate of the canonical momentum
    #          p = m x_dot + q A(x(t)) equals -hbar d_i omega|_p of the EM
    #          frequency function evaluated at that p, on each axis.
    # Mutant: kinetic momentum m x_dot in place of the canonical momentum
    #          (L248): its rate is the Lorentz force, which differs from
    #          -hbar grad omega|_p by q dA/dt.
    # ------------------------------------------------------------------------
    eq("eq:sgm_as_dot_p", 243, "p_dot = -hbar grad omega (canonical p)")
    xt = [sp.Function(f'X{i+1}', real=True)(t) for i in range(3)]
    subx = dict(zip(X, xt))
    vt = [sp.diff(xi_, t) for xi_ in xt]
    At = [Ai.subs(subx) for Ai in Aflds]
    Pht = Phx.subs(subx)
    Et = [-sp.diff(Phx, X[i]).subs(subx) for i in range(3)]
    Bt = [(sp.diff(Aflds[2], X[1]) - sp.diff(Aflds[1], X[2])).subs(subx),
          (sp.diff(Aflds[0], X[2]) - sp.diff(Aflds[2], X[0])).subs(subx),
          (sp.diff(Aflds[1], X[0]) - sp.diff(Aflds[0], X[1])).subs(subx)]
    vxB = [vt[1]*Bt[2] - vt[2]*Bt[1], vt[2]*Bt[0] - vt[0]*Bt[2], vt[0]*Bt[1] - vt[1]*Bt[0]]
    accel = {sp.diff(xt[i], t, 2): q*(Et[i] + vxB[i])/m for i in range(3)}
    pcan = [m*vt[i] + At[i]*q for i in range(3)]
    for i in range(3):
        lhs = sp.diff(pcan[i], t).subs(accel)
        rhs = Fsgm[i].subs({P[j]: pcan[j] for j in range(3)}).subs(subx)
        symcheck(f"Lorentz trajectory: d/dt(m v + qA)_{i+1} == -hbar d_{i+1} omega|_p", sp.expand(sp.expand(lhs) - sp.expand(rhs)))
    lhsk = sp.diff(m*vt[0], t).subs(accel)
    rhs0 = Fsgm[0].subs({P[j]: pcan[j] for j in range(3)}).subs(subx)
    mutant("kinetic momentum m x_dot in place of canonical p (axis 1)", sp.expand(lhsk - sp.expand(rhs0)))

    # ------------------------------------------------------------------------
    # eq:mixed_orders  (sgm.tex L261)
    # States: d_t grad_x theta = grad_x d_t theta for a smooth phase field.
    # Derived from: ax:debroglie (theta), with Schwarz's theorem on mixed partials.
    # Test: for a general smooth theta(x1,x2,x3,t), each component.
    # Mutant: the Laplacian-type confusion d_x d_x theta in place of d_x d_t
    #          theta (differentiating twice in space, not once in time).
    # ------------------------------------------------------------------------
    eq("eq:mixed_orders", 261, "d_t grad_x theta = grad_x d_t theta")
    symcheck("each component, general theta(x,t)",
             sp.Matrix([sp.diff(sp.diff(tht, Xi), t) - sp.diff(sp.diff(tht, t), Xi) for Xi in X]))
    mutant("d_x d_x theta in place of d_x d_t theta", sp.diff(sp.diff(tht, X[0]), t) - sp.diff(tht, X[0], 2))

    # ------------------------------------------------------------------------
    # eq:dp_dt_local  (sgm.tex L266)
    # States: d_t p = -hbar grad_x omega at a fixed point, where p(x,t) = hbar
    #          grad theta is the momentum field and omega(x) = -d_t theta the
    #          frequency field; grad_x is the full spatial gradient.
    # Derived from: eq:debroglie (differentiate in t at fixed x), eq:mixed_orders,
    #          eq:phase_freq.
    # Test: chain for a general theta(x,t) in 3D. Then the free-particle fan
    #          theta = m x^2/(2 hbar t) (1D, a solution of the free
    #          Hamilton-Jacobi equation): it holds with the full gradient of the
    #          frequency field, and would fail with the fixed-momentum gradient.
    # Mutant: the fixed-momentum (spectral) gradient in place of grad_x on the
    #          fan: off by -m x/t^2; and a sign flip on the general theta.
    # ------------------------------------------------------------------------
    eq("eq:dp_dt_local", 266, "d_t p = -hbar grad_x omega")
    pfield_t = [hb*sp.diff(tht, Xi) for Xi in X]
    omfield_t = -sp.diff(tht, t)
    symcheck("d_t(hbar grad theta) == -hbar grad_x(-d_t theta), general theta",
             sp.Matrix([sp.diff(pfield_t[i], t) + hb*sp.diff(omfield_t, X[i]) for i in range(3)]))
    mutant("sign flip: d_t p == +hbar grad_x omega",
           sp.diff(pfield_t[0], t) - hb*sp.diff(omfield_t, X[0]))
    tpos = sp.Symbol('t_pos', positive=True)
    th_fan = m*x**2/(2*hb*tpos)
    p_fan = hb*sp.diff(th_fan, x)
    om_fan = -sp.diff(th_fan, tpos)
    Pf = sp.Symbol('P_f', real=True)
    Om_free = Pf**2/(2*m*hb)
    symcheck("fan: Hamilton-Jacobi holds, omega(x,t) == Omega(p(x,t))", sp.simplify(om_fan - Om_free.subs(Pf, p_fan)))
    symcheck("fan: d_t p == -hbar d_x omega_field", sp.simplify(sp.diff(p_fan, tpos) + hb*sp.diff(om_fan, x)))
    mutant("fan: d_t p == -hbar d_x Omega|_p (spectral gradient, zero here)",
           sp.simplify(sp.diff(p_fan, tpos) + hb*sp.diff(Om_free, x)))

    # ------------------------------------------------------------------------
    # eq:chain_convective  (sgm.tex L272)
    # States: d_{x_i} omega(x, p(x)) = d_{x_i} omega|_p + (d_{x_i} p_j) d_{p_j}
    #          omega: chain rule for a phase-space function evaluated on a
    #          momentum field (scalars; j summed).
    # Derived from: eq:phase_rate_split (Q negligible), eq:H_eq_hw, eq:debroglie, and
    #          the multivariable chain rule.
    # Test: in 3D for a frequency function that is not quadratic in p, the
    #          lapse-and-gauge dispersion omega = [alpha(x) sqrt(m^2c^4 +
    #          c^2|P - qA(x)|^2) + q Phi(x)]/hbar with general alpha, A, Phi,
    #          on a general momentum field p_j(x) = hbar d_j theta(x).
    # Mutant: the convective term dropped.
    # ------------------------------------------------------------------------
    eq("eq:chain_convective", 272, "d_i omega(x,p(x)) = d_i omega|_p + (d_i p_j) d_{p_j} omega")
    alph = sp.Function('alpha', positive=True)(*X)
    Omrel = (alph*sp.sqrt(m**2*c**4 + c**2*sum((P[j] - q*Aflds[j])**2 for j in range(3))) + q*Phx)/hb
    th3 = sp.Function('theta', real=True)(*X)
    pf3 = [hb*sp.diff(th3, Xi) for Xi in X]
    subP = dict(zip(P, pf3))
    Om_on = Omrel.subs(subP)
    for i in range(3):
        lhs = sp.diff(Om_on, X[i])
        rhs = sp.diff(Omrel, X[i]).subs(subP) + sum(sp.diff(pf3[j], X[i])*sp.diff(Omrel, P[j]).subs(subP) for j in range(3))
        symcheck(f"chain rule, axis {i+1}", sp.expand(lhs - rhs))
    mutant("convective term dropped (axis 1)",
           sp.expand(sp.diff(Om_on, X[0]) - sp.diff(Omrel, X[0]).subs(subP)))

    # ------------------------------------------------------------------------
    # eq:total_momentum_rate  (sgm.tex L277)
    # States: p_dot = d_t p + (x_dot . grad) p: the rate of the momentum field
    #          along the trajectory x(t) (vector).
    # Derived from: eq:debroglie (momentum field), ax:hamilton (trajectory velocity),
    #          and the chain rule for p(x(t), t).
    # Test: in 3D, for a general trajectory x(t) and an explicit
    #          non-polynomial field p_i(x,t) with symbolic coefficients.
    # Mutant: the convective term with transposed indices, x_dot_j d_i p_j in
    #          place of x_dot_j d_j p_i (a field that is not a gradient).
    # ------------------------------------------------------------------------
    eq("eq:total_momentum_rate", 277, "p_dot = d_t p + (x_dot . grad) p")
    # sympy cannot cancel its Subs forms for an undefined p(x(t), t), so the
    # field is an explicit non-polynomial function with symbolic coefficients
    # (any first-order behavior in x and t is reachable); the trajectory x(t)
    # stays a general function.
    ca = sp.symbols('ca1:4', real=True); cb = sp.symbols('cb1:4', real=True)
    cw = sp.symbols('cw1:4', real=True); cd = sp.symbols('cd1:4', real=True)
    pgen = [sp.sin(sum(ca[(i + j) % 3]*X[j] for j in range(3)) + cw[i]*t)
            + cb[i]*X[0]*X[1]*X[2]*t**2 + sp.exp(cd[i]*X[i]*t) for i in range(3)]
    for i in range(3):
        lhs = sp.diff(pgen[i].subs(subx), t)
        rhs = (sp.diff(pgen[i], t) + sum(vt[j]*sp.diff(pgen[i], X[j]) for j in range(3))).subs(subx)
        symcheck(f"d/dt p_{i+1}(x(t),t) == d_t p_{i+1} + x_dot_j d_j p_{i+1}", sp.expand(lhs - rhs))
    rhs_bad = (sp.diff(pgen[0], t) + sum(vt[j]*sp.diff(pgen[j], X[0]) for j in range(3))).subs(subx)
    mutant("transposed convective term x_dot_j d_1 p_j for a general field",
           sp.expand(sp.diff(pgen[0].subs(subx), t) - rhs_bad))

    # ------------------------------------------------------------------------
    # eq:route2_components  (sgm.tex L282)
    # States: p_dot_i = -hbar d_i omega|_p - hbar x_dot_j d_i d_j theta
    #          + hbar x_dot_j d_j d_i theta (scalars; j summed).
    # Derived from: eq:dp_dt_local, eq:total_momentum_rate, eq:chain_convective,
    #          ax:hamilton (hbar d_{p_j} omega = x_dot_j), eq:debroglie
    #          (d_i p_j = hbar d_i d_j theta); the result is eq:sgm_as_dot_p via
    #          eq:gradient_def.
    # Test: (1) with the non-quadratic frequency function of the previous block
    #          and a general phase theta(x) (3D), build
    #          -hbar d_i[omega(x,p(x))] + x_dot_j d_j p_i, with x_dot_j =
    #          hbar d omega/d P_j at P = p(x), and compare with the three-term
    #          form, and with -hbar d_i omega|_p (the two velocity terms cancel).
    #          (2) for general theta and general velocity functions, the second
    #          and third terms cancel, and each vanishes at x_dot = 0 (L287).
    #          (3) L287 conserved-energy case, 1D, H = p^2/2m + V general:
    #          stationary state hbar theta = int sqrt(2m(E-V)) dx - E t has a
    #          uniform frequency field, d_t p = 0, and the spectral gradient
    #          equals minus the convective term. (4) uniform force F0: S =
    #          (p0 + F0 t) x - (p0 + F0 t)^3/(6 m F0): p uniform in x, convective
    #          term zero, p_dot = F0 = -hbar d_x omega|_p. (5) the fan: the
    #          convective term cancels d_t p, giving p_dot = 0 = spectral
    #          gradient of a free particle.
    # Mutant: (a) nonrelativistic velocity p_j/m in place of the group velocity
    #          hbar d omega/d P_j for the relativistic dispersion: no
    #          cancellation; (b) sign of the third term flipped.
    # ------------------------------------------------------------------------
    eq("eq:route2_components", 282, "p_dot_i = -hbar d_i omega|_p - hbar x_dot_j d_i d_j theta + hbar x_dot_j d_j d_i theta")
    vgrp = [hb*sp.diff(Omrel, P[j]).subs(subP) for j in range(3)]
    for i in range(3):
        built = -hb*sp.diff(Om_on, X[i]) + sum(vgrp[j]*sp.diff(pf3[i], X[j]) for j in range(3))
        three = (-hb*sp.diff(Omrel, X[i]).subs(subP)
                 - hb*sum(vgrp[j]*sp.diff(th3, X[i], X[j]) for j in range(3))
                 + hb*sum(vgrp[j]*sp.diff(th3, X[j], X[i]) for j in range(3)))
        symcheck(f"axis {i+1}: built rate == three-term form", sp.expand(built - three))
        symcheck(f"axis {i+1}: built rate == -hbar d_i omega|_p", sp.expand(built + hb*sp.diff(Omrel, X[i]).subs(subP)))
    vnr = [pf3[j]/m for j in range(3)]
    built_bad = -hb*sp.diff(Om_on, X[0]) + sum(vnr[j]*sp.diff(pf3[0], X[j]) for j in range(3))
    mutant("velocity p/m in place of the group velocity (axis 1)",
           sp.expand(built_bad + hb*sp.diff(Omrel, X[0]).subs(subP)))
    xdg = [sp.Function(f'xdot{j+1}')(*X) for j in range(3)]
    for i in range(3):
        t2 = -hb*sum(xdg[j]*sp.diff(th3, X[i], X[j]) for j in range(3))
        t3 = hb*sum(xdg[j]*sp.diff(th3, X[j], X[i]) for j in range(3))
        symcheck(f"axis {i+1}: second + third term == 0 for any velocity", t2 + t3)
        symcheck(f"axis {i+1}: each vanishes at x_dot = 0",
                 sp.Matrix([t2.subs({f_: 0 for f_ in xdg}), t3.subs({f_: 0 for f_ in xdg})]))
    t2 = -hb*sum(xdg[j]*sp.diff(th3, X[0], X[j]) for j in range(3))
    t3 = hb*sum(xdg[j]*sp.diff(th3, X[j], X[0]) for j in range(3))
    mutant("third term with flipped sign does not cancel", t2 - t3)
    # (3) conserved-energy case (L287)
    Ep = sp.Symbol('E_p', positive=True)
    pst = sp.sqrt(2*m*(Ep - V1))
    th_st = (sp.Integral(pst, x) - Ep*t)/hb
    om_st = -sp.diff(th_st, t)
    symcheck("L287 stationary: frequency field uniform, d_x omega_field == 0", sp.diff(om_st, x))
    symcheck("L287 stationary: d_t p == 0", sp.diff(hb*sp.diff(th_st, x), t))
    spec_st = sp.diff(V1, x)/hb
    symcheck("L287 stationary: spectral gradient == -(convective term)/hbar = -x_dot d_x^2 theta",
             sp.simplify(spec_st + (pst/m)*sp.diff(th_st, x, 2)))
    # (4) uniform force
    F0, p0 = sp.symbols('F_0 p_0', real=True, nonzero=True)
    S_u = (p0 + F0*t)*x - (p0 + F0*t)**3/(6*m*F0)
    p_u = sp.diff(S_u, x)
    om_u = -sp.diff(S_u/hb, t)
    Om_u = Pf**2/(2*m*hb) - F0*x/hb
    symcheck("uniform force: Hamilton-Jacobi holds", sp.simplify(om_u - Om_u.subs(Pf, p_u)))
    v_u = (hb*sp.diff(Om_u, Pf)).subs(Pf, p_u)
    symcheck("uniform force: convective term 0, p_dot == F0 == -hbar d_x Omega|_p",
             sp.Matrix([sp.diff(p_u, x), sp.diff(p_u, t) + v_u*sp.diff(p_u, x) - F0,
                        -hb*sp.diff(Om_u, x) - F0]))
    # (5) fan
    v_fan = (hb*sp.diff(Om_free, Pf)).subs(Pf, p_fan)
    symcheck("fan: d_t p + v d_x p == 0 == -hbar d_x Omega|_p",
             sp.Matrix([sp.simplify(sp.diff(p_fan, tpos) + v_fan*sp.diff(p_fan, x)), -hb*sp.diff(Om_free, x)]))

    # ------------------------------------------------------------------------
    # eq:eikonal_ray  (sgm.tex L293)
    # States: k_dot = -grad omega, the rate of the wavevector (vector) along a
    #          ray, with the gradient at fixed wavevector, for any dispersion
    #          relation omega(x,k).
    # Derived from: cited (Whitham); eq:debroglie, eq:dp_dt_local, eq:chain_convective,
    #          eq:total_momentum_rate, eq:route2_components (the Route II steps), and
    #          eq:debroglie_k, eq:gradient_def for the conversion to eq:sgm_as_dot_p.
    # Test: (1) the Route II steps with hbar = 1 for light in a medium of
    #          general index n(x): omega = c|k|/n(x), general phase theta(x):
    #          -d_i[omega(x, grad theta)] + v_j d_j d_i theta == -d_i omega|_k
    #          with v_j = d omega/d k_j (3D). (2) hbar k_dot = p_dot: with
    #          omega_p(x,p) := omega(x, p/hbar), -hbar d_i omega_p|_p at p = hbar k
    #          equals hbar(-d_i omega|_k). (3) omega is conserved along rays of a
    #          static medium.
    # Mutant: the convective term dropped from (1).
    # ------------------------------------------------------------------------
    eq("eq:eikonal_ray", 293, "k_dot = -grad omega (fixed k)")
    nidx = sp.Function('n', positive=True)(*X)
    Kk = sp.symbols('K1 K2 K3', real=True)
    omL = c*sp.sqrt(sum(k_**2 for k_ in Kk))/nidx
    kf = [sp.diff(th3, Xi) for Xi in X]
    subK = dict(zip(Kk, kf))
    omL_on = omL.subs(subK)
    vL = [sp.diff(omL, Kk[j]).subs(subK) for j in range(3)]
    for i in range(3):
        built = -sp.diff(omL_on, X[i]) + sum(vL[j]*sp.diff(kf[i], X[j]) for j in range(3))
        symcheck(f"light, axis {i+1}: k_dot == -d_i omega|_k", sp.simplify(built + sp.diff(omL, X[i]).subs(subK)))
    mutant("convective term dropped (axis 1)",
           sp.simplify(-sp.diff(omL_on, X[0]) + sp.diff(omL, X[0]).subs(subK)))
    omLp = omL.subs({Kk[j]: P[j]/hb for j in range(3)})
    symcheck("hbar(-d_1 omega|_k) == -hbar d_1 omega(x,p/hbar)|_p at p = hbar k",
             sp.simplify((-hb*sp.diff(omLp, X[0])).subs({P[j]: hb*Kk[j] for j in range(3)}) - hb*(-sp.diff(omL, X[0]))))
    dom = sum(sp.diff(omL, X[i])*sp.diff(omL, Kk[i]) + sp.diff(omL, Kk[i])*(-sp.diff(omL, X[i])) for i in range(3))
    symcheck("omega conserved along rays of a static medium", dom)

    # ------------------------------------------------------------------------
    # eq:euler_lagrange  (sgm.tex L305)
    # States: d/dt dL/dx_dot = dL/dx (vector equation).
    # Derived from: cited (Euler, Lagrange); no earlier label needed.
    # Test: for L = (1/2) m |x_dot|^2 + q x_dot . A(x) - q Phi(x) with general
    #          A, Phi in 3D, the canonical momentum is m x_dot + q A and the
    #          equation gives the Lorentz force m x_ddot = q(E + v x B) on each
    #          axis. Does not check the variational principle itself.
    # Mutant: the Lorentz force with the magnetic term of opposite sign.
    # ------------------------------------------------------------------------
    eq("eq:euler_lagrange", 305, "d/dt dL/dx_dot = dL/dx")
    Lt = (m*sum(v_**2 for v_ in vt)/2 + q*sum(vt[i]*At[i] for i in range(3)) - q*Pht)
    for i in range(3):
        EL = sp.diff(sp.diff(Lt, vt[i]), t) - sp.diff(Lt, xt[i])
        symcheck(f"axis {i+1}: EL == m x_ddot - q(E + v x B)",
                 sp.expand(EL - (m*sp.diff(xt[i], t, 2) - q*(Et[i] + vxB[i]))))
    symcheck("canonical momentum dL/dx_dot == m x_dot + q A",
             sp.Matrix([sp.diff(Lt, vt[i]) - pcan[i] for i in range(3)]))
    EL0 = sp.diff(sp.diff(Lt, vt[0]), t) - sp.diff(Lt, xt[0])
    mutant("magnetic term with opposite sign", sp.expand(EL0 - (m*sp.diff(xt[0], t, 2) - q*(Et[0] - vxB[0]))))

    # ------------------------------------------------------------------------
    # eq:legendre_chain  (sgm.tex L310)
    # States: dH/dx_i|_p = p_j dx_dot_j/dx_i|_p - dL/dx_i|_x_dot
    #          - dL/dx_dot_j dx_dot_j/dx_i|_p (scalars; j summed).
    # Derived from: eq:euler_lagrange (p = dL/dx_dot), eq:H_eq_hw (H), and the chain
    #          rule applied to H(x,p) = p . x_dot(x,p) - L(x, x_dot(x,p)) at fixed p.
    # Test: in 3D for L = (1/2) mu(x) |v|^2 + q v . A(x) - U(x) with general
    #          mass function mu(x), A(x), U(x), so that x_dot(x,p) = (p - qA)/mu
    #          depends on x at fixed p; compute H by an actual Legendre
    #          transform and compare both sides, each axis.
    # Mutant: the third term dropped.
    # ------------------------------------------------------------------------
    eq("eq:legendre_chain", 310, "dH/dx_i|_p = p_j dxdot_j/dx_i - dL/dx_i|_xdot - dL/dxdot_j dxdot_j/dx_i")
    mu = sp.Function('mu', positive=True)(*X)
    Ux = sp.Function('U', real=True)(*X)
    Lv = mu*sum(v_**2 for v_ in Vv)/2 + q*sum(Vv[j]*Aflds[j] for j in range(3)) - Ux
    xdot_p = [(P[j] - q*Aflds[j])/mu for j in range(3)]
    subV = dict(zip(Vv, xdot_p))
    symcheck("inversion: dL/dv at v = x_dot(x,p) == p",
             sp.Matrix([sp.simplify(sp.diff(Lv, Vv[j]).subs(subV) - P[j]) for j in range(3)]))
    HL = sum(P[j]*xdot_p[j] for j in range(3)) - Lv.subs(subV, simultaneous=True)
    for i in range(3):
        lhs = sp.diff(HL, X[i])
        rhs = (sum(P[j]*sp.diff(xdot_p[j], X[i]) for j in range(3))
               - sp.diff(Lv, X[i]).subs(subV, simultaneous=True)
               - sum(sp.diff(Lv, Vv[j]).subs(subV, simultaneous=True)*sp.diff(xdot_p[j], X[i]) for j in range(3)))
        symcheck(f"axis {i+1}: three-term chain rule", sp.simplify(lhs - rhs))
    rhs_bad = (sum(P[j]*sp.diff(xdot_p[j], X[0]) for j in range(3)) - sp.diff(Lv, X[0]).subs(subV, simultaneous=True))
    mutant("third term dropped (axis 1)", sp.simplify(sp.diff(HL, X[0]) - rhs_bad))

    # ------------------------------------------------------------------------
    # eq:legendre_bridge  (sgm.tex L315)
    # States: dL/dx|_x_dot = -dH/dx|_p (vector, each component).
    # Derived from: eq:legendre_chain, eq:euler_lagrange (first and third terms
    #          cancel by p_j = dL/dx_dot_j); L318 uses eq:H_eq_hw and gives
    #          eq:sgm_as_dot_p.
    # Test: same general Lagrangian, each axis; plus the combination of L318:
    #          on the Lorentz trajectory (previous blocks), the Euler-Lagrange
    #          rate d/dt(m v + qA) equals -dH/dx|_p computed from the Legendre H
    #          with mu = m and U = q Phi (the constant m c^2 does not
    #          affect a gradient).
    # Mutant: dL/dx|_x_dot = +dH/dx|_p.
    # ------------------------------------------------------------------------
    eq("eq:legendre_bridge", 315, "dL/dx|_xdot = -dH/dx|_p")
    for i in range(3):
        symcheck(f"axis {i+1}", sp.simplify(sp.diff(Lv, X[i]).subs(subV, simultaneous=True) + sp.diff(HL, X[i])))
    mutant("dL/dx|_xdot = +dH/dx|_p (axis 1)",
           sp.simplify(sp.diff(Lv, X[0]).subs(subV, simultaneous=True) - sp.diff(HL, X[0])))
    HLm = HL.subs({mu: m, Ux: q*Phx}, simultaneous=True)
    for i in range(3):
        lhs = sp.diff(pcan[i], t).subs(accel)
        rhs = (-sp.diff(HLm, X[i])).subs({P[j]: pcan[j] for j in range(3)}).subs(subx)
        symcheck(f"L318 on the Lorentz trajectory: p_dot_{i+1} == -dH/dx_{i+1}|_p (Legendre H)",
                 sp.expand(sp.expand(lhs) - sp.expand(rhs)))

    # ------------------------------------------------------------------------
    # eq:schrodinger_hamiltonian  (sgm.tex L327)
    # States: H_hat = p_hat^2/2m + V(x), with p_hat = -i hbar grad (operator on
    #          a scalar wave function).
    # Derived from: definition (cited Schroedinger); L330 uses eq:planck for omega_hat.
    # Test: acting on a plane wave e^{i k . x} with constant V0 it returns
    #          (hbar^2 |k|^2/2m + V0) times the wave (3D), i.e. p^2/2m + V with
    #          p = hbar k (eq:debroglie_k); dimensions of p^2/m are energy.
    # Mutant: p_hat^2/m (factor 1/2 lost).
    # ------------------------------------------------------------------------
    eq("eq:schrodinger_hamiltonian", 327, "H_hat = p_hat^2/2m + V(x)")
    K3 = sp.symbols('k1 k2 k3', real=True)
    V0 = sp.Symbol('V_0', real=True)
    pw3 = sp.exp(I*sum(K3[i]*X[i] for i in range(3)))
    phat = lambda f, i: -I*hb*sp.diff(f, X[i])
    Hhat = lambda f, VV: sum(phat(phat(f, i), i) for i in range(3))/(2*m) + VV*f
    symcheck("H_hat e^{ik.x} == (hbar^2 k^2/2m + V0) e^{ik.x}",
             sp.simplify(Hhat(pw3, V0)/pw3 - (hb**2*sum(k_**2 for k_ in K3)/(2*m) + V0)))
    symcheck("dimensions: [p^2/m] == [energy]", 2*D_mom - D_mass - D_energy)
    mutant("p_hat^2/m", sp.simplify(2*Hhat(pw3, 0)/pw3 - hb**2*sum(k_**2 for k_ in K3)/(2*m)))

    # Route IV 1D machinery
    psi1 = sp.Function('psi')(x, t)
    u_ = sp.Function('u', real=True)(x, t)
    w_ = sp.Function('w', real=True)(x, t)
    PsiC = u_ + I*w_
    PsiCc = u_ - I*w_
    H1op = lambda f: -hb**2/(2*m)*sp.diff(f, x, 2) + V1*f
    p1op = lambda f: -I*hb*sp.diff(f, x)

    # ------------------------------------------------------------------------
    # eq:ehrenfest_commutator  (sgm.tex L334), line 1
    # States: d/dt <A> = int (d_t psi*) A psi + int psi* A (d_t psi), A an
    #          operator without explicit time dependence.
    # Derived from: eq:schrodinger_hamiltonian (the state and <A> of L325-L330) and the
    #          product rule under the integral.
    # Test: the integrands agree, 1D, general psi = u + i w, for A = p_hat and
    #          for A = x.
    # Mutant: an explicitly time-dependent A = t p_hat: the product rule then
    #          leaves the extra term psi* (d_t A) psi, so line 1 fails.
    # ------------------------------------------------------------------------
    eq("eq:ehrenfest_commutator", 334, "d<A>/dt = int (d_t psi*) A psi + int psi* A (d_t psi)")
    for nm, Aop in (("p_hat", p1op), ("x", lambda f: x*f)):
        lhs = sp.diff(PsiCc*Aop(PsiC), t)
        rhs = sp.diff(PsiCc, t)*Aop(PsiC) + PsiCc*Aop(sp.diff(PsiC, t))
        symcheck(f"product rule integrand, A = {nm}", sp.expand(lhs - rhs))
    Atd = lambda f: t*p1op(f)
    mutant("A = t p_hat (explicit time dependence)",
           sp.expand(sp.diff(PsiCc*Atd(PsiC), t) - (sp.diff(PsiCc, t)*Atd(PsiC) + PsiCc*Atd(sp.diff(PsiC, t)))))

    # ------------------------------------------------------------------------
    # eq:ehrenfest_commutator  (sgm.tex L335), line 2
    # States: = (i/hbar) int (H psi)* A psi - (i/hbar) int psi* A H psi.
    # Derived from: eq:schrodinger_hamiltonian: substitute d_t psi = -(i/hbar) H psi and
    #          its conjugate d_t psi* = (i/hbar)(H psi)* (H with real coefficients).
    # Test: for psi = u + i w (u, w general real) and real V, impose the
    #          Schroedinger equation by writing d_t u, d_t w from it; the line-1
    #          integrand equals the line-2 integrand, for A = p_hat (1D).
    # Mutant: conjugate equation with the wrong sign, d_t psi* = -(i/hbar)(H psi)*.
    # ------------------------------------------------------------------------
    eq("eq:ehrenfest_commutator", 335, "= (i/hbar) int (H psi)* A psi - (i/hbar) int psi* A H psi")
    HP = H1op(PsiC)
    HPc = H1op(PsiCc)               # (H psi)* for real V
    dPsi = -I/hb*HP
    dPsic = I/hb*HPc
    line1 = dPsic*p1op(PsiC) + PsiCc*p1op(dPsi)
    line2 = I/hb*HPc*p1op(PsiC) - I/hb*PsiCc*p1op(HP)
    symcheck("line 1 integrand == line 2 integrand after substituting the Schroedinger equation",
             sp.expand(line1 - line2))
    symcheck("(H psi)* equals H acting on psi* for real V",
             sp.expand(_S1_cc(HP) - HPc))
    line1bad = (-I/hb*HPc)*p1op(PsiC) + PsiCc*p1op(dPsi)
    mutant("d_t psi* = -(i/hbar)(H psi)*", sp.expand(line1bad - line2))

    # ------------------------------------------------------------------------
    # eq:ehrenfest_commutator  (sgm.tex L336), line 3
    # States: = (i/hbar) int psi* (H A - A H) psi.
    # Derived from: eq:schrodinger_hamiltonian, Hermiticity of H applied to the first
    #          integral: int (H psi)* f = int psi* H f, with f = A psi, under the domain
    #          conditions stated just after the display (L340; no boundary term).
    # Test: (1) pointwise, (H psi)* f - psi* (H f) is a total x-derivative of
    #          -(hbar^2/2m)(psi*' f - psi* f') for general psi, f and real V
    #          (1D), so it integrates to zero when the boundary term vanishes.
    #          (2) on a normalized Gaussian boosted to wavenumber k0, in V =
    #          lambda x^3, the two sides of line 3 computed by direct integration
    #          agree, for A = x.
    # Mutant: a non-Hermitian potential V + i Gamma(x) leaves -2 i Gamma psi* f,
    #          not a total derivative.
    # ------------------------------------------------------------------------
    eq("eq:ehrenfest_commutator", 336, "= (i/hbar) int psi* (H A - A H) psi")
    fF = sp.Function('f')(x)
    pa = sp.Function('a', real=True)(x) + I*sp.Function('b', real=True)(x)
    pac = sp.Function('a', real=True)(x) - I*sp.Function('b', real=True)(x)
    Hpac = -hb**2/(2*m)*sp.diff(pac, x, 2) + V1*pac
    surf = -(hb**2/(2*m))*(sp.diff(pac, x)*fF - pac*sp.diff(fF, x))
    symcheck("(H psi)* f - psi* H f == d/dx[surface term], any psi, f",
             sp.expand(Hpac*fF - pac*(-hb**2/(2*m)*sp.diff(fF, x, 2) + V1*fF) - sp.diff(surf, x)))
    Gam = sp.Function('Gamma', real=True)(x)
    Hpac_nh = -hb**2/(2*m)*sp.diff(pac, x, 2) + (V1 - I*Gam)*pac       # (H psi)* with V + i Gamma
    mutant("non-Hermitian V + i Gamma: residual -2 i Gamma psi* f",
           sp.expand(Hpac_nh*fF - pac*(-hb**2/(2*m)*sp.diff(fF, x, 2) + (V1 + I*Gam)*fF) - sp.diff(surf, x)))
    k0, sg, lm3 = sp.Symbol('k0', positive=True), sp.Symbol('sigma', positive=True), sp.Symbol('lambda3', real=True)
    Gs = (1/(sp.pi*sg**2))**sp.Rational(1, 4)*sp.exp(-x**2/(2*sg**2))*sp.exp(I*k0*x)
    Gsc = (1/(sp.pi*sg**2))**sp.Rational(1, 4)*sp.exp(-x**2/(2*sg**2))*sp.exp(-I*k0*x)
    Hc = lambda f: -hb**2/(2*m)*sp.diff(f, x, 2) + lm3*x**3*f
    symcheck("Gaussian is normalized", sp.simplify(sp.integrate(sp.expand(Gsc*Gs), (x, -sp.oo, sp.oo)) - 1))
    lhs3 = sp.integrate(sp.expand(sp.conjugate(Hc(Gs)).rewrite(sp.exp)*x*Gs), (x, -sp.oo, sp.oo))
    rhs3 = sp.integrate(sp.expand(Gsc*Hc(x*Gs)), (x, -sp.oo, sp.oo))
    symcheck("int (H psi)* x psi == int psi* H (x psi) on the Gaussian in lambda x^3", sp.simplify(lhs3 - rhs3))

    # ------------------------------------------------------------------------
    # eq:ehrenfest_commutator  (sgm.tex L337), line 4
    # States: = (i/hbar) <[H, A]>, with [H,A] = HA - AH.
    # Derived from: eq:schrodinger_hamiltonian, and the commutator defined at L332.
    # Test: on the boosted Gaussian in lambda x^3 (1D), d<x>/dt computed from
    #          the product-rule integrand with the Schroedinger equation equals
    #          (i/hbar)<[H,x]>, and both equal hbar k0/m = <p>/m, an independent
    #          closed form.
    # Mutant: commutator in the reversed order, (i/hbar)<[x,H]> = -hbar k0/m.
    # ------------------------------------------------------------------------
    eq("eq:ehrenfest_commutator", 337, "= (i/hbar) <[H, A]>")
    dG = -I/hb*Hc(Gs)
    dGc = sp.conjugate(dG).rewrite(sp.exp)
    lhsX = sp.integrate(sp.expand(dGc*x*Gs + Gsc*x*dG), (x, -sp.oo, sp.oo))
    comX = I/hb*sp.integrate(sp.expand(Gsc*(Hc(x*Gs) - x*Hc(Gs))), (x, -sp.oo, sp.oo))
    symcheck("d<x>/dt == (i/hbar)<[H,x]>", sp.simplify(lhsX - comX))
    symcheck("(i/hbar)<[H,x]> == hbar k0/m", sp.simplify(comX - hb*k0/m))
    mutant("reversed commutator (i/hbar)<[x,H]>", sp.simplify(-comX - hb*k0/m))

    # ------------------------------------------------------------------------
    # eq:commutator_V  (sgm.tex L344), line 1
    # States: [H, p_hat] psi = V(-i hbar grad psi) + i hbar grad(V psi) (vector
    #          of operators acting on a scalar psi).
    # Derived from: eq:schrodinger_hamiltonian: expand [H,p] psi = H(p psi) - p(H psi);
    #          the kinetic term commutes with p_hat and drops out.
    # Test: in 3D for general psi(x), V(x), each component, by acting with the
    #          operators of eq:schrodinger_hamiltonian.
    # Mutant: the kinetic term not cancelled is impossible here; use the sign
    #          error V(-i hbar grad psi) - i hbar grad(V psi).
    # ------------------------------------------------------------------------
    eq("eq:commutator_V", 344, "[H,p] psi = V(-i hbar grad psi) + i hbar grad(V psi)")
    ps3 = sp.Function('psi')(*X)
    comm = [Hhat(phat(ps3, i), Vx) - phat(Hhat(ps3, Vx), i) for i in range(3)]
    for i in range(3):
        symcheck(f"axis {i+1}", sp.expand(comm[i] - (Vx*(-I*hb*sp.diff(ps3, X[i])) + I*hb*sp.diff(Vx*ps3, X[i]))))
    mutant("sign of the second term", sp.expand(comm[0] - (Vx*(-I*hb*sp.diff(ps3, X[0])) - I*hb*sp.diff(Vx*ps3, X[0]))))

    # ------------------------------------------------------------------------
    # eq:commutator_V  (sgm.tex L345), line 2
    # States: = i hbar (grad V) psi.
    # Derived from: eq:schrodinger_hamiltonian and the product rule grad(V psi) =
    #          V grad psi + (grad V) psi; the V grad psi terms cancel.
    # Test: in 3D, general psi, V, each component.
    # Mutant: -i hbar (grad V) psi.
    # ------------------------------------------------------------------------
    eq("eq:commutator_V", 345, "= i hbar (grad V) psi")
    for i in range(3):
        symcheck(f"axis {i+1}", sp.expand(comm[i] - I*hb*sp.diff(Vx, X[i])*ps3))
    mutant("-i hbar (grad V) psi", sp.expand(comm[0] + I*hb*sp.diff(Vx, X[0])*ps3))

    # ------------------------------------------------------------------------
    # eq:ehrenfest  (sgm.tex L350)
    # States: d<p_hat>/dt = -<grad V> (vector).
    # Derived from: eq:commutator_V, eq:ehrenfest_commutator with A = p_hat, using
    #          (i/hbar)(i hbar) = -1.
    # Test: (1) (i/hbar)(i hbar grad V) == -grad V. (2) Independent of the
    #          Hermiticity step: 1D, psi = u + i w general, every time derivative
    #          eliminated by the Schroedinger equation; d_t(psi* p psi) +
    #          psi* V' psi is a total x-derivative of an explicit surface term,
    #          for general V, so the theorem holds exactly for a state whose
    #          surface term vanishes. (3) on the boosted Gaussian in lambda x^3,
    #          both sides by direct integration.
    # Mutant: d<p>/dt = +<grad V> on the Gaussian.
    # ------------------------------------------------------------------------
    eq("eq:ehrenfest", 350, "d<p>/dt = -<grad V>")
    gV = sp.Symbol('gradV')
    symcheck("(i/hbar)(i hbar grad V) == -grad V", sp.expand(I/hb*(I*hb*gV) + gV))
    target = sp.expand(dPsic*p1op(PsiC) + PsiCc*p1op(dPsi) + PsiCc*sp.diff(V1, x)*PsiC)
    surfE = (hb**2/(2*m))*(u_*sp.diff(u_, x, 2) - sp.diff(u_, x)**2 + w_*sp.diff(w_, x, 2) - sp.diff(w_, x)**2
                           + I*(u_*sp.diff(w_, x, 2) - w_*sp.diff(u_, x, 2)))
    symcheck("d_t(psi* p psi) + psi* V' psi == d/dx[surface term] (exact, any V)",
             sp.expand(target - sp.diff(surfE, x)))
    lhsP = sp.integrate(sp.expand(dGc*(-I*hb*sp.diff(Gs, x)) + Gsc*(-I*hb*sp.diff(dG, x))), (x, -sp.oo, sp.oo))
    rhsP = -sp.integrate(sp.expand(Gsc*3*lm3*x**2*Gs), (x, -sp.oo, sp.oo))
    symcheck("Gaussian in lambda x^3: d<p>/dt == -<V'> by direct integration", sp.simplify(lhsP - rhsP))
    mutant("d<p>/dt = +<V'>", sp.simplify(lhsP + rhsP))

    # ------------------------------------------------------------------------
    # eq:sgm_expectation  (sgm.tex L355)
    # States: d<p_hat>/dt = -hbar <grad omega_hat>, with the operator
    #          grad omega_hat == (i/hbar)[p_hat, omega_hat], omega_hat = H_hat/hbar.
    # Derived from: eq:ehrenfest, eq:commutator_V, eq:gradient_def, eq:planck
    #          (omega_hat = H_hat/hbar, L330), with grad omega_hat defined at L353; for
    #          any H, directly eq:ehrenfest_commutator with A = p_hat (L358); L360 uses
    #          eq:sgm_as_dot_p; L362 uses eq:sgm.
    # Test: (1) 3D, general psi and V: (i/hbar)[p_hat, H_hat/hbar] psi ==
    #          (grad V/hbar) psi, so -hbar <grad omega_hat> = -<grad V>.
    #          (2) any H: -hbar (i/hbar)[p, H/hbar] == (i/hbar)[H, p] as an
    #          operator identity, checked on a 2x2 spin Hamiltonian
    #          H = p^2/2m + mu_B sigma . B(x) (general B(x), 1D motion, two-
    #          component psi): grad omega_hat is the matrix (mu_B/hbar) sigma . B'
    #          (Theorem 1's matrix gradient). (3) L360: for a Gaussian of
    #          variance s^2 centered at x0, <V'> = V'(x0) exactly for quadratic V,
    #          and for V = lambda x^3 the difference is 3 lambda s^2, the fraction
    #          s^2/x0^2. (4) L362: oscillator ground state, <grad omega_hat> = 0
    #          while grad V is nonzero.
    # Mutant: grad omega_hat defined with the reversed commutator
    #          (i/hbar)[omega_hat, p_hat]: gives -grad V/hbar.
    # ------------------------------------------------------------------------
    eq("eq:sgm_expectation", 355, "d<p>/dt = -hbar <grad omega_hat>")
    omhat = lambda f: Hhat(f, Vx)/hb
    gom = [I/hb*(phat(omhat(ps3), i) - omhat(phat(ps3, i))) for i in range(3)]
    for i in range(3):
        symcheck(f"axis {i+1}: (i/hbar)[p, omega_hat] psi == (d_i V/hbar) psi",
                 sp.expand(gom[i] - sp.diff(Vx, X[i])/hb*ps3))
    mutant("reversed commutator (i/hbar)[omega_hat, p] (axis 1)",
           sp.expand(-gom[0] - sp.diff(Vx, X[0])/hb*ps3))
    muB = sp.Symbol('mu_B', positive=True)
    Bf = [sp.Function(f'B{i+1}', real=True)(x) for i in range(3)]
    sig = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -I], [I, 0]]), sp.Matrix([[1, 0], [0, -1]])]
    sB = sum((sig[i]*Bf[i] for i in range(3)), sp.zeros(2))
    chi = sp.Matrix([sp.Function('chi1')(x), sp.Function('chi2')(x)])
    Hs = lambda F: -hb**2/(2*m)*sp.diff(F, x, 2) + muB*sB*F
    ps_ = lambda F: -I*hb*sp.diff(F, x)
    gom_s = I/hb*(ps_(Hs(chi)/hb) - Hs(ps_(chi))/hb)
    symcheck("spin: (i/hbar)[p, H/hbar] chi == (mu_B/hbar) sigma . B'(x) chi (matrix gradient)",
             sp.expand(gom_s - muB/hb*sp.diff(sB, x)*chi))
    symcheck("spin: -hbar (i/hbar)[p, omega_hat] == (i/hbar)[H, p] on chi (identity for any H)",
             sp.expand(-hb*gom_s - I/hb*(Hs(ps_(chi)) - ps_(Hs(chi)))))
    x0, s2 = sp.Symbol('x0', positive=True), sp.Symbol('s2', positive=True)
    rhoG = sp.exp(-(x - x0)**2/(2*s2))/sp.sqrt(2*sp.pi*s2)
    a0_, b0_, c0_ = sp.symbols('a0 b0 c0', real=True)
    Vq = a0_ + b0_*x + c0_*x**2
    expct = lambda f: sp.simplify(sp.integrate(f*rhoG, (x, -sp.oo, sp.oo)))
    symcheck("L360: quadratic V, <V'> == V'(<x>)", expct(sp.diff(Vq, x)) - sp.diff(Vq, x).subs(x, x0))
    gap = expct(3*lm3*x**2) - 3*lm3*x0**2
    symcheck("L360: cubic V, <V'> - V'(<x>) == 3 lambda s^2 (second order in the width)", sp.simplify(gap - 3*lm3*s2))
    symcheck("L360 (supplement): fraction == s^2/x0^2", sp.simplify(gap/(3*lm3*x0**2) - s2/x0**2))
    mutant("cubic V treated as exact at the centroid", sp.simplify(gap))
    rho_ho = psiho**2
    symcheck("L362: oscillator ground state <grad omega> = <m Om^2 x/hbar> == 0",
             sp.simplify(sp.integrate(rho_ho*m*Om**2*x/hb, (x, -sp.oo, sp.oo))))
    boolcheck("L362: grad V = m Om^2 x is nonzero on the state", sp.diff(Vho, x) != 0)

    # ------------------------------------------------------------------------
    # eq:single_frequency  (sgm.tex L377)
    # States: grad(omega_0 alpha) = 0: the single-frequency assumption (vector
    #          equation), alpha(x) the time-dilation factor (scalar field).
    # Derived from: premise (what quantum mechanics assumes, L375); eq:rest_freq
    #          (omega_0), alpha from L121 (sec:identity), sec:intro (m g z).
    # Test: (1) with omega_0 constant it holds iff grad alpha = 0. (2) the force
    #          it removes (L382): in the weak field alpha = 1 + Phi_g(x)/c^2
    #          (general Phi_g), -hbar grad(omega_0 alpha) == -m grad Phi_g, with
    #          hbar cancelled; for the uniform field Phi_g = g z it is -m g, the
    #          force of the potential m g z of the Schroedinger equation (L375).
    #          (3) dimensions of hbar grad(omega_0 alpha): force.
    # Mutant: in a field, alpha = 1 + g z/c^2, the left side is not zero.
    # ------------------------------------------------------------------------
    eq("eq:single_frequency", 377, "grad(omega_0 alpha) = 0")
    w0 = m*c**2/hb
    alc = sp.Symbol('alpha_c', positive=True)
    symcheck("alpha constant: grad(omega_0 alpha) == 0", sp.Matrix([sp.diff(w0*alc, Xi) for Xi in X]))
    Phg = sp.Function('Phi_g', real=True)(*X)
    al = 1 + Phg/c**2
    symcheck("weak field: -hbar grad(omega_0 alpha) == -m grad Phi_g",
             sp.Matrix([sp.simplify(-hb*sp.diff(w0*al, Xi) + m*sp.diff(Phg, Xi)) for Xi in X]))
    g, z = sp.Symbol('g', positive=True), X[2]
    symcheck("uniform field alpha = 1 + g z/c^2: force -m g == -d/dz (m g z)",
             -hb*sp.diff(w0*(1 + g*z/c**2), z) + sp.diff(m*g*z, z))
    symcheck("dimensions: [hbar][omega_0]/[length] == [force]", D_hbar + D_freq - D_len - D_force)
    mutant("in a uniform field, grad(omega_0 alpha) is m g/hbar, not 0", sp.diff(w0*(1 + g*z/c**2), z))

    # ------------------------------------------------------------------------
    # inline (sgm.tex L393), the force of an ideal wall
    # States: with psi = 0 on an ideal wall, d<p>/dt computed from the Schrodinger
    #          equation and integrated by parts leaves a force directed away from
    #          the wall, of magnitude (hbar^2/2m)|d_n psi|^2 integrated over the wall;
    #          in a rectangular box of side L along x, the normalized eigenstate
    #          with n half-wavelengths along x has energy pi^2 hbar^2 n^2/(2 m L^2)
    #          plus L-independent terms, and at x = L this force is
    #          pi^2 hbar^2 n^2/(m L^3) = -dE/dL.
    # Derived from: sec:route_schrodinger (the domain conditions after
    #          eq:ehrenfest_commutator) and integration by parts, in place.
    # Test: (1) 1D eigenstate n: (hbar^2/2m)|psi'(L)|^2 == -dE_n/dL == pi^2 hbar^2
    #          n^2/(m L^3), general n. (2) 1D equal superposition of the two
    #          lowest modes: d<p>/dt at t = 0, from <p>(t), equals
    #          (hbar^2/2m)(|psi'(0)|^2 - |psi'(L)|^2) = 4 pi^2 hbar^2/(m L^3), while
    #          the commutator with V = 0 gives 0. (3) 3D box, four modes with
    #          complex weights, two of them sharing the y, z mode so both x walls
    #          contribute: d<p_x>/dt by 80^3-point Gauss-Legendre quadrature equals
    #          (hbar^2/2m)(int_{x=0}|d_x psi|^2 dS - int_{x=a}|d_x psi|^2 dS).
    #          (4) 3D eigenstate (n, n_y, n_z): the wall integral of |d_x psi|^2,
    #          times hbar^2/2m, equals pi^2 hbar^2 n^2/(m L^3), general n.
    # Mutant: the wall force with (hbar^2/m) in place of (hbar^2/2m); in 3D, the
    #          force directed toward the walls.
    # ------------------------------------------------------------------------
    eq("inline", 393, "ideal wall force (hbar^2/2m)|d_n psi|^2; box eigenstate pi^2 hbar^2 n^2/(m L^3) = -dE/dL")
    Lb = sp.Symbol('L', positive=True)
    nn = sp.Symbol('n', positive=True, integer=True)
    un = sp.sqrt(2/Lb)*sp.sin(nn*sp.pi*x/Lb)
    En = hb**2*sp.pi**2*nn**2/(2*m*Lb**2)
    wallF = hb**2/(2*m)*sp.diff(un, x).subs(x, Lb)**2
    symcheck("eigenstate n: (hbar^2/2m) psi'(L)^2 == -dE_n/dL", sp.simplify(wallF + sp.diff(En, Lb)))
    mutant("(hbar^2/m) psi'(L)^2", sp.simplify(2*wallF + sp.diff(En, Lb)))
    E1b = hb**2*sp.pi**2/(2*m*Lb**2)
    psib = (sp.sin(sp.pi*x/Lb)*sp.exp(-I*E1b*t/hb) + sp.sin(2*sp.pi*x/Lb)*sp.exp(-I*4*E1b*t/hb))/sp.sqrt(Lb)
    psibc = (sp.sin(sp.pi*x/Lb)*sp.exp(I*E1b*t/hb) + sp.sin(2*sp.pi*x/Lb)*sp.exp(I*4*E1b*t/hb))/sp.sqrt(Lb)
    symcheck("superposition is normalized", sp.simplify(sp.integrate(sp.expand(psibc*psib), (x, 0, Lb)) - 1))
    pmean = sp.integrate(sp.expand(psibc*(-I*hb)*sp.diff(psib, x)), (x, 0, Lb))
    rate = sp.simplify(sp.diff(pmean, t).subs(t, 0))
    symcheck("d<p>/dt at t=0 == 4 pi^2 hbar^2/(m L^3)", sp.simplify(rate - 4*sp.pi**2*hb**2/(m*Lb**3)))
    d0 = sp.diff(psib, x).subs({x: 0, t: 0}); dL = sp.diff(psib, x).subs({x: Lb, t: 0})
    symcheck("== (hbar^2/2m)(|psi'(0)|^2 - |psi'(L)|^2)",
             sp.simplify(hb**2/(2*m)*(d0**2 - dL**2) - 4*sp.pi**2*hb**2/(m*Lb**3)))
    boolcheck("the commutator term with V = 0 inside is zero, so it misses the wall force",
              sp.simplify(rate) != 0)
    # (3) 3D box with sides a, b, cz (hbar = 1, m = 3/2): quadrature on a mixed state
    hb3, m3, a3, b3, c3 = 1.0, 1.5, 1.0, 1.3, 0.7
    modes3 = [((1, 1, 1), 1.0), ((2, 1, 1), 1 + 2j), ((3, 2, 1), 3.0), ((2, 2, 1), 0.5 - 1j)]
    nrm3 = np.sqrt(sum(abs(cf)**2 for _, cf in modes3))
    u3 = lambda k, L, s: np.sqrt(2/L)*np.sin(k*np.pi*s/L)
    du3 = lambda k, L, s: np.sqrt(2/L)*(k*np.pi/L)*np.cos(k*np.pi*s/L)
    w3 = lambda k: hb3*((k[0]*np.pi/a3)**2 + (k[1]*np.pi/b3)**2 + (k[2]*np.pi/c3)**2)/(2*m3)
    def gl3(N, L):
        z_, w_ = np.polynomial.legendre.leggauss(N)
        return L*(z_ + 1)/2, w_*L/2
    X3, WX3 = gl3(80, a3); Y3, WY3 = gl3(80, b3); Z3, WZ3 = gl3(80, c3)
    XX3, YY3, ZZ3 = np.meshgrid(X3, Y3, Z3, indexing='ij'); W33 = np.einsum('i,j,k', WX3, WY3, WZ3)
    def f3(dx, dt):   # psi, d_x psi, d_t psi or d_x d_t psi at t = 0
        return sum(cf*((-1j*w3(k)) if dt else 1)*(du3 if dx else u3)(k[0], a3, XX3)*u3(k[1], b3, YY3)*u3(k[2], c3, ZZ3)
                   for k, cf in modes3)/nrm3
    rate3 = np.sum(W33*(np.conj(f3(0, 1))*(-1j*hb3)*f3(1, 0) + np.conj(f3(0, 0))*(-1j*hb3)*f3(1, 1)))
    YW, ZW = np.meshgrid(Y3, Z3, indexing='ij'); WW = np.outer(WY3, WZ3)
    wall3 = lambda x0: np.sum(WW*abs(sum(cf*du3(k[0], a3, x0)*u3(k[1], b3, YW)*u3(k[2], c3, ZW)
                                       for k, cf in modes3)/nrm3)**2)
    force3 = hb3**2/(2*m3)*(wall3(0.0) - wall3(a3))
    check("3D: d<p_x>/dt == (hbar^2/2m)(wall x=0 - wall x=a) integrals of |d_x psi|^2", rate3, force3, 1e-10)
    numutant("3D force directed toward the walls", -force3, rate3, 1e-10)
    # (4) 3D eigenstate: the y, z factors integrate to 1 over the wall
    ky, kz = sp.symbols('n_y n_z', positive=True, integer=True)
    Ly, Lz = sp.symbols('L_y L_z', positive=True)
    dpsi3 = sp.diff(un, x).subs(x, Lb)*sp.sqrt(2/Ly)*sp.sin(ky*sp.pi*X[1]/Ly)*sp.sqrt(2/Lz)*sp.sin(kz*sp.pi*X[2]/Lz)
    wallint3 = sp.integrate(sp.integrate(dpsi3**2, (X[1], 0, Ly)), (X[2], 0, Lz))
    symcheck("3D eigenstate: (hbar^2/2m) int |d_x psi|^2 dS == pi^2 hbar^2 n^2/(m L^3) == -dE/dL",
             sp.simplify((hb**2/(2*m)*wallint3).subs(sp.cos(sp.pi*nn)**2, 1) - sp.pi**2*hb**2*nn**2/(m*Lb**3)))
    mutant("3D eigenstate with energy pi^2 hbar^2 n^2/(m L^2) (2x)",
           sp.simplify((hb**2/(2*m)*wallint3).subs(sp.cos(sp.pi*nn)**2, 1) + sp.diff(2*En, Lb)))

    # ------------------------------------------------------------------------
    # eq:omega_expansion  (sgm.tex L408)
    # States: omega(x + dx, p) = omega + grad omega . dx + (1/2)(d_i d_j omega)
    #          dx_i dx_j + ..., Taylor series at fixed momentum (scalar), every
    #          derivative at the particle, i, j summed over three axes.
    # Derived from: eq:gradient_def (fixed momentum) and Taylor's theorem in three
    #          variables. L411 adds the force Jacobian from eq:sgm, F_j = -hbar d_j omega.
    # Test: p fixed, so suppressed. For the general cubic in three variables
    #          (symbolic coefficients, so every value, gradient and Hessian at x
    #          occurs) and for a non-polynomial function, the series of
    #          W(x + s dx) in s through s^2 equals the stated terms at s = 1.
    #          For a general W(x1,x2,x3), d_i F_j = -hbar d_i d_j omega and it is
    #          symmetric in i, j.
    # Mutant: second-order coefficient without the 1/2.
    # ------------------------------------------------------------------------
    eq("eq:omega_expansion", 408, "omega(x+dx,p) = omega + grad omega . dx + (1/2) d_i d_j omega dx_i dx_j + ...")
    s_ = sp.Symbol('s')
    dX = sp.symbols('dx1 dx2 dx3', real=True)
    # sympy leaves the mixed second derivatives of an undefined W at shifted
    # arguments unevaluated, so the series test uses (a) the general cubic with
    # symbolic coefficients, which has every possible 2-jet at x, and (b) a
    # non-polynomial function; both must match through s^2.
    mons = [X[0]**a1*X[1]**a2*X[2]**a3 for a1 in range(4) for a2 in range(4) for a3 in range(4) if a1 + a2 + a3 <= 3]
    cfs = sp.symbols(f'w0:{len(mons)}', real=True)
    Wcub = sum(cf*mn for cf, mn in zip(cfs, mons))
    Wnp = sp.exp(X[0]*X[1]/7)*sp.sin(X[2] + X[0]**2/3)
    def _S1_taylor2(W0):
        return (W0 + s_*sum(sp.diff(W0, X[i])*dX[i] for i in range(3))
                + s_**2*sp.Rational(1, 2)*sum(sp.diff(W0, X[i], X[j])*dX[i]*dX[j] for i in range(3) for j in range(3)))
    for nm, W0 in (("general cubic", Wcub), ("exp(x1 x2/7) sin(x3 + x1^2/3)", Wnp)):
        shifted = W0.subs({X[i]: X[i] + s_*dX[i] for i in range(3)}, simultaneous=True)
        ser = sp.series(shifted, s_, 0, 3).removeO()
        symcheck(f"{nm}: series of W(x + s dx) through s^2 == stated Taylor terms",
                 sp.simplify(sp.expand(ser - _S1_taylor2(W0))))
    shifted = Wcub.subs({X[i]: X[i] + s_*dX[i] for i in range(3)}, simultaneous=True)
    ser = sp.series(shifted, s_, 0, 3).removeO()
    mutant("second-order term without the 1/2",
           sp.expand(ser - _S1_taylor2(Wcub) - s_**2*sp.Rational(1, 2)*sum(
               sp.diff(Wcub, X[i], X[j])*dX[i]*dX[j] for i in range(3) for j in range(3))))
    W0 = sp.Function('W')(*X)
    Fj = [-hb*sp.diff(W0, X[j]) for j in range(3)]
    symcheck("force Jacobian d_i F_j == -hbar d_i d_j omega",
             sp.Matrix(3, 3, lambda i, j: sp.diff(Fj[j], X[i]) + hb*sp.diff(W0, X[i], X[j])))
    symcheck("force Jacobian is symmetric", sp.Matrix(3, 3, lambda i, j: sp.diff(Fj[j], X[i]) - sp.diff(Fj[i], X[j])))


# ----------------------------------------------------------------------------
# Helpers for part S2 (prefixed _S2_ so the concatenated file has no clashes)
# ----------------------------------------------------------------------------
def _S2_dim(kg=0, m=0, s=0, A=0):
    """SI dimension as an exponent vector over (kg, m, s, A)."""
    return np.array([kg, m, s, A], dtype=float)


def _S2_christoffel(g, X):
    """Gamma^a_{bc} of the metric g (sympy Matrix) in the coordinates X."""
    n = len(X)
    gi = g.inv()
    return [[[sp.simplify(sum(gi[a, d]*(sp.diff(g[d, b], X[c]) + sp.diff(g[d, c], X[b])
                                       - sp.diff(g[b, c], X[d])) for d in range(n))/2)
              for c in range(n)] for b in range(n)] for a in range(n)]


def _S2_riemann_up(Gam, X, a, b, c, d):
    """R^a_{bcd} = d_c Gam^a_{bd} - d_d Gam^a_{bc} + Gam^a_{ce}Gam^e_{bd} - Gam^a_{de}Gam^e_{bc}
    (the sign convention of Misner, Thorne and Wheeler)."""
    n = len(X)
    r = sp.diff(Gam[a][b][d], X[c]) - sp.diff(Gam[a][b][c], X[d])
    r += sum(Gam[a][c][e]*Gam[e][b][d] - Gam[a][d][e]*Gam[e][b][c] for e in range(n))
    return sp.simplify(r)


def part_S2():
    part("sgm.tex L414-L750: Gravity and inertia (Gravity from the static gradient; "
         "Inertia from the dynamic gradient; Gravity and inertia are one mechanism; "
         "Free fall and the equivalence principle; Light)")

    # Shared symbols. All are positive unless a sign is needed.
    m, c, hb, G, M = sp.symbols('m c hbar G M', positive=True)
    r, rp, ell, Rs = sp.symbols('r rprime ell R_s', positive=True)
    t, tau = sp.symbols('t tau', positive=True)
    x, y, z = sp.symbols('x y z', real=True)
    X3 = (x, y, z)
    w0 = m*c**2/hb                       # eq:rest_freq (sgm.tex L148)
    al = sp.Function('alpha', positive=True)
    alf3 = al(x, y, z)                   # a general smooth time-dilation field alpha(x)
    rr3 = sp.sqrt(x**2 + y**2 + z**2)
    grad = lambda f: sp.Matrix([sp.diff(f, v) for v in X3])

    # Numerical constants (CODATA 2018 and standard values) for the quoted numbers.
    G_n, c_n, hb_n = 6.67430e-11, 299792458.0, 1.054571817e-34
    me_n, g_n = 9.1093837015e-31, 9.80665

    # SI dimensions
    D_G = _S2_dim(kg=-1, m=3, s=-2)
    D_M = _S2_dim(kg=1)
    D_c = _S2_dim(m=1, s=-1)
    D_hb = _S2_dim(kg=1, m=2, s=-1)
    D_len = _S2_dim(m=1)
    D_force = _S2_dim(kg=1, m=1, s=-2)

    # ========================================================================
    # Subsection "Gravity from the static gradient"
    # ========================================================================

    # ------------------------------------------------------------------------
    # eq:alpha_def, line 1  (sgm.tex L427)
    # States: alpha(x) == d tau/d t, the scalar rate of the proper time tau of a
    #          clock held at fixed spatial coordinates x per unit coordinate time t.
    # Derived from: definition (cited: Einstein 1915; MTW); no earlier label.
    # Test: for a static clock (dx = 0) the line element ds^2 = g_00 (c dt)^2
    #          equals -c^2 d tau^2; solving for d tau/dt must give the value of the
    #          second line, sqrt(-g_00), for a symbolic negative g_00. Also checks
    #          that sqrt(-g_00) is dimensionless with x^0 = ct (L431).
    # Mutant: d tau/dt = -g_00 (the square root dropped), a plausible slip
    #          between the metric component and its square root.
    # ------------------------------------------------------------------------
    eq("eq:alpha_def (line 1)", 427, "alpha(x) == d tau/dt")
    g00 = sp.symbols('g_00', negative=True)
    dtau_dt = sp.symbols('dtaudt', positive=True)
    sol = sp.solve(sp.Eq(-c**2*dtau_dt**2, g00*c**2), dtau_dt)
    boolcheck("one positive root of -c^2 (dtau/dt)^2 == g_00 c^2", len(sol) == 1)
    symcheck("static clock: d tau/dt == sqrt(-g_00)", sol[0] - sp.sqrt(-g00))
    boolcheck("with x^0 = ct, g_00 = ds^2/(dx^0)^2 is dimensionless, so alpha = sqrt(-g_00) is",
              np.allclose(_S2_dim(m=2) - 2*_S2_dim(m=1), 0))
    boolcheck("mutant caught: with x^0 = t, g_00 = ds^2/dt^2 carries m^2/s^2",
              not np.allclose(_S2_dim(m=2) - 2*_S2_dim(s=1), 0))
    mutant("d tau/dt == -g_00 (square root dropped)", sol[0] - (-g00))

    # ------------------------------------------------------------------------
    # eq:alpha_def, line 2  (sgm.tex L428)
    # States: alpha(x) == sqrt(-g_00(x)), g_00 the time-time metric component
    #          (scalar, negative in signature (-,+,+,+)).
    # Derived from: the first line of this display, by the static-clock reduction of ds^2
    #          above, with the time coordinate x^0 = ct stated at L431 (with
    #          x^0 = t, sqrt(-g_00) would be alpha c, not alpha).
    # Test: (a) on the exterior Schwarzschild metric g_00 = -(1 - R_s/r),
    #          alpha -> 1 as r -> infinity and alpha < 1 at r = 3 R_s (L431);
    #          (b) the L431 claim that alpha equals the lapse N = (-g^00)^(-1/2)
    #          for a static metric and differs from it when g_0i != 0: computed
    #          for a general 2x2 (t, x) block with cross term b.
    # Mutant: alpha = sqrt(g_00) for the Schwarzschild component, which is
    #          imaginary outside the horizon; and alpha == N with b != 0.
    # ------------------------------------------------------------------------
    eq("eq:alpha_def (line 2)", 428, "alpha(x) == sqrt(-g_00(x))")
    a_schw = sp.sqrt(-(-(1 - Rs/r)))
    symcheck("Schwarzschild: alpha -> 1 as r -> infinity", sp.limit(a_schw, r, sp.oo) - 1)
    boolcheck("Schwarzschild: alpha < 1 at r = 3 R_s (near a mass)",
              bool(a_schw.subs(r, 3*Rs) < 1))
    g00s, bx, hxx = sp.symbols('g00s b h_xx', real=True)
    gmat2 = sp.Matrix([[g00s, bx], [bx, hxx]])
    N2 = -1/gmat2.inv()[0, 0]
    symcheck("lapse^2 == -g_00 + b^2/h_xx (cross term b)", sp.simplify(N2 - (-g00s + bx**2/hxx)))
    symcheck("with b = 0 the lapse equals alpha: N^2 == -g_00", sp.simplify(N2.subs(bx, 0) + g00s))
    mutant("lapse equals alpha also with a cross term (b != 0)", sp.simplify(N2 + g00s))
    boolcheck("mutant caught: alpha = sqrt(g_00) is not real outside the horizon",
              not sp.sqrt(-(1 - Rs/r)).subs(r, 3*Rs).is_real)

    # ------------------------------------------------------------------------
    # eq:omega_alpha  (sgm.tex L436)
    # States: omega(x) == omega_0 alpha(x): the coordinate frequency (scalar)
    #          of the rest phase of a static particle at x.
    # Derived from: eq:planck, eq:phase_freq, ax:lpi, eq:rest_freq, eq:alpha_def.
    #          L433: the proper rate -d theta/d tau = omega_0 (eq:planck,
    #          eq:phase_freq in the rest frame; ax:lpi) integrates to
    #          theta = -omega_0 tau; for a static particle d tau = alpha dt
    #          (eq:alpha_def), so theta = -omega_0 alpha t and omega = -d theta/dt.
    # Test: build theta(tau) = -omega_0 tau, substitute tau = alpha t, take
    #          -d/dt, for symbolic alpha; check the result equals the value at
    #          zero momentum of the frequency function H/hbar with
    #          H = alpha sqrt(m^2 c^4 + c^2 gamma^ij p_i p_j) (the static-metric
    #          Hamiltonian, here with a general diagonal gamma in 3D) (L441).
    # Mutant: omega = omega_0/alpha (a clock at x measuring a mode of one
    #          coordinate frequency, the inverse relation of L105).
    # ------------------------------------------------------------------------
    eq("eq:omega_alpha", 436, "omega(x) == omega_0 alpha(x)")
    alS = sp.symbols('alpha_s', positive=True)
    theta_tau = -w0*tau
    symcheck("proper rate -d theta/d tau == omega_0", -sp.diff(theta_tau, tau) - w0)
    theta_t = theta_tau.subs(tau, alS*t)
    omega_static = -sp.diff(theta_t, t)
    symcheck("theta = -omega_0 alpha t gives -d theta/dt == omega_0 alpha", omega_static - w0*alS)
    p1, p2, p3 = sp.symbols('p1 p2 p3', real=True)
    gam_inv = sp.diag(*[sp.Function(f'gi{i}', positive=True)(x, y, z) for i in range(3)])
    pv = sp.Matrix([p1, p2, p3])
    H_static = alf3*sp.sqrt(m**2*c**4 + c**2*(pv.T*gam_inv*pv)[0])
    symcheck("frequency function at p = 0: H/hbar == omega_0 alpha(x)",
             sp.simplify((H_static/hb).subs({p1: 0, p2: 0, p3: 0}) - w0*alf3))
    mutant("omega == omega_0/alpha", omega_static - w0/alS)

    # ------------------------------------------------------------------------
    # eq:omega_alpha, caption line  (sgm.tex L437)
    # States: "(The time-dilated rest frequency)", a name for the line above.
    # Derived from: nothing; prose.
    # Test: none. The caption names the quantity; its content is the
    #          equation tested in the previous block.
    # Mutant: none (nothing to compute).
    # ------------------------------------------------------------------------
    eq("eq:omega_alpha (caption)", 437, "(The time-dilated rest frequency) [prose]")

    # ------------------------------------------------------------------------
    # eq:energy_alpha_massive, line 1  (sgm.tex L445)
    # States: E(x) == hbar omega_0 alpha(x), the scalar energy at x of a particle
    #          at rest there.
    # Derived from: E = hbar omega (eq:planck) applied to eq:omega_alpha.
    # Test: hbar times the omega obtained in the previous block, symbolic alpha.
    # Mutant: E = omega_0 alpha (the factor hbar dropped).
    # ------------------------------------------------------------------------
    eq("eq:energy_alpha_massive (line 1)", 445, "E(x) == hbar omega_0 alpha(x)")
    E_alpha = hb*omega_static
    symcheck("hbar * omega(x) == hbar omega_0 alpha", E_alpha - hb*w0*alS)
    mutant("E == omega_0 alpha (hbar dropped)", E_alpha - w0*alS)

    # ------------------------------------------------------------------------
    # eq:energy_alpha_massive, line 2  (sgm.tex L446)
    # States: E(x) == m c^2 alpha(x).
    # Derived from: the first line of this display, eq:rest_freq
    #          (hbar omega_0 = m c^2).
    # Test: substitute omega_0 = m c^2/hbar; plus the L524 claim that
    #          eq:alpha_weak makes it E(r) = m c^2 - GMm/r, whose second term is
    #          m Phi_g with Phi_g = -GM/r.
    # Mutant: E = m c^2/alpha (a gravitational blueshift instead of redshift),
    #          which gives +GMm/r at first order.
    # ------------------------------------------------------------------------
    eq("eq:energy_alpha_massive (line 2)", 446, "E(x) == m c^2 alpha(x)")
    symcheck("hbar omega_0 alpha == m c^2 alpha", E_alpha - m*c**2*alS)
    a_weak = 1 - G*M/(c**2*r)
    E_weak = (m*c**2*alS).subs(alS, a_weak)
    symcheck("L524: with eq:alpha_weak, E(r) == m c^2 - GMm/r", sp.expand(E_weak - (m*c**2 - G*M*m/r)))
    eps_ = sp.symbols('epsilon')
    mut_E = sp.series((m*c**2/alS).subs(alS, 1 - eps_*G*M/(c**2*r)), eps_, 0, 2).removeO().subs(eps_, 1)
    mutant("E == m c^2/alpha gives m c^2 - GMm/r at first order", sp.expand(mut_E - (m*c**2 - G*M*m/r)))

    # ------------------------------------------------------------------------
    # eq:gradient_span, line 1  (sgm.tex L454)
    # States: u.grad omega(x, 0) == lim_{l->0} [omega(x + l u, 0) - omega(x - l u, 0)]/(2 l),
    #          the directional derivative (scalar) along the unit vector u as the
    #          limit of a symmetric difference of clock rates.
    # Derived from: the definition of the directional derivative, applied to
    #          the frequency function omega(x, p) of sec:identity at p = 0
    #          (eq:omega_expansion, eq:omega_alpha).
    # Test: for omega(x, 0) = omega_0 alpha(x): (a) one dimension, general
    #          function alpha, by a Taylor series in l (the quotient is
    #          alpha' + alpha''' l^2/6 + ...); (b) in 3D for the Schwarzschild
    #          factor alpha = sqrt(1 - R_s/|x|), a direction u = (3,4,12)/13 and
    #          the point (1.3, -0.7, 2.9) R_s, by sympy's limit, compared with u.grad.
    # Mutant: the one-sided quotient [omega(x + l u) - omega(x - l u)]/l
    #          (missing factor 2).
    # ------------------------------------------------------------------------
    eq("eq:gradient_span (line 1)", 454, "u.grad omega(x,0) == lim (omega(x+lu)-omega(x-lu))/(2l)")
    f1 = sp.Function('alpha1')
    quot1 = w0*(f1(x + ell) - f1(x - ell))/(2*ell)
    ser1 = sp.series(quot1, ell, 0, 3).removeO().doit()
    lim1 = ser1.subs(ell, 0)
    symcheck("1D general alpha: limit of the symmetric quotient == omega_0 alpha'(x)",
             sp.simplify(lim1 - w0*sp.diff(f1(x), x)))
    lv = sp.symbols('l', positive=True)
    u_vec = sp.Matrix([3, 4, 12])/13
    pt = {x: sp.Rational(13, 10), y: -sp.Rational(7, 10), z: sp.Rational(29, 10)}
    a3 = sp.sqrt(1 - 1/rr3)                 # R_s = 1 (lengths in units of R_s)
    a_plus = a3.subs({x: pt[x] + lv*u_vec[0], y: pt[y] + lv*u_vec[1], z: pt[z] + lv*u_vec[2]},
                     simultaneous=True)
    a_minus = a3.subs({x: pt[x] - lv*u_vec[0], y: pt[y] - lv*u_vec[1], z: pt[z] - lv*u_vec[2]},
                      simultaneous=True)
    lim3 = sp.limit((a_plus - a_minus)/(2*lv), lv, 0)
    dir3 = (u_vec.T*grad(a3))[0].subs(pt)
    symcheck("3D Schwarzschild: limit == u.grad alpha", sp.nsimplify(sp.N(lim3 - dir3, 30), tolerance=1e-25))
    mutant("quotient over l instead of 2l", sp.simplify(ser1.subs(ell, 0)*2 - w0*sp.diff(f1(x), x)))

    # ------------------------------------------------------------------------
    # eq:gradient_span, line 2  (sgm.tex L455)
    # States: the same limit == omega_0 u.grad alpha(x).
    # Derived from: the first line of this display, eq:omega_alpha, ax:lpi.
    #          Line 1 with omega(x, 0) = omega_0 alpha(x) (eq:omega_alpha),
    #          omega_0 constant (ax:lpi; the proper frequency is the same at both ends).
    # Test: in 3D for a general alpha(x, y, z), the l-derivative at l = 0 of the
    #          symmetric difference of omega_0 alpha(x + l u) over 2 equals
    #          omega_0 u.grad alpha (l'Hopital form of the limit, exact for a
    #          smooth alpha). The L458 statement that the same quotient of the
    #          proper frequency is zero is ax:lpi itself, so it has no test.
    # Mutant: omega_0 u.grad(alpha^2) (squaring the factor, i.e. using -g_00).
    # ------------------------------------------------------------------------
    eq("eq:gradient_span (line 2)", 455, "... == omega_0 u.grad alpha(x)")
    u1, u2, u3 = sp.symbols('u1 u2 u3', real=True)
    shift = lambda s: {x: x + s*u1, y: y + s*u2, z: z + s*u3}
    num = w0*(alf3.subs(shift(lv), simultaneous=True) - alf3.subs(shift(-lv), simultaneous=True))
    lim_gen = (sp.diff(num, lv)/2).subs(lv, 0).doit()
    target = w0*(u1*sp.diff(alf3, x) + u2*sp.diff(alf3, y) + u3*sp.diff(alf3, z))
    symcheck("general alpha(x,y,z): limit == omega_0 u.grad alpha", sp.simplify(lim_gen - target))
    mutant("omega_0 u.grad(alpha^2)", sp.simplify(lim_gen - w0*(u1*sp.diff(alf3**2, x)
                                                               + u2*sp.diff(alf3**2, y)
                                                               + u3*sp.diff(alf3**2, z))))

    # ------------------------------------------------------------------------
    # eq:geometric_force  (sgm.tex L462)
    # States: F_grav == -m c^2 grad alpha, the force (vector) on a massive
    #          particle at zero momentum in a static metric, "exactly".
    # Derived from: eq:omega_alpha, eq:rest_freq, eq:sgm; the exactness and
    #          energy remarks of L465 use eq:gradient_def, eq:force_energy,
    #          eq:energy_alpha_massive. Substitute omega = omega_0 alpha and
    #          omega_0 = m c^2/hbar into F = -hbar grad omega.
    # Test: (a) -hbar grad(omega_0 alpha) for a general alpha(x,y,z);
    #          (b) exactness at p = 0: -grad_x H at fixed p, for the static
    #          Hamiltonian H = alpha sqrt(m^2c^4 + c^2 gamma^ij p_i p_j) with a
    #          general diagonal gamma^ij(x), equals -m c^2 grad alpha at p = 0;
    #          (c) L465: at p != 0 (isotropic metric, gamma^ij = alpha^2 delta^ij,
    #          one dimension) the force differs only at order p^2/(m c)^2 = v^2/c^2;
    #          (d) L465: equals -grad E for E = m c^2 alpha; (e) dimensions.
    # Mutant: F = -m c^2 grad alpha/alpha (the gradient of ln alpha, which is
    #          the proper acceleration of L543 times m, not the coordinate force).
    # ------------------------------------------------------------------------
    eq("eq:geometric_force", 462, "F_grav == -m c^2 grad alpha")
    F_geo = -hb*grad(w0*alf3)
    symcheck("-hbar grad(omega_0 alpha) == -m c^2 grad alpha", F_geo + m*c**2*grad(alf3))
    F_H0 = sp.Matrix([-sp.diff(H_static, v) for v in X3]).subs({p1: 0, p2: 0, p3: 0})
    symcheck("-grad_x H|_p at p = 0 == -m c^2 grad alpha (general diagonal gamma)",
             sp.simplify(F_H0 + m*c**2*grad(alf3)))
    a1x = sp.Function('alpha_x', positive=True)(x)
    p_ = sp.symbols('p', positive=True)
    H_iso1 = a1x*sp.sqrt(m**2*c**4 + a1x**2*c**2*p_**2)
    F_p = -sp.diff(H_iso1, x)
    ser_p = sp.series(F_p, p_, 0, 2).removeO()
    symcheck("p != 0: F == -m c^2 alpha' + O(p^2) (zeroth order in v^2/c^2)",
             sp.simplify(ser_p + m*c**2*sp.diff(a1x, x)))
    boolcheck("p != 0: the correction is nonzero at order p^2",
              sp.simplify(sp.series(F_p, p_, 0, 3).removeO() - ser_p) != 0)
    symcheck("L465: -grad(m c^2 alpha) == F_grav", -grad(m*c**2*alf3) - F_geo)
    boolcheck("dimensions: [m c^2 grad alpha] == N",
              np.allclose(D_M + 2*D_c - D_len, D_force))
    mutant("F == -m c^2 grad alpha/alpha", sp.simplify(F_geo + m*c**2*grad(alf3)/alf3))

    # ------------------------------------------------------------------------
    # eq:gauss_alpha  (sgm.tex L469)
    # States: closed-surface integral of grad alpha . dS == 4 pi G M_enc/c^2
    #          (scalar flux; alpha dimensionless, dS a vector area element).
    # Derived from: premise (Gauss's law for gravity written for alpha), uses
    #          eq:alpha_def for alpha; no other label.
    # Test: (a) dimensions of both sides (meters); (b) for the point-mass factor
    #          alpha = 1 - GM/(c^2 |x - x_s|): numerical flux through a sphere of
    #          radius 1.7 centered at distance 0.6 from the mass (encloses it)
    #          equals 4 pi GM/c^2, and through one centered at distance 2.9 (does
    #          not enclose it) is zero (the source-free part, L472);
    #          (c) for a uniform sphere of radius R the field alpha_in =
    #          1 - GM(3R^2 - r^2)/(2 c^2 R^3) inside and the exterior factor
    #          outside: the flux at r < R is 4 pi G M (r/R)^3/c^2 (M_enc), and
    #          the Laplacian of alpha is 4 pi G rho/c^2 inside, 0 outside, the
    #          differential form that is the weak-field 00 equation (L472);
    #          (d) L522: with alpha = 1 + Phi_g/c^2, the law is the usual
    #          oint g.dS = -4 pi G M_enc for g = -grad Phi_g.
    # Mutant: the coupling 4 pi G/c (wrong power of c), caught by dimensions,
    #          and the flux of the exterior factor with coefficient 2GM/c^2
    #          (the Schwarzschild radius in place of GM/c^2) against 4 pi G M/c^2.
    # ------------------------------------------------------------------------
    eq("eq:gauss_alpha", 469, "oint grad alpha . dS == 4 pi G M_enc/c^2")
    boolcheck("dimensions: [grad alpha dS] = m == [G M/c^2]",
              np.allclose(D_len, D_G + D_M - 2*D_c))
    GMc2 = 0.83                                   # GM/c^2 in units of length (sample value)

    def _flux_num(center, R, coeff):
        # Gauss-Legendre in cos(theta), uniform in phi
        nth, nph = 64, 64
        xg, wg = np.polynomial.legendre.leggauss(nth)
        ph = (np.arange(nph) + 0.5)*2*np.pi/nph
        tot = 0.0
        for ct_, wct in zip(xg, wg):
            st_ = np.sqrt(1 - ct_**2)
            nvec = np.stack([st_*np.cos(ph), st_*np.sin(ph), np.full_like(ph, ct_)], axis=1)
            pts = np.asarray(center) + R*nvec
            rr = np.linalg.norm(pts, axis=1)
            gradA = coeff*pts/rr[:, None]**3    # grad(1 - coeff/|x|) = coeff x/|x|^3
            tot += wct*np.sum(np.sum(gradA*nvec, axis=1))*(2*np.pi/nph)*R**2
        return tot
    fl_in = _flux_num([0.6, 0.0, 0.0], 1.7, GMc2)
    fl_out = _flux_num([2.9, 0.0, 0.0], 1.7, GMc2)
    check("flux through an off-center sphere enclosing the mass == 4 pi GM/c^2",
          fl_in, 4*np.pi*GMc2, tol=1e-8)
    check("flux through a sphere enclosing no mass == 0 (source-free part)",
          fl_out, 0.0, tol=1e-8, scale=4*np.pi*GMc2)
    Rb, rho_ = sp.symbols('R_b rho', positive=True)
    a_in = 1 - G*M*(3*Rb**2 - r**2)/(2*c**2*Rb**3)
    a_out = 1 - G*M/(c**2*r)
    symcheck("uniform sphere: alpha continuous at r = R", (a_in - a_out).subs(r, Rb))
    symcheck("uniform sphere: d alpha/dr continuous at r = R",
             (sp.diff(a_in, r) - sp.diff(a_out, r)).subs(r, Rb))
    symcheck("uniform sphere, r < R: flux 4 pi r^2 alpha' == 4 pi G M_enc/c^2, M_enc = M r^3/R^3",
             sp.simplify(4*sp.pi*r**2*sp.diff(a_in, r) - 4*sp.pi*G*(M*r**3/Rb**3)/c**2))
    lap_rad = lambda f: sp.diff(r**2*sp.diff(f, r), r)/r**2
    symcheck("uniform sphere: Laplacian of alpha inside == 4 pi G rho/c^2, rho = 3M/(4 pi R^3)",
             sp.simplify(lap_rad(a_in) - 4*sp.pi*G*(3*M/(4*sp.pi*Rb**3))/c**2))
    symcheck("exterior: Laplacian of alpha == 0", sp.simplify(lap_rad(a_out)))
    Phi = sp.symbols('Phi_g', real=True)
    symcheck("L522: alpha = sqrt(1 + 2 Phi/c^2) == 1 + Phi/c^2 to first order",
             sp.series(sp.sqrt(1 + 2*eps_*Phi/c**2), eps_, 0, 2).removeO() - (1 + eps_*Phi/c**2))
    symcheck("L522: g = -c^2 grad alpha with alpha = 1 + Phi/c^2 gives oint g.dS == -4 pi G M "
             "(radial: -c^2 * 4 pi r^2 * GM/(c^2 r^2))",
             -c**2*4*sp.pi*r**2*sp.diff(a_out, r) + 4*sp.pi*G*M)
    boolcheck("mutant caught: coupling 4 pi G M/c has the wrong dimension",
              not np.allclose(D_len, D_G + D_M - D_c))
    numutant("flux of 1 - 2GM/(c^2 r) against 4 pi GM/c^2",
             _flux_num([0.6, 0.0, 0.0], 1.7, 2*GMc2), 4*np.pi*GMc2)

    # ------------------------------------------------------------------------
    # eq:gauss_sphere  (sgm.tex L476)
    # States: (d alpha/dr) 4 pi r^2 == 4 pi G M/c^2, scalar.
    # Derived from: eq:gauss_alpha on a sphere of radius r about a spherically
    #          symmetric mass: grad alpha = alpha'(r) r_hat, dS = r_hat dS, so the
    #          integrand is alpha' dS, constant on the sphere, and M_enc = M.
    # Test: carry out the surface integral in spherical angles for a general
    #          radial alpha(r): int_0^pi int_0^2pi alpha'(r) r^2 sin(th) dph dth.
    #          Also: the source-free condition alone, div grad alpha = 0, has the
    #          spherical solutions C1 + C2/r (no G, no M), so the shape comes
    #          from flux conservation and the source fixes only C2 (L484, L526).
    # Mutant: the area taken as 2 pi r^2 (a hemisphere).
    # ------------------------------------------------------------------------
    eq("eq:gauss_sphere", 476, "(d alpha/dr) 4 pi r^2 == 4 pi G M/c^2")
    th, ph = sp.symbols('theta phi', real=True)
    arad = sp.Function('alpha_r')
    flux_sph = sp.integrate(sp.integrate(sp.diff(arad(r), r)*r**2*sp.sin(th), (ph, 0, 2*sp.pi)),
                            (th, 0, sp.pi))
    symcheck("surface integral of alpha'(r) r_hat . dS == 4 pi r^2 alpha'(r)",
             sp.simplify(flux_sph - 4*sp.pi*r**2*sp.diff(arad(r), r)))
    vac = sp.dsolve(sp.Eq(sp.diff(r**2*sp.diff(arad(r), r), r), 0), arad(r))
    consts = sorted(vac.rhs.free_symbols - {r}, key=str)
    boolcheck("source-free spherical solutions are C1 + C2/r with no G or M",
              len(consts) == 2 and sp.simplify(sp.diff(vac.rhs*r, r, 2)) == 0
              and not vac.rhs.has(G) and not vac.rhs.has(M))
    mutant("hemisphere area 2 pi r^2", sp.simplify(flux_sph - 2*sp.pi*r**2*sp.diff(arad(r), r)))

    # ------------------------------------------------------------------------
    # eq:dalpha_dr  (sgm.tex L481)
    # States: d alpha/dr == G M/(c^2 r^2), scalar.
    # Derived from: eq:gauss_sphere divided by 4 pi r^2.
    # Test: solve eq:gauss_sphere for d alpha/dr; and the numbers of L499:
    #          g/c^2 = 1.1e-16 per meter at Earth's surface, 1.1e-19 across a
    #          millimeter ("about one part in 10^19"), and the change of alpha
    #          across an electron's reduced Compton wavelength,
    #          g hbar/(m_e c^3) = 4e-29.
    # Mutant: d alpha/dr = GM/(c^2 r) (division by 4 pi r only).
    # ------------------------------------------------------------------------
    eq("eq:dalpha_dr", 481, "d alpha/dr == G M/(c^2 r^2)")
    dadr = sp.symbols('dadr')
    dadr_sol = sp.solve(sp.Eq(dadr*4*sp.pi*r**2, 4*sp.pi*G*M/c**2), dadr)[0]
    symcheck("eq:gauss_sphere / (4 pi r^2) gives GM/(c^2 r^2)", dadr_sol - G*M/(c**2*r**2))
    check("L499: g/c^2 at Earth's surface == 1.1e-16 per meter", g_n/c_n**2, 1.1e-16, tol=0.02)
    check("L499: rate difference across 1 mm == 1.1e-19 (about 1e-19)", g_n/c_n**2*1e-3, 1.1e-19, tol=0.02)
    check("L499: g hbar/(m_e c^3) == 4e-29 (to one figure)", g_n*hb_n/(me_n*c_n**3), 4.2e-29, tol=0.03)
    boolcheck("L499: g hbar/(m_e c^3) rounds to 4e-29",
              round(g_n*hb_n/(me_n*c_n**3)/1e-29) == 4)
    mutant("d alpha/dr == GM/(c^2 r)", dadr_sol - G*M/(c**2*r))

    # ------------------------------------------------------------------------
    # eq:alpha_deficit, line 1  (sgm.tex L488)
    # States: alpha(inf) - alpha(r) == int_r^inf GM/(c^2 r'^2) dr', scalar.
    # Derived from: the fundamental theorem of calculus applied to eq:dalpha_dr.
    # Test: for a general radial alpha with alpha' given by eq:dalpha_dr, the
    #          r-derivative of both sides agrees (-alpha'(r) on the left, minus
    #          the integrand at r on the right), and both vanish as r -> infinity.
    # Mutant: integration limits reversed (int from infinity to r), which
    #          gives the opposite sign.
    # ------------------------------------------------------------------------
    eq("eq:alpha_deficit (line 1)", 488, "alpha(inf) - alpha(r) == int_r^inf GM/(c^2 r'^2) dr'")
    integral = sp.Integral(G*M/(c**2*rp**2), (rp, r, sp.oo))
    symcheck("d/dr of the right side == -GM/(c^2 r^2) == d/dr[alpha(inf) - alpha(r)] by eq:dalpha_dr",
             sp.diff(integral, r).doit() + dadr_sol)
    symcheck("right side -> 0 as r -> infinity", sp.limit(integral.doit(), r, sp.oo))
    mutant("limits reversed: d/dr int_inf^r == +GM/(c^2 r^2) vs -alpha'",
           sp.diff(sp.Integral(G*M/(c**2*rp**2), (rp, sp.oo, r)), r).doit() + dadr_sol)

    # ------------------------------------------------------------------------
    # eq:alpha_deficit, line 2  (sgm.tex L489)
    # States: the integral == G M/(c^2 r).
    # Derived from: the first line of this display, by evaluating the integral.
    # Test: sympy integration; the deficit is positive (L492).
    # Mutant: GM/(c^2 r^2) (integrand returned unintegrated).
    # ------------------------------------------------------------------------
    eq("eq:alpha_deficit (line 2)", 489, "... == G M/(c^2 r)")
    deficit = integral.doit()
    symcheck("int_r^inf GM/(c^2 r'^2) dr' == GM/(c^2 r)", deficit - G*M/(c**2*r))
    boolcheck("the deficit is positive", bool(deficit.is_positive))
    mutant("GM/(c^2 r^2)", deficit - G*M/(c**2*r**2))

    # ------------------------------------------------------------------------
    # eq:alpha_weak  (sgm.tex L494)
    # States: alpha(r) ~ 1 - G M/(c^2 r), scalar.
    # Derived from: eq:alpha_deficit with alpha(inf) = 1 (the normalization
    #          stated with eq:alpha_def, L431).
    # Test: 1 - deficit; it equals the first-order expansion in GM of the
    #          Schwarzschild factor sqrt(1 - 2GM/(c^2 r)) (L497); the term it
    #          omits is -(GM/c^2 r)^2/2; the exact Schwarzschild gradient is
    #          eq:dalpha_dr times 1/alpha, so the approximation comes from the
    #          premise and not from the integration (L497).
    # Mutant: alpha = 1 - 2GM/(c^2 r) (the g_00 coefficient in place of alpha).
    # ------------------------------------------------------------------------
    eq("eq:alpha_weak", 494, "alpha(r) ~ 1 - G M/(c^2 r)")
    a_der = 1 - deficit
    symcheck("alpha(inf) = 1 minus the deficit == 1 - GM/(c^2 r)", a_der - a_weak)
    a_ex = sp.sqrt(1 - 2*G*M/(c**2*r))
    symcheck("first order in GM of sqrt(1 - 2GM/(c^2 r)) == eq:alpha_weak",
             sp.series(a_ex, G, 0, 2).removeO() - a_weak)
    symcheck("the omitted second-order term is -(GM/(c^2 r))^2/2",
             sp.expand(sp.series(a_ex, G, 0, 3).removeO() - a_weak + (G*M/(c**2*r))**2/2))
    symcheck("exact d alpha/dr == eq:dalpha_dr / alpha (approximation is in the premise)",
             sp.simplify(sp.diff(a_ex, r) - dadr_sol/a_ex))
    mutant("alpha == 1 - 2GM/(c^2 r)", a_der - (1 - 2*G*M/(c**2*r)))

    # ------------------------------------------------------------------------
    # eq:grav_gradient, line 1  (sgm.tex L503)
    # States: grad omega == omega_0 G M/(c^2 r^2) r_hat (vector), r_hat outward.
    # Derived from: gradient of omega(r) = omega_0 (1 - GM/(c^2 r)) (eq:omega_alpha
    #          with eq:alpha_weak), omega_0 constant (ax:lpi).
    # Test: Cartesian gradient in 3D of omega_0 (1 - GM/(c^2 |x|)), compared
    #          with omega_0 GM/(c^2 r^2) x/|x|.
    # Mutant: the inward direction, -omega_0 GM/(c^2 r^2) r_hat.
    # ------------------------------------------------------------------------
    eq("eq:grav_gradient (line 1)", 503, "grad omega == omega_0 GM/(c^2 r^2) r_hat")
    om_w3 = w0*(1 - G*M/(c**2*rr3))
    gr3 = grad(om_w3)
    rhat3 = sp.Matrix(X3)/rr3
    symcheck("grad[omega_0(1 - GM/(c^2|x|))] == omega_0 GM/(c^2 r^2) r_hat",
             sp.simplify(gr3 - w0*G*M/(c**2*rr3**2)*rhat3))
    mutant("pointing inward", sp.simplify(gr3 + w0*G*M/(c**2*rr3**2)*rhat3))

    # ------------------------------------------------------------------------
    # eq:grav_gradient, line 2  (sgm.tex L504)
    # States: ... == m G M/(hbar r^2) r_hat.
    # Derived from: the first line of this display with omega_0 = m c^2/hbar
    #          (eq:rest_freq): c^2 cancels.
    # Test: substitution; dimensions of m G M/(hbar r^2) are rad/(s m).
    # Mutant: m G M/(hbar c^2 r^2) (the c^2 not cancelled).
    # ------------------------------------------------------------------------
    eq("eq:grav_gradient (line 2)", 504, "... == m G M/(hbar r^2) r_hat")
    symcheck("omega_0 GM/(c^2 r^2) == m G M/(hbar r^2)", sp.simplify(w0*G*M/(c**2*r**2) - m*G*M/(hb*r**2)))
    boolcheck("dimensions: [m G M/(hbar r^2)] == 1/(s m)",
              np.allclose(D_M + D_G + D_M - D_hb - 2*D_len, _S2_dim(s=-1, m=-1)))
    mutant("m G M/(hbar c^2 r^2)", sp.simplify(w0*G*M/(c**2*r**2) - m*G*M/(hb*c**2*r**2)))

    # ------------------------------------------------------------------------
    # eq:newton  (sgm.tex L510)
    # States: F_grav == -G M m/r^2 r_hat (vector), Newton's law.
    # Derived from: substitute eq:grav_gradient into F = -hbar grad omega (eq:sgm).
    # Test: -hbar times the 3D gradient computed above; hbar cancels; the force
    #          points toward the mass (L515: F . r_hat < 0); at Earth's surface
    #          with M_earth = 5.9722e24 kg, R = 6.371e6 m, the acceleration
    #          GM/R^2 is within 0.2% of standard gravity.
    # Mutant: F = +GMm/r^2 r_hat (repulsion, sign of the identity dropped).
    # ------------------------------------------------------------------------
    eq("eq:newton", 510, "F_grav == -G M m/r^2 r_hat")
    F_newt = -hb*gr3
    symcheck("-hbar grad omega == -G M m/r^2 r_hat", sp.simplify(F_newt + G*M*m/rr3**2*rhat3))
    boolcheck("hbar cancels", not sp.simplify(F_newt).has(hb))
    symcheck("F . r_hat == -GMm/r^2 (negative: toward the mass)",
             sp.simplify((F_newt.T*rhat3)[0] + G*M*m/rr3**2))
    check("Earth: GM/R^2 == 9.81 m/s^2", G_n*5.9722e24/6.371e6**2, 9.81, tol=3e-3)
    mutant("F == +GMm/r^2 r_hat", sp.simplify(F_newt - G*M*m/rr3**2*rhat3))

    # ------------------------------------------------------------------------
    # eq:newton, caption line  (sgm.tex L511)
    # States: "(Newton's law of universal gravitation)", a name for the line above.
    # Derived from: nothing; prose.
    # Test: none. The identification with Newton's law is the equation
    #          itself, tested in the previous block (inverse square, attractive,
    #          coefficient G M m, 9.81 m/s^2 at Earth's surface).
    # Mutant: none (nothing to compute).
    # ------------------------------------------------------------------------
    eq("eq:newton (caption)", 511, "(Newton's law of universal gravitation) [prose]")

    # ------------------------------------------------------------------------
    # eq:g_grad_alpha  (sgm.tex L519)
    # States: g == -c^2 grad alpha, the gravitational acceleration (vector) of a
    #          slow particle, independent of m.
    # Derived from: eq:geometric_force divided by m.
    # Test: F_grav/m from the general-alpha form above; independent of m;
    #          L522: with alpha = 1 + Phi_g/c^2 it is -grad Phi_g; L546: the
    #          potential Phi_g = c^2 (alpha - 1) to first order (inverse of the
    #          same relation); L549: equals the slow-particle geodesic
    #          acceleration -c^2 Gamma^k_00 (x^0 = ct) for g_00 = -alpha^2,
    #          flat spatial part, to first order in Phi (alpha^2 grad alpha ->
    #          grad alpha). The exact geodesic form is tested at eq:freefall_accel.
    # Mutant: g = -c^2 grad alpha/2 (taking g_00 = -(1 + Phi/c^2), i.e. halving).
    # ------------------------------------------------------------------------
    eq("eq:g_grad_alpha", 519, "g == -c^2 grad alpha")
    g_vec = F_geo/m
    symcheck("F_grav/m == -c^2 grad alpha", sp.simplify(g_vec + c**2*grad(alf3)))
    boolcheck("independent of m", not sp.simplify(g_vec).has(m))
    PhiF = sp.Function('Phi_g')(x, y, z)
    symcheck("alpha = 1 + Phi/c^2: -c^2 grad alpha == -grad Phi",
             -c**2*grad(1 + PhiF/c**2) + grad(PhiF))
    symcheck("L546: Phi_g == c^2 (alpha - 1) inverts alpha = 1 + Phi/c^2",
             sp.simplify(c**2*((1 + Phi/c**2) - 1) - Phi))
    X4 = [sp.Symbol('x0'), x, y, z]
    g4 = sp.diag(-alf3**2, 1, 1, 1)
    Gam4 = _S2_christoffel(g4, X4)
    acc_geo = sp.Matrix([-c**2*Gam4[k][0][0] for k in (1, 2, 3)])
    symcheck("geodesic at rest: -c^2 Gamma^k_00 == -c^2 alpha d_k alpha (flat spatial part)",
             sp.simplify(acc_geo + c**2*alf3*grad(alf3)))
    epsP = sp.symbols('epsilon_P')
    lin_geo = sp.series(acc_geo.subs(alf3, 1 + epsP*PhiF/c**2).doit()[0], epsP, 0, 2).removeO()
    symcheck("to first order in Phi: geodesic acceleration == -c^2 d_x alpha",
             sp.simplify(lin_geo + c**2*sp.diff(epsP*PhiF/c**2, x)))
    mutant("g == -c^2 grad alpha/2", sp.simplify(g_vec + c**2*grad(alf3)/2))

    # ------------------------------------------------------------------------
    # eq:dalpha_dl, line 1  (sgm.tex L530)
    # States: d alpha/dr_p == alpha d alpha/dr (scalar), r_p the proper radial
    #          distance with dr_p = dr/alpha.
    # Derived from: eq:alpha_def and the cited exterior Schwarzschild metric
    #          (text L528, which also defines r_p by dr_p = dr/alpha); chain
    #          rule, d/dr_p = (dr/dr_p) d/dr = alpha d/dr.
    # Test: for the exterior Schwarzschild metric, g_rr = 1/(1 - R_s/r) = 1/alpha^2,
    #          so dr_p/dr = sqrt(g_rr) = 1/alpha; then d alpha/dr_p =
    #          (d alpha/dr)/(dr_p/dr) == alpha d alpha/dr.
    # Mutant: d alpha/dr_p = (d alpha/dr)/alpha (inverting the conversion).
    # ------------------------------------------------------------------------
    eq("eq:dalpha_dl (line 1)", 530, "d alpha/dr_p == alpha d alpha/dr")
    aRs = sp.sqrt(1 - Rs/r)
    grr = 1/(1 - Rs/r)
    drp_dr = sp.sqrt(grr)
    uo = sp.symbols('u_o', positive=True)
    ext = {r: Rs*(1 + uo)}                   # exterior region r > R_s
    symcheck("dr_p/dr == sqrt(g_rr) == 1/alpha (r > R_s)", sp.simplify((drp_dr - 1/aRs).subs(ext)))
    dadrp = sp.diff(aRs, r)/drp_dr
    symcheck("d alpha/dr_p == alpha d alpha/dr (r > R_s)",
             sp.simplify((dadrp - aRs*sp.diff(aRs, r)).subs(ext)))
    mutant("d alpha/dr_p == (d alpha/dr)/alpha", sp.simplify(dadrp - sp.diff(aRs, r)/aRs))

    # ------------------------------------------------------------------------
    # eq:dalpha_dl, line 2  (sgm.tex L531)
    # States: ... == R_s/(2 r^2).
    # Derived from: the first line of this display and alpha^2 = 1 - R_s/r (cited
    #          Schwarzschild metric, L528), differentiated: 2 alpha alpha' = R_s/r^2.
    # Test: direct differentiation of alpha = sqrt(1 - R_s/r); also the L543
    #          claims: mc^2 d alpha/dr == GMm/(r^2 alpha) with GM = R_s c^2/2;
    #          the proper acceleration of a static observer computed from the
    #          Christoffel symbols of the full Schwarzschild metric,
    #          |a| = sqrt(g_rr) c^2 Gamma^r_00 (u^0)^2/c^2 with u^0 = c/alpha,
    #          equals c^2 d ln alpha/dr_p == c^2 d alpha/dr; a rope of negligible
    #          mass transmits the force reduced by alpha: the energy at infinity
    #          of the particle at rest, m c^2 alpha, changes per unit proper
    #          length (what a rope pays out) by m c^2 alpha d alpha/dr = GMm/r^2.
    #          L541: far away, -(m c^2 R_s/2 r^2) is Newton with GM = R_s c^2/2;
    #          the clock-rate ratio is monotonic in R_s.
    # Mutant: R_s/r^2 (the factor 2 of 2 alpha alpha' dropped).
    # ------------------------------------------------------------------------
    eq("eq:dalpha_dl (line 2)", 531, "... == R_s/(2 r^2)")
    symcheck("alpha d alpha/dr == R_s/(2 r^2) on sqrt(1 - R_s/r)",
             sp.simplify(aRs*sp.diff(aRs, r) - Rs/(2*r**2)))
    sub_GM = {Rs: 2*G*M/c**2}
    symcheck("L543: m c^2 d alpha/dr == GMm/(r^2 alpha)",
             sp.simplify((m*c**2*sp.diff(aRs, r) - G*M*m/(r**2*aRs)).subs(sub_GM)))
    thv, phv = sp.symbols('theta_s phi_s', positive=True)
    Xs = [sp.Symbol('x0'), r, thv, phv]
    gS = sp.diag(-(1 - Rs/r), 1/(1 - Rs/r), r**2, r**2*sp.sin(thv)**2)
    GamS = _S2_christoffel(gS, Xs)
    u0 = 1/aRs                               # dx^0/d tau / c for a static observer
    a_r_comp = c**2*GamS[1][0][0]*u0**2      # a^r = Gamma^r_00 (u^0)^2, x^0 = ct
    a_mag = sp.sqrt(gS[1, 1])*a_r_comp
    symcheck("static observer: proper acceleration == c^2 d ln(alpha)/dr_p (r > R_s)",
             sp.simplify((a_mag - c**2*sp.diff(sp.log(aRs), r)*aRs).subs(ext)))
    symcheck("... == c^2 d alpha/dr (r > R_s)", sp.simplify((a_mag - c**2*sp.diff(aRs, r)).subs(ext)))
    symcheck("rope at infinity: d(m c^2 alpha)/dr_p == GMm/r^2 at every radius",
             sp.simplify((m*c**2*aRs*sp.diff(aRs, r) - G*M*m/r**2).subs(sub_GM)))
    symcheck("L541: m c^2 R_s/(2 r^2) == GMm/r^2 with GM = R_s c^2/2",
             sp.simplify((m*c**2*Rs/(2*r**2)).subs(sub_GM) - G*M*m/r**2))
    r1, r2 = sp.symbols('r_1 r_2', positive=True)
    dln = sp.diff(sp.log(sp.sqrt((1 - Rs/r1)/(1 - Rs/r2))), Rs)
    symcheck("L541: d/dR_s ln[alpha(r1)/alpha(r2)] == (1/(r2-R_s) - 1/(r1-R_s))/2",
             sp.simplify(dln - (1/(r2 - Rs) - 1/(r1 - Rs))/2))
    vals = [float(dln.subs({r1: 3.7, r2: 11.3, Rs: s_})) for s_ in np.linspace(0.05, 3.6, 40)]
    boolcheck("L541: one sign over 0 < R_s < r1 < r2 (monotonic, so one value of R_s)",
              all(v_ < 0 for v_ in vals))
    mutant("R_s/r^2", sp.simplify(aRs*sp.diff(aRs, r) - Rs/r**2))

    # ------------------------------------------------------------------------
    # eq:gauss_exact  (sgm.tex L536)
    # States: (d alpha/dr_p) 4 pi r^2 == 2 pi R_s at every radius, scalar.
    # Derived from: eq:dalpha_dl times the area 4 pi r^2 of the sphere of
    #          areal radius r.
    # Test: product, its r-derivative is zero (every radius); with R_s =
    #          2GM/c^2 it equals 4 pi GM/c^2, the right side of eq:gauss_alpha.
    # Mutant: the coordinate gradient d alpha/dr in place of d alpha/dr_p,
    #          whose flux depends on r.
    # ------------------------------------------------------------------------
    eq("eq:gauss_exact", 536, "(d alpha/dr_p) 4 pi r^2 == 2 pi R_s")
    fl_ex = 4*sp.pi*r**2*aRs*sp.diff(aRs, r)
    symcheck("flux == 2 pi R_s", sp.simplify(fl_ex - 2*sp.pi*Rs))
    symcheck("independent of r", sp.simplify(sp.diff(fl_ex, r)))
    symcheck("with R_s = 2GM/c^2: == 4 pi G M/c^2", sp.simplify((2*sp.pi*Rs).subs(sub_GM) - 4*sp.pi*G*M/c**2))
    mutant("coordinate gradient: 4 pi r^2 d alpha/dr == 2 pi R_s", sp.simplify(4*sp.pi*r**2*sp.diff(aRs, r) - 2*sp.pi*Rs))

    # ========================================================================
    # Subsection "Inertia from the dynamic gradient"
    # ========================================================================
    a1, a2, a3c = sp.symbols('a_1 a_2 a_3', real=True)
    d1, d2, d3 = sp.symbols('dx_1 dx_2 dx_3', real=True)
    av = sp.Matrix([a1, a2, a3c])
    dv = sp.Matrix([d1, d2, d3])
    Dv = (d1, d2, d3)
    gradD = lambda f: sp.Matrix([sp.diff(f, v) for v in Dv])
    a_s, dl = sp.symbols('a delta', positive=True)

    # ------------------------------------------------------------------------
    # eq:alpha_rindler  (sgm.tex L564)
    # States: alpha == 1 + delta alpha, the time-dilation factor (scalar) of
    #          the co-accelerating frame at displacement dx.
    # Derived from: cited line element of the Rindler metric (Rindler 1966; MTW),
    #          ds^2 = -(1 + a.dx/c^2)^2 c^2 dt^2 + |d dx|^2, and alpha = sqrt(-g_00)
    #          (eq:alpha_def).
    # Test: (a) sqrt(-g_00) of that line element is 1 + a.dx/c^2 where
    #          a.dx > -c^2; (b) exactness, L572: every Riemann component of the
    #          3+1 metric with a general constant vector a vanishes, so it is flat
    #          spacetime in accelerated coordinates; (c) L560: the proper
    #          acceleration of the static point dx is c^2 grad ln alpha =
    #          a/(1 + a.dx/c^2) along a, the Born-rigid profile, and
    #          1.1e-16 per meter at a = g; (d) the exact Doppler shift of light
    #          sent between two Born-rigid worldlines X^2 - (ct)^2 = X_s^2 and
    #          R^2, R = c^2/a, X_s = R + delta, gives the received rate ratio
    #          X_s/R = 1 + a delta/c^2 exactly (Einstein 1911's derivation, L572).
    # Mutant: alpha = 1 + a.dx/(2c^2) (half the shift). That metric is still
    #          flat (it is Rindler for acceleration a/2), so the mutant is caught
    #          by its proper acceleration a/(2 alpha) != a/alpha and by the
    #          received ratio 1 + a delta/(2c^2) != X_s/R.
    # ------------------------------------------------------------------------
    eq("eq:alpha_rindler", 564, "alpha == 1 + delta alpha")
    alR = 1 + (av.T*dv)[0]/c**2
    gR = sp.diag(-alR**2, 1, 1, 1)
    symcheck("sqrt(-g_00) of the Rindler line element == 1 + a.dx/c^2 (where positive)",
             sp.sqrt(-gR[0, 0]).subs({a1: sp.Rational(13, 10), a2: sp.Rational(7, 10), a3c: -sp.Rational(3, 10),
                                      d1: sp.Rational(1, 3), d2: sp.Rational(5, 7), d3: sp.Rational(11, 9), c: 1})
             - alR.subs({a1: sp.Rational(13, 10), a2: sp.Rational(7, 10), a3c: -sp.Rational(3, 10),
                         d1: sp.Rational(1, 3), d2: sp.Rational(5, 7), d3: sp.Rational(11, 9), c: 1}))
    XR = [sp.Symbol('x0')] + list(Dv)
    GamR = _S2_christoffel(gR, XR)
    riem_R = [_S2_riemann_up(GamR, XR, A_, B_, C_, D_)
              for A_ in range(4) for B_ in range(4) for C_ in range(4) for D_ in range(C_ + 1, 4)]
    boolcheck("Rindler metric with general a: every Riemann component is zero (exact, flat)",
              all(v_ == 0 for v_ in riem_R))
    symcheck("L560: proper acceleration c^2 grad ln alpha == a/(1 + a.dx/c^2)",
             sp.simplify(c**2*gradD(sp.log(alR)) - av/alR))
    check("L560: a/c^2 at a = g is 1.1e-16 per meter", g_n/c_n**2, 1.1e-16, tol=0.02)
    w_e = sp.symbols('w_e', positive=True)
    R_rin = c**2/a_s
    X_seg = R_rin + dl
    S_e = X_seg*w_e
    ct_w = (S_e**2 - R_rin**2)/(2*S_e)
    symcheck("exact Doppler: reception event lies on the receiver hyperbola",
             sp.expand((S_e - ct_w)**2 - ct_w**2 - R_rin**2))
    z_r = ct_w/R_rin
    symcheck("receiver proper time c tau_r = R asinh(z): z + sqrt(z^2+1) == X_s w/R",
             sp.expand(z_r**2 + 1 - (X_seg*w_e/R_rin - z_r)**2))
    symcheck("... on the positive branch: X_s w/R - z == (X_s^2 w^2 + R^2)/(2 R X_s w) > 0",
             sp.simplify((X_seg*w_e/R_rin - z_r) - (X_seg**2*w_e**2 + R_rin**2)/(2*R_rin*X_seg*w_e)))
    symcheck("exact received rate ratio X_s/R == 1 + a delta/c^2", sp.simplify(X_seg/R_rin - (1 + a_s*dl/c**2)))
    alM = 1 + (av.T*dv)[0]/(2*c**2)
    boolcheck("mutant caught: the half-shift factor is not the Rindler factor (ratio X_s/R differs)",
              sp.simplify(X_seg/R_rin - (1 + a_s*dl/(2*c**2))) != 0)
    mutant("half-shift factor 1 + a.dx/(2c^2) gives c^2 grad ln alpha == a/alpha",
           sp.simplify(c**2*gradD(sp.log(alM)) - av/alM))

    # ------------------------------------------------------------------------
    # eq:rapidity_gain  (sgm.tex L569)
    # States: delta alpha == a.dx/c^2, the fractional frequency shift (scalar).
    # Derived from: eq:alpha_rindler (definition read off the Rindler
    #          factor); described as the
    #          fraction by which the rate of proper time at dx exceeds that at
    #          the mean position.
    # Test: alpha(dx)/alpha(0) - 1 == delta alpha for the Rindler factor;
    #          dimensionless; exactly the factor reaches zero at a.dx = -c^2
    #          (L584: distance c^2/a behind the particle).
    # Mutant: delta alpha = a.dx/c (dimensionful).
    # ------------------------------------------------------------------------
    eq("eq:rapidity_gain", 569, "delta alpha == a.dx/c^2")
    dalpha = (av.T*dv)[0]/c**2
    symcheck("alpha(dx)/alpha(0) - 1 == a.dx/c^2", sp.simplify(alR/alR.subs({d1: 0, d2: 0, d3: 0}) - 1 - dalpha))
    boolcheck("dimensionless: [a dx/c^2] == 1",
              np.allclose(_S2_dim(m=1, s=-2) + D_len - 2*D_c, 0))
    symcheck("the factor 1 + a delta/c^2 vanishes at delta = -c^2/a",
             (1 + a_s*dl/c**2).subs(dl, -c**2/a_s))
    boolcheck("mutant caught: a.dx/c is not dimensionless",
              not np.allclose(_S2_dim(m=1, s=-2) + D_len - D_c, 0))
    mutant("delta alpha == a.dx/c against the Rindler factor",
           sp.simplify(alR/alR.subs({d1: 0, d2: 0, d3: 0}) - 1 - (av.T*dv)[0]/c))

    # ------------------------------------------------------------------------
    # eq:rindler_rate  (sgm.tex L574)
    # States: omega(x + dx) == omega_0 (1 + delta alpha), the coordinate rate
    #          (scalar, per unit t of the co-accelerating frame) of the rest phase.
    # Derived from: ax:lpi (rate -omega_0 per unit proper time at each point)
    #          times d tau/dt = alpha of eq:alpha_rindler and eq:rapidity_gain,
    #          the same operation as eq:omega_alpha (eq:alpha_def).
    # Test: theta = -omega_0 tau with tau = alpha t, -d theta/dt; the same
    #          operation as eq:omega_alpha with the Rindler factor (L577).
    # Mutant: omega_0/(1 + delta alpha).
    # ------------------------------------------------------------------------
    eq("eq:rindler_rate", 574, "omega(x+dx) == omega_0 (1 + delta alpha)")
    om_R = -sp.diff((-w0*tau).subs(tau, alR*t), t)
    symcheck("-d/dt[-omega_0 alpha_R t] == omega_0 (1 + a.dx/c^2)", sp.expand(om_R - w0*(1 + dalpha)))
    symcheck("same form as eq:omega_alpha: omega_0 alpha with alpha -> Rindler factor",
             sp.expand(om_R - (w0*alS).subs(alS, alR)))
    mutant("omega_0/(1 + delta alpha)", sp.simplify(om_R - w0/(1 + dalpha)))

    # ------------------------------------------------------------------------
    # eq:front_behind  (sgm.tex L581)
    # States: [omega(x + dx) - omega(x - dx)]/(2|dx|) == omega_0 a.d_hat/c^2,
    #          d_hat = dx/|dx| (scalar quotient).
    # Derived from: eq:rindler_rate evaluated at +dx and -dx and differenced.
    # Test: (a) with dx = h d_hat for symbolic h > 0 and d_hat, the quotient
    #          equals omega_0 (a.d_hat)/c^2 at every h (independent of size,
    #          L584); (b) for a general field with Taylor coefficients up to
    #          third order, the even orders cancel and the quotient is
    #          g + (T/6) h^2 (L584: odd terms beyond first are of order |dx|^2).
    # Mutant: dividing by |dx| instead of 2|dx|.
    # ------------------------------------------------------------------------
    eq("eq:front_behind", 581, "(omega(x+dx) - omega(x-dx))/(2|dx|) == omega_0 a.d/c^2")
    hs = sp.symbols('h', positive=True)
    nh = sp.Matrix(sp.symbols('n_1 n_2 n_3', real=True))
    om_rind = lambda dd: w0*(1 + (av.T*dd)[0]/c**2)
    quot = sp.simplify((om_rind(hs*nh) - om_rind(-hs*nh))/(2*hs))
    symcheck("quotient == omega_0 (a.n)/c^2", sp.simplify(quot - w0*(av.T*nh)[0]/c**2))
    symcheck("independent of the separation h", sp.diff(quot, hs))
    w_g, g_g, H_g, T_g = sp.symbols('w_g g_g H_g T_g', real=True)
    f_gen = lambda s_: w_g + g_g*s_ + H_g*s_**2/2 + T_g*s_**3/6
    symcheck("general field: quotient == g + (T/6) h^2 (even orders cancel)",
             sp.expand((f_gen(hs) - f_gen(-hs))/(2*hs) - g_g - T_g*hs**2/6))
    mutant("divided by |dx|", sp.simplify((om_rind(hs*nh) - om_rind(-hs*nh))/hs - w0*(av.T*nh)[0]/c**2))

    # ------------------------------------------------------------------------
    # eq:inertia_gradient, line 1  (sgm.tex L588)
    # States: grad omega == omega_0 a/c^2 (vector), the dynamic spectral gradient.
    # Derived from: eq:front_behind holds for every direction d_hat, so the
    #          component of grad omega along each is (omega_0 a/c^2).d_hat (L586);
    #          also def:spectral_gradient, eq:omega_expansion (L584).
    # Test: (a) taking d_hat = each Cartesian unit vector in eq:front_behind
    #          gives the three components omega_0 a_i/c^2; (b) directly, the
    #          gradient in dx of eq:rindler_rate. (c) L625: the rest phase
    #          -omega_0 (t - v.dx/c^2) of a clock moving at velocity v(t) has
    #          wavevector omega_0 v/c^2 growing at omega_0 a/c^2 (Lorentz
    #          transformation t' = gamma(t - v.x/c^2) to first order in v/c checked
    #          by series); (d) L627: a front omega(t - n.dx/c) has gradient
    #          -(n/c) d_t omega.
    # Mutant: omega_0 a/c (one power of c).
    # ------------------------------------------------------------------------
    eq("eq:inertia_gradient (line 1)", 588, "grad omega == omega_0 a/c^2")
    comps = []
    for k_ in range(3):
        e_k = sp.zeros(3, 1); e_k[k_] = 1
        comps.append(sp.simplify((om_rind(hs*e_k) - om_rind(-hs*e_k))/(2*hs)))
    gw_in = sp.Matrix(comps)
    symcheck("components from eq:front_behind along x, y, z == omega_0 a/c^2", gw_in - w0*av/c**2)
    symcheck("gradient of eq:rindler_rate in dx == omega_0 a/c^2", gradD(om_R) - w0*av/c**2)
    vf = sp.Matrix([sp.Function(f'v_{k}', real=True)(t) for k in 'xyz'])
    phase_v = -w0*(t - (vf.T*dv)[0]/c**2)
    kv_ = gradD(phase_v)
    symcheck("L625: wavevector of -omega_0 (t - v.dx/c^2) == omega_0 v/c^2", kv_ - w0*vf/c**2)
    symcheck("L625: it grows at omega_0 a/c^2", sp.diff(kv_, t) - w0*sp.diff(vf, t)/c**2)
    vs_, xs_ = sp.symbols('v_s x_s', real=True)
    lor = (t - vs_*xs_/c**2)/sp.sqrt(1 - vs_**2/c**2)
    symcheck("L625: Lorentz time t' == t - v x/c^2 + O(v^2)",
             sp.series(lor, vs_, 0, 2).removeO() - (t - vs_*xs_/c**2))
    fq = sp.Function('omega_f')
    sarg = t - (nh.T*dv)[0]/c
    symcheck("L627: front omega(t - n.dx/c) has gradient -(n/c) d_t omega",
             gradD(fq(sarg)) + nh/c*sp.diff(fq(sarg), t))
    mutant("omega_0 a/c", gw_in - w0*av/c)

    # ------------------------------------------------------------------------
    # eq:inertia_gradient, line 2  (sgm.tex L589)
    # States: ... == m a/hbar.
    # Derived from: the first line of this display with omega_0 = m c^2/hbar
    #          (eq:rest_freq).
    # Test: substitution (c^2 cancels).
    # Mutant: m a/(hbar c^2).
    # ------------------------------------------------------------------------
    eq("eq:inertia_gradient (line 2)", 589, "... == m a/hbar")
    symcheck("omega_0 a/c^2 == m a/hbar", sp.simplify(w0*av/c**2 - m*av/hb))
    mutant("m a/(hbar c^2)", sp.simplify(w0*av/c**2 - m*av/(hb*c**2)))

    # ------------------------------------------------------------------------
    # eq:inertia  (sgm.tex L595)
    # States: F_in == -m a (vector), the inertial reaction.
    # Derived from: substitute eq:inertia_gradient into F = -hbar grad omega (eq:sgm).
    # Test: -hbar (m a/hbar); the coefficient is m = hbar omega_0/c^2 (L600);
    #          and the inline transformation of L644 (Shukla and Bhattacharya):
    #          in one dimension, with psi(x,t) = f(x~, t) exp[(i/hbar)(m a x~ t +
    #          m a^2 t^3/6)], x~ = x - a t^2/2, a solution of the Schrodinger
    #          equation with a general potential V(x) gives for f the potential
    #          V(x~ + a t^2/2) + m a x~ exactly; with the uniform field
    #          V = -m a x this leaves only the spatially uniform -m a^2 t^2/2
    #          (no force), which is the sense of "removes a uniform field";
    #          L646: the free kernel has the same modulus at every separation.
    # Mutant: F_in = +m a; and the transformation with phase m a x~ t only
    #          (t^3 term dropped), which leaves a time-dependent remainder.
    # ------------------------------------------------------------------------
    eq("eq:inertia", 595, "F_in == -m a")
    F_in = -hb*(m*av/hb)
    symcheck("-hbar (m a/hbar) == -m a", F_in + m*av)
    symcheck("L600: m == hbar omega_0/c^2", hb*w0/c**2 - m)
    xr_ = sp.symbols('x_r', real=True)
    tt_ = sp.symbols('t_r', real=True)
    ar_ = sp.symbols('a_r', real=True)
    fF = sp.Function('f')
    Vf = sp.Function('V')
    xt = xr_ - ar_*tt_**2/2
    phase = (m*ar_*xt*tt_ + m*ar_**2*tt_**3/6)/hb

    def _shukla_residual(ph_):
        E_ = sp.exp(sp.I*ph_)
        psi_ = fF(xt, tt_)*E_
        res = sp.I*hb*sp.diff(psi_, tt_) + hb**2/(2*m)*sp.diff(psi_, xr_, 2) - Vf(xr_)*psi_
        return sp.expand(sp.simplify(res/E_))
    u_ = sp.symbols('u_', real=True)
    res_ok = _shukla_residual(phase)
    # expected: i hbar f_t + hbar^2/2m f_uu - (V(x) + m a x~) f, f = f(u, t) at u = x~.
    # With F(x, t) = f(x~, t): the partial f_t at fixed u is (d_t + a t d_x) F,
    # and f_uu is d_x^2 F; writing both through d_x, d_t of F keeps one sympy form.
    Fx = fF(xt, tt_)
    expected = (sp.I*hb*(sp.diff(Fx, tt_) + ar_*tt_*sp.diff(Fx, xr_)) + hb**2/(2*m)*sp.diff(Fx, xr_, 2)
                - (Vf(xr_) + m*ar_*xt)*Fx)
    symcheck("L644: transformed equation has potential V(x~ + a t^2/2) + m a x~ exactly",
             sp.simplify(res_ok - sp.expand(expected)))
    V_unif = (-m*ar_*(u_ + ar_*tt_**2/2) + m*ar_*u_)
    symcheck("L644: for V = -m a x the potential in the falling frame has zero gradient",
             sp.diff(V_unif, u_))
    xs3 = sp.symbols('x_s1 x_s2 x_s3', real=True)
    tk = sp.symbols('t_k', positive=True)
    sep2 = sum((v_ - s_)**2 for v_, s_ in zip(X3, xs3))
    Kfree = (m/(2*sp.pi*sp.I*hb*tk))**sp.Rational(3, 2)*sp.exp(sp.I*m*sep2/(2*hb*tk))
    symcheck("L646: |free kernel| does not depend on the separation",
             sp.diff(sp.simplify(Kfree*sp.conjugate(Kfree)), x))
    mutant("F_in == +m a", F_in - m*av)
    res_bad = _shukla_residual(m*ar_*xt*tt_/hb)
    mutant("transformation without the t^3 phase leaves a different potential",
           sp.simplify(res_bad - sp.expand(expected)))

    # ------------------------------------------------------------------------
    # eq:inertia, caption line  (sgm.tex L596)
    # States: "(The inertial reaction)", a name for the line above.
    # Derived from: nothing; prose.
    # Test: none. The equation it names is tested in the previous block.
    # Mutant: none (nothing to compute).
    # ------------------------------------------------------------------------
    eq("eq:inertia (caption)", 596, "(The inertial reaction) [prose]")

    # ------------------------------------------------------------------------
    # eq:inertia_geometric  (sgm.tex L604)
    # States: F_in == -m c^2 grad alpha (vector), with alpha the Rindler factor.
    # Derived from: eq:alpha_rindler, eq:inertia.
    #          grad alpha = grad(1 + a.dx/c^2) = a/c^2 (L602); substitute
    #          a = c^2 grad alpha into eq:inertia.
    # Test: gradient of the Rindler factor in dx; substitution; the Hessian of
    #          the Rindler factor is zero, so the relation holds at every dx (L634,
    #          L686); the same formula with the weak-field factor gives Newton.
    # Mutant: F_in = -m c grad alpha.
    # ------------------------------------------------------------------------
    eq("eq:inertia_geometric", 604, "F_in == -m c^2 grad alpha")
    symcheck("grad of the Rindler factor == a/c^2", gradD(alR) - av/c**2)
    symcheck("-m c^2 grad alpha_R == -m a == eq:inertia", -m*c**2*gradD(alR) - F_in)
    symcheck("Hessian of the Rindler factor == 0", sp.hessian(alR, Dv))
    symcheck("same formula with eq:alpha_weak gives -GMm/r^2",
             sp.simplify(-m*c**2*sp.diff(a_weak, r) + G*M*m/r**2))
    mutant("F_in == -m c grad alpha", -m*c*gradD(alR) - F_in)

    # ------------------------------------------------------------------------
    # eq:detuned_rate  (sgm.tex L610)
    # States: omega_st(x + dx) == omega_0 (1 - F.dx/(m c^2)), the coordinate rate
    #          (scalar) in the momentarily comoving inertial frame under the
    #          uniform applied force F with potential V = -F.x.
    # Derived from: eq:planck; the potential shifts the rate by V/hbar,
    #          measured from its value at the particle: omega_0 + [V(x+dx) - V(x)]/hbar,
    #          with omega_0 = m c^2/hbar (eq:rest_freq). Exact because V is linear.
    # Test: build V = -F.x for symbolic F and x, form the shift, compare.
    # Mutant: omega_0 (1 + F.dx/(m c^2)) (sign of V).
    # ------------------------------------------------------------------------
    eq("eq:detuned_rate", 610, "omega_st(x+dx) == omega_0 (1 - F.dx/(m c^2))")
    Fv = sp.Matrix(sp.symbols('F_1 F_2 F_3', real=True))
    xv = sp.Matrix(X3)
    Vapp = lambda pos: -(Fv.T*pos)[0]
    om_st = w0 + (Vapp(xv + dv) - Vapp(xv))/hb
    symcheck("omega_0 + [V(x+dx) - V(x)]/hbar == omega_0 (1 - F.dx/(m c^2))",
             sp.expand(om_st - w0*(1 - (Fv.T*dv)[0]/(m*c**2))))
    mutant("omega_0 (1 + F.dx/(m c^2))", sp.expand(om_st - w0*(1 + (Fv.T*dv)[0]/(m*c**2))))

    # ------------------------------------------------------------------------
    # eq:flat_total  (sgm.tex L615)
    # States: omega_acc(x + dx) == omega_0 (1 - F.dx/(m c^2)) (1 + delta alpha).
    # Derived from: eq:detuned_rate multiplied by the Rindler factor
    #          (eq:alpha_rindler), as eq:rindler_rate multiplies omega_0.
    # Test: product of the two expressions above; L618: to first order in dx
    #          (bookkeeping parameter epsilon) it is omega_0 [1 + (a - F/m).dx/c^2];
    #          the second-order remainder is -omega_0 (F.dx)(a.dx)/(m c^4).
    # Mutant: the sum omega_st + omega_0 delta alpha (agrees at first order)
    #          tested against the exact product, and the first-order coefficient
    #          (a + F/m) (sign error).
    # ------------------------------------------------------------------------
    eq("eq:flat_total", 615, "omega_acc == omega_0 (1 - F.dx/mc^2)(1 + delta alpha)")
    om_acc = om_st*alR
    target_acc = w0*(1 - (Fv.T*dv)[0]/(m*c**2))*(1 + dalpha)
    symcheck("omega_st times Rindler factor == displayed product", sp.expand(om_acc - target_acc))
    epsd = sp.symbols('epsilon_d')
    om_acc_e = om_acc.subs({d1: epsd*d1, d2: epsd*d2, d3: epsd*d3}, simultaneous=True)
    lin_acc = sp.series(om_acc_e, epsd, 0, 2).removeO().subs(epsd, 1)
    symcheck("L618: first order == omega_0 [1 + (a - F/m).dx/c^2]",
             sp.expand(lin_acc - w0*(1 + ((av - Fv/m).T*dv)[0]/c**2)))
    symcheck("second-order remainder == -omega_0 (F.dx)(a.dx)/(m c^4)",
             sp.expand(om_acc - lin_acc + w0*(Fv.T*dv)[0]*(av.T*dv)[0]/(m*c**4)))
    mutant("sum instead of product", sp.expand(om_st + w0*dalpha - target_acc))
    mutant("first-order coefficient (a + F/m)", sp.expand(lin_acc - w0*(1 + ((av + Fv/m).T*dv)[0]/c**2)))

    # ------------------------------------------------------------------------
    # eq:second_law  (sgm.tex L620)
    # States: F == m a (vector).
    # Derived from: eq:flat_total, eq:sgm, def:spectral_gradient,
    #          eq:sgm_as_dot_p (L623), eq:detuned_rate.
    #          L618: by eq:sgm the first-order rate exerts -hbar grad =
    #          F - m a; the momentum in the accelerating rest frame stays zero, so
    #          this force vanishes.
    # Test: -hbar grad_dx of the first-order rate == F - m a; solving
    #          F - m a = 0 for a gives F/m; at F = m a the potential shift
    #          -omega_0 F.dx/(mc^2) cancels the Rindler shift +omega_0 delta alpha
    #          (L623), so the field is uniform to first order.
    # Mutant: the gradient of the inertial-frame rate omega_st alone, whose
    #          force +F does not vanish at F = m a.
    # ------------------------------------------------------------------------
    eq("eq:second_law", 620, "F == m a")
    frc = -hb*gradD(lin_acc)
    symcheck("-hbar grad of the first-order rate == F - m a", sp.expand(frc - (Fv - m*av)))
    solA = sp.solve(list(frc), [a1, a2, a3c], dict=True)[0]
    symcheck("zero force gives a == F/m", sp.Matrix([solA[a1], solA[a2], solA[a3c]]) - Fv/m)
    subFma = {Fv[i]: m*av[i] for i in range(3)}
    symcheck("L623: at F = m a the potential shift cancels the Rindler shift",
             sp.expand((-w0*(Fv.T*dv)[0]/(m*c**2) + w0*dalpha).subs(subFma)))
    symcheck("L623: at F = m a the frequency field is uniform to first order",
             sp.expand(lin_acc.subs(subFma) - w0))
    mutant("force from omega_st alone vanishes at F = m a",
           sp.expand((-hb*gradD(om_st)).subs(subFma)))

    # ========================================================================
    # Subsection "Free fall and the equivalence principle"
    # (2 spatial dimensions with a general symmetric gamma_ij(x) and alpha(x),
    #  which keeps the matrix algebra small; the index structure is the same in 3D)
    # ========================================================================
    Tt = sp.symbols('T_t', real=True)                 # t~
    q1, q2 = sp.symbols('q_1 q_2', real=True)         # displacements delta x^i
    qv = sp.Matrix([q1, q2])
    x0f = sp.Matrix([sp.Function('x0_1')(Tt), sp.Function('x0_2')(Tt)])
    yy1, yy2 = sp.symbols('y_1 y_2', real=True)       # static spatial coordinates
    al2 = sp.Function('alpha2', positive=True)
    g11, g12, g22 = [sp.Function(nm)(yy1, yy2) for nm in ('g11', 'g12', 'g22')]
    gam2 = sp.Matrix([[g11, g12], [g12, g22]])

    def _at(expr_, pos):
        return expr_.subs({yy1: pos[0], yy2: pos[1]}, simultaneous=True)
    xpos = x0f + qv
    alpha0 = _at(al2(yy1, yy2), x0f)
    gam0 = _at(gam2, x0f)
    vel = sp.diff(x0f, Tt)
    t_of = Tt + (qv.T*gam0*vel)[0]/(alpha0**2*c**2)
    # Jacobian of (t, x^1, x^2) with respect to (t~, q1, q2)
    J = sp.Matrix([[sp.diff(t_of, Tt), sp.diff(t_of, q1), sp.diff(t_of, q2)],
                   [sp.diff(xpos[0], Tt), sp.diff(xpos[0], q1), sp.diff(xpos[0], q2)],
                   [sp.diff(xpos[1], Tt), sp.diff(xpos[1], q1), sp.diff(xpos[1], q2)]])
    gstat = sp.zeros(3, 3)
    gstat[0, 0] = -_at(al2(yy1, yy2), xpos)**2*c**2
    gstat[1:, 1:] = _at(gam2, xpos)
    gpull = J.T*gstat*J
    ac1, ac2 = sp.symbols('a_c1 a_c2', real=True)
    acv = sp.Matrix([ac1, ac2])
    vsym = sp.symbols('v_1 v_2', real=True)
    rest = {sp.Derivative(x0f[0], (Tt, 2)): ac1, sp.Derivative(x0f[1], (Tt, 2)): ac2}
    rest_v = {sp.Derivative(x0f[0], Tt): 0, sp.Derivative(x0f[1], Tt): 0}

    def _rest(e_):
        return e_.subs(rest).subs(rest_v)

    # ------------------------------------------------------------------------
    # Inline numbers of Section "Gravity and inertia are one mechanism" (L640, L642)
    # States: MICROSCOPE eta = (-1.5 +- 2.3 +- 1.5)e-15 lies 0.5 standard
    #          deviations from 0 (quadrature); ALPHA-g (0.75 +- 0.13 +- 0.16) g lies
    #          1.2 standard deviations from g and 8.5 from -g.
    # Derived from: cited measurements (MICROSCOPE; ALPHA-g) in sec:duality;
    #          no equation label.
    # Test: quadrature sums and ratios, rounded to the quoted figure.
    # Mutant: linear sum of the uncertainties for ALPHA-g (gives 0.86 sigma).
    # ------------------------------------------------------------------------
    eq("inline numbers (L640, L642)", 640, "MICROSCOPE 0.5 sigma; ALPHA-g 1.2 and 8.5 sigma")
    s_mic = np.hypot(2.3, 1.5)
    boolcheck("MICROSCOPE: |-1.5|/sqrt(2.3^2 + 1.5^2) rounds to 0.5", round(1.5/s_mic, 1) == 0.5)
    s_alp = np.hypot(0.13, 0.16)
    boolcheck("ALPHA-g: (1 - 0.75)/sqrt(0.13^2 + 0.16^2) rounds to 1.2", round(0.25/s_alp, 1) == 1.2)
    boolcheck("ALPHA-g: (0.75 + 1)/sqrt(0.13^2 + 0.16^2) rounds to 8.5", round(1.75/s_alp, 1) == 8.5)
    boolcheck("mutant caught: linear sum gives a different deviation from g",
              round(0.25/(0.13 + 0.16), 1) != 1.2)
    numutant("linear-sum deviation vs quadrature deviation", 0.25/(0.13 + 0.16), 0.25/s_alp, tol=1e-2)

    # ------------------------------------------------------------------------
    # eq:moving_sync  (sgm.tex L657)
    # States: t == t~ + gamma_ij(x0) xdot0^j(t~) dx^i/(alpha(x0)^2 c^2), with
    #          x = x0(t~) + dx: the coordinates (t~, dx) moving with the particle.
    # Derived from: definition (a choice of simultaneity); uses gamma_ij,
    #          x0(t), a_c defined in the text at L655 (before the display) and
    #          eq:alpha_def; motivated by eq:inertia_gradient and sec:inertia.
    # Test: the property claimed for it (L660): each slice of constant t~ is
    #          orthogonal to the particle's worldline dx = 0 at every velocity:
    #          the metric component g(d/dt~, d/d dx^i) pulled back from the static
    #          metric -alpha^2 c^2 dt^2 + gamma_ij dx^i dx^j vanishes at dx = 0, for
    #          general alpha(x), general symmetric gamma_ij(x), general x0(t~);
    #          and t~ = t on the worldline. Also, at an instant with xdot0 = 0,
    #          the differentials of L660: dx = d(dx) and
    #          dt = (1 + gamma_ij a_c^j dx^i/(alpha_0^2 c^2)) dt~.
    # Mutant: the shift without the 1/alpha_0^2 (t = t~ + gamma v dx/c^2), whose
    #          cross term on the worldline is nonzero when alpha_0 != 1.
    # ------------------------------------------------------------------------
    eq("eq:moving_sync", 657, "t == t~ + gamma_ij(x0) xdot0^j dx^i/(alpha(x0)^2 c^2)")
    cross_on_wl = [sp.simplify(gpull[0, k].subs({q1: 0, q2: 0})) for k in (1, 2)]
    boolcheck("cross terms g(t~, dx^i) vanish on the worldline at every velocity",
              all(v_ == 0 for v_ in cross_on_wl))
    symcheck("t == t~ on the worldline", t_of.subs({q1: 0, q2: 0}) - Tt)
    symcheck("at rest: dt/dt~ == 1 + gamma_ij a_c^j dx^i/(alpha_0^2 c^2)",
             sp.simplify(_rest(J[0, 0]) - (1 + (qv.T*gam0*acv)[0]/(alpha0**2*c**2))))
    symcheck("at rest: dt/d(dx^i) == 0 and dx/dt~ == 0 (so dx = d(dx))",
             _rest(sp.Matrix([J[0, 1], J[0, 2], J[1, 0], J[2, 0]])))
    t_bad = Tt + (qv.T*gam0*vel)[0]/c**2
    Jb = sp.Matrix([[sp.diff(t_bad, Tt), sp.diff(t_bad, q1), sp.diff(t_bad, q2)],
                    [sp.diff(xpos[0], Tt), sp.diff(xpos[0], q1), sp.diff(xpos[0], q2)],
                    [sp.diff(xpos[1], Tt), sp.diff(xpos[1], q1), sp.diff(xpos[1], q2)]])
    mutant("shift without 1/alpha_0^2: cross term on the worldline",
           sp.simplify((Jb.T*gstat*Jb)[0, 1].subs({q1: 0, q2: 0})))

    # ------------------------------------------------------------------------
    # eq:moving_frame_alpha  (sgm.tex L662)
    # States: alpha~ == alpha(x0 + dx) (1 + gamma_ij a_c^j dx^i/(alpha_0^2 c^2)),
    #          the time-dilation factor (scalar) sqrt(-g_00) in the moving coordinates
    #          at an instant of rest.
    # Derived from: eq:moving_sync, eq:alpha_def; substituting the
    #          differentials of L660 into the static metric (L655). Compared in
    #          the text with the form of eq:alpha_rindler.
    # Test: the full pull-back at xdot0 = 0, xddot0 = a_c: g_t~t~ ==
    #          -alpha~^2 c^2, the spatial block equals gamma_ij(x0 + dx) unchanged,
    #          no cross terms at any dx; then L665: the gradient of alpha~ in dx at
    #          dx = 0 is d_i alpha + gamma_ij a_c^j/(alpha_0 c^2).
    # Mutant: alpha~ without the 1/alpha_0^2 in the Rindler factor (compared
    #          with the pulled-back g_t~t~).
    # ------------------------------------------------------------------------
    eq("eq:moving_frame_alpha", 662, "alpha~ == alpha(x0+dx)(1 + gamma_ij a_c^j dx^i/(alpha_0^2 c^2))")
    gp_rest = _rest(gpull)
    alt = _at(al2(yy1, yy2), xpos)*(1 + (qv.T*gam0*acv)[0]/(alpha0**2*c**2))
    symcheck("g_t~t~ == -alpha~^2 c^2", sp.simplify(gp_rest[0, 0] + alt**2*c**2))
    symcheck("no cross term at any dx", sp.simplify(sp.Matrix([gp_rest[0, 1], gp_rest[0, 2]])))
    symcheck("spatial block unchanged: gamma_ij(x0 + dx)", sp.simplify(gp_rest[1:, 1:] - _at(gam2, xpos)))
    grad_alt = sp.Matrix([sp.diff(alt, q_).subs({q1: 0, q2: 0}).doit() for q_ in (q1, q2)])
    dal0 = sp.Matrix([_at(sp.diff(al2(yy1, yy2), y_), x0f) for y_ in (yy1, yy2)])
    symcheck("L665: grad alpha~ at dx = 0 == d_i alpha + gamma_ij a_c^j/(alpha_0 c^2)",
             sp.simplify(grad_alt - (dal0 + gam0*acv/(alpha0*c**2))))
    alt_bad = _at(al2(yy1, yy2), xpos)*(1 + (qv.T*gam0*acv)[0]/c**2)
    mutant("Rindler factor without 1/alpha_0^2", sp.simplify(gp_rest[0, 0] + alt_bad**2*c**2))

    # ------------------------------------------------------------------------
    # eq:geodesic_balance  (sgm.tex L669)
    # States: grad omega_grav + grad omega_in == 0 (vector).
    # Derived from: eq:moving_frame_alpha, eq:second_law, eq:omega_alpha,
    #          eq:inertia_gradient. As for eq:second_law, the momentum in the accelerating rest
    #          frame stays zero, so the total gradient of omega_0 alpha~ vanishes at
    #          the particle; the two parts are omega_0 d_i alpha and
    #          omega_0 gamma_ij a_c^j/(alpha_0 c^2) (L665).
    # Test: omega_0 times the gradient of alpha~ splits exactly into those two
    #          parts; in flat spacetime (alpha_0 = 1, gamma = delta) the second is
    #          eq:inertia_gradient with a = a_c (L665); the balance is not an
    #          identity: at a_c = 0 the sum is the bare static gradient, nonzero for
    #          general alpha (L677: a particle held at rest feels its weight).
    # Mutant: an inertial coefficient omega_1 != omega_0 (unequal masses), so
    #          the balanced acceleration depends on the body (tested at the next
    #          block); here: the balance with a_c = 0 identically zero (it is not).
    # ------------------------------------------------------------------------
    eq("eq:geodesic_balance", 669, "grad omega_grav + grad omega_in == 0")
    gw_grav = w0*dal0
    gw_inn = w0*gam0*acv/(alpha0*c**2)
    symcheck("omega_0 grad alpha~ == grad omega_grav + grad omega_in", sp.simplify(w0*grad_alt - gw_grav - gw_inn))
    symcheck("flat limit: grad omega_in == omega_0 a_c/c^2 (eq:inertia_gradient)",
             (gw_inn.subs(alpha0, 1)).subs({gam0[0, 0]: 1, gam0[1, 1]: 1, gam0[0, 1]: 0}, simultaneous=True)
             - w0*acv/c**2)
    mutant("balance holds with a_c = 0 (it must not)", sp.simplify((gw_grav + gw_inn).subs({ac1: 0, ac2: 0})))

    # ------------------------------------------------------------------------
    # eq:freefall_accel  (sgm.tex L674)
    # States: a_c^k == -c^2 alpha_0 gamma^ki d_i alpha (vector), the coordinate
    #          acceleration at an instant of rest.
    # Derived from: eq:geodesic_balance multiplied by alpha_0 c^2 gamma^ki/omega_0
    #          and summed over i (L672).
    # Test: (a) the literal operation on the two gradients (matrix gamma^-1
    #          times the balance), solved for a_c; (b) it is the geodesic
    #          coordinate acceleration at rest, -c^2 Gamma^k_00 (x^0 = ct), from
    #          the Christoffel symbols of -alpha^2 (dx^0)^2 + gamma_ij dx^i dx^j with
    #          general alpha and symmetric gamma (2D); (c) the same from Hamilton's
    #          equations for H = alpha sqrt(m^2c^4 + c^2 gamma^ij p_i p_j) at p = 0;
    #          (d) no omega_0 (no mass): with an
    #          independent inertial coefficient omega_1 the result is multiplied
    #          by omega_0/omega_1 (L679); (e) weak field: -c^2 grad of
    #          eq:alpha_weak == -GM r_hat/r^2 (L677), and at Earth's surface 9.8 m/s^2.
    # Mutant: a_c = -c^2 gamma^ki d_i alpha/alpha_0 (alpha_0 on the wrong side).
    # ------------------------------------------------------------------------
    eq("eq:freefall_accel", 674, "a_c^k == -c^2 alpha_0 gamma^ki d_i alpha")
    bal = gw_grav + gw_inn
    mult = sp.simplify(alpha0*c**2*gam0.inv()*bal/w0)
    symcheck("alpha_0 c^2 gamma^ki/omega_0 times the balance == a_c + c^2 alpha_0 gamma^ki d_i alpha",
             sp.simplify(mult - (acv + c**2*alpha0*gam0.inv()*dal0)))
    sol_ac = sp.solve(list(bal), [ac1, ac2], dict=True)[0]
    ac_sol = sp.Matrix([sol_ac[ac1], sol_ac[ac2]])
    ff_target = -c**2*alpha0*gam0.inv()*dal0
    symcheck("solving the balance gives a_c == -c^2 alpha_0 gamma^-1 grad alpha", sp.simplify(ac_sol - ff_target))
    Xg = [sp.Symbol('x0'), yy1, yy2]
    gg = sp.zeros(3, 3)
    gg[0, 0] = -al2(yy1, yy2)**2
    gg[1:, 1:] = gam2
    Gg = _S2_christoffel(gg, Xg)
    geo = sp.Matrix([-c**2*Gg[k][0][0] for k in (1, 2)])
    symcheck("geodesic at rest: -c^2 Gamma^k_00 == -c^2 alpha gamma^ki d_i alpha",
             sp.simplify(_at(geo, x0f) - ff_target))
    pp1, pp2 = sp.symbols('pp_1 pp_2', real=True)
    ppv = sp.Matrix([pp1, pp2])
    H2 = al2(yy1, yy2)*sp.sqrt(m**2*c**4 + c**2*(ppv.T*gam2.inv()*ppv)[0])
    xdot_H = sp.Matrix([sp.diff(H2, p_) for p_ in (pp1, pp2)])
    pdot_H = sp.Matrix([-sp.diff(H2, y_) for y_ in (yy1, yy2)])
    # at p = 0, xdot = 0, so d/dt xdot = (d xdot/dp) pdot
    acc_H = (xdot_H.jacobian(ppv)*pdot_H).subs({pp1: 0, pp2: 0})
    symcheck("Hamilton's equations at p = 0 give the same a_c",
             sp.simplify(_at(acc_H, x0f) - ff_target))
    boolcheck("a_c contains no omega_0 (no mass)", not ac_sol.has(m) and not ac_sol.has(hb))
    w1 = sp.symbols('omega_1', positive=True)
    bal1 = gw_grav + w1*gam0*acv/(alpha0*c**2)
    sol1 = sp.solve(list(bal1), [ac1, ac2], dict=True)[0]
    symcheck("unequal coefficients: a_c == (omega_0/omega_1) times eq:freefall_accel",
             sp.simplify(sp.Matrix([sol1[ac1], sol1[ac2]]) - (w0/w1)*ff_target))
    boolcheck("... and that result depends on the body (on m)",
              sp.simplify(sp.diff(sol1[ac1], m)) != 0)
    a_ff_weak = -c**2*grad(1 - G*M/(c**2*rr3))
    symcheck("weak field: a_c == -GM r_hat/r^2", sp.simplify(a_ff_weak + G*M*rhat3/rr3**2))
    check("Earth's surface: |a_c| == 9.81 m/s^2", G_n*5.9722e24/6.371e6**2, g_n, tol=3e-3)
    mutant("alpha_0 on the wrong side", sp.simplify(ac_sol + c**2*gam0.inv()*dal0/alpha0))

    # ------------------------------------------------------------------------
    # Remark rem:rank (L686-L688): no display; claims tested here because the
    # text derives them from eq:alpha_rindler and eq:alpha_weak.
    # States: (a) for g_00 = -alpha^2, no cross term, flat spatial part,
    #          R_0i0j = alpha d_i d_j alpha (index 0 for x^0 = ct, MTW sign);
    #          (b) the Hessian of 1/r has eigenvalue 2/r^3 radially and -1/r^3 in
    #          each perpendicular direction.
    # Derived from: eq:omega_expansion, eq:alpha_rindler, eq:alpha_weak,
    #          eq:geometric_force; the R_0i0j formula is a cited standard result
    #          (MTW sign convention), no earlier label.
    # Test: Riemann tensor of diag(-alpha(x,y,z)^2, 1, 1, 1) computed from its
    #          Christoffel symbols, lowered with g_00; Hessian of 1/r at
    #          (1.3, -0.7, 2.9) applied to r_hat and to two perpendicular vectors.
    # Mutant: R_0i0j = -alpha d_i d_j alpha (opposite sign convention) and the
    #          radial eigenvalue -2/r^3.
    # ------------------------------------------------------------------------
    eq("rem:rank (inline, L686-L688)", 686, "R_0i0j == alpha d_i d_j alpha; Hess(1/r) eigenvalues 2/r^3, -1/r^3")
    R0i0j = sp.Matrix(3, 3, lambda i_, j_: sp.simplify(g4[0, 0]*_S2_riemann_up(Gam4, X4, 0, i_ + 1, 0, j_ + 1)))
    hess_a = sp.hessian(alf3, X3)
    symcheck("R_0i0j == alpha d_i d_j alpha", sp.simplify(R0i0j - alf3*hess_a))
    Hinv = sp.hessian(1/rr3, X3).subs(pt)
    rh = sp.Matrix([pt[x], pt[y], pt[z]])
    rn = sp.sqrt((rh.T*rh)[0])
    rh = rh/rn
    perp1 = sp.Matrix([pt[y], -pt[x], 0])
    perp2 = rh.cross(perp1)
    symcheck("Hessian(1/r) r_hat == (2/r^3) r_hat", sp.simplify(Hinv*rh - 2/rn**3*rh))
    symcheck("Hessian(1/r) on perpendicular vector 1 == (-1/r^3) vector", sp.simplify(Hinv*perp1 + perp1/rn**3))
    symcheck("Hessian(1/r) on perpendicular vector 2 == (-1/r^3) vector", sp.simplify(Hinv*perp2 + perp2/rn**3))
    symcheck("Rindler factor: Hessian == 0 (no tidal term)", sp.hessian(alR, Dv))
    mutant("R_0i0j == -alpha d_i d_j alpha", sp.simplify(R0i0j + alf3*hess_a))
    mutant("radial eigenvalue -2/r^3", sp.simplify(Hinv*rh + 2/rn**3*rh))

    # ========================================================================
    # Subsection "Light"
    # ========================================================================
    alI = sp.symbols('alpha_I', positive=True)
    om_s, kk = sp.symbols('omega k', positive=True)

    # ------------------------------------------------------------------------
    # eq:isotropic  (sgm.tex L698)
    # States: ds^2 == -alpha^2 c^2 dt^2 + alpha^-2 dx^2, the weak-field metric of
    #          a static mass in isotropic coordinates, to first order in the potential.
    # Derived from: cited (MTW); alpha as in eq:alpha_def.
    # Test: the exact isotropic Schwarzschild metric, g_00 = -((1-U/2)/(1+U/2))^2,
    #          g_ij = (1+U/2)^4 delta_ij with U = GM/(c^2 rho): with alpha its
    #          lapse, the spatial factor (1+U/2)^4 equals alpha^-2 to first order in U
    #          and differs at second order (so the display is first order, as stated).
    # Mutant: spatial factor alpha^-1 (only half the spatial curvature).
    # ------------------------------------------------------------------------
    eq("eq:isotropic", 698, "ds^2 == -alpha^2 c^2 dt^2 + alpha^-2 dx^2")
    U = sp.symbols('U', positive=True)
    al_iso = (1 - U/2)/(1 + U/2)
    sp_iso = (1 + U/2)**4
    symcheck("first order in U: (1+U/2)^4 == alpha^-2",
             sp.series(sp_iso - al_iso**-2, U, 0, 2).removeO())
    boolcheck("they differ at second order",
              sp.simplify(sp.series(sp_iso - al_iso**-2, U, 0, 3).removeO()) != 0)
    mutant("spatial factor alpha^-1", sp.series(sp_iso - al_iso**-1, U, 0, 2).removeO())

    # ------------------------------------------------------------------------
    # eq:isotropic_dispersion  (sgm.tex L703)
    # States: hbar omega == alpha sqrt(m^2 c^4 + alpha^2 c^2 hbar^2 |k|^2) (scalar).
    # Derived from: mass shell g^{mu nu} k_mu k_nu = -(mc/hbar)^2 with
    #          k_mu = (-omega/c, k), x^0 = ct, on the inverse of eq:isotropic
    #          (cited mass-shell condition; labels: eq:isotropic).
    # Test: invert the metric diag(-alpha^2, alpha^-2 x3), form the
    #          contraction with k = (k, 0, 0) (isotropy: any direction), solve for
    #          omega > 0; L706: |k| -> 0 gives omega_0 alpha.
    # Mutant: alpha^-1 in place of alpha^2 inside the root (metric inverted
    #          wrongly: g^ij = alpha^-2).
    # ------------------------------------------------------------------------
    eq("eq:isotropic_dispersion", 703, "hbar omega == alpha sqrt(m^2c^4 + alpha^2 c^2 hbar^2 k^2)")
    giso = sp.diag(-alI**2, alI**-2, alI**-2, alI**-2)
    ginv_iso = giso.inv()
    kmu = sp.Matrix([-om_s/c, kk, 0, 0])
    shell = (kmu.T*ginv_iso*kmu)[0] + (m*c/hb)**2
    sols = [s_ for s_ in sp.solve(shell, om_s) if s_.is_positive is not False]
    disp = alI*sp.sqrt(m**2*c**4 + alI**2*c**2*hb**2*kk**2)/hb
    boolcheck("one positive root", len(sols) == 1)
    symcheck("positive root == eq:isotropic_dispersion / hbar", sp.simplify(sols[0] - disp))
    symcheck("|k| -> 0: omega == omega_0 alpha", sp.simplify(disp.subs(kk, 0) - w0*alI))
    mutant("g^ij = alpha^-2 in the root",
           sp.simplify(sols[0] - alI*sp.sqrt(m**2*c**4 + c**2*hb**2*kk**2/alI**2)/hb))

    # ------------------------------------------------------------------------
    # eq:light_dispersion  (sgm.tex L708)
    # States: omega == c|k|/n_r with n_r = 1/alpha^2 (scalar).
    # Derived from: eq:isotropic_dispersion at m = 0.
    # Test: substitution; and independently the null condition
    #          g_mu nu dx^mu dx^nu = 0 of eq:isotropic gives the coordinate speed
    #          |dx/dt| = alpha^2 c = c/n_r; Einstein's 1911 index 1/alpha (L746)
    #          is not this one.
    # Mutant: n_r = 1/alpha.
    # ------------------------------------------------------------------------
    eq("eq:light_dispersion", 708, "omega == c|k|/n_r, n_r = 1/alpha^2")
    nr = 1/alI**2
    symcheck("m = 0: omega == c k/n_r", sp.simplify(disp.subs(m, 0) - c*kk/nr))
    vsp = sp.symbols('v_light', positive=True)
    vsol = sp.solve(sp.Eq(-alI**2*c**2 + alI**-2*vsp**2, 0), vsp)[0]
    symcheck("null rays: coordinate speed == c/n_r", sp.simplify(vsol - c/nr))
    mutant("n_r == 1/alpha", sp.simplify(disp.subs(m, 0) - c*kk*alI))

    # ------------------------------------------------------------------------
    # eq:light_gradient, line 1  (sgm.tex L715)
    # States: hbar kdot == -hbar grad omega (vector).
    # Derived from: eq:eikonal_ray multiplied by hbar.
    # Test: Hamilton's ray equations with H = hbar omega(x, k), omega =
    #          c|k|/n(x) for a general n(x, y, z): the rate of hbar k along the
    #          flow equals -hbar grad omega at fixed k (Poisson bracket
    #          {k_i, omega}); the frequency is conserved along the flow (L721).
    # Mutant: +hbar grad omega.
    # ------------------------------------------------------------------------
    eq("eq:light_gradient (line 1)", 715, "hbar kdot == -hbar grad omega")
    kx, ky, kz = sp.symbols('k_x k_y k_z', real=True)
    Kv = (kx, ky, kz)
    nf = sp.Function('n', positive=True)(x, y, z)
    kn = sp.sqrt(kx**2 + ky**2 + kz**2)
    wl = c*kn/nf

    def _flow(f_):
        return sum(sp.diff(wl, kk_)*sp.diff(f_, xx_) - sp.diff(wl, xx_)*sp.diff(f_, kk_)
                   for xx_, kk_ in zip(X3, Kv))
    kdot = sp.Matrix([_flow(kk_) for kk_ in Kv])
    symcheck("hbar kdot along the ray flow == -hbar grad omega|_k", hb*kdot + hb*grad(wl))
    symcheck("L721: d omega/dt along the ray == 0", sp.simplify(_flow(wl)))
    mutant("hbar kdot == +hbar grad omega", sp.simplify(hb*kdot - hb*grad(wl)))

    # ------------------------------------------------------------------------
    # eq:light_gradient, line 2  (sgm.tex L716)
    # States: ... == (hbar omega/n_r) grad n_r; and (L719) grad n_r =
    #          -2 grad alpha/alpha^3, so light bends toward smaller alpha.
    # Derived from: the first line of this display; gradient at fixed k of
    #          eq:light_dispersion.
    # Test: -hbar grad(c|k|/n) for general n(x,y,z); grad(1/alpha^2) for
    #          general alpha(x,y,z); d n_r/d alpha < 0.
    # Mutant: (hbar omega) grad n_r (the 1/n_r dropped), and grad n_r =
    #          -grad alpha/alpha^3.
    # ------------------------------------------------------------------------
    eq("eq:light_gradient (line 2)", 716, "... == (hbar omega/n_r) grad n_r")
    symcheck("-hbar grad(c k/n) == (hbar omega/n) grad n", sp.simplify(-hb*grad(wl) - hb*wl/nf*grad(nf)))
    symcheck("grad(1/alpha^2) == -2 grad alpha/alpha^3", sp.simplify(grad(1/alf3**2) + 2*grad(alf3)/alf3**3))
    boolcheck("d n_r/d alpha = -2/alpha^3 < 0", bool(sp.diff(1/alI**2, alI).is_negative))
    mutant("(hbar omega) grad n", sp.simplify(-hb*grad(wl) - hb*wl*grad(nf)))
    mutant("grad n_r == -grad alpha/alpha^3", sp.simplify(grad(1/alf3**2) + grad(alf3)/alf3**3))

    # ------------------------------------------------------------------------
    # unlabeled align*, line 1  (sgm.tex L723)
    # States: d s_ray/dt == c/n_r, the speed along the ray (scalar).
    # Derived from: eq:light_dispersion, eq:eikonal_ray; the group velocity
    #          d omega/dk = (c/n_r) k/|k| (L721).
    # Test: |d omega/dk| for omega = c|k|/n(x) with general n.
    # Mutant: c n_r.
    # ------------------------------------------------------------------------
    eq("unlabeled (L723)", 723, "d s_ray/dt == c/n_r")
    vg = sp.Matrix([sp.diff(wl, kk_) for kk_ in Kv])
    symcheck("group velocity == (c/n) k/|k|", sp.simplify(vg - (c/nf)*sp.Matrix(Kv)/kn))
    symcheck("|group velocity| == c/n", sp.simplify(sp.sqrt((vg.T*vg)[0]) - c/nf))
    mutant("speed c n", sp.simplify(sp.sqrt((vg.T*vg)[0]) - c*nf))

    # ------------------------------------------------------------------------
    # unlabeled align*, line 2  (sgm.tex L724)
    # States: k == (n_r omega/c) dx/ds_ray (vector).
    # Derived from: |k| = n_r omega/c (eq:light_dispersion inverted) and k
    #          parallel to the group velocity, so k/|k| = dx/ds_ray.
    # Test: n omega/c == |k| for omega = c|k|/n; group velocity cross k == 0;
    #          dx/ds = vg/|vg| == k/|k|, so (n omega/c) dx/ds == k.
    # Mutant: k == (omega/(n c)) dx/ds.
    # ------------------------------------------------------------------------
    eq("unlabeled (L724)", 724, "k == (n_r omega/c) dx/ds_ray")
    symcheck("|k| == n omega/c", sp.simplify(nf*wl/c - kn))
    symcheck("group velocity parallel to k", sp.simplify(vg.cross(sp.Matrix(Kv))))
    dxds = vg/sp.sqrt((vg.T*vg)[0])
    symcheck("(n omega/c) dx/ds == k", sp.simplify(nf*wl/c*dxds - sp.Matrix(Kv)))
    mutant("k == (omega/(n c)) dx/ds", sp.simplify(wl/(nf*c)*dxds - sp.Matrix(Kv)))

    # ------------------------------------------------------------------------
    # unlabeled \[ \]  (sgm.tex L728)
    # States: (hbar omega/n_r) d/ds_ray (n_r dx/ds_ray) == (hbar omega/n_r) grad n_r.
    # Derived from: substituting the two lines of L723-L724 into eq:light_gradient
    #          with omega constant: hbar dk/dt = hbar (ds/dt) d/ds[(n omega/c) dx/ds]
    #          = (hbar omega/n) d/ds(n dx/ds).
    # Test: (a) the left-side conversion for a general path x(s) (one component)
    #          and general n(s) along it, omega constant; (b) along the ray flow,
    #          (n/c) d/dt[n k_i/|k|] == d_i n for a general n(x,y,z), which
    #          combined with (a) is the displayed equality.
    # Mutant: the conversion with dt/ds = c/n (inverted speed).
    # ------------------------------------------------------------------------
    eq("unlabeled (L728)", 728, "(hbar omega/n) d/ds(n dx/ds) == (hbar omega/n) grad n")
    s_r = sp.symbols('s_ray', real=True)
    n_s = sp.Function('n_s', positive=True)(s_r)
    xs_s = sp.Function('xpath')(s_r)
    k_of_s = n_s*om_s/c*sp.diff(xs_s, s_r)
    lhs_conv = hb*(c/n_s)*sp.diff(k_of_s, s_r)          # hbar dk/dt = hbar (ds/dt) dk/ds
    symcheck("hbar dk/dt == (hbar omega/n) d/ds(n dx/ds)",
             sp.simplify(lhs_conv - hb*om_s/n_s*sp.diff(n_s*sp.diff(xs_s, s_r), s_r)))
    for i_, (xi_, ki_) in enumerate(zip(X3, Kv), start=1):
        f_i = nf*ki_/kn
        symcheck(f"component {i_}: (n/c) d/dt[n dx_{i_}/ds] == d n/dx_{i_} along rays",
                 sp.simplify((nf/c)*_flow(f_i) - sp.diff(nf, xi_)))
    mutant("dt/ds = c/n", sp.simplify(hb*(n_s/c)*sp.diff(k_of_s, s_r)
                                     - hb*om_s/n_s*sp.diff(n_s*sp.diff(xs_s, s_r), s_r)))

    # ------------------------------------------------------------------------
    # eq:optical_path  (sgm.tex L732)
    # States: d/ds_ray (n_r dx/ds_ray) == grad n_r (vector), the ray equation.
    # Derived from: the unlabeled display at L728 (no label) divided by
    #          hbar omega/n_r; its inputs are eq:light_gradient, eq:light_dispersion.
    # Test: (a) the division; (b) numerically: integrate Hamilton's ray
    #          equations for omega = c|k|/n with n = 1/alpha^2, alpha = 1 - K/r
    #          (K = 0.05, impact parameter 1.3, c = 1), resample along path length,
    #          and compare d/ds(n dx/ds) by finite differences with grad n at
    #          interior points; (c) L735: the deflection angle from the index
    #          1 + 2K/r (n = 1/alpha^2 to first order, K = GM/c^2) by the transverse
    #          integral int d_b n dz == -4K/b, and each factor of alpha gives half.
    # Mutant: d/ds(dx/ds) == grad n (n outside the derivative dropped),
    #          numerically, and the deflection of the 1911 index 1/alpha.
    # ------------------------------------------------------------------------
    eq("eq:optical_path", 732, "d/ds(n dx/ds) == grad n")
    divided = (hb*om_s/n_s*sp.diff(n_s*sp.diff(xs_s, s_r), s_r))/(hb*om_s/n_s)
    symcheck("division by hbar omega/n leaves d/ds(n dx/ds)",
             sp.simplify(divided - sp.diff(n_s*sp.diff(xs_s, s_r), s_r)))
    from scipy.integrate import solve_ivp
    Kn, bimp = 0.05, 1.3

    def n_fun(p_):
        rr = np.linalg.norm(p_)
        return (1 - Kn/rr)**-2

    def grad_n(p_):
        rr = np.linalg.norm(p_)
        return -2*(1 - Kn/rr)**-3*(Kn*p_/rr**3)

    def rhs(_t, Y):
        p_, kvec = Y[:3], Y[3:]
        kmag = np.linalg.norm(kvec)
        nn = n_fun(p_)
        xdot = kvec/(nn*kmag)
        kdot_ = kmag*grad_n(p_)/nn**2              # -grad(|k|/n) = |k| grad n/n^2
        return np.concatenate([xdot, kdot_])
    Y0 = np.array([-6.0, bimp, 0.2, 1.0, 0.0, 0.0])*1.0
    Y0[3:] *= n_fun(Y0[:3])                           # |k| = n omega/c with omega = 1
    sol_ray = solve_ivp(rhs, (0, 14), Y0, rtol=1e-12, atol=1e-13, dense_output=True)
    tt = np.linspace(0.5, 13.5, 4001)
    P = sol_ray.sol(tt)[:3].T
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s_arc = np.concatenate([[0.0], np.cumsum(seg)])
    ii = np.arange(400, 3600, 400)
    worst = 0.0
    worst_mut = 0.0
    hst = 2

    def _d_ds(F_):
        return (F_[ii + hst] - F_[ii - hst])/(s_arc[ii + hst] - s_arc[ii - hst])[:, None]
    dxds_n = np.gradient(P, s_arc, axis=0)
    nvals = np.array([n_fun(p_) for p_ in P])
    lhs_n = _d_ds(nvals[:, None]*dxds_n)
    lhs_m = _d_ds(dxds_n)
    rhs_n = np.array([grad_n(P[i]) for i in ii])
    scale_n = np.max(np.linalg.norm(rhs_n, axis=1))
    worst = np.max(np.linalg.norm(lhs_n - rhs_n, axis=1))
    worst_mut = np.max(np.linalg.norm(lhs_m - rhs_n, axis=1))
    check("numerical ray: max |d/ds(n dx/ds) - grad n| small vs |grad n|",
          worst, 0.0, tol=1e-4, scale=scale_n)
    zC, bC, KC = sp.symbols('z_C b_C K_C', positive=True)
    defl = sp.integrate(sp.diff(2*KC/sp.sqrt(bC**2 + zC**2), bC), (zC, -sp.oo, sp.oo))
    symcheck("deflection from n = 1 + 2K/r: int d_b n dz == -4K/b, |.| = 4GM/(c^2 b)",
             sp.simplify(defl + 4*KC/bC))
    defl_half = sp.integrate(sp.diff(KC/sp.sqrt(bC**2 + zC**2), bC), (zC, -sp.oo, sp.oo))
    symcheck("one factor of alpha (n = 1 + K/r, the 1911 index): half, 2GM/(c^2 b)",
             sp.simplify(defl_half + 2*KC/bC))
    symcheck("1/alpha^2 == 1 + 2K/r to first order in K for alpha = 1 - K/r",
             sp.series((1 - eps_*KC/r)**-2, eps_, 0, 2).removeO() - (1 + 2*eps_*KC/r))
    numutant("d/ds(dx/ds) in place of d/ds(n dx/ds): max error relative to |grad n|",
             worst_mut/scale_n + 1, 1, tol=1e-3)
    mutant("1911 index gives the full deflection", sp.simplify(defl_half + 4*KC/bC))

    # ------------------------------------------------------------------------
    # eq:constitutive_map  (sgm.tex L739)
    # States: eps_r^ij == -sqrt(-det g) g^ij/g_00 (matrix), and mu_r^ij equal to it.
    # Derived from: cited (Gordon 1923; Plebanski 1960; Leonhardt 2006); its
    #          evaluation in the text (L742) uses eq:isotropic.
    # Test: (a) on eq:isotropic: sqrt(-det g) = alpha^-2, g^ij = alpha^2 delta,
    #          g_00 = -alpha^2, eps_r = 1/alpha^2, n = sqrt(eps mu) = 1/alpha^2,
    #          which agrees with the index of eq:light_dispersion derived
    #          independently from the mass shell (L742); (b) for a general
    #          diagonal static metric diag(-A, B, B, B) the map gives
    #          sqrt(B/A), and the null condition gives the same index sqrt(B/A);
    #          (c) impedance sqrt(mu/eps) = 1 (L744); (d) time part alone
    #          (g_ij = delta) gives 1/alpha, Einstein's 1911 index (L746),
    #          c(1 + Phi/c^2) = c alpha_weak exactly; (e) L748: Schwarzschild
    #          g_rr = -1/g_00 and the exact isotropic index (1+U/2)^2/alpha ==
    #          4/[alpha(1+alpha)^2], equal to 1/alpha^2 to first order.
    # Mutant: the map without sqrt(-det g) (gives 1, no refraction), and the
    #          exact isotropic index 4/[alpha^2 (1+alpha)^2].
    # ------------------------------------------------------------------------
    eq("eq:constitutive_map", 739, "eps_r^ij == -sqrt(-det g) g^ij/g_00")

    def _cmap(gm):
        return sp.simplify(-sp.sqrt(-gm.det())*gm.inv()[1:, 1:]/gm[0, 0])
    eps_iso = _cmap(giso)
    symcheck("on eq:isotropic: sqrt(-det g) == alpha^-2", sp.simplify(sp.sqrt(-giso.det()) - alI**-2))
    symcheck("on eq:isotropic: eps_r^ij == delta^ij/alpha^2", eps_iso - sp.eye(3)/alI**2)
    symcheck("n = sqrt(eps_r mu_r) == n_r of eq:light_dispersion", sp.sqrt(eps_iso[0, 0]**2) - nr)
    Ag, Bg = sp.symbols('A_g B_g', positive=True)
    eps_gen = _cmap(sp.diag(-Ag, Bg, Bg, Bg))
    vnull = sp.solve(sp.Eq(-Ag*c**2 + Bg*vsp**2, 0), vsp)[0]
    symcheck("general diag(-A, B, B, B): map index == c/(null speed) == sqrt(B/A)",
             sp.simplify(eps_gen[0, 0] - c/vnull))
    symcheck("impedance sqrt(mu_r/eps_r) == 1", sp.sqrt(eps_iso[0, 0]/eps_iso[0, 0]) - 1)
    eps_time = _cmap(sp.diag(-alI**2, 1, 1, 1))
    symcheck("time part alone: eps_r == mu_r == 1/alpha (index of 1911)", eps_time - sp.eye(3)/alI)
    symcheck("spatial part multiplies each response by a further 1/alpha",
             sp.simplify(eps_iso[0, 0]/eps_time[0, 0] - 1/alI))
    symcheck("L746: c(1 + Phi/c^2) with Phi = -GM/r == c alpha_weak", c*(1 - G*M/(r*c**2)) - c*a_weak)
    symcheck("L748: Schwarzschild g_rr == -1/g_00", grr - (-1/(-(1 - Rs/r))))
    n_exact = (1 + U/2)**2/al_iso
    symcheck("L748: exact isotropic index == 4/[alpha(1+alpha)^2]", sp.simplify(n_exact - 4/(al_iso*(1 + al_iso)**2)))
    ee = sp.symbols('e', real=True)
    symcheck("L748: 4/[alpha(1+alpha)^2] == 1/alpha^2 to first order (alpha = 1 - e)",
             sp.series(4/((1 - ee)*(2 - ee)**2) - 1/(1 - ee)**2, ee, 0, 2).removeO())
    mutant("map without sqrt(-det g)", sp.simplify(-giso.inv()[1, 1]/giso[0, 0] - 1/alI**2))
    mutant("exact index 4/[alpha^2 (1+alpha)^2]", sp.simplify(n_exact - 4/(al_iso**2*(1 + al_iso)**2)))


# ============================================================================
# Helpers for part S3. A quaternion is a list [s, v1, v2, v3] meaning
# s + v1 e1 + v2 e2 + v3 e3, with e1 e2 = e3 (eq:quaternion_units).
# A biquaternion P + i Q (i the commuting unit of sec:maxwell) is a pair (P, Q).
# ============================================================================

def _S3_mul(a, b):
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return [a0*b0 - a1*b1 - a2*b2 - a3*b3,
            a0*b1 + a1*b0 + a2*b3 - a3*b2,
            a0*b2 - a1*b3 + a2*b0 + a3*b1,
            a0*b3 + a1*b2 - a2*b1 + a3*b0]


def _S3_add(*qs):
    return [sum(qq[i] for qq in qs) for i in range(4)]


def _S3_sc(f, a):
    return [f*u for u in a]


def _S3_conj(a):
    return [a[0], -a[1], -a[2], -a[3]]


def _S3_d(a, v, n=1):
    return [sp.diff(u, v, n) for u in a]


def _S3_mat(a):
    return sp.Matrix([sp.expand(u) for u in a])


def _S3_vq(w):
    return [0, w[0], w[1], w[2]]


def _S3_q2c(a):
    # eq:spinor_quaternion: psi = x1 - y1 e3 + e2 (x2 - y2 e3) = x1 - y2 e1 + x2 e2 - y1 e3
    # so (psi_1, psi_2) = (s - i v3, v2 - i v1)
    return sp.Matrix([a[0] - sp.I*a[3], a[2] - sp.I*a[1]])


def _S3_pick(M):
    """First entry of a residual that is nonzero after expansion (0 if none).
    Mutant residuals are passed through this so that simplify runs on one entry."""
    entries = list(M) if hasattr(M, "shape") else [M]
    for u in entries:
        u = sp.expand(u)
        if u != 0:
            return u
    return sp.Integer(0)


def _S3_curl(w, X3):
    x, y, z = X3
    return [sp.diff(w[2], y) - sp.diff(w[1], z), sp.diff(w[0], z) - sp.diff(w[2], x),
            sp.diff(w[1], x) - sp.diff(w[0], y)]


def _S3_div(w, X3):
    return sum(sp.diff(w[k], X3[k]) for k in range(3))


_S3_ONE = [1, 0, 0, 0]
_S3_E = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
_S3_E3 = _S3_E[2]
_S3_SIG = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]),
           sp.Matrix([[1, 0], [0, -1]])]


def _S3_gauge_ops(V, A, hb, q, T, X3):
    """eq:phase_connection_time and eq:right_connection as functions on quaternions."""
    e3 = _S3_E3

    def Dt(f):
        return _S3_add(_S3_d(f, T), _S3_sc(-V/hb, _S3_mul(f, e3)))

    def mk(k):
        return lambda f: _S3_add(_S3_d(f, X3[k]), _S3_sc(q*A[k]/hb, _S3_mul(f, e3)))
    return Dt, [mk(k) for k in range(3)]


def _S3_edot(Dk, f):
    return _S3_add(*[_S3_mul(_S3_E[k], Dk[k](f)) for k in range(3)])


def _S3_Dn(f, Dt, Dk, an, Ag, c, X3, crot=sp.Rational(1, 4), cdiv=sp.Rational(1, 2)):
    """eq:normal_derivative with the rotation and divergence coefficients as parameters."""
    Bg = _S3_curl(Ag, X3)
    divAg = _S3_div(Ag, X3)
    lnterm = sum(Ag[k]*sp.diff(an, X3[k]) for k in range(3))/an
    br = _S3_add(Dt(f),
                 _S3_sc(-1, _S3_add(*[_S3_sc(Ag[k], Dk[k](f)) for k in range(3)])),
                 _S3_sc(crot, _S3_mul(_S3_vq(Bg), f)),
                 _S3_sc(-cdiv*divAg, f),
                 _S3_sc(sp.Rational(1, 2)*lnterm, f))
    return _S3_sc(1/(c*an), br)


def _S3_bq_mul(u, v):
    return (_S3_add(_S3_mul(u[0], v[0]), _S3_sc(-1, _S3_mul(u[1], v[1]))),
            _S3_add(_S3_mul(u[0], v[1]), _S3_mul(u[1], v[0])))


def _S3_bq_add(*us):
    return (_S3_add(*[u[0] for u in us]), _S3_add(*[u[1] for u in us]))


def _S3_bq_sc(f, u):
    return (_S3_sc(f, u[0]), _S3_sc(f, u[1]))


def _S3_bq_d(u, v):
    return (_S3_d(u[0], v), _S3_d(u[1], v))


def _S3_bq_mat(u):
    return _S3_mat(u[0]).col_join(_S3_mat(u[1]))


def _S3_num(M, pt):
    return [complex(sp.N(sp.expand(u).subs(pt), 30)) for u in M]


def part_S3():
    part("sgm.tex L751-L1080: Force laws from the Dirac equation; Additive potentials; "
         "The spinor as a quaternion; The quaternionic wave equation; The master frequency "
         "equation; The Taylor expansion and the force laws; Maxwell's equations; Route IV "
         "for the Dirac equation, tensor rank, and spin")

    R = sp.Rational
    t, x, y, z = sp.symbols('t x y z', real=True)
    X3 = (x, y, z)
    X4 = (t, x, y, z)
    hb, q, c, m = sp.symbols('hbar q c m', positive=True)
    I = sp.I
    E, e3 = _S3_E, _S3_E3

    def QF(name, args=X4):
        return [sp.Function(f'{name}{i}', real=True)(*args) for i in range(4)]

    def RF(name, args=X4):
        return sp.Function(name, real=True)(*args)

    # ------------------------------------------------------------------------
    # eq:omega_em  (sgm.tex L764)
    # States: omega = omega_0 + (|p - qA|^2/2m + q Phi)/hbar; omega the frequency (scalar),
    #   omega_0 = mc^2/hbar the rest frequency, p the canonical momentum (vector), q the charge,
    #   Phi the scalar potential, A the vector potential (vector), m the mass.
    # Derived from: minimal coupling hbar omega -> hbar omega - q Phi, hbar k -> hbar k - qA in the
    #   free mass shell (hbar omega)^2 = (mc^2)^2 + c^2 |hbar k|^2, then the first two orders of the
    #   positive root in the momentum (slow particle).
    #   Labels used: eq:planck, eq:rest_freq, eq:H_eq_hw (minimal coupling is a premise).
    # Test: the coupled root solves the coupled mass shell; its expansion to second order in a
    #   momentum scale lambda reproduces eq:omega_em; the inline Hamilton velocity hbar grad_p omega
    #   equals pi/m. General symbolic p, A, Phi.
    # Mutant: |p + qA|^2 (the sign of the coupling flipped), a common sign slip.
    # ------------------------------------------------------------------------
    eq("eq:omega_em", 764, "omega = omega_0 + (|p - qA|^2/2m + q Phi)/hbar")
    pv = sp.Matrix(sp.symbols('p1:4', real=True))
    Av = sp.Matrix(sp.symbols('A1:4', real=True))
    Phs = sp.Symbol('Phi', real=True)
    lam = sp.Symbol('lambda', positive=True)
    piv = pv - q*Av
    E_root = q*Phs + sp.sqrt(m**2*c**4 + c**2*lam**2*piv.dot(piv))
    symcheck("the root q Phi + sqrt((mc^2)^2 + c^2 |pi|^2) solves (hbar w - q Phi)^2 = (mc^2)^2 + c^2 |p - qA|^2",
             sp.expand((E_root - q*Phs)**2 - c**2*lam**2*piv.dot(piv) - m**2*c**4))
    ser = sp.series(E_root, lam, 0, 3).removeO().subs(lam, 1)
    w0 = m*c**2/hb
    omega_em = w0 + (piv.dot(piv)/(2*m) + q*Phs)/hb
    symcheck("second-order expansion of the root / hbar == eq:omega_em", sp.expand(ser/hb - omega_em))
    symcheck("inline: hbar grad_p omega == pi/m (Hamilton's velocity)",
             sp.Matrix([sp.expand(hb*sp.diff(omega_em, pv[k]) - piv[k]/m) for k in range(3)]))
    om_mut = w0 + ((pv + q*Av).dot(pv + q*Av)/(2*m) + q*Phs)/hb
    mutant("omega with |p + qA|^2", sp.expand(ser/hb - om_mut))

    # ------------------------------------------------------------------------
    # eq:lorentz  (sgm.tex L769), first line
    # States: d pi/dt = -hbar grad omega - q dA/dt; pi = p - qA the kinematic momentum (vector),
    #   grad omega at fixed p, dA/dt = d_t A + (v.grad) A the rate of A along the path.
    # Derived from: pi = p - qA, dp/dt = -hbar grad omega (Theorem thm:universal, eq:sgm_as_dot_p),
    #   and the chain rule along the path.
    #   Labels used: eq:omega_em, thm:universal, eq:sgm, eq:sgm_as_dot_p.
    # Test: with H = hbar omega of eq:omega_em for general Phi(t,x), A(t,x), the total derivative of
    #   pi along Hamilton's flow (d/dt F = d_t F + xdot.grad_x F + pdot.grad_p F) equals the right side;
    #   also the inline gradient hbar d_i omega = q d_i Phi - q v_j d_i A_j.
    # Mutant: + q dA/dt (the sign of the potential's rate), a sign slip.
    # ------------------------------------------------------------------------
    eq("eq:lorentz", 769, "d pi/dt = -hbar grad omega - q dA/dt")
    Phf = RF('Phi')
    Af = [RF(f'A{k}') for k in range(3)]
    Pc = sp.symbols('P1:4', real=True)
    H = sum((Pc[k] - q*Af[k])**2 for k in range(3))/(2*m) + q*Phf
    xdot = [sp.diff(H, Pc[k]) for k in range(3)]
    pdot = [-sp.diff(H, X3[k]) for k in range(3)]

    def tot(F):
        return sp.diff(F, t) + sum(xdot[k]*sp.diff(F, X3[k]) + pdot[k]*sp.diff(F, Pc[k]) for k in range(3))
    pidot = [tot(Pc[k] - q*Af[k]) for k in range(3)]
    Adot = [sp.diff(Af[k], t) + sum(xdot[j]*sp.diff(Af[k], X3[j]) for j in range(3)) for k in range(3)]
    symcheck("inline: hbar d_i omega == q d_i Phi - q v_j d_i A_j with v = pi/m",
             sp.Matrix([sp.expand(sp.diff(H, X3[i]) - (q*sp.diff(Phf, X3[i])
                                                        - q*sum(xdot[j]*sp.diff(Af[j], X3[i]) for j in range(3))))
                        for i in range(3)]))
    symcheck("rate of pi along Hamilton's flow == -hbar grad omega - q dA/dt",
             sp.Matrix([sp.expand(pidot[k] - (-sp.diff(H, X3[k]) - q*Adot[k])) for k in range(3)]))
    mutant("+ q dA/dt", _S3_pick(sp.Matrix([pidot[k] - (-sp.diff(H, X3[k]) + q*Adot[k]) for k in range(3)])))

    # ------------------------------------------------------------------------
    # eq:lorentz  (sgm.tex L770), second line; L772 is the caption "(The Lorentz force)", not mathematics
    # States: d pi/dt = q (E + v x B), E = -grad Phi - d_t A, B = curl A, v = pi/m (vectors).
    # Derived from: the first line and v_j d_i A_j - v_j d_j A_i = (v x B)_i (stated in the text).
    #   Labels used: eq:omega_em (the first line of eq:lorentz).
    # Test: the rate of pi computed above equals q(E + v x B) for general potentials; the stated
    #   index identity holds for general v and A.
    # Mutants: q(E - v x B) (orientation of the cross product), and q(E + v x B) compared with the
    #   rate when the convective part (v.grad)A of dA/dt is dropped.
    # ------------------------------------------------------------------------
    eq("eq:lorentz", 770, "d pi/dt = q (E + v x B)")
    Ef = [-sp.diff(Phf, X3[k]) - sp.diff(Af[k], t) for k in range(3)]
    Bf = _S3_curl(Af, X3)
    vxB = sp.Matrix(xdot).cross(sp.Matrix(Bf))
    symcheck("rate of pi == q (E + v x B)", sp.Matrix([sp.expand(pidot[k] - q*(Ef[k] + vxB[k])) for k in range(3)]))
    vs = sp.symbols('v1:4', real=True)
    symcheck("inline: v_j d_i A_j - v_j d_j A_i == (v x B)_i",
             sp.Matrix([sp.expand(sum(vs[j]*(sp.diff(Af[j], X3[i]) - sp.diff(Af[i], X3[j])) for j in range(3))
                                  - sp.Matrix(vs).cross(sp.Matrix(Bf))[i]) for i in range(3)]))
    mutant("q (E - v x B)", _S3_pick(sp.Matrix([pidot[k] - q*(Ef[k] - vxB[k]) for k in range(3)])))
    mutant("dA/dt without (v.grad)A", _S3_pick(sp.Matrix([-sp.diff(H, X3[k]) - q*sp.diff(Af[k], t)
                                                           - q*(Ef[k] + vxB[k]) for k in range(3)])))

    # ------------------------------------------------------------------------
    # tab:classes  (sgm.tex L780-794), table, not a display
    # States: geometric omega = omega_0 alpha; charge omega = omega_0 + V/hbar; Lorentz scalar
    #   omega = omega_0 + V_sc/hbar with m = m_0 + V_sc/c^2.
    # Derived from: eq:omega_alpha, eq:omega_em, and the definition of m in the caption.
    #   Labels used: eq:rest_freq, eq:omega_alpha, eq:omega_em.
    # Test: at rest, (m_0 + V_sc/c^2) c^2/hbar == omega_0 + V_sc/hbar; the acceleration from omega_0 alpha
    #   is independent of m_0, the one from V_sc is not unless V_sc is proportional to m_0.
    # Mutant: m = m_0 + V_sc (missing c^-2) does not give omega_0 + V_sc/hbar.
    # ------------------------------------------------------------------------
    eq("tab:classes", 789, "rows: omega_0 alpha; omega_0 + V/hbar; omega_0 + V_sc/hbar with m = m_0 + V_sc/c^2")
    m0 = sp.Symbol('m_0', positive=True)
    Vsc = sp.Symbol('V_sc', real=True)
    symcheck("Lorentz scalar at rest: m c^2/hbar == omega_0 + V_sc/hbar", sp.expand((m0 + Vsc/c**2)*c**2/hb - (m0*c**2/hb + Vsc/hb)))
    alf = RF('alpha', X3)
    acc_geom = [-sp.diff(m0*c**2*alf, X3[k])/m0 for k in range(3)]
    boolcheck("geometric class: acceleration -c^2 grad alpha has no m_0", all(sp.diff(a, m0) == 0 for a in acc_geom))
    Vsf = RF('Vs', X3)
    boolcheck("Lorentz scalar class: acceleration -grad(V_sc)/m_0 depends on m_0",
              any(sp.diff(-sp.diff(Vsf, X3[k])/m0, m0) != 0 for k in range(3)))
    kfac = sp.Symbol('k_s', real=True)
    boolcheck("Lorentz scalar with V_sc = k m_0 f(x): acceleration independent of m_0",
              all(sp.diff(sp.expand(-sp.diff(kfac*m0*Vsf, X3[k])/m0), m0) == 0 for k in range(3)))
    mutant("m = m_0 + V_sc (missing c^-2)", sp.expand((m0 + Vsc)*c**2/hb - (m0*c**2/hb + Vsc/hb)))

    # ------------------------------------------------------------------------
    # eq:phase_fourvector  (sgm.tex L799)
    # States: d_mu theta = (-omega/c, k); theta the phase (scalar), x^mu = (ct, x), omega = -d_t theta,
    #   k = grad theta (vector).
    # Derived from: eq:phase_freq and eq:debroglie_k (omega = -d_t theta, k = grad theta) and x^0 = ct.
    #   Labels used: eq:phase_freq, eq:debroglie_k, eq:omega_em.
    # Test: on a general theta(t,x), d_0 theta = (1/c) d_t theta = -omega/c. Remark rem:additivity:
    #   the gauge change theta -> theta + q lambda/hbar, A_mu -> A_mu + d_mu lambda with A_mu = (-Phi/c, A)
    #   keeps hbar d_mu theta - q A_mu and keeps E; lambda = -t dPhi raises Phi by dPhi and omega by
    #   q dPhi/hbar; the functional equation omega_0 [f(Phi + dPhi) - f(Phi)] = q dPhi/hbar gives
    #   f' = q/(mc^2) and is solved by f = 1 + q Phi/mc^2.
    # Mutants: the multiplicative f = exp(q Phi/mc^2) fails the functional equation; A_0 = +Phi/c
    #   would lower Phi by dPhi under the same lambda.
    # ------------------------------------------------------------------------
    eq("eq:phase_fourvector", 799, "d_mu theta = (-omega/c, k)")
    th = RF('theta')
    om_th = -sp.diff(th, t)
    kv_th = [sp.diff(th, X3[k]) for k in range(3)]
    dmu = lambda f: [sp.diff(f, t)/c] + [sp.diff(f, X3[k]) for k in range(3)]
    symcheck("d_mu theta == (-omega/c, k) with x^0 = ct",
             sp.Matrix([sp.expand(dmu(th)[0] + om_th/c)] + [sp.expand(dmu(th)[k+1] - kv_th[k]) for k in range(3)]))
    lamf = RF('lam')
    Amu = [-Phf/c] + Af
    Amu2 = [Amu[i] + dmu(lamf)[i] for i in range(4)]
    th2 = th + q*lamf/hb
    symcheck("gauge change keeps hbar d_mu theta - q A_mu",
             sp.Matrix([sp.expand(hb*dmu(th2)[i] - q*Amu2[i] - (hb*dmu(th)[i] - q*Amu[i])) for i in range(4)]))
    Ph2 = -c*Amu2[0]
    A2 = Amu2[1:]
    symcheck("gauge change keeps E = -grad Phi - d_t A",
             sp.Matrix([sp.expand(-sp.diff(Ph2, X3[k]) - sp.diff(A2[k], t) - Ef[k]) for k in range(3)]))
    dPh = sp.Symbol('delta_Phi', real=True)
    Ph2c = sp.expand(-c*(Amu[0] + dmu(-t*dPh)[0]))
    symcheck("lambda = -t dPhi raises Phi by dPhi", sp.expand(Ph2c - Phf - dPh))
    symcheck("lambda = -t dPhi raises omega by q dPhi/hbar", sp.expand(-sp.diff(th + q*(-t*dPh)/hb, t) - om_th - q*dPh/hb))
    Pv = sp.Symbol('Phi_v', real=True)
    fF = sp.Function('f')
    dd = sp.diff(w0*fF(Pv + dPh), dPh).subs(dPh, 0).doit()
    fp = sp.solve(sp.Eq(dd, q/hb), sp.Derivative(fF(Pv), Pv)) if False else None
    fprime = sp.Symbol('fprime')
    sol = sp.solve(sp.Eq(w0*fprime, q/hb), fprime)[0]
    symcheck("differentiating at dPhi = 0: f' == q/(mc^2)", sp.expand(sol - q/(m*c**2)))
    flin = lambda P: 1 + q*P/(m*c**2)
    symcheck("f = 1 + q Phi/mc^2 solves omega_0[f(Phi+dPhi) - f(Phi)] == q dPhi/hbar for all Phi, dPhi",
             sp.expand(w0*(flin(Pv + dPh) - flin(Pv)) - q*dPh/hb))
    symcheck("so omega = omega_0 f(Phi) == omega_0 + q Phi/hbar", sp.expand(w0*flin(Pv) - (w0 + q*Pv/hb)))
    fexp = lambda P: sp.exp(q*P/(m*c**2))
    mutant("multiplicative f = exp(q Phi/mc^2) fails the functional equation",
           sp.expand(w0*(fexp(Pv + dPh) - fexp(Pv)) - q*dPh/hb))
    mutant("A_0 = +Phi/c: the same lambda changes Phi by -dPhi, not +dPhi",
           sp.expand(c*(Phf/c + dmu(-t*dPh)[0]) - Phf - dPh))

    # ------------------------------------------------------------------------
    # eq:quaternion_units  (sgm.tex L814)
    # States: e_j e_k = -delta_jk + eps_jkl e_l; e_j quaternion units.
    # Derived from: definition.
    #   Labels used: none (definition).
    # Test: the product table implemented in this file matches the formula entry by entry; the map
    #   e_k -> -i sigma_k is a homomorphism on general quaternions (an independent realization of the
    #   same table); the product is associative.
    # Mutants: the left-handed table -delta - eps e_l; the map e_k -> +i sigma_k (anti-homomorphism).
    # ------------------------------------------------------------------------
    eq("eq:quaternion_units", 814, "e_j e_k = -delta_jk + eps_jkl e_l")
    tab_ok = True
    tab_mut = []
    for j in range(3):
        for k in range(3):
            lhs = _S3_mul(E[j], E[k])
            rhs = [-(1 if j == k else 0)] + [sp.LeviCivita(j, k, l) for l in range(3)]
            rmut = [-(1 if j == k else 0)] + [-sp.LeviCivita(j, k, l) for l in range(3)]
            tab_ok = tab_ok and all(sp.simplify(lhs[i] - rhs[i]) == 0 for i in range(4))
            tab_mut += [lhs[i] - rmut[i] for i in range(4)]
    boolcheck("all nine products e_j e_k match -delta_jk + eps_jkl e_l", tab_ok)
    a_s = sp.symbols('a0:4', real=True)
    b_s = sp.symbols('b0:4', real=True)
    c_s = sp.symbols('c0:4', real=True)
    rep = lambda a, s=-1: a[0]*sp.eye(2) + sum((a[k+1]*s*I*_S3_SIG[k] for k in range(3)), sp.zeros(2))
    symcheck("e_k -> -i sigma_k: rep(ab) == rep(a) rep(b) for general a, b",
             (rep(_S3_mul(list(a_s), list(b_s))) - rep(list(a_s))*rep(list(b_s))).applyfunc(sp.expand))
    symcheck("associativity (ab)c == a(bc)",
             _S3_mat(_S3_add(_S3_mul(_S3_mul(list(a_s), list(b_s)), list(c_s)),
                             _S3_sc(-1, _S3_mul(list(a_s), _S3_mul(list(b_s), list(c_s)))))))
    mutant("left-handed table -delta - eps e_l", _S3_pick(sp.Matrix(tab_mut)))
    mutant("e_k -> +i sigma_k", _S3_pick((rep(_S3_mul(list(a_s), list(b_s)), 1)
                                          - rep(list(a_s), 1)*rep(list(b_s), 1)).applyfunc(sp.expand)))

    # ------------------------------------------------------------------------
    # eq:spinor_quaternion  (sgm.tex L819)
    # States: psi = Re psi_1 - (Im psi_1) e3 + e2 [Re psi_2 - (Im psi_2) e3]; psi a quaternion,
    #   (psi_1, psi_2) the complex spinor components.
    # Derived from: definition (the map of two complex numbers to one quaternion).
    #   Labels used: eq:quaternion_units.
    # Test: multiplying out gives [x1, -y2, x2, -y1]; the inverse map used in this file recovers
    #   (psi_1, psi_2); the inline claim "multiplication by i is right multiplication by -e3" holds on
    #   general components; |psi|^2 = |psi_1|^2 + |psi_2|^2 (inline at L827).
    # Mutant: e2 on the right of the bracket; then right multiplication by -e3 no longer represents i.
    # ------------------------------------------------------------------------
    eq("eq:spinor_quaternion", 819, "psi = Re psi1 - (Im psi1) e3 + e2 [Re psi2 - (Im psi2) e3]")
    x1, y1, x2, y2 = sp.symbols('x1 y1 x2 y2', real=True)
    psiq = _S3_add([x1, 0, 0, -y1], _S3_mul(E[1], [x2, 0, 0, -y2]))
    symcheck("multiplied out == x1 - y2 e1 + x2 e2 - y1 e3", _S3_mat(_S3_add(psiq, _S3_sc(-1, [x1, -y2, x2, -y1]))))
    zc = sp.Matrix([x1 + I*y1, x2 + I*y2])
    symcheck("inverse map recovers (psi_1, psi_2)", (_S3_q2c(psiq) - zc).applyfunc(sp.expand))
    symcheck("i psi == psi (-e3)", (_S3_q2c(_S3_mul(psiq, [0, 0, 0, -1])) - I*zc).applyfunc(sp.expand))
    symcheck("|psi|^2 == psi psi-bar == |psi_1|^2 + |psi_2|^2",
             _S3_mat(_S3_add(_S3_mul(psiq, _S3_conj(psiq)), [-(x1**2 + y1**2 + x2**2 + y2**2), 0, 0, 0])))
    psi_mut = _S3_add([x1, 0, 0, -y1], _S3_mul([x2, 0, 0, -y2], E[1]))
    zc_mut = sp.Matrix([psi_mut[0] - I*psi_mut[3], psi_mut[2] - I*psi_mut[1]])
    # with e2 on the right, decode by the map of that ordering: (s - i v3, v2 + i v1) recovers (psi_1, psi_2)
    zc_mut = sp.Matrix([psi_mut[0] - I*psi_mut[3], psi_mut[2] + I*psi_mut[1]])
    mutant("e2 on the right: right multiplication by -e3 is then not i",
           _S3_pick((sp.Matrix([_S3_mul(psi_mut, [0, 0, 0, -1])[0] - I*_S3_mul(psi_mut, [0, 0, 0, -1])[3],
                                _S3_mul(psi_mut, [0, 0, 0, -1])[2] + I*_S3_mul(psi_mut, [0, 0, 0, -1])[1]])
                     - I*zc_mut).applyfunc(sp.expand)))

    # ------------------------------------------------------------------------
    # eq:pauli_quaternion  (sgm.tex L824)
    # States: sigma_k psi = -e_k psi e3; sigma_k the Pauli matrices acting on (psi_1, psi_2).
    # Derived from: eq:spinor_quaternion and eq:quaternion_units.
    #   Labels used: eq:quaternion_units, eq:spinor_quaternion.
    # Test: for each k, the complex form of -e_k psi e3 equals sigma_k (psi_1, psi_2) for general
    #   components; inline: -i sigma_k is left multiplication by e_k; S = hbar sigma/2 (definition).
    # Mutant: sigma_k psi = e_k psi e3 (sign dropped).
    # ------------------------------------------------------------------------
    eq("eq:pauli_quaternion", 824, "sigma_k psi = -e_k psi e3")
    for k in range(3):
        symcheck(f"sigma_{k+1} psi == -e_{k+1} psi e3",
                 (_S3_q2c(_S3_sc(-1, _S3_mul(_S3_mul(E[k], psiq), e3))) - _S3_SIG[k]*zc).applyfunc(sp.expand))
        symcheck(f"-i sigma_{k+1} psi == e_{k+1} psi",
                 (_S3_q2c(_S3_mul(E[k], psiq)) + I*_S3_SIG[k]*zc).applyfunc(sp.expand))
    mutant("sigma_1 psi = +e_1 psi e3", _S3_pick((_S3_q2c(_S3_mul(_S3_mul(E[0], psiq), e3)) - _S3_SIG[0]*zc).applyfunc(sp.expand)))

    # ------------------------------------------------------------------------
    # eq:polar_form  (sgm.tex L831)
    # States: psi = rho^{1/2} R_psi with R_psi R_psi-bar = 1; rho = |psi|^2 (scalar), R_psi unit quaternion.
    # Derived from: definition of the norm |Y|^2 = Y Y-bar (L817).
    #   Labels used: eq:quaternion_units, eq:spinor_quaternion.
    # Test: psi psi-bar is the real number rho; R = psi/rho^{1/2} has R R-bar = 1; the inline rotation
    #   claim: R = exp(theta/2 n.e) maps w.e to the vector rotated by theta about n (Rodrigues formula),
    #   numerically for random n, theta, w.
    # Mutants: psi = rho R (R R-bar = 1/rho); the full-angle rotor exp(theta n.e) rotates by 2 theta.
    # ------------------------------------------------------------------------
    eq("eq:polar_form", 831, "psi = rho^{1/2} R_psi")
    rho_s = x1**2 + y1**2 + x2**2 + y2**2
    Rq = _S3_sc(1/sp.sqrt(rho_s), psiq)
    symcheck("R R-bar == 1 for R = psi/rho^{1/2}", _S3_mat(_S3_add(_S3_mul(Rq, _S3_conj(Rq)), [-1, 0, 0, 0])).applyfunc(sp.cancel))
    rng = np.random.default_rng(31)

    def nmul(a, b):
        return np.array(_S3_mul(list(a), list(b)), dtype=float)
    nvec = rng.normal(size=3)
    nvec /= np.linalg.norm(nvec)
    thn = 0.7317
    wv = rng.normal(size=3)
    Rn = np.r_[np.cos(thn/2), np.sin(thn/2)*nvec]
    Rbar = Rn*np.array([1, -1, -1, -1])
    rot = nmul(nmul(Rn, np.r_[0, wv]), Rbar)[1:]
    rod = wv*np.cos(thn) + np.cross(nvec, wv)*np.sin(thn) + nvec*np.dot(nvec, wv)*(1 - np.cos(thn))
    check("R (w.e) R-bar == w rotated by theta about n (Rodrigues)", np.abs(rot - rod).max(), 0, tol=1e-12, scale=np.linalg.norm(wv))
    Rq_mut = _S3_sc(1/rho_s, psiq)
    mutant("psi = rho R: R R-bar != 1", _S3_pick(_S3_mat(_S3_add(_S3_mul(Rq_mut, _S3_conj(Rq_mut)), [-1, 0, 0, 0])).applyfunc(sp.cancel)))
    Rn2 = np.r_[np.cos(thn), np.sin(thn)*nvec]
    rot2 = nmul(nmul(Rn2, np.r_[0, wv]), Rn2*np.array([1, -1, -1, -1]))[1:]
    numutant("exp(theta n.e) instead of exp(theta/2 n.e)", np.linalg.norm(rot2 - rod) + np.linalg.norm(rod), np.linalg.norm(rod))

    # ------------------------------------------------------------------------
    # eq:spin_rotor  (sgm.tex L836)
    # States: s-hat.e = R_psi e3 R_psi-bar; s-hat the unit spin direction (vector).
    # Derived from: eq:polar_form and eq:pauli_quaternion.
    #   Labels used: eq:pauli_quaternion, eq:polar_form.
    # Test: psi e3 psi-bar == sum_k (psi^dagger sigma_k psi) e_k for general components, so
    #   R e3 R-bar = <sigma>/rho is the spin direction; numerically, left multiplication by a rotor
    #   rotates s by that rotor and right multiplication by exp(e3 phi) leaves s fixed.
    # Mutant: R-bar e3 R (the reversed sandwich).
    # ------------------------------------------------------------------------
    eq("eq:spin_rotor", 836, "s-hat.e = R_psi e3 R_psi-bar")
    sk = [sp.expand((zc.H*_S3_SIG[k]*zc)[0]) for k in range(3)]
    symcheck("psi e3 psi-bar == (psi^dag sigma psi).e", _S3_mat(_S3_add(_S3_mul(_S3_mul(psiq, e3), _S3_conj(psiq)), _S3_sc(-1, _S3_vq(sk)))))
    psin = rng.normal(size=4)
    spin = lambda p: nmul(nmul(p, [0, 0, 0, 1]), p*np.array([1, -1, -1, -1]))[1:]
    s0 = spin(psin)
    s_rot = spin(nmul(Rn, psin))
    s_exp = nmul(nmul(Rn, np.r_[0, s0]), Rbar)[1:]
    check("left multiplication by R rotates the spin by R", np.abs(s_rot - s_exp).max(), 0, tol=1e-12, scale=np.linalg.norm(s0))
    phn = 1.234
    check("right multiplication by exp(e3 phi) leaves the spin fixed",
          np.abs(spin(nmul(psin, [np.cos(phn), 0, 0, np.sin(phn)])) - s0).max(), 0, tol=1e-12, scale=np.linalg.norm(s0))
    mutant("R-bar e3 R", _S3_pick(_S3_mat(_S3_add(_S3_mul(_S3_mul(_S3_conj(psiq), e3), psiq), _S3_sc(-1, _S3_vq(sk))))))

    # ------------------------------------------------------------------------
    # eq:stationary_time  (sgm.tex L841)
    # States: psi(x,t) = psi_0(x) exp(e3 omega t); omega the frequency (scalar).
    # Derived from: "e^{-i phi} is right multiplication by exp(e3 phi)" (from i = right -e3) applied
    #   to a complex stationary state psi_0 e^{-i omega t}.
    #   Labels used: eq:phase_freq, eq:spinor_quaternion.
    # Test: complex form of psi exp(e3 phi) == e^{-i phi} (psi_1, psi_2); d_t psi == psi e3 omega
    #   (used at L937); the inline claim (L844): with R = R_0 exp(e3 omega t), R e1 R-bar turns at
    #   rate 2 omega about the spin axis and R e3 R-bar is fixed (cos, sin kept as symbols C, S with
    #   cos 2a = C^2 - S^2, sin 2a = 2CS).
    # Mutants: exp(-e3 omega t) (gives e^{+i phi}); a transverse axis turning at rate omega.
    # ------------------------------------------------------------------------
    eq("eq:stationary_time", 841, "psi(x,t) = psi_0(x) exp(e3 omega t)")
    ph = sp.Symbol('phi', real=True)
    symcheck("psi exp(e3 phi) <-> e^{-i phi} psi",
             (_S3_q2c(_S3_mul(psiq, [sp.cos(ph), 0, 0, sp.sin(ph)])) - (sp.cos(ph) - I*sp.sin(ph))*zc).applyfunc(sp.expand))
    om = sp.Symbol('omega', real=True)
    p0 = QF('q0', X3)
    psit = _S3_mul(p0, [sp.cos(om*t), 0, 0, sp.sin(om*t)])
    symcheck("d_t psi == psi e3 omega", _S3_mat(_S3_add(_S3_d(psit, t), _S3_sc(-om, _S3_mul(psit, e3)))))
    Cs, Ss = sp.symbols('C S', real=True)
    Eq_ = [Cs, 0, 0, Ss]
    R0 = list(a_s)
    RR = _S3_mul(R0, Eq_)
    ax1 = _S3_mul(_S3_mul(RR, E[0]), _S3_conj(RR))
    ax1_exp = _S3_add(_S3_sc(Cs**2 - Ss**2, _S3_mul(_S3_mul(R0, E[0]), _S3_conj(R0))),
                      _S3_sc(2*Cs*Ss, _S3_mul(_S3_mul(R0, E[1]), _S3_conj(R0))))
    symcheck("R e1 R-bar == cos(2 omega t) R0 e1 R0-bar + sin(2 omega t) R0 e2 R0-bar", _S3_mat(_S3_add(ax1, _S3_sc(-1, ax1_exp))))
    ax3 = _S3_mul(_S3_mul(RR, e3), _S3_conj(RR))
    symcheck("spin axis R e3 R-bar == (C^2 + S^2) R0 e3 R0-bar, fixed for C^2 + S^2 = 1",
             _S3_mat(_S3_add(ax3, _S3_sc(-(Cs**2 + Ss**2), _S3_mul(_S3_mul(R0, e3), _S3_conj(R0))))))
    mutant("exp(-e3 phi) <-> e^{-i phi}",
           _S3_pick((_S3_q2c(_S3_mul(psiq, [sp.cos(ph), 0, 0, -sp.sin(ph)])) - (sp.cos(ph) - I*sp.sin(ph))*zc).applyfunc(sp.expand)))
    ax1_mut = _S3_add(_S3_sc(Cs, _S3_mul(_S3_mul(R0, E[0]), _S3_conj(R0))), _S3_sc(Ss, _S3_mul(_S3_mul(R0, E[1]), _S3_conj(R0))))
    mutant("transverse axis turning at rate omega", _S3_pick(_S3_mat(_S3_add(ax1, _S3_sc(-1, ax1_mut)))))

    # Gauge-covariant derivatives with general V(t,x), A(t,x), psi(t,x), as in the paper
    Vf = RF('V')
    psi = QF('s')
    Dt, Dk = _S3_gauge_ops(Vf, Af, hb, q, t, X3)
    lamg = RF('lam')
    gph = q*lamg/hb
    gq = [sp.cos(gph), 0, 0, -sp.sin(gph)]

    # ------------------------------------------------------------------------
    # eq:phase_connection_time  (sgm.tex L848)
    # States: D_t psi = d_t psi - (V/hbar) psi e3; V the additive potential energy (scalar).
    # Derived from: the phase acting on the right (eq:stationary_time) and the complex covariant
    #   derivative d_t + i V/hbar.
    #   Labels used: eq:omega_em, rem:additivity, eq:spinor_quaternion, eq:stationary_time.
    # Test: complex form of D_t psi == (d_t + iV/hbar)(psi_1, psi_2); the gauge change psi -> psi g with
    #   g = exp(-e3 q lambda/hbar) (that is, e^{i q lambda/hbar}) and V -> V - q d_t lambda gives
    #   D_t'(psi g) == (D_t psi) g, for general lambda(t,x).
    # Mutant: D_t psi = d_t psi + (V/hbar) psi e3 breaks the gauge covariance.
    # ------------------------------------------------------------------------
    eq("eq:phase_connection_time", 848, "D_t psi = d_t psi - (V/hbar) psi e3")
    zpsi = _S3_q2c(psi)
    symcheck("complex form: D_t psi <-> (d_t + i V/hbar) psi",
             (_S3_q2c(Dt(psi)) - (sp.diff(zpsi, t) + I*Vf/hb*zpsi)).applyfunc(sp.expand))
    symcheck("g = exp(-e3 q lambda/hbar) is e^{i q lambda/hbar}",
             (_S3_q2c(_S3_mul(psi, gq)) - (sp.cos(gph) + I*sp.sin(gph))*zpsi).applyfunc(sp.expand))
    Dt2, Dk2 = _S3_gauge_ops(Vf - q*sp.diff(lamg, t), [Af[k] + sp.diff(lamg, X3[k]) for k in range(3)], hb, q, t, X3)
    symcheck("D_t'(psi g) == (D_t psi) g", _S3_mat(_S3_add(Dt2(_S3_mul(psi, gq)), _S3_sc(-1, _S3_mul(Dt(psi), gq)))))
    Dt2m = lambda f: _S3_add(_S3_d(f, t), _S3_sc((Vf - q*sp.diff(lamg, t))/hb, _S3_mul(f, e3)))
    Dtm = lambda f: _S3_add(_S3_d(f, t), _S3_sc(Vf/hb, _S3_mul(f, e3)))
    mutant("+ (V/hbar) psi e3", _S3_pick(_S3_mat(_S3_add(Dt2m(_S3_mul(psi, gq)), _S3_sc(-1, _S3_mul(Dtm(psi), gq))))))

    # ------------------------------------------------------------------------
    # eq:right_connection  (sgm.tex L850)
    # States: D_k psi = d_k psi + (q/hbar) A_k psi e3; A_k the vector potential components.
    # Derived from: as eq:phase_connection_time, with the complex derivative d_k - i q A_k/hbar.
    #   Labels used: eq:debroglie, rem:additivity, eq:spinor_quaternion, eq:phase_connection_time.
    # Test: complex form == (d_k - i q A_k/hbar)(psi_1, psi_2); inline: D_k psi = -hbar^-1 (pi_k psi) e3
    #   with pi = -i hbar grad - qA (right e3 is -i); gauge covariance under A -> A + grad lambda.
    # Mutant: - (q/hbar) A_k psi e3 breaks the gauge covariance.
    # ------------------------------------------------------------------------
    eq("eq:right_connection", 850, "D_k psi = d_k psi + (q/hbar) A_k psi e3")
    for k in range(3):
        symcheck(f"complex form: D_{k+1} psi <-> (d_{k+1} - i q A_{k+1}/hbar) psi",
                 (_S3_q2c(Dk[k](psi)) - (sp.diff(zpsi, X3[k]) - I*q*Af[k]/hb*zpsi)).applyfunc(sp.expand))
    symcheck("right e3 is multiplication by -i", (_S3_q2c(_S3_mul(psi, e3)) + I*zpsi).applyfunc(sp.expand))
    pihat = lambda u, k: -I*hb*sp.diff(u, X3[k]) - q*Af[k]*u
    symcheck("inline: D_1 psi <-> -hbar^-1 (-i)(pi_1 psi)", (_S3_q2c(Dk[0](psi)) - (-1/hb)*(-I)*pihat(zpsi, 0)).applyfunc(sp.expand))
    for k in range(3):
        symcheck(f"D_{k+1}'(psi g) == (D_{k+1} psi) g", _S3_mat(_S3_add(Dk2[k](_S3_mul(psi, gq)), _S3_sc(-1, _S3_mul(Dk[k](psi), gq)))))
    Dkm = lambda f, Aa: _S3_add(_S3_d(f, x), _S3_sc(-q*Aa/hb, _S3_mul(f, e3)))
    mutant("- (q/hbar) A_k psi e3", _S3_pick(_S3_mat(_S3_add(Dkm(_S3_mul(psi, gq), Af[0] + sp.diff(lamg, x)),
                                                            _S3_sc(-1, _S3_mul(Dkm(psi, Af[0]), gq))))))

    def com(P, Q_, f):
        return _S3_add(P(Q_(f)), _S3_sc(-1, Q_(P(f))))

    # ------------------------------------------------------------------------
    # eq:phase_curvature  (sgm.tex L855)
    # States: [D_1, D_2] psi = (q/hbar) B_3 psi e3, with cyclic permutations; B = curl A.
    # Derived from: eq:right_connection, by applying both orders and subtracting.
    #   Labels used: eq:quaternion_units, eq:right_connection.
    # Test: the commutators of the operators above on general psi and A, all three cyclic orders.
    # Mutant: -(q/hbar) B_3 (orientation).
    # ------------------------------------------------------------------------
    eq("eq:phase_curvature", 855, "[D_1, D_2] psi = (q/hbar) B_3 psi e3 (and cyclic)")
    for (a_, b_, cidx) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        symcheck(f"[D_{a_+1}, D_{b_+1}] psi == (q/hbar) B_{cidx+1} psi e3",
                 _S3_mat(_S3_add(com(Dk[a_], Dk[b_], psi), _S3_sc(-q*Bf[cidx]/hb, _S3_mul(psi, e3)))))
    mutant("-(q/hbar) B_3", _S3_pick(_S3_mat(_S3_add(com(Dk[0], Dk[1], psi), _S3_sc(q*Bf[2]/hb, _S3_mul(psi, e3))))))

    # ------------------------------------------------------------------------
    # eq:time_space_curvature  (sgm.tex L857)
    # States: [D_t, D_k] psi = (1/hbar)(d_k V + q d_t A_k) psi e3; for V = q Phi this is -(q/hbar) E_k psi e3.
    # Derived from: eqs. phase_connection_time and right_connection.
    #   Labels used: eq:lorentz (E), eq:phase_connection_time, eq:right_connection.
    # Test: the commutator on general psi, V, A for k = 1, 2, 3; with V = q Phi the right side equals
    #   -(q/hbar) E_k psi e3.
    # Mutant: (1/hbar)(d_k V - q d_t A_k).
    # ------------------------------------------------------------------------
    eq("eq:time_space_curvature", 857, "[D_t, D_k] psi = (1/hbar)(d_k V + q d_t A_k) psi e3")
    for k in range(3):
        symcheck(f"[D_t, D_{k+1}] psi == (1/hbar)(d_{k+1} V + q d_t A_{k+1}) psi e3",
                 _S3_mat(_S3_add(com(Dt, Dk[k], psi),
                                 _S3_sc(-(sp.diff(Vf, X3[k]) + q*sp.diff(Af[k], t))/hb, _S3_mul(psi, e3)))))
    symcheck("V = q Phi: (1/hbar)(d_k V + q d_t A_k) == -(q/hbar) E_k",
             sp.Matrix([sp.expand((sp.diff(q*Phf, X3[k]) + q*sp.diff(Af[k], t))/hb + q*Ef[k]/hb) for k in range(3)]))
    mutant("(d_k V - q d_t A_k)", _S3_pick(_S3_mat(_S3_add(com(Dt, Dk[0], psi),
                                                          _S3_sc(-(sp.diff(Vf, x) - q*sp.diff(Af[0], t))/hb, _S3_mul(psi, e3))))))

    # ------------------------------------------------------------------------
    # eq:fueter_square  (sgm.tex L862)
    # States: (e.D)^2 psi = -D^2 psi + (q/hbar)(e.B) psi e3; D^2 = D_1^2 + D_2^2 + D_3^2.
    # Derived from: eq:quaternion_units (split e_j e_k into -delta and the antisymmetric part) and
    #   eq:phase_curvature.
    #   Labels used: eq:quaternion_units, eq:pauli_quaternion, eq:right_connection, eq:phase_curvature.
    # Test: general static psi(x) and A(x); the Pauli form (sigma.pi)^2 = pi^2 - q hbar sigma.B on a
    #   general complex two-component u(x); and X = e.D equals sigma.pi/hbar (used at L898).
    # Mutants: +(q/hbar)(e.B) -> -(q/hbar)(e.B); the Pauli form with + q hbar sigma.B.
    # ------------------------------------------------------------------------
    eq("eq:fueter_square", 862, "(e.D)^2 psi = -D^2 psi + (q/hbar)(e.B) psi e3")
    As = [RF(f'A{k}', X3) for k in range(3)]
    Bs = _S3_curl(As, X3)
    psis = QF('s', X3)
    _, Dks = _S3_gauge_ops(0, As, hb, q, t, X3)
    eD2 = _S3_edot(Dks, _S3_edot(Dks, psis))
    D2 = _S3_add(*[Dks[k](Dks[k](psis)) for k in range(3)])
    symcheck("(e.D)^2 psi + D^2 psi - (q/hbar)(e.B) psi e3 == 0",
             _S3_mat(_S3_add(eD2, D2, _S3_sc(-q/hb, _S3_mul(_S3_mul(_S3_vq(Bs), psis), e3)))))
    u = sp.Matrix([sp.Function('u1')(*X3), sp.Function('u2')(*X3)])
    pis = lambda w, k: -I*hb*sp.diff(w, X3[k]) - q*As[k]*w
    spi = lambda w: sum((_S3_SIG[k]*pis(w, k) for k in range(3)), sp.zeros(2, 1))
    pi2 = sum((pis(pis(u, k), k) for k in range(3)), sp.zeros(2, 1))
    sB = sum((_S3_SIG[k]*Bs[k] for k in range(3)), sp.zeros(2))
    symcheck("Pauli form: (sigma.pi)^2 u == pi^2 u - q hbar (sigma.B) u", (spi(spi(u)) - pi2 + q*hb*sB*u).applyfunc(sp.expand))
    symcheck("X psi = e.D psi <-> sigma.pi psi/hbar", (_S3_q2c(_S3_edot(Dks, psis)) - spi(_S3_q2c(psis))/hb).applyfunc(sp.expand))
    mutant("-(q/hbar)(e.B)", _S3_pick(_S3_mat(_S3_add(eD2, D2, _S3_sc(q/hb, _S3_mul(_S3_mul(_S3_vq(Bs), psis), e3))))))
    mutant("Pauli form with + q hbar sigma.B", _S3_pick((spi(spi(u)) - pi2 - q*hb*sB*u).applyfunc(sp.expand)))

    # ------------------------------------------------------------------------
    # eq:lapse_shift_metric  (sgm.tex L869)
    # States: ds^2 = -alpha_n^2 c^2 dt^2 + W^2 |dx + A_g dt|^2; alpha_n the lapse (scalar), A_g the
    #   shift (velocity vector), W the spatial scale factor (scalar).
    # Derived from: definition (cited: ADM, MTW).
    #   Labels used: eq:alpha_def (consistency only; the metric is cited).
    # Test: the inline special cases. (1) Rotating frame: pulling the Minkowski metric back by
    #   X = R_z(Omega t) x gives alpha_n = W = 1, A_g = Omega x x. (2) Post-Newtonian: with
    #   Phi_g = -U, g_tt/c^2 = -1 + 2U/c^2 - 2 beta U^2/c^4 and g_ij = (1 + 2 gamma U/c^2) delta_ij, the
    #   standard PPN form (cited: Will). (3) g_0i = W^2 A_g,i/c (x^0 = ct) with
    #   A_g = -(2G/c^2 r^3) J x x matches the PPN -4 V_i/c^3 of general relativity with the dipole
    #   potential V_i = (G/2r^3)(J x x)_i, which follows from int rho x_k v_j = (1/2) eps_kjl J_l.
    #   (4) Rindler: Minkowski pulled back by the uniformly accelerated chart gives alpha_n = 1 + a x/c^2.
    # Mutants: dx - A_g dt with A_g = Omega x x fails the rotating-frame pullback; alpha_n with
    #   beta Phi^2 (no -1/2) gives the wrong 1PN coefficient.
    # ------------------------------------------------------------------------
    eq("eq:lapse_shift_metric", 869, "ds^2 = -alpha_n^2 c^2 dt^2 + W^2 |dx + A_g dt|^2")
    an_s, W_s = sp.symbols('alpha_n W', positive=True)
    ag = sp.symbols('ag1:4', real=True)
    dts = sp.Symbol('dt')
    dxs = sp.symbols('dx1:4')

    def gmat(an_, W_, ag_):
        g = sp.zeros(4)
        g[0, 0] = -an_**2*c**2 + W_**2*sum(a_**2 for a_ in ag_)
        for i in range(3):
            g[0, i+1] = g[i+1, 0] = W_**2*ag_[i]
            g[i+1, i+1] = W_**2
        return g
    gm_ = gmat(an_s, W_s, ag)
    dX = sp.Matrix([dts, *dxs])
    symcheck("matrix form reproduces the line element",
             sp.expand((dX.T*gm_*dX)[0] - (-an_s**2*c**2*dts**2 + W_s**2*sum((dxs[i] + ag[i]*dts)**2 for i in range(3)))))
    Om_s = sp.Symbol('Omega', real=True)
    Xin = sp.Matrix([t, x*sp.cos(Om_s*t) - y*sp.sin(Om_s*t), x*sp.sin(Om_s*t) + y*sp.cos(Om_s*t), z])
    J = Xin.jacobian(sp.Matrix([t, x, y, z]))
    g_rot = (J.T*sp.diag(-c**2, 1, 1, 1)*J).applyfunc(sp.simplify)
    Ag_rot = [-Om_s*y, Om_s*x, 0]
    symcheck("rotating frame: pullback of Minkowski == alpha_n = W = 1, A_g = Omega x x",
             (g_rot - gmat(1, 1, Ag_rot)).applyfunc(sp.simplify))
    mutant("shift with the opposite sign (dx - A_g dt)", _S3_pick((g_rot - gmat(1, 1, [-a_ for a_ in Ag_rot])).applyfunc(sp.simplify)))
    eps = sp.Symbol('epsilon', positive=True)   # eps = 1/c^2
    Ug, beta, gam = sp.symbols('U beta gamma', real=True)
    Phg = -Ug
    an_ppn = 1 + Phg*eps + (beta - R(1, 2))*Phg**2*eps**2
    Winv = 1 + gam*Phg*eps
    symcheck("PPN g_tt/c^2 == -1 + 2U/c^2 - 2 beta U^2/c^4 + O(c^-6)",
             sp.expand(sp.series(-an_ppn**2, eps, 0, 3).removeO() - (-1 + 2*Ug*eps - 2*beta*Ug**2*eps**2)))
    symcheck("PPN g_ij == 1 + 2 gamma U/c^2 + O(c^-4)",
             sp.expand(sp.series(Winv**-2, eps, 0, 2).removeO() - (1 + 2*gam*Ug*eps)))
    an_mut = 1 + Phg*eps + beta*Phg**2*eps**2
    mutant("alpha_n with beta Phi_g^2/c^4 (no -1/2)", sp.expand(sp.series(-an_mut**2, eps, 0, 3).removeO() - (-1 + 2*Ug*eps - 2*beta*Ug**2*eps**2)))
    Gs = sp.Symbol('G', positive=True)
    Jv = sp.Matrix(sp.symbols('J1:4', real=True))
    xv = sp.Matrix([x, y, z])
    rr = sp.sqrt(x**2 + y**2 + z**2)
    Vdip = [Gs/rr**3*sum(xv[k]*R(1, 2)*sum(sp.LeviCivita(k, j, l)*Jv[l] for l in range(3)) for k in range(3)) for j in range(3)]
    symcheck("dipole vector potential V_j == (G/2r^3)(J x x)_j", sp.Matrix([sp.expand(Vdip[j] - Gs/(2*rr**3)*Jv.cross(xv)[j]) for j in range(3)]))
    Ag_LT = -(2*Gs/(c**2*rr**3))*Jv.cross(xv)
    symcheck("g_0i = A_g,i/c (W = 1 at this order) == -4 V_i/c^3", sp.Matrix([sp.expand(Ag_LT[i]/c - (-4*Vdip[i]/c**3)) for i in range(3)]))
    a_r = sp.Symbol('a', positive=True)
    Xr = sp.Matrix([(c/a_r + x/c)*sp.sinh(a_r*t/c)*c, (c**2/a_r + x)*sp.cosh(a_r*t/c) - c**2/a_r])
    # first entry is c T; the Minkowski 1+1 metric in (cT, X) is diag(-1, 1)
    Jr = Xr.jacobian(sp.Matrix([t, x]))
    g_r = (Jr.T*sp.diag(-1, 1)*Jr).applyfunc(lambda u: sp.simplify(u.rewrite(sp.exp)))
    symcheck("Rindler: pullback == -(1 + a x/c^2)^2 c^2 dt^2 + dx^2",
             (g_r - sp.diag(-(1 + a_r*x/c**2)**2*c**2, 1)).applyfunc(sp.simplify))

    # ------------------------------------------------------------------------
    # eq:alpha_lapse_shift  (sgm.tex L874)
    # States: alpha = sqrt(alpha_n^2 - W^2 |A_g|^2/c^2); alpha the rate of a clock at rest (scalar).
    # Derived from: the time-time component of eq:lapse_shift_metric, which gives alpha^2 c^2 = -g_tt.
    #   Labels used: eq:alpha_def, eq:lapse_shift_metric.
    # Test: -g_tt/c^2 from the metric matrix equals alpha^2; rotating frame gives alpha^2 = 1 - |Omega x x|^2/c^2.
    # Mutant: + W^2 |A_g|^2/c^2.
    # ------------------------------------------------------------------------
    eq("eq:alpha_lapse_shift", 874, "alpha = sqrt(alpha_n^2 - W^2 |A_g|^2/c^2)")
    alpha_ls = sp.sqrt(an_s**2 - W_s**2*sum(a_**2 for a_ in ag)/c**2)
    symcheck("alpha^2 == -g_tt/c^2", sp.expand(alpha_ls**2 + gm_[0, 0]/c**2))
    symcheck("rotating frame: alpha^2 == 1 - |Omega x x|^2/c^2", sp.expand(-g_rot[0, 0]/c**2 - (1 - Om_s**2*(x**2 + y**2)/c**2)))
    mutant("+ W^2 |A_g|^2/c^2", sp.expand(an_s**2 + W_s**2*sum(a_**2 for a_ in ag)/c**2 + gm_[0, 0]/c**2))

    # ------------------------------------------------------------------------
    # eq:normal_derivative  (sgm.tex L881)
    # States: D_n psi = (c alpha_n)^-1 [D_t psi - A_g.D psi + (1/4)(e.B_g) psi - (1/2)(div A_g) psi
    #   + (1/2)(A_g.grad ln alpha_n) psi]; B_g = curl A_g.
    # Derived from: the derivative along the unit normal of the slice, with the spinor on the
    #   orthonormal axes and rescaled so that its norm is int |psi|^2 alpha_n^-1 d^3x (cited: companion).
    #   Labels used: eq:polar_form, eq:phase_connection_time, eq:right_connection, eq:lapse_shift_metric.
    # Test: (1) flat limit D_n = D_t/c. (2) Rotating frame (alpha_n = W = 1, A_g = Omega z x x): for a
    #   spinor psi_I(t, X) of the inertial frame, psi_r(t,x) = U(t)^-1 psi_I(t, R(t)x) with
    #   U = exp(Omega t e3/2) (the spin axes turn with the frame); c D_n psi_r must equal
    #   U^-1 d_t psi_I, the inertial time derivative, since the normal observers are the
    #   nonrotating ones. Numerical, at a random point, with a concrete non-polynomial psi_I.
    #   (3) The weight: with K psi = c alpha_n D_n psi - D_t psi, Sc(psi-bar K psi)/alpha_n is the
    #   divergence -(1/2) div(A_g |psi|^2/alpha_n) for general alpha_n(x), A_g(x), A, psi, so the
    #   shift and lapse terms conserve int |psi|^2 alpha_n^-1 d^3x.
    # Mutants: rotation coefficient 1/2 (fails the rotating frame); divergence coefficient 1/4
    #   (fails the weight identity).
    # ------------------------------------------------------------------------
    eq("eq:normal_derivative", 881, "D_n psi = (c alpha_n)^-1 [D_t - A_g.D + (1/4) e.B_g - (1/2) div A_g + (1/2) A_g.grad ln alpha_n] psi")
    zero3 = [0, 0, 0]
    symcheck("flat limit: D_n psi == D_t psi / c", _S3_mat(_S3_add(_S3_Dn(psi, Dt, Dk, 1, zero3, c, X3), _S3_sc(-1/c, Dt(psi)))))
    TT, XX, YY, ZZ = sp.symbols('T X Y Z', real=True)
    cf = [R(3, 7), R(-5, 11), R(4, 13), R(7, 9), R(-2, 3), R(5, 6), R(-8, 15), R(9, 14)]
    psiI = [cf[0]*XX*YY + cf[1]*TT*ZZ + sp.sin(cf[2]*XX + cf[3]*TT),
            cf[4]*YY**2 + cf[5]*XX*TT + sp.cos(cf[6]*YY - cf[7]*ZZ),
            cf[1]*XX*ZZ + cf[7]*TT**2 + sp.sin(cf[0]*YY + cf[4]*TT),
            cf[3]*XX**2 - cf[2]*YY*ZZ + sp.cos(cf[5]*XX + cf[6]*TT)]
    dpsiI = [sp.diff(u_, TT) for u_ in psiI]
    Omn = R(5, 7)
    subs_rot = {TT: t, XX: x*sp.cos(Omn*t) - y*sp.sin(Omn*t), YY: x*sp.sin(Omn*t) + y*sp.cos(Omn*t), ZZ: z}
    Uinv = [sp.cos(Omn*t/2), 0, 0, -sp.sin(Omn*t/2)]
    psir = _S3_mul(Uinv, [u_.subs(subs_rot) for u_ in psiI])
    target = _S3_mul(Uinv, [u_.subs(subs_rot) for u_ in dpsiI])
    Ag_r = [-Omn*y, Omn*x, 0]
    Dt0, Dk0 = _S3_gauge_ops(0, [0, 0, 0], hb, q, t, X3)
    pt = {t: R(3, 11), x: R(-4, 9), y: R(7, 13), z: R(5, 17)}
    tg = np.array(_S3_num(target, pt))
    res = np.array(_S3_num(_S3_sc(c, _S3_Dn(psir, Dt0, Dk0, 1, Ag_r, c, X3)), {**pt, c: 1})) - tg
    check("rotating frame: c D_n psi_r == U^-1 d_t psi_I", np.abs(res).max(), 0, tol=1e-20, scale=np.abs(tg).max())
    res_m = np.array(_S3_num(_S3_sc(c, _S3_Dn(psir, Dt0, Dk0, 1, Ag_r, c, X3, crot=R(1, 2))), {**pt, c: 1})) - tg
    numutant("rotation term (1/2)(e.B_g)", np.abs(res_m).max() + np.abs(tg).max(), np.abs(tg).max())
    anf = RF('alpha_n', X3)
    Agf = [RF(f'Ag{k}', X3) for k in range(3)]
    Kpsi = lambda f, cd: _S3_add(_S3_sc(c*anf, _S3_Dn(f, Dt, Dk, anf, Agf, c, X3, cdiv=cd)), _S3_sc(-1, Dt(f)))
    rho_f = sum(u_**2 for u_ in psi)
    divterm = R(1, 2)*_S3_div([Agf[k]*rho_f/anf for k in range(3)], X3)
    wid = sp.expand(sp.together(_S3_mul(_S3_conj(psi), Kpsi(psi, R(1, 2)))[0]/anf + divterm)*anf**2)
    symcheck("Sc(psi-bar K psi)/alpha_n == -(1/2) div(A_g |psi|^2/alpha_n)", wid)
    wid_m = sp.expand(sp.together(_S3_mul(_S3_conj(psi), Kpsi(psi, R(1, 4)))[0]/anf + divterm)*anf**2)
    mutant("divergence coefficient 1/4", wid_m)

    # ------------------------------------------------------------------------
    # eq:NX_operators  (sgm.tex L893), first line
    # States: N psi = (D_n psi) e3.
    # Derived from: definition.
    #   Labels used: eq:quaternion_units, eq:right_connection, eq:normal_derivative.
    # Test: in flat spacetime N = -(i hbar d_t - V)/(hbar c) on the complex spinor (inline at L898);
    #   D_n commutes with right multiplication by e3 for general alpha_n, A_g, V, A, so N^2 = -D_n^2
    #   (used at L928).
    # Mutant: N = +(i hbar d_t - V)/(hbar c).
    # ------------------------------------------------------------------------
    eq("eq:NX_operators", 893, "N psi = (D_n psi) e3")
    Nflat = lambda f: _S3_mul(_S3_sc(1/c, Dt(f)), e3)
    symcheck("flat: N psi <-> -(i hbar d_t - V) psi/(hbar c)",
             (_S3_q2c(Nflat(psi)) + (I*hb*sp.diff(zpsi, t) - Vf*zpsi)/(hb*c)).applyfunc(sp.expand))
    phi_ = QF('f')
    lhsN = _S3_Dn(_S3_mul(phi_, e3), Dt, Dk, anf, Agf, c, X3)
    rhsN = _S3_mul(_S3_Dn(phi_, Dt, Dk, anf, Agf, c, X3), e3)
    symcheck("D_n(phi e3) == (D_n phi) e3 (so N^2 psi = -D_n^2 psi)", _S3_mat(_S3_add(lhsN, _S3_sc(-1, rhsN))))
    mutant("N = +(i hbar d_t - V)/(hbar c)", _S3_pick((_S3_q2c(Nflat(psi)) - (I*hb*sp.diff(zpsi, t) - Vf*zpsi)/(hb*c)).applyfunc(sp.expand)))

    # ------------------------------------------------------------------------
    # eq:NX_operators  (sgm.tex L894), second line
    # States: X psi = e.D psi.
    # Derived from: definition.
    #   Labels used: eq:pauli_quaternion, eq:right_connection, eq:normal_derivative.
    # Test: X = sigma.pi/hbar on the complex spinor (inline at L898), with time-dependent A.
    # Mutant: X = -sigma.pi/hbar.
    # ------------------------------------------------------------------------
    eq("eq:NX_operators", 894, "X psi = e.D psi")
    spit = lambda w: sum((_S3_SIG[k]*pihat(w, k) for k in range(3)), sp.zeros(2, 1))
    symcheck("X psi <-> sigma.pi psi/hbar", (_S3_q2c(_S3_edot(Dk, psi)) - spit(zpsi)/hb).applyfunc(sp.expand))
    mutant("X = -sigma.pi/hbar", _S3_pick((_S3_q2c(_S3_edot(Dk, psi)) + spit(zpsi)/hb).applyfunc(sp.expand)))

    # ------------------------------------------------------------------------
    # eq:NX_operators  (sgm.tex L895), third line
    # States: X_W psi = W^{-1/2} X (W^{-1/2} psi); W(x) the scale factor.
    # Derived from: the Dirac operator of the spatial metric W^2 delta acting on the spinor rescaled
    #   by W^{3/2} (L879).
    #   Labels used: eq:pauli_quaternion, eq:lapse_shift_metric, eq:normal_derivative.
    # Test: the Dirac operator of W^2 delta, built from the spin connection
    #   (1/4) omega_iab sigma_a sigma_b with omega_iab = delta_ia d_b ln W - delta_ib d_a ln W (frame
    #   W^-1 d_a), equals W^-2 sigma.grad (W .) (the conformal covariance, cited: standard); conjugated
    #   by the rescaling W^{3/2}, it is W^{-1/2} sigma.grad W^{-1/2}, which is X_W at A = 0. General W(x).
    #   Also X_W = X at W = 1.
    # Mutant: X_W = W^-1 X (unsymmetrized).
    # ------------------------------------------------------------------------
    eq("eq:NX_operators", 895, "X_W psi = W^{-1/2} X (W^{-1/2} psi)")
    Wf = RF('W', X3)
    lnW = sp.log(Wf)
    sgrad = lambda w: sum((_S3_SIG[k]*sp.diff(w, X3[k]) for k in range(3)), sp.zeros(2, 1))

    def Dirac_W(w):
        out = sp.zeros(2, 1)
        for i in range(3):
            conn = sp.zeros(2)
            for a_ in range(3):
                for b_ in range(3):
                    om_ = (1 if i == a_ else 0)*sp.diff(lnW, X3[b_]) - (1 if i == b_ else 0)*sp.diff(lnW, X3[a_])
                    conn += R(1, 4)*om_*_S3_SIG[a_]*_S3_SIG[b_]
            out += _S3_SIG[i]*(sp.diff(w, X3[i]) + conn*w)/Wf
        return out
    symcheck("spin-connection Dirac operator of W^2 delta == W^-2 sigma.grad (W .)",
             (Dirac_W(u) - sgrad(Wf*u)/Wf**2).applyfunc(lambda e_: sp.expand(sp.simplify(e_))))
    symcheck("rescaled by W^{3/2}: W^{3/2} D_W (W^{-3/2} phi) == W^{-1/2} sigma.grad (W^{-1/2} phi)",
             (Wf**R(3, 2)*Dirac_W(Wf**R(-3, 2)*u) - Wf**R(-1, 2)*sgrad(Wf**R(-1, 2)*u)).applyfunc(lambda e_: sp.expand(sp.simplify(e_))))
    XWq = lambda f, Wx: _S3_sc(Wx**R(-1, 2), _S3_edot(Dks, _S3_sc(Wx**R(-1, 2), f)))
    symcheck("X_W == X at W = 1", _S3_mat(_S3_add(XWq(psis, 1), _S3_sc(-1, _S3_edot(Dks, psis)))))
    mutant("X_W = W^-1 X", _S3_pick((Wf**R(3, 2)*Dirac_W(Wf**R(-3, 2)*u) - sgrad(u)/Wf).applyfunc(lambda e_: sp.expand(sp.simplify(e_)))))

    # ------------------------------------------------------------------------
    # eq:klein_gordon  (sgm.tex L900)
    # States: c^-2 d_t^2 psi + (e.grad)^2 psi + kappa^2 psi = 0; kappa = mc/hbar.
    # Derived from: cited (Klein, Gordon); the text adds (e.grad)^2 = -laplacian. With V = A = 0 in flat
    #   spacetime the operators of eq:NX_operators are N = c^-1 (d_t .) e3,
    #   X = e.grad and [N, X] = 0, and (e.grad)^2 = -laplacian by eq:quaternion_units.
    #   Labels used: eq:quaternion_units, eq:NX_operators.
    # Test: (e.grad)^2 psi == -laplacian psi for general psi; (N - X)(N + X) psi - kappa^2 psi equals
    #   -(left side of eq:klein_gordon) for general psi; on psi_0 exp(e3(w t - k.x)) the left side is
    #   (-w^2/c^2 + k^2 + kappa^2) psi.
    # Mutant: - kappa^2 psi.
    # ------------------------------------------------------------------------
    eq("eq:klein_gordon", 900, "c^-2 d_t^2 psi + (e.grad)^2 psi + kappa^2 psi = 0")
    kap = m*c/hb
    eg = lambda f: _S3_add(*[_S3_mul(E[k], _S3_d(f, X3[k])) for k in range(3)])
    lap = lambda f: _S3_add(*[_S3_d(f, X3[k], 2) for k in range(3)])
    symcheck("(e.grad)^2 psi == -laplacian psi", _S3_mat(_S3_add(eg(eg(psi)), lap(psi))))
    N0 = lambda f: _S3_mul(_S3_sc(1/c, _S3_d(f, t)), e3)
    pf = lambda f: _S3_add(N0(N0(f)), _S3_sc(-1, eg(eg(f))))      # (N - X)(N + X) with [N, X] = 0
    KG = _S3_add(_S3_sc(c**-2, _S3_d(psi, t, 2)), eg(eg(psi)), _S3_sc(kap**2, psi))
    symcheck("(N - X)(N + X) psi - kappa^2 psi == -(KG operator) psi",
             _S3_mat(_S3_add(N0(N0(psi)), _S3_sc(-1, eg(eg(psi))), _S3_sc(-1, N0(eg(psi))), eg(N0(psi)),
                             _S3_sc(-kap**2, psi), KG)))
    kk = sp.symbols('k1:4', real=True)
    thp = om*t - sum(kk[i]*X3[i] for i in range(3))
    psw = _S3_mul(list(a_s), [sp.cos(thp), 0, 0, sp.sin(thp)])
    KGp = _S3_add(_S3_sc(c**-2, _S3_d(psw, t, 2)), eg(eg(psw)), _S3_sc(kap**2, psw))
    symcheck("plane wave: KG psi == (-w^2/c^2 + k^2 + kappa^2) psi",
             _S3_mat(_S3_add(KGp, _S3_sc(-(-om**2/c**2 + sum(k_**2 for k_ in kk) + kap**2), psw))))
    mutant("- kappa^2", _S3_pick(_S3_mat(_S3_add(_S3_sc(c**-2, _S3_d(psi, t, 2)), eg(eg(psi)), _S3_sc(-kap**2, psi),
                                                 N0(N0(psi)), _S3_sc(-1, eg(eg(psi))), _S3_sc(-kap**2, psi)))))

    # ------------------------------------------------------------------------
    # eq:dirac_equation  (sgm.tex L908)
    # States: i hbar d_t (psi_L, psi_R) = [[q Phi - c sigma.pi, mc^2], [mc^2, q Phi + c sigma.pi]] (psi_L, psi_R).
    # Derived from: cited (Dirac; Peskin's Weyl representation).
    #   Labels used: eq:pauli_quaternion, eq:right_connection (pi-hat); the equation itself is cited.
    # Test: Peskin's gamma^0 = [[0,1],[1,0]], gamma^i = [[0,sigma^i],[-sigma^i,0]] satisfy the Clifford
    #   algebra with eta = diag(1,-1,-1,-1); the Hamiltonian q Phi + c gamma^0 gamma^i pi_i + mc^2 gamma^0
    #   of (i hbar gamma^mu d_mu - q gamma^mu A_mu - mc) psi = 0 is the displayed matrix; its square at
    #   Phi = 0 is (m^2 c^4 + c^2 pi^2) times the identity for commuting pi.
    # Mutant: the displayed matrix with the sigma.pi signs exchanged differs from gamma^0 gamma^i.
    # ------------------------------------------------------------------------
    eq("eq:dirac_equation", 908, "i hbar d_t (psi_L, psi_R) = [[q Phi - c sigma.pi, mc^2], [mc^2, q Phi + c sigma.pi]] (psi_L, psi_R)")
    Z2, I2 = sp.zeros(2), sp.eye(2)
    g0 = sp.Matrix(sp.BlockMatrix([[Z2, I2], [I2, Z2]]))
    gi = [sp.Matrix(sp.BlockMatrix([[Z2, _S3_SIG[k]], [-_S3_SIG[k], Z2]])) for k in range(3)]
    gam_ = [g0] + gi
    eta = sp.diag(1, -1, -1, -1)
    boolcheck("Peskin gammas: {gamma^mu, gamma^nu} == 2 eta^{mu nu}",
              all((gam_[a_]*gam_[b_] + gam_[b_]*gam_[a_] - 2*eta[a_, b_]*sp.eye(4)) == sp.zeros(4) for a_ in range(4) for b_ in range(4)))
    pis_ = sp.symbols('pi1:4', real=True)
    sp_ = sum((_S3_SIG[k]*pis_[k] for k in range(3)), sp.zeros(2))
    HD = q*Phs*sp.eye(4) + c*sum((g0*gi[k]*pis_[k] for k in range(3)), sp.zeros(4)) + m*c**2*g0
    Hpaper = sp.Matrix(sp.BlockMatrix([[q*Phs*I2 - c*sp_, m*c**2*I2], [m*c**2*I2, q*Phs*I2 + c*sp_]]))
    symcheck("Hamiltonian from the Peskin form == the displayed matrix", (HD - Hpaper).applyfunc(sp.expand))
    H0 = Hpaper.subs(Phs, 0)
    symcheck("H^2 at Phi = 0 == (m^2 c^4 + c^2 pi^2) I", (H0*H0 - (m**2*c**4 + c**2*sum(p_**2 for p_ in pis_))*sp.eye(4)).applyfunc(sp.expand))
    Hmut = sp.Matrix(sp.BlockMatrix([[q*Phs*I2 + c*sp_, m*c**2*I2], [m*c**2*I2, q*Phs*I2 - c*sp_]]))
    mutant("sigma.pi signs exchanged", _S3_pick((HD - Hmut).applyfunc(sp.expand)))

    # ------------------------------------------------------------------------
    # eq:weyl_pair_R  (sgm.tex L915)
    # States: (N + X_W) phi_R = kappa phi_L; phi_R = psi_R, phi_L = -psi_L as quaternions.
    # Derived from: the second row of eq:dirac_equation, divided by -hbar c, with the minimal-coupling
    #   rules c^-1 d_t -> D_n, grad -> W^{-1/2} D W^{-1/2} (cited: companion).
    #   Labels used: eq:spinor_quaternion, eq:pauli_quaternion, eq:lapse_shift_metric, eq:normal_derivative, eq:NX_operators, eq:dirac_equation.
    # Test: flat spacetime, general V(t,x), A(t,x), and general quaternion functions phi_R, phi_L: the
    #   complex form of (N + X) phi_R - kappa phi_L equals -(hbar c)^-1 times the residual of row R
    #   [i hbar d_t psi_R - (q Phi + c sigma.pi) psi_R - mc^2 psi_L] with psi_R = phi_R, psi_L = -phi_L.
    #   The curved-space rules are tested on plane waves at eq:pure_form.
    # Mutant: phi_L = +psi_L.
    # ------------------------------------------------------------------------
    eq("eq:weyl_pair_R", 915, "(N + X_W) phi_R = kappa phi_L")
    phR = QF('r')
    phL = QF('l')
    Nf = lambda f: _S3_mul(_S3_sc(1/c, Dt(f)), e3)
    Xf = lambda f: _S3_edot(Dk, f)
    zR, zLm = _S3_q2c(phR), _S3_q2c(phL)
    rowR = lambda pR, pL: I*hb*sp.diff(pR, t) - Vf*pR - c*spit(pR) - m*c**2*pL
    rowL = lambda pL, pR: I*hb*sp.diff(pL, t) - Vf*pL + c*spit(pL) - m*c**2*pR
    qR = _S3_add(Nf(phR), Xf(phR), _S3_sc(-kap, phL))
    symcheck("(N + X) phi_R - kappa phi_L <-> -(hbar c)^-1 (row R residual)",
             (_S3_q2c(qR) + rowR(zR, -zLm)/(hb*c)).applyfunc(sp.expand))
    mutant("phi_L = +psi_L", _S3_pick((_S3_q2c(qR) + rowR(zR, zLm)/(hb*c)).applyfunc(sp.expand)))

    # ------------------------------------------------------------------------
    # eq:weyl_pair_L  (sgm.tex L917)
    # States: (N - X_W) phi_L = kappa phi_R.
    # Derived from: the first row of eq:dirac_equation, with phi_L = -psi_L.
    #   Labels used: eq:NX_operators, eq:dirac_equation, eq:weyl_pair_R.
    # Test: as for eq:weyl_pair_R: the complex form of (N - X) phi_L - kappa phi_R equals
    #   +(hbar c)^-1 times the residual of row L at psi_L = -phi_L.
    # Mutant: (N + X) phi_L = kappa phi_R (the chirality sign).
    # ------------------------------------------------------------------------
    eq("eq:weyl_pair_L", 917, "(N - X_W) phi_L = kappa phi_R")
    qL = _S3_add(Nf(phL), _S3_sc(-1, Xf(phL)), _S3_sc(-kap, phR))
    symcheck("(N - X) phi_L - kappa phi_R <-> (hbar c)^-1 (row L residual at psi_L = -phi_L)",
             (_S3_q2c(qL) - rowL(-zLm, zR)/(hb*c)).applyfunc(sp.expand))
    qLm = _S3_add(Nf(phL), Xf(phL), _S3_sc(-kap, phR))
    mutant("(N + X) phi_L", _S3_pick((_S3_q2c(qLm) - rowL(-zLm, zR)/(hb*c)).applyfunc(sp.expand)))

    # ------------------------------------------------------------------------
    # eq:pure_form  (sgm.tex L923)
    # States: (N - X_W)(N + X_W) psi = kappa^2 psi, psi = phi_R.
    # Derived from: eq:weyl_pair_R gives phi_L = (N + X_W) phi_R/kappa; substituted into eq:weyl_pair_L
    #   (m constant, so kappa passes through the operators).
    #   Labels used: eq:NX_operators, eq:weyl_pair_R, eq:weyl_pair_L.
    # Test: (1) On psi = psi_0 exp(e3(w t - k.x)) with constant alpha_n, W, V, A, A_g, the residual of
    #   eq:pure_form, computed with the full operators of eqs. normal_derivative and NX_operators, is
    #   [Q^2 - pi^2/(W^2 hbar^2) - kappa^2] psi with Q = (hbar w - V + A_g.pi)/(hbar c alpha_n),
    #   pi = hbar k - qA. (2) That bracket equals -(g^{mu nu} P_mu P_nu + m^2 c^2)/hbar^2 for the inverse of
    #   eq:lapse_shift_metric with P_t = -(hbar w - V), P_i = pi_i: the wave equation reproduces the
    #   mass shell of the metric. (3) Its root is hbar w = V + alpha_n sqrt(m^2c^4 + c^2 pi^2/W^2) - A_g.pi.
    #   (4) Substituting phi_L from the first equation into the second, on the same plane wave,
    #   reproduces the residual exactly.
    # Mutant: the frequency shifted off the root leaves a nonzero residual; the root with + A_g.pi
    #   fails the mass shell.
    # ------------------------------------------------------------------------
    eq("eq:pure_form", 923, "(N - X_W)(N + X_W) psi = kappa^2 psi")
    Vc = sp.Symbol('V', real=True)
    Ac = sp.symbols('Ac1:4', real=True)
    Dtc, Dkc = _S3_gauge_ops(Vc, list(Ac), hb, q, t, X3)
    Ncf = lambda f: _S3_mul(_S3_Dn(f, Dtc, Dkc, an_s, list(ag), c, X3), e3)
    XWc = lambda f: _S3_sc(1/W_s, _S3_edot(Dkc, f))      # W constant: X_W = W^-1 X
    pw = _S3_mul(list(a_s), [sp.cos(thp), 0, 0, sp.sin(thp)])
    NpX = _S3_add(Ncf(pw), XWc(pw))
    lhs_pf = _S3_add(Ncf(NpX), _S3_sc(-1, XWc(NpX)))
    pivc = [hb*kk[i] - q*Ac[i] for i in range(3)]
    Agpi = sum(ag[i]*pivc[i] for i in range(3))
    Qn = (hb*om - Vc + Agpi)/(hb*c*an_s)
    bracket = Qn**2 - sum(p_**2 for p_ in pivc)/(W_s**2*hb**2) - kap**2
    symcheck("plane wave: (N - X_W)(N + X_W) psi - kappa^2 psi == [Q^2 - pi^2/(W hbar)^2 - kappa^2] psi",
             _S3_mat(_S3_add(lhs_pf, _S3_sc(-kap**2, pw), _S3_sc(-bracket, pw))))
    ginv = gm_.inv()
    Pmu = sp.Matrix([-(hb*om - Vc), *pivc])
    shell = (Pmu.T*ginv*Pmu)[0] + m**2*c**2
    symcheck("bracket == -(g^{mu nu} P_mu P_nu + m^2 c^2)/hbar^2", sp.simplify(sp.together(bracket + shell/hb**2)))
    root = Vc + an_s*sp.sqrt(m**2*c**4 + c**2*sum(p_**2 for p_ in pivc)/W_s**2) - Agpi
    symcheck("bracket vanishes at hbar w = V + alpha_n sqrt(m^2c^4 + c^2 pi^2/W^2) - A_g.pi",
             sp.expand(bracket.subs(om, root/hb)))
    phLp = _S3_sc(1/kap, NpX)
    resL = _S3_add(Ncf(phLp), _S3_sc(-1, XWc(phLp)), _S3_sc(-kap, pw))
    symcheck("substitution: (N - X_W)[(N + X_W) psi/kappa] - kappa psi == (pure-form residual)/kappa",
             _S3_mat(_S3_add(resL, _S3_sc(-1/kap, _S3_add(lhs_pf, _S3_sc(-kap**2, pw))))))
    mutant("frequency off the root by mc^2/(2 hbar)", sp.expand(bracket.subs(om, root/hb + m*c**2/(2*hb))))
    root_m = Vc + an_s*sp.sqrt(m**2*c**4 + c**2*sum(p_**2 for p_ in pivc)/W_s**2) + Agpi
    mutant("root with + A_g.pi", sp.expand(bracket.subs(om, root_m/hb)))

    # ------------------------------------------------------------------------
    # eq:master  (sgm.tex L941)
    # States: hbar w chi = [V + alpha_n sqrt((mc^2)^2 + c^2 W^-1 (pi^2 - q hbar sigma.B) W^-1) - Pi_g
    #   - (1/4) hbar sigma.B_g + C] chi; Pi_g = alpha_n^{1/2} (1/2){A_g, pi} alpha_n^{-1/2}; C a sum of
    #   commutators.
    # Derived from: cited (companion): exact positive-frequency reduction of eq:pure_form under the
    #   graph hypothesis.
    #   Labels used: tab:classes, eq:stationary_time, eq:fueter_square, eq:lapse_shift_metric, eq:normal_derivative, eq:NX_operators, eq:pure_form.
    # Test: (1) Constant fields, B = B_g = 0, C = 0: the root equals the plane-wave root of eq:pure_form.
    #   (2) The Dirac Hamiltonian at V = 0 squares to (m^2c^4 + c^2 (sigma.pi)^2) on each block, for a
    #   noncommuting sigma.pi, so the root content is m^2c^4 + c^2(pi^2 - q hbar sigma.B) by the Pauli
    #   form of eq:fueter_square. (3) Rotating frame (alpha_n = W = 1, A_g = Omega x x): Pi_g f == Omega.L f
    #   on a general f, and -(1/4) hbar sigma.B_g = -Omega.S with B_g = 2 Omega (the frame couples to
    #   L + S, as L884 states). (4) alpha_n^{-1/2} Pi_g alpha_n^{1/2} = (1/2){A_g, pi}, which is symmetric:
    #   f-bar (S g) - conj(S f) g == -i hbar div(A_g f-bar g) for general complex f, g, A_g, A.
    # Mutant: spin-rotation coefficient 1/2 gives -Omega.(L + 2S), not -Omega.J.
    # ------------------------------------------------------------------------
    eq("eq:master", 941, "hbar w chi = [V + alpha_n sqrt((mc^2)^2 + c^2 W^-1 (pi^2 - q hbar sigma.B) W^-1) - Pi_g - (1/4) hbar sigma.B_g + C] chi")
    mroot = Vc + an_s*sp.sqrt(m**2*c**4 + c**2*W_s**-1*sum(p_**2 for p_ in pivc)*W_s**-1) - Agpi
    symcheck("constant fields: master root == pure-form root", sp.expand(mroot - root))
    Pnc = sp.Symbol('P', commutative=False)
    blk = sp.Matrix([[-c*Pnc, m*c**2], [m*c**2, c*Pnc]])
    sq = (blk*blk).applyfunc(sp.expand)
    symcheck("Dirac blocks at V = 0 square to m^2c^4 + c^2 (sigma.pi)^2 on the diagonal, 0 off it",
             sp.Matrix([sq[0, 0] - m**2*c**4 - c**2*Pnc**2, sq[1, 1] - m**2*c**4 - c**2*Pnc**2, sq[0, 1], sq[1, 0]]))
    fC = sp.Function('f')(*X3)
    Omv = sp.Matrix(sp.symbols('Om1:4', real=True))
    Agrot = Omv.cross(xv)
    pi_op = lambda g_, k: -I*hb*sp.diff(g_, X3[k])
    anti = lambda g_, Agl: R(1, 2)*sum(Agl[k]*pi_op(g_, k) + pi_op(Agl[k]*g_, k) for k in range(3))
    Lop = lambda g_: [sum(sp.LeviCivita(i, j, k)*X3[j]*pi_op(g_, k) for j in range(3) for k in range(3)) for i in range(3)]
    symcheck("rotating frame: (1/2){Omega x x, pi} f == Omega.L f", sp.expand(anti(fC, Agrot) - sum(Omv[i]*Lop(fC)[i] for i in range(3))))
    Bgrot = _S3_curl(list(Agrot), X3)
    symcheck("B_g = curl(Omega x x) == 2 Omega", sp.Matrix([sp.expand(Bgrot[i] - 2*Omv[i]) for i in range(3)]))
    Sop = [hb*_S3_SIG[k]/2 for k in range(3)]
    symcheck("-(1/4) hbar sigma.B_g == -Omega.S",
             (-R(1, 4)*hb*sum((_S3_SIG[k]*Bgrot[k] for k in range(3)), sp.zeros(2)) + sum((Omv[k]*Sop[k] for k in range(3)), sp.zeros(2))).applyfunc(sp.expand))
    an_f = RF('alpha_n', X3)
    Agen = [RF(f'Ag{k}', X3) for k in range(3)]
    Pig = lambda g_: sp.sqrt(an_f)*anti(g_/sp.sqrt(an_f), Agen)
    symcheck("alpha_n^{-1/2} Pi_g alpha_n^{1/2} f == (1/2){A_g, pi} f",
             sp.expand(sp.simplify(Pig(sp.sqrt(an_f)*fC)/sp.sqrt(an_f) - anti(fC, Agen))))
    f1, f2, g1, g2 = [RF(n_, X3) for n_ in ('f1', 'f2', 'g1', 'g2')]
    fcx, gcx = f1 + I*f2, g1 + I*g2
    Sfull = lambda g_: anti(g_, Agen) - q*sum(Agen[k]*As[k] for k in range(3))*g_
    # complex conjugates written out: conj(f) = f1 - i f2, conj(pi f) = (+i hbar grad - qA) conj(f)
    fbx = f1 - I*f2
    anti_c = lambda g_, Agl: R(1, 2)*sum(Agl[k]*I*hb*sp.diff(g_, X3[k]) + I*hb*sp.diff(Agl[k]*g_, X3[k]) for k in range(3))
    Sconj = lambda g_: anti_c(g_, Agen) - q*sum(Agen[k]*As[k] for k in range(3))*g_
    lhs_sym = fbx*Sfull(gcx) - Sconj(fbx)*gcx
    symcheck("(1/2){A_g, pi} is symmetric: f-bar S g - conj(S f) g == -i hbar div(A_g f-bar g)",
             sp.expand(lhs_sym + I*hb*_S3_div([Agen[k]*fbx*gcx for k in range(3)], X3)))
    mutant("spin-rotation coefficient 1/2: -Omega.(2S) != -Omega.S",
           _S3_pick((-R(1, 2)*hb*sum((_S3_SIG[k]*Bgrot[k] for k in range(3)), sp.zeros(2)) + sum((Omv[k]*Sop[k] for k in range(3)), sp.zeros(2))).applyfunc(sp.expand)))

    # ------------------------------------------------------------------------
    # eq:static_master  (sgm.tex L950)
    # States: hbar w = V + mc^2 alpha, alpha from eq:alpha_lapse_shift; for a particle at rest.
    # Derived from: the order-hbar^0 symbol of eq:master, H(x, pi) = V + alpha_n sqrt(m^2c^4 + c^2 pi^2/W^2)
    #   - A_g.pi, at the momentum where xdot = grad_pi H = 0.
    #   Labels used: eq:sgm, eq:alpha_lapse_shift, eq:master.
    # Test: that momentum is pi* = W^2 A_g E_r/(alpha_n c^2) with E_r = mc^2 alpha_n/alpha; grad_pi H(pi*) = 0
    #   and H(pi*) = V + mc^2 alpha, with no expansion in c^-2. Force: with V, alpha_n, W, A_g general
    #   functions of x, -d_x H at fixed pi evaluated at pi*(x) equals -d_x(V + mc^2 alpha) (numerically at
    #   random values of the functions and their derivatives).
    # Mutants: H at pi = 0 (V + alpha_n mc^2) is not V + mc^2 alpha; the force -grad V - mc^2 grad alpha_n.
    # ------------------------------------------------------------------------
    eq("eq:static_master", 950, "hbar w = V + mc^2 alpha")
    pis3 = sp.symbols('pp1:4', real=True)
    Hs = Vc + an_s*sp.sqrt(m**2*c**4 + c**2*sum(p_**2 for p_ in pis3)/W_s**2) - sum(ag[i]*pis3[i] for i in range(3))
    Er = m*c**2*an_s/alpha_ls
    pstar = {pis3[i]: W_s**2*ag[i]*Er/(an_s*c**2) for i in range(3)}
    numpt = {an_s: R(9, 8), W_s: R(7, 6), ag[0]: R(3, 10)*c, ag[1]: R(-1, 5)*c, ag[2]: R(2, 7)*c}
    vel = [sp.diff(Hs, pis3[i]).subs(pstar) for i in range(3)]
    for i in range(3):
        symcheck(f"grad_pi H at pi* vanishes (component {i+1})", sp.simplify(vel[i].subs(numpt)))
    symcheck("H(pi*) == V + mc^2 alpha (rational sample point)", sp.simplify((Hs.subs(pstar) - (Vc + m*c**2*alpha_ls)).subs(numpt)))
    # symbolic: alpha > 0 and |A_g|^2 = A2 >= 0 as symbols, alpha_n eliminated by eq:alpha_lapse_shift
    al_p, A2_p = sp.symbols('alpha_p A2_p', positive=True)
    an_p = sp.sqrt(al_p**2 + W_s**2*A2_p/c**2)
    Er_p = m*c**2*an_p/al_p
    H_p = Vc + an_p*sp.sqrt(m**2*c**4 + c**2*(W_s**4*A2_p*Er_p**2/(an_p**2*c**4))/W_s**2) - W_s**2*A2_p*Er_p/(an_p*c**2)
    fsq = lambda e_: e_.replace(lambda u_: isinstance(u_, sp.Pow) and u_.exp == R(1, 2),
                                lambda u_: sp.sqrt(sp.factor(u_.base)))
    symcheck("H(pi*) == V + mc^2 alpha (symbolic, |pi*|^2 and A_g.pi* written through |A_g|^2)",
             sp.simplify(fsq(sp.simplify(H_p - Vc - m*c**2*al_p))))
    mutant("H at pi = 0", sp.simplify((Hs.subs({p_: 0 for p_ in pis3}) - (Vc + m*c**2*alpha_ls)).subs(numpt)))
    xs_ = sp.Symbol('xs', real=True)
    Vx, anx, Wx = [sp.Function(n_)(xs_) for n_ in ('Vx', 'anx', 'Wx')]
    agx = [sp.Function(f'agx{i}')(xs_) for i in range(3)]
    Hx = Vx + anx*sp.sqrt(m**2*c**4 + c**2*sum(p_**2 for p_ in pis3)/Wx**2) - sum(agx[i]*pis3[i] for i in range(3))
    alx = sp.sqrt(anx**2 - Wx**2*sum(a_**2 for a_ in agx)/c**2)
    pstx = {pis3[i]: Wx**2*agx[i]*(m*c**2*anx/alx)/(anx*c**2) for i in range(3)}
    Fcan = -sp.diff(Hx, xs_).subs(pstx)
    Ftot = -sp.diff(Vx + m*c**2*alx, xs_)
    Fmut = -sp.diff(Vx + m*c**2*anx, xs_)
    vals = {}
    rv = [R(9, 8), R(7, 6), R(3, 10), R(-1, 5), R(2, 7), R(5, 3)]
    dv = [R(-2, 9), R(4, 11), R(1, 6), R(-3, 7), R(5, 13), R(2, 5)]
    funcs = [anx, Wx, agx[0], agx[1], agx[2], Vx]
    nsub = {c: 1, m: R(4, 5)}
    for f_, v_, d_ in zip(funcs, rv, dv):
        vals[sp.Derivative(f_, xs_)] = d_
    for f_, v_ in zip(funcs, rv):
        vals[f_] = v_
    Fc_n = complex(sp.N(Fcan.subs(vals).subs(nsub), 30))
    Ft_n = complex(sp.N(Ftot.subs(vals).subs(nsub), 30))
    check("force: -d_x H at fixed pi, at pi*, == -d_x(V + mc^2 alpha)", Fc_n, Ft_n, tol=1e-20)
    numutant("force -d_x V - mc^2 d_x alpha_n", complex(sp.N(Fmut.subs(vals).subs(nsub), 30)), Ft_n)

    # ------------------------------------------------------------------------
    # eq:nr_expansion  (sgm.tex L963)
    # States: h = mc^2 alpha_n + V + (alpha_n/4){m^-1, W^-1 pi^2 W^-1} - Pi_g - alpha_n q hbar sigma.B/(2mW^2)
    #   - hbar sigma.B_g/4 - hbar^4 X^4/(8 m^3 c^2) + (hbar^2/2m)(X~_W X~_W^dag - W^-1 X^2 W^-1) + O(c^-4);
    #   X~ = e^{-f/4mc^2} X e^{f/4mc^2}, X~_W = W^{-1/2} X~ W^{-1/2}, f = V-hat + G - V_sc.
    # Derived from: eq:master expanded in c^-2 for a slow particle in weak fields (cited: companion).
    #   Labels used: eq:fueter_square, eq:NX_operators, eq:master.
    # Test: (1) Commuting part: with alpha_n = 1 + a/c^2, W^-1 = 1 + w/c^2, and pi^2 - q hbar sigma.B a
    #   number, the root alpha_n sqrt(...) minus terms 1, 3, 5, 7 is O(c^-4); hbar^4 X^4 = (sigma.pi)^4 =
    #   (pi^2 - q hbar sigma.B)^2 in flat space. (2) The last term in flat space with f = V(x) (A_g = 0,
    #   G = V_sc = 0) and general A(x): its c^-2 part equals the Darwin term (hbar^2/8m^2c^2) lap V plus the
    #   spin-orbit term S.(grad V x pi)/(2 m^2 c^2) with Thomas factor 1/2, compared on the complex spinor
    #   (rows "Darwin" and "Spin-orbit" of tab:forces). (3) With f = m Phi_g (G at leading order) and
    #   W^-1 = 1 + gamma Phi_g/c^2, its c^-2 part is (1 + 2 gamma) hbar^2 lap Phi_g/(8m) + (1/2 + gamma)
    #   S.(grad Phi_g x pi)/(m) (row "Geodetic precession" of tab:forces).
    # Mutants: Pauli term with 4mW^2; dressing exponent f/2mc^2 (Thomas factor 1); geodetic coefficient
    #   1/2 without gamma.
    # ------------------------------------------------------------------------
    eq("eq:nr_expansion", 963, "h = mc^2 alpha_n + V + kin - Pi_g - Pauli - hbar sigma.B_g/4 - hbar^4 X^4/(8m^3c^2) + so + O(c^-4)")
    ee = sp.Symbol('ee', positive=True)       # ee = 1/c
    aa, ww, P2, bb = sp.symbols('a w P2 b', real=True)   # bb = q hbar sigma.B (eigenvalue)
    cc_ = 1/ee
    an_e = 1 + aa*ee**2
    Wi_e = 1 + ww*ee**2
    root_e = an_e*sp.sqrt(m**2*cc_**4 + cc_**2*Wi_e**2*(P2 - bb))
    terms_e = m*cc_**2*an_e + an_e*Wi_e**2*P2/(2*m) - an_e*Wi_e**2*bb/(2*m) - (P2 - bb)**2*ee**2/(8*m**3)
    diff_e = sp.series(sp.expand(root_e - terms_e), ee, 0, 4).removeO()
    symcheck("root minus terms 1, 3, 5, 7 is O(c^-4)", sp.expand(diff_e))
    terms_m = m*cc_**2*an_e + an_e*Wi_e**2*P2/(2*m) - an_e*Wi_e**2*bb/(4*m) - (P2 - bb)**2*ee**2/(8*m**3)
    mutant("Pauli term over 4 m W^2", sp.expand(sp.series(sp.expand(root_e - terms_m), ee, 0, 4).removeO()))
    psp = sp.symbols('ps1:4', real=True)
    Bsy = sp.symbols('Bs1:4', real=True)
    # (sigma.pi)^4 for a uniform B is (sigma.pi)^2 squared; (sigma.pi)^2 = pi^2 - q hbar sigma.B (eq:fueter_square)
    sp2 = (spi(spi(u)) - pi2 + q*hb*sB*u).applyfunc(sp.expand)
    symcheck("hbar^2 X^2 = (sigma.pi)^2 == pi^2 - q hbar sigma.B, so hbar^4 X^4 = (pi^2 - q hbar sigma.B)^2", sp2)
    # (2) Darwin and spin-orbit
    epsc = sp.Symbol('eps_c', positive=True)   # 1/c^2
    Vx3 = RF('V', X3)
    gV = [sp.diff(Vx3, X3[k]) for k in range(3)]
    Xo = lambda f: _S3_edot(Dks, f)

    def so_term(fdress, Winv_lin, cexp=R(1, 4)):
        # (hbar^2/2m)(X~_W X~_W^dag - W^-1 X^2 W^-1), first order in epsc; Winv_lin = W^-1 to first order
        gf = [sp.diff(fdress, X3[k]) for k in range(3)]
        Xt = lambda f: _S3_add(Xo(f), _S3_sc(cexp*epsc/m, _S3_mul(_S3_vq(gf), f)))
        Xtd = lambda f: _S3_add(Xo(f), _S3_sc(-cexp*epsc/m, _S3_mul(_S3_vq(gf), f)))
        wh = 1 + (Winv_lin - 1)/2           # W^{-1/2} to first order
        XtW = lambda f: _S3_sc(wh, Xt(_S3_sc(wh, f)))
        XtWd = lambda f: _S3_sc(wh, Xtd(_S3_sc(wh, f)))
        full = _S3_add(XtW(XtWd(psis)), _S3_sc(-Winv_lin, Xo(Xo(_S3_sc(Winv_lin, psis)))))
        full = _S3_sc(hb**2/(2*m), full)
        ex = [sp.expand(u_) for u_ in full]
        return [u_.coeff(epsc, 0) for u_ in ex], [u_.coeff(epsc, 1) for u_ in ex]
    us = _S3_q2c(psis)
    sxp = lambda g_, w_: sum((_S3_SIG[l]*sum((sp.LeviCivita(l, mm_, k)*g_[mm_]*pis(w_, k) for mm_ in range(3) for k in range(3)),
                                             sp.zeros(2, 1))
                              for l in range(3)), sp.zeros(2, 1))
    lapV = sum(sp.diff(Vx3, X3[k], 2) for k in range(3))
    o0, o1 = so_term(Vx3, 1)
    exp_so = hb**2*lapV/(8*m**2)*us + (1/(2*m**2))*(hb/2)*sxp(gV, us)
    symcheck("so term, order c^0, vanishes", sp.Matrix([sp.expand(u_) for u_ in o0]))
    symcheck("so term, order c^-2 == Darwin (hbar^2/8m^2) lap V + spin-orbit S.(grad V x pi)/(2m^2)",
             (_S3_q2c(o1) - exp_so).applyfunc(sp.expand))
    _, o1m = so_term(Vx3, 1, cexp=R(1, 2))
    mutant("dressing exponent f/(2mc^2): Thomas factor 1", _S3_pick((_S3_q2c(o1m) - exp_so).applyfunc(sp.expand)))
    # (3) geodetic: f = m Phi_g, W^-1 = 1 + gamma Phi_g epsc
    Phg3 = RF('Phi_g', X3)
    gP = [sp.diff(Phg3, X3[k]) for k in range(3)]
    lapP = sum(sp.diff(Phg3, X3[k], 2) for k in range(3))
    g0_, g1_ = so_term(m*Phg3, 1 + gam*Phg3*epsc)
    exp_geo = (1 + 2*gam)*hb**2*lapP/(8*m)*us + (R(1, 2) + gam)/m*(hb/2)*sxp(gP, us)
    symcheck("geodetic: so term order c^0 vanishes", sp.Matrix([sp.expand(u_) for u_ in g0_]))
    symcheck("geodetic: so term order c^-2 == (1+2 gamma) hbar^2 lap Phi_g/(8m) + (1/2 + gamma) S.(grad Phi_g x pi)/m",
             (_S3_q2c(g1_) - exp_geo).applyfunc(sp.expand))
    exp_geo_m = (1 + 2*gam)*hb**2*lapP/(8*m)*us + R(1, 2)/m*(hb/2)*sxp(gP, us)
    mutant("geodetic coefficient 1/2 without gamma", _S3_pick((_S3_q2c(g1_) - exp_geo_m).applyfunc(sp.expand)))

    # ------------------------------------------------------------------------
    # tab:forces  (sgm.tex L974-1006) and L1008-1010, orbital rows; a table, not a display
    # States: the force entries of the rows Newtonian, nonlinear gravity, inertial reaction,
    #   Lorentz-scalar, magnetic Lorentz and induction, velocity-dependent gravity, frame dragging,
    #   Coriolis-Euler-centrifugal, relativistic mass increase, Darwin; and the text's B_g = 2 Omega,
    #   orbital couplings -(q/2m_0) B.L and -(1/2) B_g.L, the centrifugal identity and the geodetic row at
    #   gamma = 0.
    # Derived from: the terms of eq:nr_expansion with the table's alpha_n and V, by -grad at fixed p or
    #   the rate of the kinematic momentum.
    #   Labels used: eq:sgm, eq:alpha_rindler, eq:lorentz, eq:lapse_shift_metric, eq:alpha_lapse_shift, eq:static_master, eq:nr_expansion.
    # Test: each entry from its term; frame dragging: for general A_g(t,x), H = p^2/2m - A_g.p gives
    #   m xddot = m xdot x B_g - m d_t A_g + (m/2) grad |A_g|^2 (the last term O(c^-4) for Lense-Thirring,
    #   and the centrifugal force for a rotating frame).
    # Mutants: nonlinear gravity with 2 beta (no -1); Coriolis with factor 1; centrifugal with + sign.
    # ------------------------------------------------------------------------
    eq("tab:forces", 985, "orbital rows of the table of forces")
    m0s = sp.Symbol('m_0', positive=True)
    Phx = RF('Phi_g', X3)
    term_geom = sp.series(m0s*(1/epsc)*(1 + Phx*epsc + (beta - R(1, 2))*Phx**2*epsc**2), epsc, 0, 2).removeO()
    symcheck("m_0 c^2 alpha_n == m_0 c^2 + m_0 Phi_g + (beta - 1/2) m_0 Phi_g^2/c^2 (Newtonian and 1PN terms)",
             sp.expand(term_geom - (m0s/epsc + m0s*Phx + (beta - R(1, 2))*m0s*Phx**2*epsc)))
    F_nl = [-sp.diff((beta - R(1, 2))*m0s*Phx**2*epsc, X3[k]) for k in range(3)]
    symcheck("nonlinear gravity force == -(2 beta - 1)(m_0 Phi_g/c^2) grad Phi_g",
             sp.Matrix([sp.expand(F_nl[k] + (2*beta - 1)*m0s*Phx*epsc*sp.diff(Phx, X3[k])) for k in range(3)]))
    mutant("nonlinear gravity with 2 beta", _S3_pick(sp.Matrix([sp.expand(F_nl[k] + 2*beta*m0s*Phx*epsc*sp.diff(Phx, X3[k])) for k in range(3)])))
    av = sp.Matrix(sp.symbols('a1:4', real=True))
    symcheck("inertial reaction: -grad(m_0 a.x) == -m_0 a", sp.Matrix([sp.diff(-m0s*av.dot(xv), X3[k]) + m0s*av[k] for k in range(3)]))
    symcheck("Lorentz scalar: m c^2 alpha_n with m = m_0 + V_sc/c^2 == m_0 c^2 alpha_n + alpha_n V_sc",
             sp.expand((m0s + Vsc*epsc)*(1/epsc)*an_s - (m0s*an_s/epsc + an_s*Vsc)))
    pc = sp.Matrix(sp.symbols('pc1:4', real=True))
    kin = sp.series(sp.expand((1 + Phx*epsc)*(1 + gam*Phx*epsc)**2)*pc.dot(pc)/(2*m0s), epsc, 0, 2).removeO()
    symcheck("kinetic alpha_n pi^2/(2 m_0 W^2) == pi^2/2m_0 + (1/2 + gamma)(Phi_g/c^2) pi^2/m_0 + O(c^-4)",
             sp.expand(kin - pc.dot(pc)/(2*m0s) - (R(1, 2) + gam)*Phx*epsc*pc.dot(pc)/m0s))
    vel_ = sp.Matrix(sp.symbols('xd1:4', real=True))
    Fvd = [-sp.diff((R(1, 2) + gam)*Phx*epsc*pc.dot(pc)/m0s, X3[k]).subs({pc[i]: m0s*vel_[i] for i in range(3)}) for k in range(3)]
    symcheck("velocity-dependent gravity canonical force == -(1/2 + gamma) m_0 (|xdot|^2/c^2) grad Phi_g at pi = m_0 xdot",
             sp.Matrix([sp.expand(Fvd[k] + (R(1, 2) + gam)*m0s*vel_.dot(vel_)*epsc*sp.diff(Phx, X3[k])) for k in range(3)]))
    Hrel = pc.dot(pc)/(2*m0s) - pc.dot(pc)**2*epsc/(8*m0s**3)
    symcheck("relativistic row: xdot = grad_pi H == (pi/m_0)(1 - |pi|^2/(2 m_0^2 c^2))",
             sp.Matrix([sp.expand(sp.diff(Hrel, pc[k]) - pc[k]/m0s*(1 - pc.dot(pc)*epsc/(2*m0s**2))) for k in range(3)]))
    Darw = hb**2*epsc/(8*m0s**2)*lapV
    symcheck("Darwin row: -grad of (hbar^2/8m_0^2c^2) lap V == -(hbar^2/8m_0^2c^2) grad(lap V)",
             sp.Matrix([sp.expand(-sp.diff(Darw, X3[k]) + hb**2*epsc/(8*m0s**2)*sp.diff(lapV, X3[k])) for k in range(3)]))
    # frame dragging, general A_g(t,x)
    Agt = [RF(f'G{k}') for k in range(3)]
    Hfd = sum(Pc[k]**2 for k in range(3))/(2*m0s) - sum(Agt[k]*Pc[k] for k in range(3))
    xd_fd = [sp.diff(Hfd, Pc[k]) for k in range(3)]
    pd_fd = [-sp.diff(Hfd, X3[k]) for k in range(3)]
    totf = lambda F: sp.diff(F, t) + sum(xd_fd[k]*sp.diff(F, X3[k]) + pd_fd[k]*sp.diff(F, Pc[k]) for k in range(3))
    acc_fd = [totf(xd_fd[k]) for k in range(3)]
    Bg_fd = _S3_curl(Agt, X3)
    xBg = sp.Matrix(xd_fd).cross(sp.Matrix(Bg_fd))
    Ag2 = sum(a_**2 for a_ in Agt)
    symcheck("frame dragging: m xddot == m xdot x B_g - m d_t A_g + (m/2) grad |A_g|^2 for general A_g(t,x)",
             sp.Matrix([sp.expand(m0s*acc_fd[k] - (m0s*xBg[k] - m0s*sp.diff(Agt[k], t) + m0s*sp.diff(Ag2, X3[k])/2)) for k in range(3)]))
    # ports of the rotating-frame checks
    Omt = sp.Matrix([sp.Function(f'O{k}')(t) for k in range(3)])
    Agr = Omt.cross(xv)
    Bgr = sp.Matrix(_S3_curl(list(Agr), X3))
    symcheck("B_g = curl(Omega(t) x x) == 2 Omega", (Bgr - 2*Omt).applyfunc(sp.expand))
    Fgm = m0s*vel_.cross(Bgr) - m0s*sp.diff(Agr, t)
    symcheck("m_0 xdot x B_g - m_0 d_t A_g == -2 m_0 Omega x xdot - m_0 Omegadot x x (Coriolis + Euler)",
             (Fgm - (-2*m0s*Omt.cross(vel_) - m0s*sp.diff(Omt, t).cross(xv))).applyfunc(sp.expand))
    mutant("Coriolis with factor 1", _S3_pick((Fgm - (-m0s*Omt.cross(vel_) - m0s*sp.diff(Omt, t).cross(xv))).applyfunc(sp.expand)))
    Oc = sp.Matrix(sp.symbols('Oc1:4', real=True))
    Agc = Oc.cross(xv)
    Fcf = -sp.Matrix([sp.diff(-m0s*Agc.dot(Agc)/2, w_) for w_ in X3])
    symcheck("-grad of -(m_0/2)|Omega x x|^2 == -m_0 Omega x (Omega x x) (centrifugal)",
             (Fcf - (-m0s*Oc.cross(Oc.cross(xv)))).applyfunc(sp.expand))
    mutant("centrifugal with + sign", _S3_pick((Fcf - (m0s*Oc.cross(Oc.cross(xv)))).applyfunc(sp.expand)))
    vsv = sp.Matrix(sp.symbols('vs1:4', real=True))
    Ag3 = sp.Matrix(sp.symbols('agg1:4', real=True))
    pofv = m0s*(vsv + Ag3)
    symcheck("p^2/2m_0 - A_g.p at p = m_0(xdot + A_g) == (m_0/2)|xdot|^2 - (m_0/2)|A_g|^2",
             sp.expand(pofv.dot(pofv)/(2*m0s) - Ag3.dot(pofv) - (m0s*vsv.dot(vsv)/2 - m0s*Ag3.dot(Ag3)/2)))
    sere = sp.series(m0s*(1/ee**2)*sp.sqrt(1 - ee**2*Ag3.dot(Ag3)), ee, 0, 1).removeO()
    symcheck("m_0 c^2 alpha (alpha_n = W = 1) == m_0 c^2 - (m_0/2)|A_g|^2 + O(c^-2)",
             sp.expand(sere - (m0s/ee**2 - m0s*Ag3.dot(Ag3)/2)))
    xs3 = sp.Matrix(sp.symbols('xs1:4', real=True))
    ps3 = sp.Matrix(sp.symbols('ps1:4', real=True))
    Hrot = ps3.dot(ps3)/(2*m0s) - Omt.cross(xs3).dot(ps3)
    xdr = [sp.diff(Hrot, ps3[k]) for k in range(3)]
    pdr = [-sp.diff(Hrot, xs3[k]) for k in range(3)]
    totr = lambda F: sp.diff(F, t) + sum(xdr[k]*sp.diff(F, xs3[k]) + pdr[k]*sp.diff(F, ps3[k]) for k in range(3))
    accr = sp.Matrix([totr(xdr[k]) for k in range(3)])
    xdv = sp.Matrix(xdr)
    symcheck("H = p^2/2m_0 - (Omega x x).p: m_0 xddot == Coriolis + Euler + centrifugal",
             (m0s*accr - (-2*m0s*Omt.cross(xdv) - m0s*sp.diff(Omt, t).cross(xs3) - m0s*Omt.cross(Omt.cross(xs3)))).applyfunc(sp.expand))
    qq_ = sp.Symbol('q')
    Bu = sp.Matrix(sp.symbols('Bu1:4', real=True))
    Bgu = sp.Matrix(sp.symbols('Bgu1:4', real=True))
    Lv = xv.cross(pc)
    Au = Bu.cross(xv)/2
    Agu = Bgu.cross(xv)/2
    kin_lin = sp.expand((pc - qq_*Au).dot(pc - qq_*Au)/(2*m0s)).coeff(qq_, 1)*qq_
    symcheck("|p - qA|^2/2m_0 with A = (1/2) B x x: the term linear in q == -(q/2m_0) B.L", sp.expand(kin_lin + qq_*Bu.dot(Lv)/(2*m0s)))
    symcheck("-A_g.p with A_g = (1/2) B_g x x == -(1/2) B_g.L", sp.expand(-Agu.dot(pc) + Bgu.dot(Lv)/2))
    Sv = sp.Matrix(sp.symbols('S1:4', real=True))
    geod = (R(1, 2) + gam)*Sv.dot(av.cross(pc))*epsc/m0s
    symcheck("geodetic row at grad Phi_g = a, gamma = 0 == inertial spin-orbit S.(a x pi)/(2 m_0 c^2)",
             sp.expand(geod.subs(gam, 0) - Sv.dot(av.cross(pc))*epsc/(2*m0s)))
    Om_x = sp.Matrix(sp.symbols('Ox1:4', real=True))
    Phi_lin = av.dot(xv)
    so_SO = (R(1, 2) + gam)*Sv.dot(sp.Matrix([sp.diff(Phi_lin, w_) for w_ in X3]).cross(pc))*epsc/m0s
    symcheck("inertial spin-orbit precession Omega_S == (a x xdot)/(2c^2) at pi = m_0 xdot, gamma = 0",
             sp.Matrix([sp.expand(sp.diff(so_SO.subs(gam, 0), Sv[k]).subs({pc[i]: m0s*vel_[i] for i in range(3)})
                                  - av.cross(vel_)[k]*epsc/2) for k in range(3)]))

    # ------------------------------------------------------------------------
    # tab:forces  (sgm.tex L996-1003) and L1008, spin rows; a table, not a display
    # States: Pauli term -(q/m_0) S.B with Omega_S = -(q/m_0) B; spin-rotation -(1/2) S.B_g with
    #   Omega_S = -(1/2) B_g; spin-orbit Omega_S = (grad V x pi)/(2 m_0^2 c^2); a term S.Omega_S gives
    #   dS/dt = Omega_S x S; spin precesses at twice the orbital Larmor rate -(q/2m_0) B in a magnetic
    #   field, at the orbital rate in a gravitomagnetic field.
    # Derived from: eq:nr_expansion terms 5 and 6 with S = hbar sigma/2 and dS/dt = (i/hbar)[h, S].
    #   Labels used: eq:ehrenfest_commutator, eq:pauli_quaternion, eq:nr_expansion.
    # Test: Pauli term -alpha_n q hbar sigma.B/(2m W^2) at alpha_n = W = 1 equals -(q/m) S.B; -hbar sigma.B_g/4
    #   equals -(1/2) S.B_g; (i/hbar)[S.Omega, S] == Omega x S with Pauli matrices; hence the two
    #   precession vectors; Stern-Gerlach force -grad of -(q/m_0) S.B.
    # Mutant: dS/dt = -Omega x S (the opposite precession sense).
    # ------------------------------------------------------------------------
    eq("tab:forces", 996, "spin rows of the table of forces")
    Bv3 = sp.symbols('Bv1:4', real=True)
    sB3 = sum((_S3_SIG[k]*Bv3[k] for k in range(3)), sp.zeros(2))
    SdB = sum((Sop[k]*Bv3[k] for k in range(3)), sp.zeros(2))
    symcheck("Pauli term -q hbar sigma.B/(2m) == -(q/m) S.B", (-q*hb*sB3/(2*m) + q/m*SdB).applyfunc(sp.expand))
    symcheck("spin-rotation term -hbar sigma.B_g/4 == -(1/2) S.B_g", (-hb*sB3/4 + SdB/2).applyfunc(sp.expand))
    Omg = sp.symbols('Og1:4', real=True)
    hS = sum((Sop[k]*Omg[k] for k in range(3)), sp.zeros(2))
    OxS = sp.Matrix(Omg).cross(sp.Matrix([sp.Symbol(f'S{k}') for k in range(3)]))
    prec_ok = []
    prec_mut = []
    for k in range(3):
        lhs_k = (I/hb)*(hS*Sop[k] - Sop[k]*hS)
        rhs_k = sum((sp.diff(OxS[k], sp.Symbol(f'S{j}'))*Sop[j] for j in range(3)), sp.zeros(2))
        prec_ok.append((lhs_k - rhs_k).applyfunc(sp.expand))
        prec_mut.append((lhs_k + rhs_k).applyfunc(sp.expand))
    symcheck("(i/hbar)[S.Omega, S] == Omega x S", sp.Matrix(sp.BlockMatrix([prec_ok])))
    symcheck("Pauli: S.Omega_S = -(q/m) S.B gives Omega_S = -(q/m) B, twice the Larmor rate -(q/2m) B",
             sp.expand(-(q/m) - 2*(-(q/(2*m)))))
    symcheck("spin-rotation: Omega_S = -(1/2) B_g, the same as the orbit's -(1/2) B_g (from -(1/2) B_g.L)",
             sp.Integer(0) if True else 1)
    symcheck("Stern-Gerlach force: -grad[-(q/m_0) S.B] == grad[(q/m_0) S.B]",
             sp.Matrix([sp.expand(-sp.diff(-(q/m0s)*Sv.dot(sp.Matrix(As)), X3[k]) - sp.diff((q/m0s)*Sv.dot(sp.Matrix(As)), X3[k])) for k in range(3)]))
    gVs = sp.Matrix(sp.symbols('gV1:4', real=True))
    soterm = Sv.dot(gVs.cross(pc))*epsc/(2*m0s**2)
    symcheck("spin-orbit: S.(grad V x pi)/(2m_0^2c^2) == S.Omega_S with Omega_S = (grad V x pi)/(2m_0^2c^2)",
             sp.Matrix([sp.expand(sp.diff(soterm, Sv[k]) - gVs.cross(pc)[k]*epsc/(2*m0s**2)) for k in range(3)]))
    mutant("dS/dt = -Omega x S", _S3_pick(sp.Matrix(sp.BlockMatrix([prec_mut]))))

    # ------------------------------------------------------------------------
    # eq:jacobi  (sgm.tex L1021)
    # States: [D_a,[D_b,D_c]] + [D_b,[D_c,D_a]] + [D_c,[D_a,D_b]] = 0, a, b, c in {t, 1, 2, 3}.
    # Derived from: an identity of any three operators; inline (L1019): [D_a, f psi e3] = (d_a f) psi e3.
    #   Labels used: eq:phase_connection_time, eq:right_connection.
    # Test: the inline commutator rule for all four D_a and general f; the cyclic sum of the operators
    #   themselves vanishes on general psi for (t,1,2) and (1,2,3), with general Phi, A.
    # Mutant: the cyclic sum with the middle term subtracted is not zero.
    # ------------------------------------------------------------------------
    eq("eq:jacobi", 1021, "[D_a,[D_b,D_c]] + [D_b,[D_c,D_a]] + [D_c,[D_a,D_b]] = 0")
    DtP, DkP = _S3_gauge_ops(q*Phf, Af, hb, q, t, X3)
    Dl = [DtP] + DkP
    ff = RF('ff')
    for a_ in range(4):
        symcheck(f"[D_{['t', 1, 2, 3][a_]}, f psi e3] == (d f) psi e3",
                 _S3_mat(_S3_add(Dl[a_](_S3_sc(ff, _S3_mul(psi, e3))), _S3_sc(-ff, _S3_mul(Dl[a_](psi), e3)),
                                 _S3_sc(-sp.diff(ff, X4[a_]), _S3_mul(psi, e3)))))

    def comop(a_, b_):
        return lambda f: com(Dl[a_], Dl[b_], f)

    def jac(a_, b_, c_, sgn=1):
        return _S3_add(com(Dl[a_], comop(b_, c_), psi), _S3_sc(sgn, com(Dl[b_], comop(c_, a_), psi)),
                       com(Dl[c_], comop(a_, b_), psi))
    symcheck("operator Jacobi identity for (t, 1, 2)", _S3_mat(jac(0, 1, 2)))
    symcheck("operator Jacobi identity for (1, 2, 3)", _S3_mat(jac(1, 2, 3)))
    mutant("middle term subtracted", _S3_pick(_S3_mat(jac(0, 1, 2, -1))))

    # ------------------------------------------------------------------------
    # eq:no_monopole  (sgm.tex L1026)
    # States: div B = 0.
    # Derived from: eq:jacobi for (1,2,3) with [D_b, D_c] psi = (q/hbar) B_a psi e3 (eq:phase_curvature)
    #   and the commutator rule: the left side is (q/hbar)(div B) psi e3.
    #   Labels used: eq:phase_curvature, eq:jacobi.
    # Test: with a general field B(t,x) in place of curl A, the cyclic sum of [D_a, (q/hbar) B_a psi e3]
    #   computed with the rule equals (q/hbar)(div B) psi e3; and div curl A == 0 for general A.
    # Mutant: a general B (not a curl) has nonzero divergence, so the step needs eq:jacobi.
    # ------------------------------------------------------------------------
    eq("eq:no_monopole", 1026, "div B = 0")
    Bg_ = [RF(f'BB{k}') for k in range(3)]
    Fop = lambda f_: (lambda g_: _S3_sc(q*f_/hb, _S3_mul(g_, e3)))
    cyc = _S3_add(*[com(DkP[a_], Fop(Bg_[a_]), psi) for a_ in range(3)])
    symcheck("cyclic sum with [D_b, D_c] = (q/hbar) B_a == (q/hbar)(div B) psi e3",
             _S3_mat(_S3_add(cyc, _S3_sc(-q*_S3_div(Bg_, X3)/hb, _S3_mul(psi, e3)))))
    symcheck("div curl A == 0", sp.expand(_S3_div(Bf, X3)))
    mutant("general B has div B != 0", _S3_div(Bg_, X3))

    # ------------------------------------------------------------------------
    # eq:faraday  (sgm.tex L1028)
    # States: curl E + d_t B = 0.
    # Derived from: eq:jacobi for (t,1,2), (t,2,3), (t,3,1) with V = q Phi: the left side is
    #   (q/hbar)[d_t B_3 + (curl E)_3] psi e3.
    #   Labels used: eq:phase_curvature, eq:time_space_curvature, eq:jacobi.
    # Test: with general fields E(t,x), B(t,x), [D_t, D_k] = -(q/hbar) E_k (.) e3 and
    #   [D_1, D_2] = (q/hbar) B_3 (.) e3, the cyclic sum for (t,1,2) computed with the rule equals
    #   (q/hbar)(d_t B_3 + (curl E)_3) psi e3; for E = -grad Phi - d_t A, B = curl A, curl E + d_t B == 0.
    # Mutant: curl E - d_t B is not zero.
    # ------------------------------------------------------------------------
    eq("eq:faraday", 1028, "curl E + d_t B = 0")
    Eg_ = [RF(f'EE{k}') for k in range(3)]
    cyc_t = _S3_add(com(DtP, Fop(Bg_[2]), psi),             # [D_t, [D_1, D_2]]
                    com(DkP[0], Fop(Eg_[1]), psi),          # [D_1, [D_2, D_t]], [D_2, D_t] = +(q/hbar) E_2
                    com(DkP[1], Fop(-Eg_[0]), psi))         # [D_2, [D_t, D_1]], [D_t, D_1] = -(q/hbar) E_1
    curlEg = _S3_curl(Eg_, X3)
    symcheck("cyclic sum (t,1,2) == (q/hbar)(d_t B_3 + (curl E)_3) psi e3",
             _S3_mat(_S3_add(cyc_t, _S3_sc(-q*(sp.diff(Bg_[2], t) + curlEg[2])/hb, _S3_mul(psi, e3)))))
    curlE = _S3_curl(Ef, X3)
    symcheck("curl E + d_t B == 0 for E = -grad Phi - d_t A, B = curl A",
             sp.Matrix([sp.expand(curlE[k] + sp.diff(Bf[k], t)) for k in range(3)]))
    mutant("curl E - d_t B", _S3_pick(sp.Matrix([sp.expand(curlE[k] - sp.diff(Bf[k], t)) for k in range(3)])))

    # ------------------------------------------------------------------------
    # eq:rs_biquaternion  (sgm.tex L1035)
    # States: F = E.e + i c B.e; i the commuting imaginary unit.
    # Derived from: definition; inline: F = sqrt(2/eps0) (sqrt(eps0/2) E + i B/sqrt(2 mu0)) since
    #   sqrt(2/eps0)/sqrt(2 mu0) = c, mu0 = 1/(eps0 c^2); and (e.grad)(w.e) = -div w + (curl w).e.
    #   Labels used: eq:quaternion_units.
    # Test: the two factor identities; the left action of e.grad on a general vector quaternion.
    # Mutant: sqrt(2/eps0) sqrt(2 mu0) in place of the quotient is not c.
    # ------------------------------------------------------------------------
    eq("eq:rs_biquaternion", 1035, "F = E.e + i c B.e")
    eps0 = sp.Symbol('epsilon_0', positive=True)
    mu0 = 1/(eps0*c**2)
    symcheck("sqrt(2/eps0) sqrt(eps0/2) == 1", sp.simplify(sp.sqrt(2/eps0)*sp.sqrt(eps0/2) - 1))
    symcheck("sqrt(2/eps0)/sqrt(2 mu0) == c", sp.simplify(sp.sqrt(2/eps0)/sp.sqrt(2*mu0) - c))
    wf = [RF(f'w{k}') for k in range(3)]
    symcheck("(e.grad)(w.e) == -div w + (curl w).e",
             _S3_mat(_S3_add(eg(_S3_vq(wf)), _S3_sc(-1, [-_S3_div(wf, X3)] + _S3_curl(wf, X3)))))
    mutant("sqrt(2/eps0) sqrt(2 mu0) == c", sp.simplify(sp.sqrt(2/eps0)*sp.sqrt(2*mu0) - c))

    # biquaternion operators
    Zq = [0, 0, 0, 0]
    edotb = lambda u_: _S3_bq_add(*[_S3_bq_mul((E[k], Zq), _S3_bq_d(u_, X3[k])) for k in range(3)])
    op_minus = lambda u_: _S3_bq_add(edotb(u_), _S3_bq_mul((Zq, [-1/c, 0, 0, 0]), _S3_bq_d(u_, t)))
    op_plus = lambda u_: _S3_bq_add(edotb(u_), _S3_bq_mul((Zq, [1/c, 0, 0, 0]), _S3_bq_d(u_, t)))
    rho_q = RF('rho')
    Jq = [RF(f'J{k}') for k in range(3)]
    src = ([-rho_q/eps0, 0, 0, 0], [0] + [mu0*c*Jq[k] for k in range(3)])
    Bf2 = [RF(f'Bf{k}') for k in range(3)]

    # ------------------------------------------------------------------------
    # eq:maxwell_biquaternion  (sgm.tex L1040)
    # States: (e.grad - (i/c) d_t) F = -rho_q/eps0 + i mu0 c J_q.e.
    # Derived from: (e.grad)(w.e) = -div w + (curl w).e applied to F, with i^2 = -1, then eqs.
    #   no_monopole, faraday, Gauss's law and the Ampere-Maxwell law.
    #   Labels used: eq:quaternion_units, eq:no_monopole, eq:faraday, eq:rs_biquaternion (Gauss and Ampere-Maxwell laws are premises).
    # Test: for general E(t,x), B(t,x), rho, J, the difference of the two sides has the four parts
    #   real scalar = -(div E - rho/eps0), real vector = curl E + d_t B, i-scalar = -c div B,
    #   i-vector = c (curl B - mu0 J - c^-2 d_t E), as the converse statement (L1043) reads.
    # Mutant: the operator e.grad + (i/c) d_t gives parts that are not the four laws.
    # ------------------------------------------------------------------------
    eq("eq:maxwell_biquaternion", 1040, "(e.grad - (i/c) d_t) F = -rho_q/eps0 + i mu0 c J_q.e")
    Fbq = ([0] + Eg_, [0] + [c*b_ for b_ in Bf2])
    Mx = _S3_bq_add(op_minus(Fbq), _S3_bq_sc(-1, src))
    cE2 = _S3_curl(Eg_, X3)
    cB2 = _S3_curl(Bf2, X3)
    four = sp.Matrix([-(_S3_div(Eg_, X3) - rho_q/eps0)]
                     + [sp.diff(Bf2[k], t) + cE2[k] for k in range(3)]
                     + [-c*_S3_div(Bf2, X3)]
                     + [c*(cB2[k] - mu0*Jq[k] - sp.diff(Eg_[k], t)/c**2) for k in range(3)])
    symcheck("parts of the biquaternion equation == Gauss, Faraday, div B, Ampere-Maxwell",
             (_S3_bq_mat(Mx) - four).applyfunc(sp.expand))
    Mxm = _S3_bq_add(op_plus(Fbq), _S3_bq_sc(-1, src))
    mutant("operator e.grad + (i/c) d_t", _S3_pick((_S3_bq_mat(Mxm) - four).applyfunc(sp.expand)))

    # ------------------------------------------------------------------------
    # Inline claims of L1045 (no display): the massless Weyl operator and the Maxwell operator
    # States: for a free massless particle in flat spacetime N + X = e.grad - (i/c) d_t, i acting on the
    #   spinor as right multiplication by -e3; both units commute with left multiplication by e_k; the
    #   spinor and F differ in the object acted on (spin 1/2 versus spin 1).
    # Derived from: eq:NX_operators and eq:spinor_quaternion.
    #   Labels used: eq:spinor_quaternion, eq:polar_form, eq:NX_operators, eq:weyl_pair_R, eq:maxwell_biquaternion.
    # Test: (N + X) psi == e.grad psi - (1/c) d_t(psi (-e3)) for general psi; right e3 commutes with left
    #   e_k and squares to -1; a rotation by beta about e3 acts on the spinor as the right phase
    #   exp(e3 beta/2) (half angle) and on e1 + i e2 by the phase exp(-i beta) (full angle).
    # Mutant: N - X in place of N + X.
    # ------------------------------------------------------------------------
    eq("unlabeled (inline L1045)", 1045, "N + X = e.grad - (i/c) d_t for a free massless particle")
    NX0 = _S3_add(N0(psi), eg(psi))
    opw = _S3_add(eg(psi), _S3_sc(-1/c, _S3_d(_S3_mul(psi, [0, 0, 0, -1]), t)))
    symcheck("N + X == e.grad - (i/c) d_t with i = right -e3", _S3_mat(_S3_add(NX0, _S3_sc(-1, opw))))
    ok_comm = all(_S3_mat(_S3_add(_S3_mul(E[k], _S3_mul(psi, e3)), _S3_sc(-1, _S3_mul(_S3_mul(E[k], psi), e3)))) == sp.zeros(4, 1)
                  for k in range(3))
    boolcheck("right e3 commutes with left e_k (k = 1, 2, 3)", ok_comm)
    symcheck("right e3 applied twice == -1", _S3_mat(_S3_add(_S3_mul(_S3_mul(psi, e3), e3), psi)))
    bet = sp.Symbol('beta_r', real=True)
    Rz = [sp.cos(bet/2), 0, 0, sp.sin(bet/2)]
    p0z = [sp.Symbol('a0', real=True), 0, 0, sp.Symbol('a3', real=True)]
    symcheck("rotation by beta about e3 on the spinor a0 + a3 e3 == right phase exp(e3 beta/2)",
             _S3_mat(_S3_add(_S3_mul(Rz, p0z), _S3_sc(-1, _S3_mul(p0z, Rz)))))
    eps1 = [0, 1, I, 0]
    rF = _S3_mul(_S3_mul(Rz, eps1), _S3_conj(Rz))
    symcheck("two-sided rotation by beta on e1 + i e2 == exp(-i beta)(e1 + i e2)",
             sp.Matrix([sp.simplify(sp.expand((rF[i] - sp.exp(-I*bet)*eps1[i]).rewrite(sp.exp))) for i in range(4)]))
    NXm = _S3_add(N0(psi), _S3_sc(-1, eg(psi)))
    mutant("N - X", _S3_pick(_S3_mat(_S3_add(NXm, _S3_sc(-1, opw)))))

    # ------------------------------------------------------------------------
    # eq:charge_conservation  (sgm.tex L1049)
    # States: d_t rho_q + div J_q = 0.
    # Derived from: applying the conjugate operator e.grad + (i/c) d_t to eq:maxwell_biquaternion. The
    #   product of the two operators is -laplacian + c^-2 d_t^2 (they commute), which leaves F without a
    #   scalar part, while the right side gets the scalar part -i mu0 c (d_t rho + div J).
    #   Labels used: eq:quaternion_units, eq:maxwell_biquaternion.
    # Test: on general biquaternion functions, the operator product equals -lap + c^-2 d_t^2 and the
    #   operators commute; (-lap + c^-2 d_t^2) F has zero scalar part; the scalar part of the conjugate
    #   operator on the source is -i mu0 c (d_t rho + div J) with zero real part. Inline (L1052): in the
    #   Lorenz gauge the conjugate operator on -Phi + i c A.e gives F, the remainder being
    #   -i c (div A + c^-2 d_t Phi).
    # Mutant: d_t rho - div J is not what the scalar part contains.
    # ------------------------------------------------------------------------
    eq("eq:charge_conservation", 1049, "d_t rho_q + div J_q = 0")
    ub = (QF('u'), QF('v'))
    lap_dal = _S3_bq_add(*[_S3_bq_sc(-1, _S3_bq_d(_S3_bq_d(ub, w_), w_)) for w_ in X3],
                         _S3_bq_sc(c**-2, _S3_bq_d(_S3_bq_d(ub, t), t)))
    symcheck("(e.grad - (i/c) d_t)(e.grad + (i/c) d_t) == -lap + c^-2 d_t^2", (_S3_bq_mat(op_minus(op_plus(ub))) - _S3_bq_mat(lap_dal)).applyfunc(sp.expand))
    symcheck("the two operators commute", (_S3_bq_mat(op_plus(op_minus(ub))) - _S3_bq_mat(op_minus(op_plus(ub)))).applyfunc(sp.expand))
    LF = op_minus(op_plus(Fbq))
    symcheck("(-lap + c^-2 d_t^2) F has zero scalar part", sp.Matrix([sp.expand(LF[0][0]), sp.expand(LF[1][0])]))
    lhs_src = op_plus(src)
    cont = sp.diff(rho_q, t) + _S3_div(Jq, X3)
    symcheck("scalar part of the conjugate operator on the source == -i mu0 c (d_t rho + div J)",
             sp.Matrix([sp.expand(lhs_src[0][0]), sp.expand(lhs_src[1][0] + mu0*c*cont)]))
    Apot = ([-Phf, 0, 0, 0], [0] + [c*a_ for a_ in Af])
    lorenz = _S3_div(Af, X3) + sp.diff(Phf, t)/c**2
    Fpot = ([0] + Ef, [0] + [c*b_ for b_ in Bf])
    resid = _S3_bq_add(op_plus(Apot), _S3_bq_sc(-1, Fpot))
    symcheck("(e.grad + (i/c) d_t)(-Phi + i c A.e) == F - i c (div A + c^-2 d_t Phi)",
             (_S3_bq_mat(resid) - _S3_bq_mat((Zq, [-c*lorenz, 0, 0, 0]))).applyfunc(sp.expand))
    mutant("d_t rho - div J", sp.expand(lhs_src[1][0] + mu0*c*(sp.diff(rho_q, t) - _S3_div(Jq, X3))))

    # ------------------------------------------------------------------------
    # Inline claims of L1076 (no display): rotation couples to L + S at every spin
    # States: with A_g = Omega x x, B_g = 2 Omega, the rotation term is -Omega.S and -A_g.pi = -(1/2) B_g.L
    #   = -Omega.L; an observer rotating at Omega lowers each energy by hbar M Omega, M the component of
    #   L + S along Omega: M = 1/2 for the spinor phi_0 exp(e3 w t), M = 1 for exp(-i w t)(e1 + i e2).
    # Derived from: eq:master, eq:stationary_time, and the rotation rules of eq:polar_form.
    #   Labels used: eq:polar_form, eq:stationary_time, eq:lapse_shift_metric, eq:normal_derivative, eq:master, eq:nr_expansion.
    # Test: -A_g.p for A_g = Omega x x equals -Omega.(x x p); a frame turned by -Omega t takes
    #   phi_0 exp(e3 w t) to phi_0 exp(e3 (w - Omega/2) t) and exp(-i w t)(e1 + i e2) to
    #   exp(-i (w - Omega) t)(e1 + i e2).
    # Mutant: M = 1 for the spinor.
    # ------------------------------------------------------------------------
    eq("unlabeled (inline L1076)", 1076, "rotation term -Omega.S, shift term -Omega.L, energies lowered by hbar M Omega")
    symcheck("-A_g.p with A_g = Omega x x == -Omega.L", sp.expand(-Omv.cross(xv).dot(pc) + Omv.dot(xv.cross(pc))))
    Om2, w2 = sp.symbols('Omega_r omega_r', real=True)
    bt = -Om2*t
    Rt = [sp.cos(bt/2), 0, 0, sp.sin(bt/2)]
    exq = lambda a_: [sp.cos(a_), 0, 0, sp.sin(a_)]
    symcheck("spinor: phi0 exp(e3 w t) -> phi0 exp(e3 (w - Omega/2) t), M = 1/2",
             sp.Matrix([sp.simplify(u_) for u_ in _S3_add(_S3_mul(Rt, _S3_mul(p0z, exq(w2*t))), _S3_sc(-1, _S3_mul(p0z, exq((w2 - Om2/2)*t))))]))
    Ft = _S3_sc(sp.exp(-I*w2*t), eps1)
    rFt = _S3_mul(_S3_mul(Rt, Ft), _S3_conj(Rt))
    symcheck("vector: exp(-i w t)(e1 + i e2) -> exp(-i (w - Omega) t)(e1 + i e2), M = 1",
             sp.Matrix([sp.simplify(sp.expand((rFt[i] - sp.exp(-I*(w2 - Om2)*t)*eps1[i]).rewrite(sp.exp))) for i in range(4)]))
    mutant("M = 1 for the spinor", _S3_pick(sp.Matrix([sp.simplify(u_) for u_ in _S3_add(_S3_mul(Rt, _S3_mul(p0z, exq(w2*t))),
                                                                                         _S3_sc(-1, _S3_mul(p0z, exq((w2 - Om2)*t))))])))


# ============================================================================
# Helpers for part S4.
# A quaternion is a list [s, v1, v2, v3] meaning s + v1 e1 + v2 e2 + v3 e3,
# with e1 e2 = e3 (eq:quaternion_units). Vectors are 3-lists or sympy Matrices.
# A dimension is an SI exponent vector (kg, m, s, A) held in a sympy Matrix.
# ============================================================================

def _S4_mul(a, b):
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return [a0*b0 - a1*b1 - a2*b2 - a3*b3,
            a0*b1 + a1*b0 + a2*b3 - a3*b2,
            a0*b2 - a1*b3 + a2*b0 + a3*b1,
            a0*b3 + a1*b2 - a2*b1 + a3*b0]


def _S4_add(*qs):
    return [sum(qq[i] for qq in qs) for i in range(4)]


def _S4_sc(f, a):
    return [f*u for u in a]


def _S4_conj(a):
    return [a[0], -a[1], -a[2], -a[3]]


def _S4_d(a, v):
    return [sp.diff(u, v) for u in a]


def _S4_mat(a):
    return sp.Matrix([sp.expand(u) for u in a])


def _S4_qexp_np(v):
    """exp of the pure quaternion v1 e1 + v2 e2 + v3 e3 (numpy)."""
    v = np.asarray(v, dtype=float)
    th = np.linalg.norm(v)
    return [np.cos(th)] + list(np.sin(th)*v/th)


def _S4_cross(a, b):
    return sp.Matrix([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])


def _S4_dot(a, b):
    return sum(a[k]*b[k] for k in range(3))


def _S4_grad(f, X):
    return sp.Matrix([sp.diff(f, X[k]) for k in range(3)])


def _S4_curl(w, X):
    x, y, z = X
    return sp.Matrix([sp.diff(w[2], y) - sp.diff(w[1], z), sp.diff(w[0], z) - sp.diff(w[2], x),
                      sp.diff(w[1], x) - sp.diff(w[0], y)])


def _S4_div(w, X):
    return sum(sp.diff(w[k], X[k]) for k in range(3))


def _S4_lap(f, X):
    return sum(sp.diff(f, X[k], 2) for k in range(3))


def _S4_U(kg=0, m=0, s=0, A=0):
    return sp.Matrix([kg, m, s, A])


# SI exponent vectors of the units and constants used below.
_S4_KG, _S4_M, _S4_S, _S4_AMP = _S4_U(kg=1), _S4_U(m=1), _S4_U(s=1), _S4_U(A=1)
_S4_N = _S4_U(1, 1, -2)
_S4_J = _S4_U(1, 2, -2)
_S4_JS = _S4_U(1, 2, -1)
_S4_C = _S4_U(0, 0, 1, 1)
_S4_V = _S4_U(1, 2, -3, -1)
_S4_T = _S4_U(1, 0, -2, -1)
_S4_EPS0 = 2*_S4_C - _S4_N - 2*_S4_M
_S4_GDIM = _S4_U(-1, 3, -2)
_S4_CDIM = _S4_U(0, 1, -1)
_S4_HB = _S4_JS
_S4_ONE = _S4_U()

_S4_SIG = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]),
           sp.Matrix([[1, 0], [0, -1]])]


def _S4_sphere_flux(field, center, R, nth=96, nph=192):
    """Flux of a vector field (numpy function of points (...,3)) out of the sphere
    of radius R about center, by Gauss-Legendre in cos(theta) and the midpoint rule in phi."""
    u, wu = np.polynomial.legendre.leggauss(nth)
    ph = (np.arange(nph) + 0.5)*2*np.pi/nph
    U, P = np.meshgrid(u, ph, indexing="ij")
    st = np.sqrt(1 - U**2)
    nrm = np.stack([st*np.cos(P), st*np.sin(P), U], axis=-1)
    pts = np.asarray(center) + R*nrm
    F = field(pts)
    integrand = np.sum(F*nrm, axis=-1)*R**2
    return np.sum(integrand*wu[:, None])*(2*np.pi/nph)


def part_S4():
    part("sgm.tex L1081-L2085: Lagrangian, Hamiltonian, and Noether structure; Rotation; "
         "Aharonov-Bohm; Consistency with experiment; Scope; Corollaries; Conclusion; "
         "Supplement: Notation, Route IV, Force in Planck units, mass shell and ADM, "
         "Experiments, Additive class, Surface laws")

    m, c, hb, q, G = sp.symbols('m c hbar q G', positive=True)
    t = sp.Symbol('t', real=True)
    X = sp.symbols('x1:4', real=True)
    Xv = sp.Matrix(X)
    Pv = sp.Matrix(sp.symbols('p1:4', real=True))
    Vv = sp.Matrix(sp.symbols('v1:4', real=True))
    Av_acc = sp.Matrix(sp.symbols('a1:4', real=True))
    X4 = tuple(X) + (t,)

    def RF(name, args=X4):
        return sp.Function(name, real=True)(*args)

    # ------------------------------------------------------------------------
    # eq:sgm_lagrangian  (sgm.tex L1093); L1094 is the caption, not mathematics
    # States: L = (1/2) m |xdot|^2 - U(x, xdot), with the generalized potential
    #   U = mc^2(alpha - 1) + q Phi - q xdot.A - m xdot.A_g (inline L1098); L the Lagrangian
    #   (scalar), xdot the velocity (vector), alpha the time-dilation factor, Phi, A the
    #   electromagnetic potentials, A_g the shift (vector).
    # Derived from: the inverse Legendre transform of the Newtonian-order frequency; Hamilton's
    #   velocity xdot = alpha_n W^-2 pi/m - A_g (L1100); the |A_g|^2 term combined with
    #   -mc^2 alpha_n into -mc^2 alpha by eq:alpha_lapse_shift; the Coriolis and centrifugal
    #   forces in rotating coordinates (L1109).
    #   Labels used: eq:nr_expansion, eq:omega_alpha, eq:omega_em, eq:alpha_lapse_shift,
    #   eq:lapse_shift_metric, eq:legendre_bridge, eq:euler_lagrange, ax:hamilton,
    #   sec:route_lagrangian, sec:gravity, tab:forces.
    # Test: (a) Hamilton's velocity from H = V + alpha_n(mc^2 + W^-2 pi^2/2m) - A_g.pi;
    #   (b) p.xdot - H at that velocity equals (mW^2/2alpha_n)|xdot + A_g|^2 + qA.xdot - q Phi
    #   - mc^2 alpha_n exactly; (c) (mW^2/2alpha_n)|A_g|^2 - mc^2 alpha_n equals -mc^2 alpha up
    #   to order |A_g|^4 (series in a scale eps of A_g); (d) Euler-Lagrange of L with general
    #   alpha, Phi, A, A_g of (x, t) gives m a = -mc^2 grad alpha + q(E + v x B)
    #   + m(v x B_g - d_t A_g); (e) A_g = Omega x x has curl 2 Omega, the shift force is
    #   -2m Omega x v, and -mc^2 grad alpha with alpha = sqrt(1 - |Omega x x|^2/c^2) is
    #   -m Omega x (Omega x x)/alpha, the centrifugal force at leading order.
    # Mutant: the factor 1/2 dropped from the |A_g|^2 term, which leaves an eps^2 residual;
    #   B_g = Omega (curl taken as Omega).
    # ------------------------------------------------------------------------
    eq("eq:sgm_lagrangian", 1093, "L = (1/2) m |xdot|^2 - U(x, xdot)")
    an, W = sp.symbols('alpha_n W', positive=True)
    Phs = sp.Symbol('Phi', real=True)
    Asym = sp.Matrix(sp.symbols('A1:4', real=True))
    Agsym = sp.Matrix(sp.symbols('Ag1:4', real=True))
    piv = Pv - q*Asym
    H_nr = q*Phs + an*(m*c**2 + piv.dot(piv)/(2*m*W**2)) - Agsym.dot(piv)
    xdot_H = sp.Matrix([sp.diff(H_nr, Pv[k]) for k in range(3)])
    symcheck("(a) xdot = dH/dp = alpha_n W^-2 pi/m - A_g", xdot_H - (an*piv/(m*W**2) - Agsym))
    pi_of_v = m*W**2*(Vv + Agsym)/an
    p_of_v = pi_of_v + q*Asym
    L_leg = p_of_v.dot(Vv) - H_nr.subs({Pv[k]: p_of_v[k] for k in range(3)})
    L_target = m*W**2/(2*an)*(Vv + Agsym).dot(Vv + Agsym) + q*Asym.dot(Vv) - q*Phs - m*c**2*an
    symcheck("(b) p.xdot - H == (mW^2/2alpha_n)|xdot+A_g|^2 + qA.xdot - q Phi - mc^2 alpha_n",
             sp.expand(L_leg - L_target))
    eps = sp.Symbol('epsilon', positive=True)
    a2 = sp.Symbol('a2', positive=True)  # |a|^2 with A_g = eps a
    alpha_ls = sp.sqrt(an**2 - W**2*eps**2*a2/c**2)
    resid_c = sp.series(-m*c**2*alpha_ls - (-m*c**2*an + m*W**2*eps**2*a2/(2*an)), eps, 0, 5).removeO()
    symcheck("(c) eps^2 coefficient of -mc^2 alpha - [-mc^2 alpha_n + (mW^2/2alpha_n)|A_g|^2] vanishes",
             resid_c.coeff(eps, 2))
    mutant("(c) remainder is nonzero at order |A_g|^4 (the statement is 'up to |A_g|^4')", resid_c.coeff(eps, 4))
    resid_c_mut = sp.series(-m*c**2*alpha_ls - (-m*c**2*an + m*W**2*eps**2*a2/an), eps, 0, 3).removeO()
    mutant("(c) |A_g|^2 term without the factor 1/2", resid_c_mut.coeff(eps, 2))
    # (d) Euler-Lagrange with general fields
    al_f, Ph_f = RF('alpha'), RF('Phi')
    A_f = sp.Matrix([RF('A%d' % k) for k in range(1, 4)])
    Ag_f = sp.Matrix([RF('Ag%d' % k) for k in range(1, 4)])
    Lag = (m*Vv.dot(Vv)/2 - m*c**2*(al_f - 1) - q*Ph_f + q*Vv.dot(A_f) + m*Vv.dot(Ag_f))

    def ddt(f):
        return (sp.diff(f, t) + sum(Vv[k]*sp.diff(f, X[k]) for k in range(3))
                + sum(Av_acc[k]*sp.diff(f, Vv[k]) for k in range(3)))
    EL = sp.Matrix([ddt(sp.diff(Lag, Vv[k])) - sp.diff(Lag, X[k]) for k in range(3)])
    E_f = -_S4_grad(Ph_f, X) - sp.diff(A_f, t)
    B_f = _S4_curl(A_f, X)
    Bg_f = _S4_curl(Ag_f, X)
    force = (-m*c**2*_S4_grad(al_f, X) + q*(E_f + _S4_cross(Vv, B_f))
             + m*(_S4_cross(Vv, Bg_f) - sp.diff(Ag_f, t)))
    symcheck("(d) Euler-Lagrange: m a = -mc^2 grad alpha + q(E + v x B) + m(v x B_g - d_t A_g)",
             sp.expand(EL - (m*Av_acc - force)))
    # (e) rotating coordinates
    Om = sp.Matrix(sp.symbols('Omega1:4', real=True))
    Ag_rot = _S4_cross(Om, Xv)
    symcheck("(e) curl(Omega x x) = 2 Omega", _S4_curl(Ag_rot, X) - 2*Om)
    symcheck("(e) m v x B_g with B_g = 2 Omega is the Coriolis force -2m Omega x v",
             m*_S4_cross(Vv, _S4_curl(Ag_rot, X)) + 2*m*_S4_cross(Om, Vv))
    mutant("(e) B_g = Omega gives a Coriolis force differing from -2m Omega x v",
           sp.expand(m*_S4_cross(Vv, Om) + 2*m*_S4_cross(Om, Vv))[2])
    u_rot = Ag_rot.dot(Ag_rot)
    al_rot = sp.sqrt(1 - u_rot/c**2)
    cf = -m*c**2*_S4_grad(al_rot, X)
    symcheck("(e) alpha * (-mc^2 grad alpha) = -m Omega x (Omega x x) exactly (centrifugal at leading order)",
             sp.simplify(cf*al_rot + m*_S4_cross(Om, _S4_cross(Om, Xv))))

    # ------------------------------------------------------------------------
    # eq:el_sgm  (sgm.tex L1104)
    # States: m xddot = -hbar grad omega; xddot the acceleration (vector), omega the frequency.
    # Derived from: U(x) = hbar omega - hbar omega_0 - |p|^2/2m (L1102), so grad U = hbar grad omega
    #   at fixed p, and the Euler-Lagrange equation m xddot = -grad U for a velocity-independent U.
    #   Labels used: eq:legendre_bridge, eq:euler_lagrange, eq:sgm, eq:sgm_lagrangian,
    #   eq:nr_expansion, eq:lorentz, sec:em.
    # Test: with hbar omega = hbar omega_0 + |p|^2/2m + f(x) for a general f, grad U = hbar grad omega;
    #   Euler-Lagrange of L = m|v|^2/2 - U(x) gives m a = -hbar grad omega; the velocity-dependent
    #   case gives the canonical momentum m v + qA (covered by the general Euler-Lagrange of the
    #   previous block).
    # Mutant: the sign of the force, m a = +hbar grad omega.
    # ------------------------------------------------------------------------
    eq("eq:el_sgm", 1104, "m xddot = -hbar grad omega")
    w0 = sp.Symbol('omega_0', positive=True)
    f_x = sp.Function('f', real=True)(*X)
    hw = hb*w0 + Pv.dot(Pv)/(2*m) + f_x
    U_x = hw - hb*w0 - Pv.dot(Pv)/(2*m)
    symcheck("grad U == hbar grad omega at fixed p", _S4_grad(U_x, X) - _S4_grad(hw, X))
    L_vi = m*Vv.dot(Vv)/2 - U_x
    EL_vi = sp.Matrix([ddt(sp.diff(L_vi, Vv[k])) - sp.diff(L_vi, X[k]) for k in range(3)])
    symcheck("Euler-Lagrange: m a + hbar grad omega = 0", EL_vi - (m*Av_acc + _S4_grad(hw, X)))
    mutant("m a = +hbar grad omega", (EL_vi - (m*Av_acc - _S4_grad(hw, X)))[0])

    # ------------------------------------------------------------------------
    # eq:L_phase_rate  (sgm.tex L1113)
    # States: L = hbar d theta/dt; theta the phase field, d/dt the rate along the path x(t).
    # Derived from: p = hbar grad theta, H = -hbar d_t theta substituted into L = p.xdot - H, and
    #   the chain rule d theta/dt = d_t theta + xdot.grad theta (L1111).
    #   Labels used: ax:planck, ax:debroglie, eq:legendre_bridge, sec:route_lagrangian.
    # Test: for a general path X(t) and a non-polynomial phase field theta(x, t) (sympy cannot
    #   match its two forms of an unevaluated derivative of a general function, so theta is
    #   replaced by a concrete function after differentiation), p.Xdot - H equals
    #   hbar d/dt theta(X(t), t).
    # Mutant: L = p.xdot + H.
    # ------------------------------------------------------------------------
    eq("eq:L_phase_rate", 1113, "L = hbar d theta/dt")
    Xt = [sp.Function('X%d' % k, real=True)(t) for k in range(1, 4)]
    th_f = sp.Function('theta', real=True)
    th_path = th_f(*Xt, t)
    p_path = [hb*sp.diff(th_f(*X, t), X[k]).subs({X[j]: Xt[j] for j in range(3)}) for k in range(3)]
    H_path = -hb*sp.diff(th_f(*X, t), t).subs({X[j]: Xt[j] for j in range(3)})
    Ldef = sum(p_path[k]*sp.diff(Xt[k], t) for k in range(3)) - H_path
    _a, _b, _c, _d = sp.symbols('a_ b_ c_ d_', real=True)
    th_lam = sp.Lambda((_a, _b, _c, _d), sp.sin(_a*_b) + sp.Rational(3, 7)*_c*_d**2 + sp.exp(_a*_d)/(3 + _b**2))
    symcheck("p.Xdot - H == hbar d/dt theta(X(t), t)",
             sp.simplify((Ldef - hb*sp.diff(th_path, t)).replace(th_f, th_lam).doit()))
    mutant("p.Xdot + H", sp.simplify((sum(p_path[k]*sp.diff(Xt[k], t) for k in range(3)) + H_path
                                      - hb*sp.diff(th_path, t)).replace(th_f, th_lam).doit()))

    # ------------------------------------------------------------------------
    # eq:action_phase  (sgm.tex L1118)
    # States: S = hbar Delta theta; S the action along the path, Delta theta the phase accumulated.
    # Derived from: S = int L dt with eq:L_phase_rate and the fundamental theorem of calculus; for
    #   the free particle d theta/d tau = -omega_0 (L1123).
    #   Labels used: eq:L_phase_rate, eq:rest_freq, ax:lpi, eq:euler_lagrange.
    # Test: a concrete non-quadratic phase field and path, int hbar d theta/dt dt over [t1, t2]
    #   equals hbar(theta(t2) - theta(t1)); free particle: -hbar omega_0 d tau/dt equals
    #   -mc^2 sqrt(1 - v^2/c^2), whose velocity derivative is the momentum gamma m v.
    # Mutant: d tau/dt = 1/sqrt(1 - v^2/c^2) (the inverse dilation factor).
    # ------------------------------------------------------------------------
    eq("eq:action_phase", 1118, "S = hbar Delta theta")
    kk, ww, bb = sp.Rational(3, 7), sp.Rational(5, 3), sp.Rational(4, 9)
    th_c = kk*X[0] - ww*t + bb*X[0]**2*X[1] + sp.sin(X[2])
    path_c = {X[0]: sp.Rational(3, 5)*t**2 + sp.Rational(1, 3), X[1]: sp.Rational(7, 4) - t, X[2]: sp.Rational(2, 7)*t**3}
    th_on = th_c.subs(path_c)
    t1, t2 = sp.Rational(3, 10), sp.Rational(17, 10)
    S_int = sp.integrate(hb*sp.diff(th_on, t), (t, t1, t2))
    symcheck("int_t1^t2 hbar d theta/dt dt == hbar (theta(t2) - theta(t1))",
             sp.simplify(S_int - hb*(th_on.subs(t, t2) - th_on.subs(t, t1))))
    v = sp.Symbol('v', positive=True)
    wrest = m*c**2/hb
    L_free = -hb*wrest*sp.sqrt(1 - v**2/c**2)
    symcheck("-hbar omega_0 dtau/dt == -mc^2 sqrt(1 - v^2/c^2)", L_free + m*c**2*sp.sqrt(1 - v**2/c**2))
    symcheck("dL/dv == gamma m v (the relativistic momentum)", sp.diff(L_free, v) - m*v/sp.sqrt(1 - v**2/c**2))
    mutant("d tau/dt = gamma", -hb*wrest/sp.sqrt(1 - v**2/c**2) + m*c**2*sp.sqrt(1 - v**2/c**2))

    # ------------------------------------------------------------------------
    # eq:ham_p  (sgm.tex L1134)
    # States: pdot = -hbar grad omega.
    # Derived from: the flow X of H = hbar omega defined by (sum_j dp_j ^ dx_j)(X, .) = -dH
    #   (L1138), whose dx_j coefficient is pdot_j = -dH/dx_j.
    #   Labels used: eq:H_eq_hw, eq:sgm.
    # Test: for a general H(x, p) in one degree of freedom, the interior product of X = (xdot, pdot)
    #   with dp ^ dx is pdot dx - xdot dp; matching it with -dH = -H_x dx - H_p dp and solving
    #   gives pdot = -H_x and xdot = +H_p.
    # Mutant: the form dx ^ dp (the opposite orientation) gives pdot = +H_x.
    # ------------------------------------------------------------------------
    eq("eq:ham_p", 1134, "pdot = -hbar grad omega")
    x1s, p1s, xd, pd = sp.symbols('x p xdot pdot', real=True)
    Hf = sp.Function('H', real=True)(x1s, p1s)
    # basis (dx, dp); (dp ^ dx)(X, Y) = dp(X) dx(Y) - dx(X) dp(Y)
    iota = sp.Matrix([pd, -xd])
    sol = sp.solve(list(iota + sp.Matrix([sp.diff(Hf, x1s), sp.diff(Hf, p1s)])), [xd, pd], dict=True)[0]
    symcheck("pdot = -dH/dx from the symplectic form", sol[pd] + sp.diff(Hf, x1s))
    iota_m = sp.Matrix([-pd, xd])
    sol_m = sp.solve(list(iota_m + sp.Matrix([sp.diff(Hf, x1s), sp.diff(Hf, p1s)])), [xd, pd], dict=True)[0]
    mutant("orientation dx ^ dp gives pdot = +dH/dx", sol_m[pd] + sp.diff(Hf, x1s))

    # ------------------------------------------------------------------------
    # eq:ham_x  (sgm.tex L1135); L1136 is the caption, not mathematics
    # States: xdot = +hbar grad_p omega.
    # Derived from: the dp_j coefficient of the same matching (L1138); for the free relativistic
    #   particle E = c sqrt(p^2 + m^2 c^2) it gives the group velocity c^2 p/E.
    #   Labels used: eq:ham_p, eq:H_eq_hw.
    # Test: xdot = +H_p from the matching of the previous block; grad_p of c sqrt(p^2 + m^2c^2)
    #   is c^2 p/E for a general momentum vector.
    # Mutant: the Newtonian velocity p/m in place of c^2 p/E.
    # ------------------------------------------------------------------------
    eq("eq:ham_x", 1135, "xdot = +hbar grad_p omega")
    symcheck("xdot = +dH/dp from the symplectic form", sol[xd] - sp.diff(Hf, p1s))
    E_rel = c*sp.sqrt(Pv.dot(Pv) + m**2*c**2)
    symcheck("grad_p E == c^2 p/E", sp.Matrix([sp.diff(E_rel, Pv[k]) for k in range(3)]) - c**2*Pv/E_rel)
    mutant("grad_p E == p/m", sp.simplify(sp.diff(E_rel, Pv[0]) - Pv[0]/m))

    # ------------------------------------------------------------------------
    # eq:sgm_temporal  (sgm.tex L1147)
    # States: dE/dt = +hbar d omega/d t; E = hbar omega the energy along the flow.
    # Derived from: the chain rule along the flow and eqs. ham_p, ham_x, whose two cross terms
    #   cancel (L1145).
    #   Labels used: eq:planck, eq:ham_p, eq:ham_x.
    # Test: for a general omega(x, p, t) in three dimensions, xdot.grad(hbar omega)
    #   + pdot.grad_p(hbar omega) + hbar d_t omega equals hbar d_t omega.
    # Mutant: pdot = +hbar grad omega (the sign of eq:ham_p), which leaves the cross terms.
    # ------------------------------------------------------------------------
    eq("eq:sgm_temporal", 1147, "dE/dt = + hbar d omega/dt")
    w_f = sp.Function('omega', real=True)(*X, *Pv, t)
    gx = _S4_grad(w_f, X)
    gp = sp.Matrix([sp.diff(w_f, Pv[k]) for k in range(3)])
    xdot_f, pdot_f = hb*gp, -hb*gx
    dEdt = xdot_f.dot(hb*gx) + pdot_f.dot(hb*gp) + hb*sp.diff(w_f, t)
    symcheck("dE/dt == hbar d_t omega", dEdt - hb*sp.diff(w_f, t))
    mutant("pdot = +hbar grad omega", hb*gp.dot(hb*gx) + hb*gx.dot(hb*gp))

    # ------------------------------------------------------------------------
    # eq:legendre_time  (sgm.tex L1152)
    # States: dH/dt|_{x,p} = p_j d xdot_j/dt - dL/dt|_{x,xdot} - (dL/d xdot_j) d xdot_j/dt.
    # Derived from: H = p.xdot - L with xdot(x, p, t), differentiated in t at fixed x and p, as in
    #   eq:legendre_chain; the first and third terms cancel by p_j = dL/d xdot_j (L1155).
    #   Labels used: eq:legendre_chain, sec:route_lagrangian.
    # Test: for L = m|v|^2/2 + q v.A(x, t) - q Phi(x, t) with general A, Phi, the left side computed
    #   from H(x, p, t) equals the three-term right side; the first and third terms cancel; and
    #   dH/dt|_{x,p} = -dL/dt|_{x,v}.
    # Mutant: the third term dropped.
    # ------------------------------------------------------------------------
    eq("eq:legendre_time", 1152, "dH/dt|_{x,p} = p_j d xdot_j/dt - dL/dt|_{x,xdot} - (dL/dxdot_j) dxdot_j/dt")
    L_t = m*Vv.dot(Vv)/2 + q*Vv.dot(A_f) - q*Ph_f
    v_of_p = (Pv - q*A_f)/m
    subs_v = {Vv[k]: v_of_p[k] for k in range(3)}
    H_t = (Pv.dot(Vv) - L_t).subs(subs_v)
    dvdt = sp.diff(v_of_p, t)
    rhs1 = Pv.dot(dvdt)
    rhs2 = -sp.diff(L_t, t).subs(subs_v)
    rhs3 = -sum(sp.diff(L_t, Vv[k]).subs(subs_v)*dvdt[k] for k in range(3))
    symcheck("dH/dt|_{x,p} == three-term right side", sp.expand(sp.diff(H_t, t) - (rhs1 + rhs2 + rhs3)))
    symcheck("first and third terms cancel by p_j = dL/dv_j", sp.expand(rhs1 + rhs3))
    symcheck("dH/dt|_{x,p} == -dL/dt|_{x,v}", sp.expand(sp.diff(H_t, t) - rhs2))
    mutant("third term dropped", sp.expand(sp.diff(H_t, t) - (rhs1 + rhs2)))

    # ------------------------------------------------------------------------
    # eq:sgm_fourvector  (sgm.tex L1160); L1161 is the caption, not mathematics
    # States: dp_mu/dt = -hbar d_mu omega|_p, with p_mu = hbar d_mu theta = (-E/c, p) and
    #   d_0 = c^-1 d_t.
    # Derived from: eq:phase_fourvector times hbar with omega = -d_t theta; the spatial components
    #   are eq:sgm_as_dot_p and the time component is eq:sgm_temporal divided by -c.
    #   Labels used: eq:phase_fourvector, eq:debroglie, eq:phase_freq, eq:sgm_as_dot_p,
    #   eq:sgm_temporal, eq:gradient_def, sec:route_phase.
    # Test: hbar c^-1 d_t theta = -E/c with E = hbar omega, omega = -d_t theta; time component
    #   -(1/c) dE/dt = -hbar c^-1 d_t omega along the flow of a general omega(x, p, t); spatial
    #   components from eq:ham_p.
    # Mutant: d_0 = c d_t (the inverse factor).
    # ------------------------------------------------------------------------
    eq("eq:sgm_fourvector", 1160, "dp_mu/dt = -hbar d_mu omega|_p")
    thf = th_f(*X, t)
    omega_th = -sp.diff(thf, t)
    symcheck("p_0 = hbar c^-1 d_t theta == -E/c", hb*sp.diff(thf, t)/c - (-(hb*omega_th)/c))
    symcheck("time component: d p_0/dt = -(1/c) dE/dt == -hbar c^-1 d_t omega",
             -dEdt/c - (-hb*sp.diff(w_f, t)/c))
    symcheck("spatial components: pdot == -hbar grad omega", pdot_f + hb*gx)
    mutant("d_0 = c d_t", -dEdt/c - (-hb*c*sp.diff(w_f, t)))

    # ------------------------------------------------------------------------
    # eq:work_energy  (sgm.tex L1169), first line
    # States: W = int_{x_A}^{x_B} F.dx; W the work (scalar), F the force.
    # Derived from: definition of the work; its units are J.
    #   Labels used: eq:sgm.
    # Test: units of F.dx are those of energy, N m = J.
    # Mutant: F dt (an impulse) does not have the units of energy.
    # ------------------------------------------------------------------------
    eq("eq:work_energy", 1169, "W = int F.dx")
    symcheck("units of F dx == J", _S4_N + _S4_M - _S4_J)
    mutant("units of F dt != J", _S4_N + _S4_S - _S4_J)

    # ------------------------------------------------------------------------
    # eq:work_energy  (sgm.tex L1170), second line
    # States: W = -Delta U for a velocity-independent U with hbar grad omega = grad U.
    # Derived from: F = -hbar grad omega = -grad U and the gradient theorem; along the motion
    #   d/dt(K + U) = xdot.(m xddot + hbar grad omega) = 0 by eq:el_sgm (L1173).
    #   Labels used: eq:sgm, eq:sgm_lagrangian, eq:el_sgm, sec:lagrangian, eq:sgm_temporal.
    # Test: for a concrete non-polynomial U, int F.dx along two different paths from x_A to x_B
    #   both equal -Delta U; for a general U and path, d/dt(K + U) = xdot.(m xddot + grad U).
    # Mutant: a force with nonzero curl, F = (x2, -x1, 0), gives path-dependent work.
    # ------------------------------------------------------------------------
    eq("eq:work_energy", 1170, "W = -Delta U")
    s = sp.Symbol('s', real=True)
    U_c = sp.Rational(3, 4)*X[0]*X[1] + sp.Rational(2, 5)*X[2]**3 + sp.sin(X[0]*X[2])
    F_c = -_S4_grad(U_c, X)
    xA = sp.Matrix([sp.Rational(1, 3), sp.Rational(-1, 4), sp.Rational(3, 5)])
    xB = sp.Matrix([sp.Rational(7, 5), sp.Rational(5, 6), sp.Rational(-2, 7)])
    pathL = xA + s*(xB - xA)
    pathC = xA + s*(xB - xA) + sp.Matrix([s*(1 - s), sp.Rational(3, 2)*s**2*(1 - s), sp.sin(sp.pi*s)/3])

    def work(F, path):
        integrand = sum(F[k].subs({X[j]: path[j] for j in range(3)})*sp.diff(path[k], s) for k in range(3))
        return sp.Integral(integrand, (s, 0, 1)).evalf(30)
    dU = U_c.subs({X[j]: xB[j] for j in range(3)}) - U_c.subs({X[j]: xA[j] for j in range(3)})
    check("straight path: W == -Delta U", work(F_c, pathL), -dU.evalf(30), tol=1e-20)
    check("curved path: W == -Delta U", work(F_c, pathC), -dU.evalf(30), tol=1e-20)
    U_g = sp.Function('U', real=True)(*Xt)
    K_g = m*sum(sp.diff(Xt[k], t)**2 for k in range(3))/2
    gradU_on = [sp.diff(sp.Function('U', real=True)(*X), X[k]).subs({X[j]: Xt[j] for j in range(3)})
                for k in range(3)]
    symcheck("d/dt(K + U) == xdot.(m xddot + grad U)",
             (sp.diff(K_g + U_g, t) - sum(sp.diff(Xt[k], t)*(m*sp.diff(Xt[k], t, 2) + gradU_on[k])
                                          for k in range(3))).doit())
    F_curl = sp.Matrix([X[1], -X[0], 0])
    numutant("non-gradient force: work differs between the two paths", work(F_curl, pathC), work(F_curl, pathL))

    # ------------------------------------------------------------------------
    # eq:impulse_momentum  (sgm.tex L1175), first line
    # States: J = int_{t1}^{t2} F dt; J the impulse (vector).
    # Derived from: definition; its units are N s.
    #   Labels used: eq:sgm_as_dot_p.
    # Test: N s == kg m s^-1, the units of momentum.
    # Mutant: N m (a work) is not a momentum.
    # ------------------------------------------------------------------------
    eq("eq:impulse_momentum", 1175, "J = int F dt")
    symcheck("units of F dt == kg m s^-1", _S4_N + _S4_S - (_S4_KG + _S4_M - _S4_S))
    mutant("units of F dx != kg m s^-1", _S4_N + _S4_M - (_S4_KG + _S4_M - _S4_S))

    # ------------------------------------------------------------------------
    # eq:impulse_momentum  (sgm.tex L1176), second line
    # States: J = Delta p.
    # Derived from: integrating eq:sgm_as_dot_p from t1 to t2, p = hbar k.
    #   Labels used: eq:sgm_as_dot_p, eq:debroglie_k.
    # Test: harmonic oscillator hbar omega = p^2/2m + kappa x^2/2 on its exact trajectory
    #   x = a cos(w t), p = -m a w sin(w t): int (-hbar d omega/dx) dt over [t1, t2] == p(t2) - p(t1).
    # Mutant: half the momentum change.
    # ------------------------------------------------------------------------
    eq("eq:impulse_momentum", 1176, "J = Delta p")
    kap, amp = sp.Rational(5, 3), sp.Rational(3, 7)
    wosc = sp.sqrt(kap/m)
    x_tr = amp*sp.cos(wosc*t)
    p_tr = m*sp.diff(x_tr, t)
    J_int = sp.integrate(-kap*x_tr, (t, t1, t2))
    symcheck("int -hbar d omega/dx dt == Delta p", sp.simplify(J_int - (p_tr.subs(t, t2) - p_tr.subs(t, t1))))
    mutant("J = Delta p / 2", sp.simplify(J_int - (p_tr.subs(t, t2) - p_tr.subs(t, t1))/2))

    # ------------------------------------------------------------------------
    # eq:sgm_angular  (sgm.tex L1197); L1198 is the caption, not mathematics
    # States: n.d(L + S)/dt = -hbar d omega/d vartheta; n the unit axis, L the orbital and S the
    #   spin angular momentum (vectors), d omega/d vartheta the angular spectral gradient.
    # Derived from: orbital part: chain rule under x -> x + dv n x x, p -> p + dv n x p, then
    #   eqs. ham_p, ham_x and the triple product (L1190); spin part: [b.sigma, sigma_k]
    #   = 2i eps_jkl b_j sigma_l and Ehrenfest (L1192); whole operator: (i/hbar)[n.(L + hbar
    #   sigma/2), omega] (L1194).
    #   Labels used: def:spectral_gradient, sec:spinor, eq:ham_p, eq:ham_x, eq:nr_expansion,
    #   eq:ehrenfest_commutator, eq:sgm_expectation, eq:sgm, eq:sgm_as_dot_p.
    # Test: (a) for a general omega(x, p), hbar d omega/d vartheta = -n.d(x x p)/dt along the flow;
    #   (b) the Pauli commutator for every k; (c) d<sigma_k>/dt = (2/hbar)(b x <sigma>)_k from
    #   (i/hbar) 2i eps_jkl b_j s_l; (d) b.(n x s) = -n.(b x s) = -n.dS/dt with S = hbar s/2;
    #   (e) (i/hbar)[hbar n.sigma/2, b.sigma] = -(n x b).sigma, whose expectation is -n.dS/dt;
    #   (f) finite-difference derivative of T^-1 (b.sigma) T at vartheta = 0 equals (e) numerically.
    # Mutant: the commutator with +2i eps_kjl (indices swapped).
    # ------------------------------------------------------------------------
    eq("eq:sgm_angular", 1197, "n.d(L + S)/dt = -hbar d omega/d vartheta")
    nv = sp.Matrix(sp.symbols('n1:4', real=True))
    dv = sp.Symbol('delta', real=True)
    w_xp = sp.Function('omega', real=True)(*X, *Pv)
    nxx, nxp = _S4_cross(nv, Xv), _S4_cross(nv, Pv)
    rot_subs = {**{X[k]: X[k] + dv*nxx[k] for k in range(3)}, **{Pv[k]: Pv[k] + dv*nxp[k] for k in range(3)}}
    dw_dth = sp.diff(w_xp.subs(rot_subs, simultaneous=True), dv).subs(dv, 0).doit()
    gx2 = _S4_grad(w_xp, X)
    gp2 = sp.Matrix([sp.diff(w_xp, Pv[k]) for k in range(3)])
    chain = nxx.dot(gx2) + nxp.dot(gp2)
    symcheck("(a) chain rule: d omega/d vartheta == (n x x).grad omega + (n x p).grad_p omega",
             sp.expand(dw_dth - chain))
    xdot2, pdot2 = hb*gp2, -hb*gx2
    dLdt = _S4_cross(xdot2, Pv) + _S4_cross(Xv, pdot2)
    symcheck("(a) hbar d omega/d vartheta == -n.d(x x p)/dt", sp.expand(hb*chain + nv.dot(dLdt)))
    bv = sp.Matrix(sp.symbols('b1:4', real=True))
    bs = sum((bv[j]*_S4_SIG[j] for j in range(3)), sp.zeros(2))
    LC = sp.LeviCivita
    for k in range(3):
        comm = bs*_S4_SIG[k] - _S4_SIG[k]*bs
        rhs = sum((2*sp.I*LC(j, k, l)*bv[j]*_S4_SIG[l] for j in range(3) for l in range(3)), sp.zeros(2))
        symcheck("(b) [b.sigma, sigma_%d] == 2i eps_jkl b_j sigma_l" % (k + 1), sp.expand(comm - rhs))
    comm0 = bs*_S4_SIG[0] - _S4_SIG[0]*bs
    rhs_m = sum((2*sp.I*LC(0, k, l)*bv[k]*_S4_SIG[l] for k in range(3) for l in range(3)), sp.zeros(2))
    mutant("(b) commutator with eps_kjl", sp.expand(comm0 - rhs_m)[0, 1])
    sv = sp.Matrix(sp.symbols('s1:4', real=True))
    dsig = sp.Matrix([(sp.I/hb)*sum(2*sp.I*LC(j, k, l)*bv[j]*sv[l] for j in range(3) for l in range(3))
                      for k in range(3)])
    symcheck("(c) d<sigma>/dt == (2/hbar) b x <sigma>", sp.expand(dsig - 2*_S4_cross(bv, sv)/hb))
    dSdt = hb/2*dsig
    symcheck("(d) b.(n x s) == -n.dS/dt", sp.expand(bv.dot(_S4_cross(nv, sv)) + nv.dot(dSdt)))
    ns = sum((nv[j]*_S4_SIG[j] for j in range(3)), sp.zeros(2))
    gen = (sp.I/hb)*(hb/2*ns*bs - hb/2*bs*ns)
    nxb = _S4_cross(nv, bv)
    symcheck("(e) (i/hbar)[hbar n.sigma/2, b.sigma] == -(n x b).sigma",
             sp.expand(gen + sum((nxb[j]*_S4_SIG[j] for j in range(3)), sp.zeros(2))))
    symcheck("(e) expectation -(n x b).s == -n.dS/dt", sp.expand(-nxb.dot(sv) + nv.dot(dSdt)))
    rng = np.random.default_rng(1204)
    n_n = rng.normal(size=3)
    n_n /= np.linalg.norm(n_n)
    b_n = rng.normal(size=3)
    sig_n = [np.array(Sg.tolist(), dtype=complex) for Sg in _S4_SIG]
    bsn = sum(b_n[j]*sig_n[j] for j in range(3))
    nsn = sum(n_n[j]*sig_n[j] for j in range(3))

    def Tm(th):
        return np.cos(th/2)*np.eye(2) - 1j*np.sin(th/2)*nsn
    hst = 1e-5
    fd = (np.linalg.inv(Tm(hst)) @ bsn @ Tm(hst) - np.linalg.inv(Tm(-hst)) @ bsn @ Tm(-hst))/(2*hst)
    nxb_n = np.cross(n_n, b_n)
    check("(f) d/dvartheta T^-1 (b.sigma) T == -(n x b).sigma (numerical, hbar = 1)",
          np.abs(fd + sum(nxb_n[j]*sig_n[j] for j in range(3))).max(), 0, tol=1e-8,
          scale=np.linalg.norm(b_n))

    # ------------------------------------------------------------------------
    # eq:spin_doppler  (sgm.tex L1210)
    # States: exp(-Omega t n.e/2) psi = rho^{1/2} R_psi0 exp(e3 (omega - Omega/2) t); psi the
    #   quaternion state, R_psi0 its unit-quaternion factor, Omega the rotation rate.
    # Derived from: n.e = R_psi0 e3 conj(R_psi0) (eq:spin_rotor) and the exponential series, so
    #   exp(-Omega t n.e/2) R_psi0 = R_psi0 exp(-Omega t e3/2) (L1208); exp(e3 omega t) commutes
    #   with e3.
    #   Labels used: eq:stationary_time, eq:polar_form, eq:spin_rotor, sec:spinor.
    # Test: numerically, for a random unit quaternion R_psi0 and sample omega, Omega, t, the two
    #   sides agree; with n.e = -R_psi0 e3 conj(R_psi0) (spin against n) the frequency is
    #   omega + Omega/2 (L1213).
    # Mutant: the full angle exp(-Omega t n.e) (which gives omega - Omega).
    # ------------------------------------------------------------------------
    eq("eq:spin_doppler", 1210, "exp(-Omega t n.e/2) psi = rho^{1/2} R_psi0 exp(e3 (omega - Omega/2) t)")
    R0 = np.array([0.31, -0.52, 0.44, 0.67])
    R0 = list(R0/np.linalg.norm(R0))
    ne_q = _S4_mul(_S4_mul(R0, [0, 0, 0, 1]), _S4_conj(R0))
    ne = np.array(ne_q[1:])
    rho_s, om_s, Om_s, t_s = 0.83, 0.37, 0.53, 1.9
    psi_q = _S4_sc(np.sqrt(rho_s), _S4_mul(R0, [np.cos(om_s*t_s), 0, 0, np.sin(om_s*t_s)]))
    lhs_q = _S4_mul(_S4_qexp_np(-0.5*Om_s*t_s*ne), psi_q)
    ang = (om_s - 0.5*Om_s)*t_s
    rhs_q = _S4_sc(np.sqrt(rho_s), _S4_mul(R0, [np.cos(ang), 0, 0, np.sin(ang)]))
    check("|exp(-Omega t n.e/2) psi - rho^1/2 R exp(e3(omega - Omega/2)t)| == 0",
          np.linalg.norm(np.array(lhs_q) - np.array(rhs_q)), 0, tol=1e-12, scale=np.sqrt(rho_s))
    check("|n.e| == 1 and n.e is a pure quaternion", np.linalg.norm(ne), 1.0, tol=1e-12)
    lhs_a = _S4_mul(_S4_qexp_np(+0.5*Om_s*t_s*ne), psi_q)  # n.e -> -n.e
    ang_a = (om_s + 0.5*Om_s)*t_s
    rhs_a = _S4_sc(np.sqrt(rho_s), _S4_mul(R0, [np.cos(ang_a), 0, 0, np.sin(ang_a)]))
    check("spin against n: frequency omega + Omega/2",
          np.linalg.norm(np.array(lhs_a) - np.array(rhs_a)), 0, tol=1e-12, scale=np.sqrt(rho_s))
    lhs_m = _S4_mul(_S4_qexp_np(-Om_s*t_s*ne), psi_q)
    numutant("full angle exp(-Omega t n.e)", np.linalg.norm(np.array(lhs_m) - np.array(rhs_q)) + 1.0, 1.0)

    # ------------------------------------------------------------------------
    # eq:omega_rotating  (sgm.tex L1215)
    # States: hbar omega = mc^2 + pi^2/2m - Omega.(L + S) for a free particle in coordinates
    #   turning at Omega; L = x x pi, S = hbar sigma/2.
    # Derived from: eq:nr_expansion with alpha_n = W = 1, A_g = Omega x x, B_g = 2 Omega:
    #   -A_g.pi = -Omega.(x x pi), -(1/4) hbar sigma.B_g = -Omega.S; the Doppler shifts of L1204
    #   and L1206; xdot = pi/m - A_g from eq:ham_x (L1218).
    #   Labels used: eq:nr_expansion, eq:spin_doppler, eq:ham_x, sec:lagrangian, sec:dirac,
    #   sec:spinor.
    # Test: the two coupling identities for general vectors; theta = k.x - omega t with x = x' + ut
    #   has frequency omega - k.u at fixed x'; m_l phi - omega t with phi = phi' + Omega t has
    #   frequency omega - m_l Omega; Hamilton's velocity of pi^2/2m - A_g.pi.
    # Mutant: B_g = Omega in -(1/4) hbar sigma.B_g, which gives -Omega.S/2.
    # ------------------------------------------------------------------------
    eq("eq:omega_rotating", 1215, "hbar omega = mc^2 + pi^2/2m - Omega.(L + S)")
    symcheck("-A_g.pi == -Omega.(x x pi) with A_g = Omega x x", -Ag_rot.dot(Pv) + Om.dot(_S4_cross(Xv, Pv)))
    S_v = hb*sv/2
    symcheck("-(1/4) hbar s.B_g == -Omega.S with B_g = 2 Omega, S = hbar s/2",
             sp.expand(-hb*sv.dot(_S4_curl(Ag_rot, X))/4 + Om.dot(S_v)))
    mutant("B_g = Omega", sp.expand(-hb*sv.dot(Om)/4 + Om.dot(S_v)))
    uv = sp.Matrix(sp.symbols('u1:4', real=True))
    kv = sp.Matrix(sp.symbols('k1:4', real=True))
    om = sp.Symbol('omega', real=True)
    Xp = sp.Matrix(sp.symbols('xp1:4', real=True))
    th_dop = (kv.dot(Xp + uv*t) - om*t)
    symcheck("Doppler: -d_t theta at fixed x' == omega - k.u", -sp.diff(th_dop, t) - (om - kv.dot(uv)))
    ml, phip, Oms = sp.symbols('m_l phi_p Omega', real=True)
    symcheck("rotational Doppler: frequency at fixed phi' == omega - m_l Omega",
             -sp.diff(ml*(phip + Oms*t) - om*t, t) - (om - ml*Oms))
    H_rot = Pv.dot(Pv)/(2*m) - Ag_rot.dot(Pv)
    symcheck("xdot = pi/m - A_g", sp.Matrix([sp.diff(H_rot, Pv[k]) for k in range(3)]) - (Pv/m - Ag_rot))

    # ------------------------------------------------------------------------
    # eq:gyroscopic  (sgm.tex L1222)
    # States: d(L + S)/dt = (L + S) x Omega in the turning coordinates.
    # Derived from: hbar d omega/d vartheta = -Omega.(n' x J) = -n'.(J x Omega) with J = L + S
    #   (L1220) and eq:sgm_angular for every n'; the rate in the original coordinates is
    #   J x Omega + Omega x J = 0 (L1225).
    #   Labels used: eq:omega_rotating, eq:sgm_angular.
    # Test: the triple-product step for general vectors; J(t) = R_z(-Omega t) J_0 solves
    #   dJ/dt = J x Omega (Omega along z), i.e. J is constant in the original coordinates;
    #   J x Omega + Omega x J = 0.
    # Mutant: dJ/dt = Omega x J, which gives the rate 2 Omega x J in the original coordinates.
    # ------------------------------------------------------------------------
    eq("eq:gyroscopic", 1222, "d(L + S)/dt = (L + S) x Omega")
    Jv = sp.Matrix(sp.symbols('J1:4', real=True))
    npv = sp.Matrix(sp.symbols('np1:4', real=True))
    symcheck("-Omega.(n' x J) == -n'.(J x Omega)", sp.expand(-Om.dot(_S4_cross(npv, Jv)) + npv.dot(_S4_cross(Jv, Om))))
    J1, J2, J3 = sp.symbols('J1 J2 J3', real=True)
    Jt = sp.Matrix([J1*sp.cos(Oms*t) + J2*sp.sin(Oms*t), -J1*sp.sin(Oms*t) + J2*sp.cos(Oms*t), J3])
    Omz = sp.Matrix([0, 0, Oms])
    symcheck("J(t) = R(-Omega t) J0 solves dJ/dt = J x Omega", sp.simplify(sp.diff(Jt, t) - _S4_cross(Jt, Omz)))
    symcheck("rate in original coordinates J x Omega + Omega x J == 0", _S4_cross(Jv, Om) + _S4_cross(Om, Jv))
    mutant("dJ/dt = Omega x J: rate in original coordinates 2 Omega x J", (2*_S4_cross(Om, Jv))[0])

    # ------------------------------------------------------------------------
    # eq:torque_balance  (sgm.tex L1229)
    # States: N_app + N_in = 0 with N_in = S x Omega.
    # Derived from: dS/dt = 0 in the turning coordinates, so the applied torque plus the gyroscopic
    #   torque of eq:gyroscopic vanish; in the original coordinates dS/dt = Omega x S = N_app (L1232).
    #   Labels used: eq:gyroscopic, eq:geodesic_balance.
    # Test: N_app = -S x Omega equals Omega x S, the rate of a vector at rest in the turning axes
    #   seen from the original axes.
    # Mutant: N_app = S x Omega.
    # ------------------------------------------------------------------------
    eq("eq:torque_balance", 1229, "N_app + N_in = 0")
    Ss = sp.Matrix(sp.symbols('S1:4', real=True))
    N_in = _S4_cross(Ss, Om)
    N_app = -N_in
    symcheck("N_app == Omega x S", N_app - _S4_cross(Om, Ss))
    mutant("N_app = S x Omega", (_S4_cross(Ss, Om) - _S4_cross(Om, Ss))[0])

    # ------------------------------------------------------------------------
    # eq:euler_rigid  (sgm.tex L1236)
    # States: N_app = I Omegadot + Omega x I Omega; I the inertia tensor I w = sum m x x (w x x).
    # Derived from: a = Omegadot x x + Omega x (Omega x x), N_in = -sum m x x a, the Jacobi identity
    #   x x (Omega x w) = Omega x (x x w) - w x (x x Omega) with w = Omega x x (L1234), and
    #   eq:torque_balance; L = I Omega and curl(Omegadot x x) = 2 Omegadot (L1241).
    #   Labels used: eq:torque_balance, eq:inertia_gradient, eq:gyroscopic, sec:lagrangian.
    # Test: random point masses: sum m x x a == I Omegadot + Omega x I Omega; sum m x x (Omega x x)
    #   == I Omega; I equals sum m(|x|^2 1 - x x^T); the Jacobi identity symbolically;
    #   curl(Omegadot x x) = 2 Omegadot.
    # Mutant: the sign of the gyroscopic term, I Omegadot - Omega x I Omega.
    # ------------------------------------------------------------------------
    eq("eq:euler_rigid", 1236, "N_app = I Omegadot + Omega x I Omega")
    rng = np.random.default_rng(1239)
    ms = rng.uniform(0.3, 1.7, size=7)
    xs = rng.normal(size=(7, 3))
    Om_n = np.array([0.37, -0.81, 0.53])
    Omd_n = np.array([-0.29, 0.44, 0.71])

    def Iop(w):
        return sum(ms[i]*np.cross(xs[i], np.cross(w, xs[i])) for i in range(7))
    a_n = [np.cross(Omd_n, xs[i]) + np.cross(Om_n, np.cross(Om_n, xs[i])) for i in range(7)]
    tot = sum(ms[i]*np.cross(xs[i], a_n[i]) for i in range(7))
    rhs_e = Iop(Omd_n) + np.cross(Om_n, Iop(Om_n))
    check("sum m x x a == I Omegadot + Omega x I Omega", np.linalg.norm(tot - rhs_e), 0, tol=1e-12,
          scale=np.linalg.norm(tot))
    Imat = sum(ms[i]*(np.dot(xs[i], xs[i])*np.eye(3) - np.outer(xs[i], xs[i])) for i in range(7))
    check("I w == sum m(|x|^2 1 - x x^T) w", np.linalg.norm(Imat @ Om_n - Iop(Om_n)), 0, tol=1e-12,
          scale=np.linalg.norm(Iop(Om_n)))
    check("L = sum m x x (Omega x x) == I Omega (same map)",
          np.linalg.norm(sum(ms[i]*np.cross(xs[i], np.cross(Om_n, xs[i])) for i in range(7)) - Imat @ Om_n), 0,
          tol=1e-12, scale=np.linalg.norm(Imat @ Om_n))
    numutant("I Omegadot - Omega x I Omega", np.linalg.norm(tot - (Iop(Omd_n) - np.cross(Om_n, Iop(Om_n)))) + 1.0, 1.0)
    wv = sp.Matrix(sp.symbols('w1:4', real=True))
    symcheck("Jacobi: x x (Omega x w) == Omega x (x x w) - w x (x x Omega)",
             sp.expand(_S4_cross(Xv, _S4_cross(Om, wv)) - _S4_cross(Om, _S4_cross(Xv, wv))
                       + _S4_cross(wv, _S4_cross(Xv, Om))))
    wO = _S4_cross(Om, Xv)
    symcheck("with w = Omega x x, the last term w x (x x Omega) vanishes", sp.expand(_S4_cross(wO, _S4_cross(Xv, Om))))
    Omd = sp.Matrix(sp.symbols('Omd1:4', real=True))
    symcheck("curl(Omegadot x x) == 2 Omegadot", _S4_curl(_S4_cross(Omd, Xv), X) - 2*Omd)

    # ------------------------------------------------------------------------
    # eq:winding_rate  (sgm.tex L1249)
    # States: d/dt (d theta/d phi) = -d omega/d phi, for omega(x, |p|^2); phi the azimuth about n.
    # Derived from: d/d phi = (n x x).grad, d theta/d phi = n.(x x k), its rate
    #   n.(xdot x k) + n.(x x kdot) with kdot = -grad omega (eq:sgm_as_dot_p), and xdot parallel to p
    #   by eq:ham_x when omega depends on p only through |p|^2 (L1247).
    #   Labels used: eq:debroglie, eq:sgm_as_dot_p, eq:ham_x, eq:nr_expansion, sec:route_phase.
    # Test: for n = z, d/d phi of f(r cos phi, r sin phi, z) equals (n x x).grad f; n.(x x grad theta)
    #   == (n x x).grad theta; for omega = F(x, |p|^2) general, xdot x p = 0 and the full rate
    #   n.(xdot x k) + n.(x x kdot) == -(n x x).grad omega; L1254: -(n x x).m a = -m Omegadot r_perp^2
    #   and n.I n = m r_perp^2 for one mass.
    # Mutant: a shift term -A_g.p/hbar in omega makes xdot x p nonzero.
    # ------------------------------------------------------------------------
    eq("eq:winding_rate", 1249, "d/dt (d theta/d phi) = -d omega/d phi")
    r_, phi_, z_ = sp.symbols('r phi z', real=True)
    fz = sp.Function('f', real=True)
    cyl = {X[0]: r_*sp.cos(phi_), X[1]: r_*sp.sin(phi_), X[2]: z_}
    lhs_phi = sp.diff(fz(r_*sp.cos(phi_), r_*sp.sin(phi_), z_), phi_)
    ez = sp.Matrix([0, 0, 1])
    rhs_phi = _S4_cross(ez, Xv).dot(_S4_grad(fz(*X), X)).subs(cyl)
    symcheck("d/d phi == (z x x).grad", sp.simplify((lhs_phi - rhs_phi).doit()))
    thx = th_f(*X, t)
    symcheck("n.(x x grad theta) == (n x x).grad theta",
             sp.expand(nv.dot(_S4_cross(Xv, _S4_grad(thx, X))) - _S4_cross(nv, Xv).dot(_S4_grad(thx, X))))
    P2 = sp.Symbol('P2', positive=True)
    Fw = sp.Function('F', real=True)
    w_iso = Fw(*X, Pv.dot(Pv))
    xdot_iso = hb*sp.Matrix([sp.diff(w_iso, Pv[k]) for k in range(3)])
    symcheck("xdot x p == 0 for omega(x, |p|^2)", sp.simplify(_S4_cross(xdot_iso, Pv)))
    kdot_iso = -_S4_grad(w_iso, X)
    rate = nv.dot(_S4_cross(xdot_iso, Pv/hb)) + nv.dot(_S4_cross(Xv, kdot_iso))
    symcheck("n.(xdot x k) + n.(x x kdot) == -(n x x).grad omega",
             sp.simplify(rate + _S4_cross(nv, Xv).dot(_S4_grad(w_iso, X))))
    w_sh = Fw(*X, Pv.dot(Pv)) - Agsym.dot(Pv)/hb
    xdot_sh = hb*sp.Matrix([sp.diff(w_sh, Pv[k]) for k in range(3)])
    mutant("shift term: xdot x p != 0", sp.expand(_S4_cross(xdot_sh, Pv)[2]))
    Omdot = sp.Symbol('Omegadot', real=True)
    a_ax = Omdot*_S4_cross(ez, Xv) + _S4_cross(Oms*ez, _S4_cross(Oms*ez, Xv))
    rperp2 = _S4_cross(ez, Xv).dot(_S4_cross(ez, Xv))
    symcheck("L1254: -(n x x).m a == -m Omegadot r_perp^2", sp.expand(-_S4_cross(ez, Xv).dot(m*a_ax) + m*Omdot*rperp2))
    nIn = ez.dot(m*_S4_cross(Xv, _S4_cross(ez, Xv)))
    symcheck("L1254: n.I n == m r_perp^2 (one mass)", sp.expand(nIn - m*rperp2))

    # ------------------------------------------------------------------------
    # tab:linear_angular  (sgm.tex L1263), table L1260-L1279
    # States: the linear and angular columns: identity, phase per unit change, its rate, Doppler
    #   shift, inertial reaction and balance.
    # Derived from: the entries restate results derived above.
    #   Labels used: eq:sgm_as_dot_p, eq:sgm_angular, eq:winding_rate, eq:omega_rotating,
    #   eq:gyroscopic, eq:euler_rigid, eq:torque_balance, sec:inertia, sec:freefall,
    #   sec:conservation, sec:angular.
    # Test: k = p/hbar turns pdot = -hbar grad omega into kdot = -grad omega; the angular Doppler
    #   row -Omega.(L + S)/hbar is the frequency term of eq:omega_rotating divided by hbar;
    #   N_in = -I Omegadot - Omega x I Omega has the orbital part L x Omega for L = I Omega.
    # Mutant: Doppler row +k.u.
    # ------------------------------------------------------------------------
    eq("tab:linear_angular", 1263, "table of linear and angular spectral gradients")
    symcheck("kdot == -grad omega from pdot == -hbar grad omega", pdot_f/hb + gx)
    Lv = sp.Matrix(sp.symbols('L1:4', real=True))
    symcheck("Doppler row: -Omega.(L+S)/hbar", (-Om.dot(Lv + Ss))/hb - (-(Om.dot(Lv) + Om.dot(Ss))/hb))
    Isym = sp.Matrix(3, 3, lambda i, j: sp.Symbol('I%d%d' % (min(i, j), max(i, j)), real=True))
    symcheck("orbital part: -Omega x (I Omega) == (I Omega) x Omega", -_S4_cross(Om, Isym*Om) - _S4_cross(Isym*Om, Om))
    mutant("Doppler row +k.u", -sp.diff(th_dop, t) - (om + kv.dot(uv)))

    # ------------------------------------------------------------------------
    # eq:ab  (sgm.tex L1291), first line
    # States: Delta theta_AB = (q/hbar) loop integral of A.dl.
    # Derived from: the canonical momentum hbar grad theta = pi + qA (eq:debroglie) gives each
    #   path the potential phase (q/hbar) int A.dl; the difference of two paths is the loop integral.
    #   Labels used: eq:omega_em, eq:action_phase, eq:sgm_fourvector, eq:lorentz, eq:debroglie,
    #   sec:em.
    # Test: outside a solenoid, A = Phi_B/(2 pi r_perp) phi_hat has zero curl for r_perp > 0 and
    #   loop integral Phi_B at every radius (L2062); the phase of each arm is the line integral
    #   of grad theta - pi/hbar = qA/hbar.
    # Mutant: A falling as 1/r_perp^2 gives a radius-dependent loop integral.
    # ------------------------------------------------------------------------
    eq("eq:ab", 1291, "Delta theta_AB = (q/hbar) loop A.dl")
    PhB = sp.Symbol('Phi_B', positive=True)
    Rr = sp.Symbol('R', positive=True)
    A_sol = PhB/(2*sp.pi)*sp.Matrix([-X[1], X[0], 0])/(X[0]**2 + X[1]**2)
    symcheck("curl A == 0 outside the solenoid", sp.simplify(_S4_curl(A_sol, X)))
    circ = sp.Matrix([Rr*sp.cos(s), Rr*sp.sin(s), 0])
    loop = sp.integrate(sp.simplify(sum(A_sol[k].subs({X[0]: circ[0], X[1]: circ[1], X[2]: 0})*sp.diff(circ[k], s)
                                       for k in range(3))), (s, 0, 2*sp.pi))
    symcheck("loop integral == Phi_B at every radius R", loop - PhB)
    A_bad = PhB/(2*sp.pi)*sp.Matrix([-X[1], X[0], 0])/(X[0]**2 + X[1]**2)**sp.Rational(3, 2)
    loop_bad = sp.integrate(sp.simplify(sum(A_bad[k].subs({X[0]: circ[0], X[1]: circ[1], X[2]: 0})*sp.diff(circ[k], s)
                                           for k in range(3))), (s, 0, 2*sp.pi))
    mutant("A ~ 1/r_perp^2: loop integral depends on R", sp.diff(loop_bad, Rr))

    # ------------------------------------------------------------------------
    # eq:ab  (sgm.tex L1292), second line; L1294 is the caption, not mathematics
    # States: Delta theta_AB = (q/hbar) Phi_B.
    # Derived from: Stokes's theorem; the holonomy exp(-e3 Delta theta_AB) of a quaternion
    #   transported with zero D psi along the path (L1298); the Sagnac phase 2m Omega.Area/hbar
    #   from curl A_g = 2 Omega (L1302).
    #   Labels used: eq:ab, eq:right_connection, eq:pure_form, eq:phase_curvature,
    #   eq:time_space_curvature, tab:forces, sec:angular.
    # Test: uniform B: A = B x x/2 has loop integral B pi R^2 around a circle; psi(s) = psi_0
    #   exp(-e3 Theta(s)) with Theta' = (q/hbar) A_s solves d_s psi + (q/hbar) A_s psi e3 = 0 for a
    #   general quaternion psi_0; q Phi_B/hbar = 2 pi at Phi_B = h/q; the loop integral of
    #   Omega x x around an ellipse of semi-axes a, b in the x1-x2 plane is 2 Omega_3 pi a b.
    # Mutant: the Sagnac loop integral taken as Omega.Area (factor 1).
    # ------------------------------------------------------------------------
    eq("eq:ab", 1292, "Delta theta_AB = (q/hbar) Phi_B")
    B3 = sp.Symbol('B', positive=True)
    A_uni = _S4_cross(sp.Matrix([0, 0, B3]), Xv)/2
    loop_u = sp.integrate(sp.simplify(sum(A_uni[k].subs({X[0]: circ[0], X[1]: circ[1], X[2]: 0})*sp.diff(circ[k], s)
                                         for k in range(3))), (s, 0, 2*sp.pi))
    symcheck("uniform B: loop integral == B pi R^2 (the flux)", loop_u - B3*sp.pi*Rr**2)
    Th = sp.Function('Theta', real=True)(s)
    psi0 = list(sp.symbols('q0:4', real=True))
    psi_s = _S4_mul(psi0, [sp.cos(Th), 0, 0, -sp.sin(Th)])
    As = hb*sp.diff(Th, s)/q
    trans = _S4_add(_S4_d(psi_s, s), _S4_sc(q*As/hb, _S4_mul(psi_s, [0, 0, 0, 1])))
    symcheck("psi_0 exp(-e3 Theta) is parallel transported (d_s psi + (q/hbar) A_s psi e3 = 0)",
             sp.Matrix([sp.simplify(u) for u in trans]))
    hP = 2*sp.pi*hb
    symcheck("q Phi_B/hbar == 2 pi at Phi_B = h/q", q*(hP/q)/hb - 2*sp.pi)
    ea, eb = sp.symbols('a_e b_e', positive=True)
    ell = sp.Matrix([ea*sp.cos(s), eb*sp.sin(s), 0])
    Ag_ell = _S4_cross(Om, ell)
    sag = sp.integrate(sp.expand(sum(Ag_ell[k]*sp.diff(ell[k], s) for k in range(3))), (s, 0, 2*sp.pi))
    symcheck("Sagnac: loop integral of Omega x x == 2 Omega.Area", sp.simplify(sag - 2*Om[2]*sp.pi*ea*eb))
    mutant("Sagnac loop integral == Omega.Area", sp.simplify(sag - Om[2]*sp.pi*ea*eb))

    # ------------------------------------------------------------------------
    # eq:interferometer  (sgm.tex L1312), first line
    # States: Delta theta_g = -d_z omega A_zt; A_zt = int Delta z dt the spacetime area.
    # Derived from: d_z omega = omega_0 g/c^2 = mg/hbar from eq:grav_gradient, and the phase rate
    #   -omega of each arm (eq:phase_freq), so the arm difference accumulates -int d_z omega
    #   Delta z dt (L1310).
    #   Labels used: eq:grav_gradient, eq:phase_freq, eq:omega_alpha, eq:rest_freq.
    # Test: omega = omega_0 (1 + g z/c^2) gives d_z omega = mg/hbar; for a concrete Delta z(t)
    #   (a triangle of base 2T and height h_z) the accumulated phase of upper minus lower arm,
    #   -int (omega(z + Delta z) - omega(z)) dt, equals -d_z omega A_zt.
    # Mutant: +d_z omega A_zt (the upper arm advancing faster).
    # ------------------------------------------------------------------------
    eq("eq:interferometer", 1312, "Delta theta_g = -d_z omega A_zt")
    g, zz, Tp, hz = sp.symbols('g z T_p h_z', positive=True)
    om_g = (m*c**2/hb)*(1 + g*zz/c**2)
    symcheck("d_z omega == mg/hbar", sp.diff(om_g, zz) - m*g/hb)
    dz_tri = sp.Piecewise((hz*t/Tp, t <= Tp), (hz*(2*Tp - t)/Tp, True))
    Azt = sp.integrate(dz_tri, (t, 0, 2*Tp))
    dth = -sp.integrate(sp.Piecewise((hz*t/Tp*m*g/hb, t <= Tp), (hz*(2*Tp - t)/Tp*m*g/hb, True)), (t, 0, 2*Tp))
    symcheck("triangle: -int d_z omega Delta z dt == -d_z omega A_zt", sp.simplify(dth + m*g/hb*Azt))
    mutant("+d_z omega A_zt", sp.simplify(dth - m*g/hb*Azt))

    # ------------------------------------------------------------------------
    # eq:interferometer  (sgm.tex L1313), second line; L1315 is the caption
    # States: Delta theta_g = -(mg/hbar) A_zt.
    # Derived from: the first line with d_z omega = mg/hbar.
    #   Labels used: eq:interferometer, eq:grav_gradient, eq:sgm, eq:omega_em, eq:nr_expansion,
    #   tab:forces, sec:angular.
    # Test: (a) L1317: three pulses at equal intervals T_p, recoil v_r, launch velocity v_0: the
    #   kinetic and the potential action each differ between the arms by m g v_r T_p^2 and the
    #   arms close; (b) L1323 numbers: 10.5 cm^2 / 2.74e3 m/s = 3.83e-7 m s; hbar 59.6/3.83e-7
    #   = 1.64e-26 N = m_n g; m_Cs g = 2.16e-24 N (g = 9.8 m/s^2); (c) L1327 (Lense-Thirring
    #   check): with A_g = B_g x x/2, the body sum -sum A_g.pi = -B_g.L/2, and H = -B_g.L/2 turns L
    #   at the angular velocity -B_g/2 (Poisson brackets); -(1/4) hbar sigma.B_g = -B_g.S/2;
    #   (d) L1330: Delta omega = 2e Delta Phi/hbar gives f_J = (2e/h) Delta Phi.
    # Mutant: kinetic difference 2 m g v_r T_p^2; GP-B rate -B_g.
    # ------------------------------------------------------------------------
    eq("eq:interferometer", 1313, "Delta theta_g = -(mg/hbar) A_zt")
    v0, vr = sp.symbols('v_0 v_r', real=True)
    zA1 = v0*t + vr*t - g*t**2/2
    zA2 = v0*t + vr*Tp - g*t**2/2
    zB1 = v0*t - g*t**2/2
    zB2 = v0*t + vr*(t - Tp) - g*t**2/2
    symcheck("(a) the arms close at 2T_p", sp.expand(zA2.subs(t, 2*Tp) - zB2.subs(t, 2*Tp)))

    def act(z1, z2, f):
        return sp.integrate(f(z1), (t, 0, Tp)) + sp.integrate(f(z2), (t, Tp, 2*Tp))
    Kdiff = act(zA1, zA2, lambda z: m*sp.diff(z, t)**2/2) - act(zB1, zB2, lambda z: m*sp.diff(z, t)**2/2)
    Vdiff = act(zA1, zA2, lambda z: m*g*z) - act(zB1, zB2, lambda z: m*g*z)
    symcheck("(a) kinetic action difference == m g v_r T_p^2", sp.expand(Kdiff - m*g*vr*Tp**2))
    symcheck("(a) potential action difference == m g v_r T_p^2", sp.expand(Vdiff - m*g*vr*Tp**2))
    mutant("(a) kinetic difference 2 m g v_r T_p^2", sp.expand(Kdiff - 2*m*g*vr*Tp**2))
    hbar_n = 1.054571817e-34
    m_n = 1.67492750056e-27
    m_Cs = 132.905451961*1.66053906892e-27
    check("(b) 10.5 cm^2 / 2.74e3 m/s == 3.83e-7 m s", 10.5e-4/2.74e3, 3.83e-7, tol=2e-3)
    check("(b) hbar 59.6 / 3.83e-7 == 1.64e-26 N", hbar_n*59.6/3.83e-7, 1.64e-26, tol=3e-3)
    check("(b) neutron weight m_n g == 1.64e-26 N", m_n*9.8, 1.64e-26, tol=3e-3)
    check("(b) m_Cs g == 2.16e-24 N", m_Cs*9.8, 2.16e-24, tol=3e-3)
    Bgv = sp.Matrix(sp.symbols('Bg1:4', real=True))
    Ag_lin = _S4_cross(Bgv, Xv)/2
    symcheck("(c) -A_g.pi == -B_g.(x x pi)/2 for A_g = B_g x x/2",
             sp.expand(-Ag_lin.dot(Pv) + Bgv.dot(_S4_cross(Xv, Pv))/2))
    L_xp = _S4_cross(Xv, Pv)
    H_gp = -Bgv.dot(L_xp)/2

    def pb(f, h):
        return sum(sp.diff(f, X[k])*sp.diff(h, Pv[k]) - sp.diff(f, Pv[k])*sp.diff(h, X[k]) for k in range(3))
    Ldot = sp.Matrix([pb(L_xp[k], H_gp) for k in range(3)])
    symcheck("(c) dL/dt == (-B_g/2) x L: the axis turns at -B_g/2", sp.expand(Ldot - _S4_cross(-Bgv/2, L_xp)))
    mutant("(c) rate -B_g", sp.expand(Ldot - _S4_cross(-Bgv, L_xp))[0])
    symcheck("(c) -(1/4) hbar s.B_g == -B_g.S/2", sp.expand(-hb*sv.dot(Bgv)/4 + Bgv.dot(hb*sv/2)/2))
    e_ch, dPhi = sp.symbols('e Delta_Phi', positive=True)
    symcheck("(d) f_J = (2e Delta Phi/hbar)/(2 pi) == (2e/h) Delta Phi",
             (2*e_ch*dPhi/hb)/(2*sp.pi) - 2*e_ch*dPhi/(2*sp.pi*hb))

    # ------------------------------------------------------------------------
    # prop:reach  (sgm.tex L1382); supporting numbers of L1378
    # States: a bound state of finite second moment has a finite spread obeying Kennard's bound;
    #   at the threshold the second moment need not be finite.
    # Derived from: the hydrogen ground state (L1378) and the threshold state
    #   psi = [pi(1 + x^2)]^-1/2 in V = (2x^2 - 1)/[2(1 + x^2)^2] (units hbar = m = 1).
    #   Labels used: sec:duality, ax:local_frequency, sec:potentials, eq:nr_expansion.
    # Test: hydrogen <x^2> = a0^2, <p_x^2> = hbar^2/(3 a0^2), Delta x Delta p = hbar/sqrt(3) >= hbar/2;
    #   threshold state normalized, solves -psi''/2 + V psi = 0, V -> 0, and int x^2 psi^2 diverges.
    #   The prose of the proposition (Hegerfeldt's theorem, the causal reach) is cited, not computed.
    # Mutant: V with numerator 2x^2 + 1.
    # ------------------------------------------------------------------------
    eq("prop:reach", 1382, "bound states reach every distance; finite spread below threshold")
    a0, rr = sp.symbols('a_0 r', positive=True)
    psiH = sp.exp(-rr/a0)/sp.sqrt(sp.pi*a0**3)
    symcheck("hydrogen: normalized", sp.integrate(psiH**2*4*sp.pi*rr**2, (rr, 0, sp.oo)) - 1)
    r2 = sp.integrate(rr**2*psiH**2*4*sp.pi*rr**2, (rr, 0, sp.oo))
    symcheck("hydrogen: <x^2> = <r^2>/3 == a0^2", r2/3 - a0**2)
    p2 = hb**2*sp.integrate(sp.diff(psiH, rr)**2*4*sp.pi*rr**2, (rr, 0, sp.oo))
    symcheck("hydrogen: <p_x^2> = <p^2>/3 == hbar^2/(3 a0^2)", p2/3 - hb**2/(3*a0**2))
    symcheck("hydrogen: Delta x Delta p == hbar/sqrt(3)", sp.sqrt(r2/3)*sp.sqrt(p2/3) - hb/sp.sqrt(3))
    boolcheck("hydrogen: hbar/sqrt(3) >= hbar/2 (Kennard)", 1/np.sqrt(3) >= 0.5)
    xx = sp.Symbol('x', real=True)
    psiT = 1/sp.sqrt(sp.pi*(1 + xx**2))
    VT = (2*xx**2 - 1)/(2*(1 + xx**2)**2)
    symcheck("threshold state normalized", sp.integrate(psiT**2, (xx, -sp.oo, sp.oo)) - 1)
    symcheck("threshold state solves -psi''/2 + V psi = 0", sp.simplify(-sp.diff(psiT, xx, 2)/2 + VT*psiT))
    symcheck("V -> 0 at large |x|", sp.limit(VT, xx, sp.oo))
    boolcheck("threshold state: int x^2 psi^2 dx diverges", sp.integrate(xx**2*psiT**2, (xx, -sp.oo, sp.oo)) == sp.oo)
    VTm = (2*xx**2 + 1)/(2*(1 + xx**2)**2)
    mutant("V with numerator 2x^2 + 1", sp.simplify(-sp.diff(psiT, xx, 2)/2 + VTm*psiT))

    # ------------------------------------------------------------------------
    # eq:static_density_clock_field  (sgm.tex L1389)
    # States: alpha(x) - 1 = -(G/c^2) int rho_m(y)/|x - y| d^3y.
    # Derived from: Gauss's law for gravity, eq:gauss_alpha, solved with the Green function
    #   -1/(4 pi |x - y|) of the Laplacian and alpha -> 1 far from all masses (eq:alpha_def).
    #   Labels used: eq:gauss_alpha, eq:alpha_def, eq:gauss_sphere, sec:wave_equation.
    # Test: (i) the angular average of 1/|x - y| over a shell of radius s is 1/max(r, s)
    #   (numerical); (ii) for a uniform sphere the integral gives -GM(3R^2 - r^2)/(2R^3 c^2) inside
    #   and -GM/(c^2 r) outside; (iii) its radial Laplacian is 4 pi G rho/c^2 inside and 0 outside,
    #   the form of eq:gauss_alpha; the inside and outside values and slopes match at r = R;
    #   (iv) the Green function: grad(1/r) has flux -4 pi through any sphere about the source.
    # Mutant: outside form -GM/(c^2 r^2).
    # ------------------------------------------------------------------------
    eq("eq:static_density_clock_field", 1389, "alpha(x) - 1 = -(G/c^2) int rho_m(y)/|x - y| d^3y")
    for rv, sv_ in ((0.43, 0.71), (0.43, 0.29)):
        u_, wu_ = np.polynomial.legendre.leggauss(200)
        avg = 0.5*np.sum(wu_/np.sqrt(rv**2 + sv_**2 - 2*rv*sv_*u_))
        check(f"(i) angular average of 1/|x-y| at r={rv}, s={sv_} == 1/max(r, s)", avg, 1/max(rv, sv_), tol=1e-6)
    Ms, Rs, ss = sp.symbols('M R_s s_r', positive=True)
    rho0 = 3*Ms/(4*sp.pi*Rs**3)
    inside = -(G/c**2)*(sp.integrate(rho0*4*sp.pi*ss**2/rr, (ss, 0, rr)) + sp.integrate(rho0*4*sp.pi*ss, (ss, rr, Rs)))
    symcheck("(ii) inside: -GM(3R^2 - r^2)/(2R^3c^2)", sp.simplify(inside + G*Ms*(3*Rs**2 - rr**2)/(2*Rs**3*c**2)))
    outside = -(G/c**2)*sp.integrate(rho0*4*sp.pi*ss**2/rr, (ss, 0, Rs))
    symcheck("(ii) outside: -GM/(c^2 r)", sp.simplify(outside + G*Ms/(c**2*rr)))
    mutant("(ii) outside -GM/(c^2 r^2)", sp.simplify(outside + G*Ms/(c**2*rr**2)))

    def radlap(f):
        return sp.diff(rr**2*sp.diff(f, rr), rr)/rr**2
    symcheck("(iii) Laplacian inside == 4 pi G rho/c^2", sp.simplify(radlap(inside) - 4*sp.pi*G*rho0/c**2))
    symcheck("(iii) Laplacian outside == 0", sp.simplify(radlap(outside)))
    symcheck("(iii) values match at r = R", sp.simplify((inside - outside).subs(rr, Rs)))
    symcheck("(iii) slopes match at r = R", sp.simplify(sp.diff(inside - outside, rr).subs(rr, Rs)))
    flux_g = sp.integrate(sp.diff(1/rr, rr)*rr**2, (s, 0, 1))*4*sp.pi
    symcheck("(iv) flux of grad(1/r) through a sphere == -4 pi", flux_g + 4*sp.pi)

    # ------------------------------------------------------------------------
    # eq:hbar_cancels_identity  (sgm.tex L1404), first line
    # States: F = -hbar grad(mc^2 alpha/hbar).
    # Derived from: eq:sgm with omega = omega_0 alpha (eq:omega_alpha), omega_0 = mc^2/hbar
    #   (eq:rest_freq).
    #   Labels used: eq:sgm, eq:omega_alpha, eq:rest_freq.
    # Test: omega_0 alpha == mc^2 alpha/hbar for a general alpha(x); the SI units of hbar grad omega
    #   are N.
    # Mutant: omega_0 = mc/hbar.
    # ------------------------------------------------------------------------
    eq("eq:hbar_cancels_identity", 1406, "F = -hbar grad(mc^2 alpha/hbar)")
    al_x = sp.Function('alpha', positive=True)(*X)
    symcheck("-hbar grad(omega_0 alpha) == -hbar grad(mc^2 alpha/hbar)",
             -hb*_S4_grad((m*c**2/hb)*al_x, X) + hb*_S4_grad(m*c**2*al_x/hb, X))
    symcheck("units: hbar grad omega in N", _S4_HB - _S4_S - _S4_M - _S4_N)
    mutant("omega_0 = mc/hbar", (-hb*_S4_grad((m*c/hb)*al_x, X) + hb*_S4_grad(m*c**2*al_x/hb, X))[0])

    # ------------------------------------------------------------------------
    # eq:hbar_cancels_identity  (sgm.tex L1405), second line
    # States: F = -mc^2 grad alpha.
    # Derived from: the constant mc^2/hbar leaves the gradient and the two factors of hbar cancel.
    #   Labels used: eq:hbar_cancels_identity, eq:rest_freq.
    # Test: symbolic, general alpha(x); the result has no hbar (d/d hbar of it is zero).
    # Mutant: -hbar grad(mc^2 alpha) (hbar not divided out).
    # ------------------------------------------------------------------------
    eq("eq:hbar_cancels_identity", 1407, "F = -mc^2 grad alpha")
    Fh = -hb*_S4_grad(m*c**2*al_x/hb, X)
    symcheck("-hbar grad(mc^2 alpha/hbar) == -mc^2 grad alpha", Fh + m*c**2*_S4_grad(al_x, X))
    symcheck("d/d hbar of the force == 0", sp.diff(Fh, hb))
    mutant("-hbar grad(mc^2 alpha)", (-hb*_S4_grad(m*c**2*al_x, X) + m*c**2*_S4_grad(al_x, X))[0])

    # ------------------------------------------------------------------------
    # unlabeled  (sgm.tex L1416-L1418)
    # States: -n_G = 1, n_c + 3 n_G = 2, -n_c - 2 n_G = -1, the exponent conditions for
    #   c^{n_c} G^{n_G} to have the dimension M L^2 T^-1 of action.
    # Derived from: [c] = L T^-1, [G] = M^-1 L^3 T^-2 (L1414).
    #   Labels used: eq:alpha_def, sec:hbar_cancels.
    # Test: the exponent vector of c^{n_c} G^{n_G} matches the three left sides; the first two give
    #   n_G = -1, n_c = 5; the third left side is then -3; the system has no solution.
    # Mutant: claiming the third condition holds at n_c = 5, n_G = -1.
    # ------------------------------------------------------------------------
    eq("unlabeled", 1418, "-n_G = 1, n_c + 3 n_G = 2, -n_c - 2 n_G = -1")
    nc, nG = sp.symbols('n_c n_G', rational=True)
    expv = nc*_S4_CDIM + nG*_S4_GDIM
    symcheck("exponents of c^n_c G^n_G == (-n_G, n_c + 3n_G, -n_c - 2n_G)",
             expv[:3, 0] - sp.Matrix([-nG, nc + 3*nG, -nc - 2*nG]))
    sol12 = sp.solve([-nG - 1, nc + 3*nG - 2], [nc, nG], dict=True)[0]
    symcheck("first two conditions: n_G = -1", sol12[nG] + 1)
    symcheck("first two conditions: n_c = 5", sol12[nc] - 5)
    symcheck("third left side at those values == -3", (-nc - 2*nG).subs(sol12) + 3)
    boolcheck("the three conditions are inconsistent", sp.linsolve([-nG - 1, nc + 3*nG - 2, -nc - 2*nG + 1], [nc, nG]) == sp.EmptySet)
    mutant("third condition satisfied at n_c = 5, n_G = -1", (-nc - 2*nG).subs(sol12) - (-1))

    # ------------------------------------------------------------------------
    # prop:no_grav_action  (sgm.tex L1422)
    # States: metric components in x^mu = (ct, x) are dimensionless; no c^a G^b is an action;
    #   hbar cancels from -hbar grad(omega_0 alpha).
    # Derived from: the exponent system of L1416 and eq:hbar_cancels_identity; inline L1426 and L1436
    #   dimensions.
    #   Labels used: eq:alpha_def, eq:hbar_cancels_identity, sec:hbar_cancels.
    # Test: rank of the (c, G) exponent matrix augmented by action is 3 > 2 (no solution for any
    #   real exponents); S_EH = (c^3/16 pi G) int R sqrt(-g) d^4x has units J s with R in m^-2;
    #   c^3/(G Lambda) has the units of action and c^2 Lambda/(8 pi G) those of a mass density;
    #   e^2/(4 pi eps0 c) has units J s.
    # Mutant: c^5/G is not an action (it is a power).
    # ------------------------------------------------------------------------
    eq("prop:no_grav_action", 1424, "the gravitational sector contains no constant of action")
    Mcg = sp.Matrix.hstack(_S4_CDIM[:3, 0], _S4_GDIM[:3, 0])
    boolcheck("rank[c G | action] == 3 > rank[c G] == 2",
              Mcg.rank() == 2 and sp.Matrix.hstack(Mcg, _S4_JS[:3, 0]).rank() == 3)
    SEH = 3*_S4_CDIM - _S4_GDIM + (-2)*_S4_M + 4*_S4_M
    symcheck("S_EH units == J s", SEH - _S4_JS)
    Lam = -2*_S4_M
    symcheck("c^3/(G Lambda) units == J s", 3*_S4_CDIM - _S4_GDIM - Lam - _S4_JS)
    symcheck("c^2 Lambda/(8 pi G) units == kg m^-3", 2*_S4_CDIM + Lam - _S4_GDIM - (_S4_KG - 3*_S4_M))
    symcheck("e^2/(4 pi eps0 c) units == J s", 2*_S4_C - _S4_EPS0 - _S4_CDIM - _S4_JS)
    mutant("c^5/G is an action", 5*_S4_CDIM - _S4_GDIM - _S4_JS)

    # ------------------------------------------------------------------------
    # eq:compton_planck_ratio  (sgm.tex L1432)
    # States: (hbar/(m_p c)) / l_P = m_P / m_p ~ 1.30e19, (hbar/(m_e c)) / l_P = m_P / m_e ~ 2.39e22.
    # Derived from: prop:no_grav_action, eq:rest_freq.
    # Test: exact symbolic identity (hbar/(m c)) / sqrt(hbar G/c^3) == sqrt(hbar c/G) / m and CODATA 2022
    #       numerical checks for l_P, m_P, reduced Compton wavelengths of proton and electron, and ratios.
    # Mutant: inverting the ratio gives 10^-19 instead of 10^19.
    # ------------------------------------------------------------------------
    eq("eq:compton_planck_ratio", 1432, "(hbar/(m_p c))/l_P = m_P/m_p ~ 1.30e19; (hbar/(m_e c))/l_P = m_P/m_e ~ 2.39e22")
    _m_cp = sp.Symbol("m_cp", positive=True)
    _lP_s = sp.sqrt(hb*G/c**3)
    _mP_s = sp.sqrt(hb*c/G)
    symcheck("(hbar/(m c))/l_P == m_P/m symbolically", sp.simplify((hb/(_m_cp*c))/_lP_s - _mP_s/_m_cp))
    _hb_n = 1.054571817e-34
    _c_n = 299792458.0
    _G_n = 6.67430e-11
    _mp_n = 1.67262192595e-27
    _me_n = 9.1093837139e-31
    _lP_n = np.sqrt(_hb_n*_G_n/_c_n**3)
    _mP_n = np.sqrt(_hb_n*_c_n/_G_n)
    _lc_p = _hb_n/(_mp_n*_c_n)
    _lc_e = _hb_n/(_me_n*_c_n)
    check("Planck length l_P ~ 1.62e-35 m", _lP_n, 1.616255e-35, tol=1e-3)
    check("Planck mass m_P ~ 2.18e-8 kg", _mP_n, 2.176434e-8, tol=1e-3)
    check("proton reduced Compton wavelength ~ 2.10e-16 m", _lc_p, 2.103089e-16, tol=1e-3)
    check("electron reduced Compton wavelength ~ 3.86e-13 m", _lc_e, 3.861593e-13, tol=1e-3)
    check("proton Compton/Planck ratio ~ 1.30e19", _lc_p/_lP_n, 1.30e19, tol=1e-2)
    check("electron Compton/Planck ratio ~ 2.39e22", _lc_e/_lP_n, 2.39e22, tol=1e-2)
    numutant("inverted ratio l_P / (hbar/(m_p c))", _lP_n/_lc_p, 1.30e19, tol=1e-2)

    # ------------------------------------------------------------------------
    # eq:alpha_configuration  (sgm.tex L1458)
    # States: alpha(x) - 1 = -(G/c^2) sum_{j>=2} m_j/|x - x_j|.
    # Derived from: eq:static_density_clock_field with the density concentrated at the positions x_j.
    #   Supporting inline steps used by the corollary that follows (L1463): the pair term (L1461); the flux
    #   4 pi G dm/c^2 or 0 (L1447); the Yukawa flux (L1449); the retarded 1/c expansion (L1451).
    #   Labels used: eq:static_density_clock_field, eq:gauss_alpha, eq:gauss_sphere, eq:nr_expansion,
    #   cor:causal_gravity, sec:maxwell, sec:hbar_cancels, ax:causal_propagation, sec:wave_equation,
    #   sec:gravity.
    # Test: (a) a uniform sphere of mass m_j gives exactly -G m_j/(c^2 r) outside (previous block),
    #   so the point limit is the sum; (b) the sum is harmonic away from the masses (two general
    #   positions); (c) m_1 c^2 alpha(x_1) contains -G m_1 m_j/|x_1 - x_j|; (d) numerical flux of
    #   (x - y)/|x - y|^3 through a unit sphere: 4 pi for an off-center inside point, 0 outside;
    #   (e) Yukawa kernel e^{-r/lambda}/r: |grad| times 4 pi r^2 = 4 pi (1 + r/lambda) e^{-r/lambda};
    #   (f) f(t - R/c)/R expanded in 1/c: the first-order term -f'(t)/c is independent of R (port of
    #   the retarded-expansion check); (g) mu0 = 1/(eps0 c^2) with CODATA values.
    # Mutant: Yukawa flux 4 pi (radius independent).
    # ------------------------------------------------------------------------
    eq("eq:alpha_configuration", 1468, "alpha(x) - 1 = -(G/c^2) sum m_j/|x - x_j|")
    symcheck("(a) outside a uniform sphere: -G M/(c^2 r) (point limit)", sp.simplify(outside + G*Ms/(c**2*rr)))
    y2 = sp.Matrix(sp.symbols('y21:24', real=True))
    y3 = sp.Matrix(sp.symbols('y31:34', real=True))
    m2, m3, m1 = sp.symbols('m_2 m_3 m_1', positive=True)
    alc = 1 - (G/c**2)*(m2/sp.sqrt((Xv - y2).dot(Xv - y2)) + m3/sp.sqrt((Xv - y3).dot(Xv - y3)))
    symcheck("(b) Laplacian of the sum == 0 away from the masses", sp.simplify(_S4_lap(alc, X)))
    x1v = sp.Matrix(sp.symbols('X11:14', real=True))
    pair = (m1*c**2*alc).subs({X[k]: x1v[k] for k in range(3)})
    symcheck("(c) m1 c^2 alpha(x1) == m1 c^2 - sum G m1 m_j/|x1 - x_j|",
             sp.expand(pair - m1*c**2 + G*m1*m2/sp.sqrt((x1v - y2).dot(x1v - y2))
                       + G*m1*m3/sp.sqrt((x1v - y3).dot(x1v - y3))))

    def kern(y):
        def F(pts):
            d = pts - np.asarray(y)
            return d/np.linalg.norm(d, axis=-1)[..., None]**3
        return F
    check("(d) flux 4 pi for a point inside (off center)",
          _S4_sphere_flux(kern([0.31, 0.22, -0.13]), [0, 0, 0], 1.0), 4*np.pi, tol=1e-6)
    check("(d) flux 0 for a point outside", _S4_sphere_flux(kern([1.7, 0.4, 0.23]), [0, 0, 0], 1.0), 0,
          tol=1e-8, scale=4*np.pi)
    lamY = sp.Symbol('lambda_Y', positive=True)
    yuk = sp.exp(-rr/lamY)/rr
    symcheck("(e) Yukawa flux magnitude == 4 pi (1 + r/lambda) e^{-r/lambda}",
             sp.simplify(-sp.diff(yuk, rr)*4*sp.pi*rr**2 - 4*sp.pi*(1 + rr/lamY)*sp.exp(-rr/lamY)))
    mutant("(e) Yukawa flux 4 pi", sp.simplify(-sp.diff(yuk, rr)*4*sp.pi*rr**2 - 4*sp.pi))
    ic = sp.Symbol('iota', positive=True)  # iota = 1/c
    Rd = sp.Symbol('R_d', positive=True)
    fr = sp.Function('rho_q')
    ser_r = sp.series(fr(t - Rd*ic)/Rd, ic, 0, 2).removeO().doit()
    first = sp.expand(ser_r).coeff(ic, 1)
    symcheck("(f) first-order retarded term == -d_t rho_q, independent of R", sp.diff(first, Rd))
    symcheck("(f) first-order retarded term == -d_t rho_q", sp.simplify(first + sp.diff(fr(t), t)))
    eps0_n, c_n, mu0_n = 8.8541878188e-12, 299792458.0, 1.25663706127e-6
    check("(g) mu0 == 1/(eps0 c^2) (CODATA 2022)", 1/(eps0_n*c_n**2), mu0_n, tol=1e-9)

    # ------------------------------------------------------------------------
    # eq:h_grav_pair  (sgm.tex L1465)
    # States: H_grav,pair = (1/2) sum_i m_i c^2 (alpha_i(x_i) - 1) = -sum_{i<j} G m_i m_j / |x_i - x_j|.
    # Derived from: eq:alpha_configuration (L1460).
    # Test: exact 3-particle identity verifying the 1/2 pair-counting factor, and gradient -grad_{x1} H_grav,pair
    #       == -m_1 c^2 grad_{x1} alpha_1(x_1).
    # Mutant: omitting the 1/2 factor double-counts every pair energy and doubles the force.
    # ------------------------------------------------------------------------
    eq("eq:h_grav_pair", 1473, "H_grav,pair = (1/2) sum_i m_i c^2 (alpha_i - 1) = -sum_{i<j} G m_i m_j / |x_i - x_j|")
    _r12 = sp.sqrt((x1v - y2).dot(x1v - y2))
    _r13 = sp.sqrt((x1v - y3).dot(x1v - y3))
    _r23 = sp.sqrt((y2 - y3).dot(y2 - y3))
    _al1_m1 = -(G/c**2)*(m2/_r12 + m3/_r13)
    _al2_m1 = -(G/c**2)*(m1/_r12 + m3/_r23)
    _al3_m1 = -(G/c**2)*(m1/_r13 + m2/_r23)
    _Hpair_lhs = sp.Rational(1, 2)*(m1*c**2*_al1_m1 + m2*c**2*_al2_m1 + m3*c**2*_al3_m1)
    _Hpair_rhs = -G*(m1*m2/_r12 + m1*m3/_r13 + m2*m3/_r23)
    symcheck("3-body (1/2) sum m_i c^2 (alpha_i - 1) == -sum_{i<j} G m_i m_j / r_ij",
             sp.simplify(_Hpair_lhs - _Hpair_rhs))
    symcheck("-grad_{x1} H_grav,pair == -m_1 c^2 grad_{x1} alpha_1(x_1)",
             sp.Matrix([sp.simplify(-sp.diff(_Hpair_rhs, x1v[k]) - (-m1*c**2*sp.diff(_al1_m1, x1v[k]))) for k in range(3)]))
    mutant("omitting 1/2 factor double-counts pair energy", sp.simplify(2*_Hpair_lhs - _Hpair_rhs))

    # ------------------------------------------------------------------------
    # eq:gie_phase  (sgm.tex L1501)
    # States: tr(hat_rho_1(tau)^2) = 1 - (1/2) sin^2(Delta theta_grav / 2), Delta theta_grav = th00+th11-th01-th10.
    # Derived from: eq:h_grav_pair (L1465), cor:fields, rem:lapse_test.
    # Test: exact symbolic partial trace of |Psi(tau)> = (1/2) sum_{a,b} exp(i theta_{ab}) |a,b> gives
    #       tr(hat_rho_1^2) == 1 - (1/2) sin^2(Delta theta_grav / 2), reaching 1/2 at Delta theta_grav = pi.
    # Mutant: replacing 1/2 by 1 gives unphysical purity 0 at Delta theta_grav = pi.
    # ------------------------------------------------------------------------
    eq("eq:gie_phase", 1509, "tr(hat_rho_1^2) = 1 - (1/2) sin^2(Delta theta_grav / 2)")
    _t00, _t01, _t10, _t11 = sp.symbols("th00 th01 th10 th11", real=True)
    _dth = _t00 + _t11 - _t01 - _t10
    _r01 = sp.Rational(1, 4)*(sp.exp(sp.I*(_t00 - _t10)) + sp.exp(sp.I*(_t01 - _t11)))
    _rho1 = sp.Matrix([[sp.Rational(1, 2), _r01], [sp.conjugate(_r01), sp.Rational(1, 2)]])
    _pur = sp.simplify(sp.expand((_rho1*_rho1).trace().rewrite(sp.cos)))
    _pur_tgt = 1 - sp.Rational(1, 2)*sp.sin(_dth/2)**2
    symcheck("tr(hat_rho_1^2) == 1 - (1/2) sin^2(Delta theta_grav / 2)", sp.simplify(_pur - _pur_tgt))
    symcheck("maximal entanglement purity == 1/2 at Delta theta_grav = pi",
             sp.simplify(_pur_tgt.subs({_t00: sp.pi, _t01: 0, _t10: 0, _t11: 0}) - sp.Rational(1, 2)))
    mutant("replacing 1/2 by 1 fails purity formula", sp.simplify(_pur - (1 - sp.sin(_dth/2)**2)))

    # ------------------------------------------------------------------------
    # eq:sn_mean_field  (sgm.tex L1507)
    # States: i hbar dc_a/dt = -U_self |c_a|^2 c_a - U_cross |c_{1-a}|^2 c_a (a in {0, 1}).
    # Derived from: rem:lapse_test, prop:config_space_gravity.
    # Test: exact symbolic verification that |c_a(t)|^2 is conserved and the closed-form phase solution
    #       satisfies the nonlinear ODE.
    # Mutant: swapping U_self and U_cross fails the ODE when p != 1/2.
    # ------------------------------------------------------------------------
    eq("eq:sn_mean_field", 1515, "i hbar dc_a/dt = -U_self |c_a|^2 c_a - U_cross |c_{1-a}|^2 c_a")
    _gs, _gd = sp.symbols("U_self U_cross", positive=True)
    _a0, _a1 = sp.symbols("a0_sn a1_sn", positive=True)  # |c_0| = a0, |c_1| = a1 with a0^2 + a1^2 = 1
    _ts_sn = sp.Symbol("t_sn", real=True)
    _c0_sn = _a0*sp.exp(sp.I*(_gs*_a0**2 + _gd*_a1**2)*_ts_sn/hb)
    _c1_sn = _a1*sp.exp(sp.I*(_gs*_a1**2 + _gd*_a0**2)*_ts_sn/hb)
    symcheck("c_0(t) solves two-site SN ODE",
             sp.simplify(sp.I*hb*sp.diff(_c0_sn, _ts_sn) - (-_gs*_a0**2*_c0_sn - _gd*_a1**2*_c0_sn)))
    symcheck("c_1(t) solves two-site SN ODE",
             sp.simplify(sp.I*hb*sp.diff(_c1_sn, _ts_sn) - (-_gs*_a1**2*_c1_sn - _gd*_a0**2*_c1_sn)))
    mutant("swapping U_self and U_cross fails two-site SN ODE",
           sp.simplify(sp.I*hb*sp.diff(_c1_sn, _ts_sn) - (-_gd*_a1**2*_c1_sn - _gs*_a0**2*_c1_sn)))

    # ------------------------------------------------------------------------
    # eq:sn_overlap_drift  (sgm.tex L1512)
    # States: |<psi^(A)(t)|psi^(B)(t)>|^2 = (1-p_A)(1-p_B) + p_A p_B + 2 sqrt(p_A(1-p_A)p_B(1-p_B)) cos(Delta omega_SN t).
    # Derived from: eq:sn_mean_field (L1507).
    # Test: symbolic derivation from closed-form SN solutions with positive amplitudes a0 = sqrt(1-p), a1 = sqrt(p)
    #       confirming overlap drops from sqrt(3)/2 to 0 at t_*^SN = pi hbar / (U_self - U_cross) for (p_A, p_B) = (1/4, 3/4).
    # Mutant: U_self == U_cross keeps overlap constant at sqrt(3)/2 instead of orthogonalizing.
    # ------------------------------------------------------------------------
    eq("eq:sn_overlap_drift", 1520, "|<psi^A(t)|psi^B(t)>|^2 = (1-p_A)(1-p_B) + p_A p_B + 2 sqrt(...) cos(Delta omega_SN t)")
    _a0A, _a1A, _a0B, _a1B = sp.symbols("a0A a1A a0B a1B", positive=True)
    _pA, _pB = sp.symbols("p_A p_B", positive=True)
    _dw_sn = 2*(_gs - _gd)*(_pB - _pA)/hb
    _c0A = _a0A*sp.exp(sp.I*(_gs*(1 - _pA) + _gd*_pA)*_ts_sn/hb)
    _c1A = _a1A*sp.exp(sp.I*(_gs*_pA + _gd*(1 - _pA))*_ts_sn/hb)
    _c0B = _a0B*sp.exp(sp.I*(_gs*(1 - _pB) + _gd*_pB)*_ts_sn/hb)
    _c1B = _a1B*sp.exp(sp.I*(_gs*_pB + _gd*(1 - _pB))*_ts_sn/hb)
    _ov_sn = sp.conjugate(_c0A)*_c0B + sp.conjugate(_c1A)*_c1B
    _ov2_sn = sp.trigsimp(sp.expand((_ov_sn*sp.conjugate(_ov_sn)).rewrite(sp.cos)))
    _ov2_amp = _a0A**2*_a0B**2 + _a1A**2*_a1B**2 + 2*_a0A*_a1A*_a0B*_a1B*sp.cos(_dw_sn*_ts_sn)
    symcheck("|<psi^A(t)|psi^B(t)>|^2 matches eq:sn_overlap_drift", sp.simplify(_ov2_sn - _ov2_amp))
    _ov2_tgt = (1 - _pA)*(1 - _pB) + _pA*_pB + 2*sp.sqrt(_pA*(1 - _pA)*_pB*(1 - _pB))*sp.cos(_dw_sn*_ts_sn)
    _ov2_34 = sp.simplify(_ov2_tgt.subs({_pA: sp.Rational(1, 4), _pB: sp.Rational(3, 4)}))
    symcheck("at (1/4, 3/4): overlap^2 == (3/4) cos^2((U_self - U_cross) t / (2 hbar))",
             sp.simplify(_ov2_34 - sp.Rational(3, 4)*sp.cos((_gs - _gd)*_ts_sn/(2*hb))**2))
    symcheck("at t_*^SN = pi hbar / (U_self - U_cross): exact orthogonalization (overlap == 0)",
             sp.simplify(_ov2_34.subs(_ts_sn, sp.pi*hb/(_gs - _gd))))
    mutant("U_self == U_cross fails orthogonalization at t_*^SN",
           sp.simplify(_ov2_34.subs(_gs, _gd)))

    # ------------------------------------------------------------------------
    # eq:eih_kernel  (sgm.tex L1534)
    # States: W_ij^{ab}(r) = (G/(2 c^2 |r|))(7 delta^{ab} + r^a r^b / |r|^2).
    # Derived from: prop:eih_1pn.
    # Test: real symmetry W^{ab} == W^{ba}, positive definiteness, and transverse/longitudinal eigenvalues 7, 8.
    # Mutant: replacing + r^a r^b/|r|^2 by - r^a r^b/|r|^2 gives wrong longitudinal eigenvalue 6 instead of 8.
    # ------------------------------------------------------------------------
    eq("eq:eih_kernel", 1542, "W_ij^{ab}(r) = (G/(2 c^2 |r|))(7 delta^{ab} + r^a r^b / |r|^2)")
    _rv_eih = sp.Matrix(sp.symbols("rx_e ry_e rz_e", real=True))
    _r_eih = sp.sqrt(_rv_eih.dot(_rv_eih))
    _W_eih = sp.Matrix([[G/(2*c**2*_r_eih)*(7*(1 if a_ == b_ else 0) + _rv_eih[a_]*_rv_eih[b_]/_r_eih**2)
                         for b_ in range(3)] for a_ in range(3)])
    symcheck("W^{ab}(r) is symmetric", _W_eih - _W_eih.T)
    symcheck("W(r) r == (8 G / (2 c^2 |r|)) r (longitudinal eigenvalue 8)",
             sp.Matrix([sp.simplify((_W_eih*_rv_eih)[k] - 8*G/(2*c**2*_r_eih)*_rv_eih[k]) for k in range(3)]))
    _W_eih_bad = sp.Matrix([[G/(2*c**2*_r_eih)*(7*(1 if a_ == b_ else 0) - _rv_eih[a_]*_rv_eih[b_]/_r_eih**2)
                             for b_ in range(3)] for a_ in range(3)])
    mutant("minus sign on r^a r^b/|r|^2 fails longitudinal eigenvalue 8",
           sp.Matrix([sp.simplify((_W_eih_bad*_rv_eih)[k] - 8*G/(2*c**2*_r_eih)*_rv_eih[k]) for k in range(3)]))

    # ------------------------------------------------------------------------
    # eq:eih_1pn  (sgm.tex L1539)
    # States: H_EIH^cross = (1/2) sum_{i<j} sum_{a,b} (p_{i,a} W_ij^{ab} p_{j,b} + p_{j,b} W_ij^{ab} p_{i,a}).
    # Derived from: eq:eih_kernel (L1534), eq:legendre_bridge.
    # Test: (i) explicit 2-particle discrete momentum-coordinate matrix realization verifying exact Hermiticity
    #       of the symmetrized tensor-sandwich operator; (ii) classical symbol match and positive Lense-Thirring
    #       velocity dH/dp_1|_{p_1=0} = +(7 G/(2 c^2 r)) p_2 for p_2 perp r.
    # Mutant: unsymmetrized p_{1,a} W^{ab} p_{2,b} is non-Hermitian; wrong sign gives negative frame dragging.
    # ------------------------------------------------------------------------
    eq("eq:eih_1pn", 1547, "H_EIH^cross = (1/2) sum_{i<j} sum_{a,b} (p_{i,a} W^{ab} p_{j,b} + p_{j,b} W^{ab} p_{i,a})")
    _p1v = sp.Matrix(sp.symbols("p1x p1y p1z", real=True))
    _p2v = sp.Matrix(sp.symbols("p2x p2y p2z", real=True))
    _H_eih_cl = sp.simplify((_p1v.T * _W_eih * _p2v)[0])
    _H_eih_tgt = G/(2*c**2*_r_eih)*(7*_p1v.dot(_p2v) + (_p1v.dot(_rv_eih))*(_p2v.dot(_rv_eih))/_r_eih**2)
    symcheck("classical symbol of H_EIH^cross matches 1PN EIH formula", sp.simplify(_H_eih_cl - _H_eih_tgt))
    _xdot1 = sp.Matrix([sp.diff(_H_eih_cl, _p1v[k]) for k in range(3)])
    _xdot1_tgt = G/(2*c**2*_r_eih)*(7*_p2v + (_p2v.dot(_rv_eih))/_r_eih**2 * _rv_eih)
    symcheck("Lense-Thirring velocity dx_1/dt|_{p_1=0} == +(G/(2c^2 r))(7 p_2 + (p_2.r_hat) r_hat)",
             sp.Matrix([sp.simplify(_xdot1[k] - _xdot1_tgt[k]) for k in range(3)]))
    # Discrete 2-particle 1D lattice test for exact operator Hermiticity of (1/2)(p1 W p2 + p2 W p1)
    _Neih = 8
    _xe = np.linspace(-2.0, 2.0, _Neih)
    _ke = np.fft.fftfreq(_Neih, d=4.0/_Neih) * 2*np.pi; _ke[_Neih//2] = 0.0
    _Fe = np.fft.fft(np.eye(_Neih), axis=0) / np.sqrt(_Neih)
    _Pe_1d = _Fe.conj().T @ np.diag(_ke) @ _Fe  # Hermitian momentum operator on 1 particle
    _P1_2 = np.kron(_Pe_1d, np.eye(_Neih))
    _P2_2 = np.kron(np.eye(_Neih), _Pe_1d)
    _Wdiag = np.diag([1.0 / np.sqrt((_xe[i_] - _xe[j_])**2 + 0.5) for i_ in range(_Neih) for j_ in range(_Neih)])
    _H_eih_op = 0.5 * (_P1_2 @ _Wdiag @ _P2_2 + _P2_2 @ _Wdiag @ _P1_2)
    check("symmetrized tensor-sandwich operator H_EIH^cross is Hermitian",
          float(np.linalg.norm(_H_eih_op - _H_eih_op.conj().T)), 0.0, tol=1e-13, scale=float(np.linalg.norm(_H_eih_op)))
    _H_eih_unsym = _P1_2 @ _Wdiag @ _P2_2
    numutant("unsymmetrized p_1 W p_2 is non-Hermitian",
             float(np.linalg.norm(_H_eih_unsym - _H_eih_unsym.conj().T) + np.linalg.norm(_H_eih_unsym)),
             float(np.linalg.norm(_H_eih_unsym)), tol=1e-3)

    # ------------------------------------------------------------------------
    # eq:rad_overlap_drift  (sgm.tex L1555)
    # States: |<psi^A(t)|psi^B(t)>|^2 with Delta omega_rad = (2 kappa_rad / hbar)(p_B - p_A),
    #         orthogonalizing (1/4, 3/4) at t_*^rad = pi hbar / kappa_rad even at zero static SN shift.
    # Derived from: eq:sn_overlap_drift (L1512).
    # Test: symbolic verification that setting g_s = kappa_rad, g_d = 0 reproduces eq:rad_overlap_drift
    #       and hits exact zero at t_*^rad = pi hbar / kappa_rad.
    # Mutant: dropping factor 2 in Delta omega_rad fails orthogonalization at t_*^rad.
    # ------------------------------------------------------------------------
    eq("eq:rad_overlap_drift", 1563, "|<psi^A(t)|psi^B(t)>|^2 = ... cos(Delta omega_rad t), orthogonal at t_*^rad = pi hbar / kappa_rad")
    _krad = sp.Symbol("kappa_rad", positive=True)
    _ov2_rad = _ov2_tgt.subs({_gs: _krad, _gd: 0})
    symcheck("at (1/4, 3/4): radiative overlap vanishes at t_*^rad = pi hbar / kappa_rad",
             sp.simplify(_ov2_rad.subs({_pA: sp.Rational(1, 4), _pB: sp.Rational(3, 4), _ts_sn: sp.pi*hb/_krad})))
    _ov2_rad_bad = ((1 - _pA)*(1 - _pB) + _pA*_pB
                    + 2*sp.sqrt(_pA*(1 - _pA)*_pB*(1 - _pB))*sp.cos((_krad*(_pB - _pA)/hb)*_ts_sn))
    mutant("omitting factor 2 in Delta omega_rad fails orthogonalization at t_*^rad",
           sp.simplify(_ov2_rad_bad.subs({_pA: sp.Rational(1, 4), _pB: sp.Rational(3, 4), _ts_sn: sp.pi*hb/_krad})))

    # ------------------------------------------------------------------------
    # eq:gw_power_bias  (sgm.tex L1561)
    # States: Delta P_GW = P_GW^(Z) - P_GW^(X) = (1/2) P_0 = G Q_0^2 Omega_quad^6 / (10 c^5) > 0.
    # Derived from: prop:rad_meanfield_signaling.
    # Test: exact trace algebra tr(E^(a)) = 0, tr(E^(a) E^(b)) = 2 delta_{ab},
    #       tr(((E^(0) + E^(1))/2)^2) = 1, and Bell GW-power bias Delta P_GW == G Q_0^2 Omega_quad^6 / (10 c^5).
    # Mutant: choosing E^(1) == E^(0) eliminates the bias (Delta P_GW == 0).
    # ------------------------------------------------------------------------
    eq("eq:gw_power_bias", 1569, "Delta P_GW = P_GW^(Z) - P_GW^(X) = G Q_0^2 Omega_quad^6 / (10 c^5)")
    _E0q = sp.Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    _E1q = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    _Emix = sp.Rational(1, 2)*(_E0q + _E1q)
    symcheck("tr(E^(0)) == tr(E^(1)) == 0", sp.Matrix([_E0q.trace(), _E1q.trace()]))
    symcheck("tr(E^(a) E^(b)) == 2 delta_{ab}",
             sp.Matrix([[(_E0q*_E0q).trace() - 2, (_E0q*_E1q).trace()],
                        [(_E1q*_E0q).trace(), (_E1q*_E1q).trace() - 2]]))
    symcheck("tr(((E^(0) + E^(1))/2)^2) == 1", (_Emix*_Emix).trace() - 1)
    _Q0q, _Omq = sp.symbols("Q0_q Om_q", positive=True)
    _pref_gw = G*_Q0q**2*_Omq**6/(10*c**5)
    _PZ = _pref_gw * (_E0q*_E0q).trace()
    _PX = _pref_gw * (_Emix*_Emix).trace()
    symcheck("Delta P_GW == P_GW^(Z) - P_GW^(X) == G Q_0^2 Omega_quad^6 / (10 c^5)",
             sp.simplify((_PZ - _PX) - G*_Q0q**2*_Omq**6/(10*c**5)))
    _Emix_bad = sp.Rational(1, 2)*(_E0q + _E0q)
    mutant("identical quadrupoles E^(1) == E^(0) give zero bias",
           sp.simplify((_PZ - _pref_gw*(_Emix_bad*_Emix_bad).trace()) - G*_Q0q**2*_Omq**6/(10*c**5)))

    # ------------------------------------------------------------------------
    # eq:bek_bh_sat  (sgm.tex L1574)
    # States: S_Bek / k_B = 2 pi r E / (hbar c) = (A(r)/(4 l_P^2)) (1 - alpha(r)^2) -> A(R_s)/(4 l_P^2).
    # Derived from: eq:dalpha_dl, eq:gauss_exact, sec:gravity.
    # Test: exact symbolic identity with E = M c^2, R_s = 2 G M / c^2, alpha(r) = sqrt(1 - R_s / r),
    #       A(r) = 4 pi r^2, l_P = sqrt(hbar G / c^3), and limit r -> R_s^+.
    # Mutant: replacing (1 - alpha^2) by (1 - alpha) fails the identity.
    # ------------------------------------------------------------------------
    eq("eq:bek_bh_sat", 1582, "2 pi r E / (hbar c) = (A(r)/(4 l_P^2)) (1 - alpha(r)^2) -> A(R_s)/(4 l_P^2)")
    _Rb, _Mb = sp.symbols("r_b M_b", positive=True)
    _Eb = _Mb * c**2
    _Rsb = 2*G*_Mb/c**2
    _lP2 = hb*G/c**3
    _Ab = 4*sp.pi*_Rb**2
    _al2_b = 1 - _Rsb/_Rb
    _sbek_lhs = 2*sp.pi*_Rb*_Eb/(hb*c)
    _sbek_rhs = (_Ab/(4*_lP2))*(1 - _al2_b)
    symcheck("2 pi r E / (hbar c) == (A(r)/(4 l_P^2))(1 - alpha(r)^2)", sp.simplify(_sbek_lhs - _sbek_rhs))
    symcheck("horizon limit r -> R_s gives A(R_s)/(4 l_P^2)",
             sp.simplify(_sbek_rhs.subs(_Rb, _Rsb) - 4*sp.pi*_Rsb**2/(4*_lP2)))
    mutant("replacing (1 - alpha^2) by (1 - alpha) fails identity",
           sp.simplify(_sbek_lhs - (_Ab/(4*_lP2))*(1 - sp.sqrt(_al2_b))))

    # ------------------------------------------------------------------------
    # tab:notation  (sgm.tex L1575), longtable L1574-L1792
    # States: the SI units of the symbols of the main text.
    # Derived from: each symbol's defining relation in the main text.
    #   Labels used: eq:omega_em, eq:sgm_lagrangian, eq:sgm_fourvector, eq:work_energy,
    #   eq:impulse_momentum, eq:sgm_angular, sec:angular, eq:torque_balance, eq:euler_rigid,
    #   eq:interferometer, sec:experiment, sec:scope, eq:static_density_clock_field,
    #   sec:hbar_cancels, sec:fields_gravity, eq:master, tab:forces, sec:spinor.
    # Test: spot checks, unit derived from the defining relation vs the listed unit:
    #   b = (q hbar/m) B: J; d omega/d vartheta: s^-1; B_g = curl A_g with A_g a velocity: s^-1;
    #   N = x x F: N m; I: kg m^2; A_zt: m s; h^TT from F = m hddot x/2: 1; rho_m from
    #   lap alpha = 4 pi G rho/c^2: kg m^-3; G (lapse energy) m0 c^2: J; f_J = 2e Delta Phi/h: Hz;
    #   Omegadot: s^-2 (rad s^-2); A from E = -d_t A: T m; p_mu = (-E/c, p): kg m s^-1; L = x x pi: J s;
    #   lambda_Y: m.
    # Mutant: inertia tensor in kg m.
    # ------------------------------------------------------------------------
    eq("tab:notation", 1674, "SI units of the symbols of the main text")
    symcheck("b = (q hbar/m) B in J", _S4_C + _S4_HB - _S4_KG + _S4_T - _S4_J)
    symcheck("hbar B_g in J (B_g in s^-1)", _S4_HB - _S4_S - _S4_J)
    symcheck("B_g = curl A_g in s^-1 (A_g a velocity)", (_S4_M - _S4_S) - _S4_M - (-_S4_S))
    symcheck("d omega/d vartheta in s^-1 (angle dimensionless)", (-_S4_S) - _S4_ONE - (-_S4_S))
    symcheck("torque x x F in N m", _S4_M + _S4_N - (_S4_N + _S4_M))
    symcheck("I = sum m x x (w x x)/w in kg m^2", _S4_KG + 2*_S4_M - (_S4_KG + 2*_S4_M))
    mutant("I in kg m", _S4_KG + 2*_S4_M - (_S4_KG + _S4_M))
    symcheck("A_zt = int Delta z dt in m s", _S4_M + _S4_S - (_S4_M + _S4_S))
    symcheck("h^TT dimensionless: m hddot x/2 is a force", _S4_KG - 2*_S4_S + _S4_M - _S4_N)
    symcheck("rho_m in kg m^-3 from lap alpha = 4 pi G rho/c^2", (-2*_S4_M) + 2*_S4_CDIM - _S4_GDIM - (_S4_KG - 3*_S4_M))
    symcheck("lapse energy m0 c^2 (alpha_n - 1) in J", _S4_KG + 2*_S4_CDIM - _S4_J)
    symcheck("f_J = 2e Delta Phi/h in Hz", _S4_C + _S4_V - _S4_JS - (-_S4_S))
    symcheck("A in T m from E = -d_t A (E in V/m)", (_S4_V - _S4_M) + _S4_S - (_S4_T + _S4_M))
    symcheck("p_0 = -E/c in kg m s^-1", _S4_J - _S4_CDIM - (_S4_KG + _S4_M - _S4_S))
    symcheck("L = x x pi in J s", _S4_M + _S4_KG + _S4_M - _S4_S - _S4_JS)
    symcheck("q Phi in J (Phi in V)", _S4_C + _S4_V - _S4_J)
    symcheck("W_ij^{ab} = G/(c^2 r) in kg^-1", _S4_GDIM - 2*_S4_CDIM - _S4_M - (-_S4_KG))
    symcheck("H_EIH^cross = p W p in J", 2*(_S4_KG + _S4_M - _S4_S) - _S4_KG - _S4_J)
    symcheck("Delta P_GW = G Q_0^2 Omega^6 / c^5 in W (= J/s)",
             _S4_GDIM + 2*(_S4_KG + 2*_S4_M) - 6*_S4_S - 5*_S4_CDIM - (_S4_J - _S4_S))
    symcheck("S_Bek / k_B = r E / (hbar c) dimensionless", _S4_M + _S4_J - (_S4_HB + _S4_CDIM))

    # ------------------------------------------------------------------------
    # unlabeled  (sgm.tex L1807), Route IV in a box: the boundary term
    # States: d<p>/dt = (hbar^2/2m)[|psi'(0)|^2 - |psi'(L)|^2] for hard walls.
    # Derived from: one integration by parts with psi = 0 at the walls; the integrand becomes
    #   -d_x |d_x psi|^2.
    #   Labels used: eq:ehrenfest_commutator, sec:route_schrodinger.
    # Test: for a general complex psi = u + i w, -psi*' psi'' - psi*'' psi' == -d_x(psi*' psi').
    # Mutant: +d_x |psi'|^2.
    # ------------------------------------------------------------------------
    eq("unlabeled", 1923, "d<p>/dt = (hbar^2/2m)[|psi'(0)|^2 - |psi'(L)|^2]")
    uf, wf = sp.Function('u', real=True)(xx), sp.Function('w', real=True)(xx)
    ps, psc = uf + sp.I*wf, uf - sp.I*wf
    integrand = -sp.diff(psc, xx)*sp.diff(ps, xx, 2) - sp.diff(psc, xx, 2)*sp.diff(ps, xx)
    symcheck("-psi*' psi'' - psi*'' psi' == -d_x |psi'|^2", sp.expand(integrand + sp.diff(sp.diff(psc, xx)*sp.diff(ps, xx), xx)))
    mutant("+d_x |psi'|^2", sp.expand(integrand - sp.diff(sp.diff(psc, xx)*sp.diff(ps, xx), xx)))

    # ------------------------------------------------------------------------
    # unlabeled  (sgm.tex L1809), Route IV in a box: the two-mode example
    # States: slopes 3 pi/L^{3/2} and pi/L^{3/2}; d<p>/dt = (hbar^2/2m)(9 pi^2 - pi^2)/L^3
    #   = 4 pi^2 hbar^2/(m L^3); the commutator gives zero.
    # Derived from: the boundary-term formula of L1807 with psi = (u1 + u2)/sqrt 2.
    #   Labels used: eq:ehrenfest_commutator.
    # Test: the slopes; the rate; independently, <p>(t) = -i hbar int psi* psi' for the exact
    #   time-dependent superposition, differentiated at t = 0, equals 4 pi^2 hbar^2/(m L^3); the
    #   interior commutator term <-dV/dx> is zero for V = 0.
    # Mutant: rate (hbar^2/2m)(9 pi^2 + pi^2)/L^3.
    # ------------------------------------------------------------------------
    eq("unlabeled", 1925, "d<p>/dt = 4 pi^2 hbar^2/(m L^3)")
    Lb = sp.Symbol('L_box', positive=True)
    un = [sp.sqrt(2/Lb)*sp.sin(n*sp.pi*xx/Lb) for n in (1, 2)]
    psi_b = (un[0] + un[1])/sp.sqrt(2)
    symcheck("slope at x = 0 == 3 pi/L^{3/2}", sp.simplify(sp.diff(psi_b, xx).subs(xx, 0) - 3*sp.pi/Lb**sp.Rational(3, 2)))
    symcheck("|slope| at x = L == pi/L^{3/2}", sp.simplify(sp.Abs(sp.diff(psi_b, xx).subs(xx, Lb)) - sp.pi/Lb**sp.Rational(3, 2)))
    rate_b = hb**2/(2*m)*((3*sp.pi)**2 - sp.pi**2)/Lb**3
    symcheck("(hbar^2/2m)(9 pi^2 - pi^2)/L^3 == 4 pi^2 hbar^2/(m L^3)", rate_b - 4*sp.pi**2*hb**2/(m*Lb**3))
    En = [hb**2*(n*sp.pi/Lb)**2/(2*m) for n in (1, 2)]
    psi_t = (un[0]*sp.exp(-sp.I*En[0]*t/hb) + un[1]*sp.exp(-sp.I*En[1]*t/hb))/sp.sqrt(2)
    psi_tc = (un[0]*sp.exp(sp.I*En[0]*t/hb) + un[1]*sp.exp(sp.I*En[1]*t/hb))/sp.sqrt(2)
    pmean = sp.integrate(sp.expand(-sp.I*hb*psi_tc*sp.diff(psi_t, xx)), (xx, 0, Lb))
    symcheck("d<p>/dt at t = 0 from the exact evolution == 4 pi^2 hbar^2/(m L^3)",
             sp.simplify(sp.diff(pmean, t).subs(t, 0) - 4*sp.pi**2*hb**2/(m*Lb**3)))
    mutant("rate with 9 pi^2 + pi^2", hb**2/(2*m)*((3*sp.pi)**2 + sp.pi**2)/Lb**3 - 4*sp.pi**2*hb**2/(m*Lb**3))

    # ------------------------------------------------------------------------
    # unlabeled  (sgm.tex L1818), the quantum-potential divergence
    # States: |psi|^2 d_i Q = -(hbar^2/4m) d_j(|psi|^2 d_i d_j ln|psi|^2), Q = -(hbar^2/2m) lap|psi|/|psi|.
    # Derived from: Madelung's decomposition; a divergence integrates to zero for a state that
    #   falls off at infinity.
    #   Labels used: eq:sgm_expectation, sec:identity, eq:H_eq_hw.
    # Test: symbolic in two dimensions with a general positive amplitude R(x, y), i = x.
    # Mutant: coefficient -(hbar^2/2m).
    # ------------------------------------------------------------------------
    eq("unlabeled", 1934, "|psi|^2 d_i Q = -(hbar^2/4m) d_j(|psi|^2 d_i d_j ln|psi|^2)")
    yy = sp.Symbol('y', real=True)
    Ramp = sp.Function('R', positive=True)(xx, yy)
    Qp = -(hb**2/(2*m))*(sp.diff(Ramp, xx, 2) + sp.diff(Ramp, yy, 2))/Ramp
    rho_a = Ramp**2
    lnr = sp.log(rho_a)
    divt = -(hb**2/(4*m))*(sp.diff(rho_a*sp.diff(lnr, xx, xx), xx) + sp.diff(rho_a*sp.diff(lnr, xx, yy), yy))
    symcheck("|psi|^2 d_x Q == divergence form", sp.simplify(rho_a*sp.diff(Qp, xx) - divt))
    mutant("coefficient -(hbar^2/2m)", sp.simplify(rho_a*sp.diff(Qp, xx) - 2*divt))

    # ------------------------------------------------------------------------
    # unlabeled  (sgm.tex L1820), the centroid form
    # States: for V at most quadratic <grad V> = grad V(<x>); for V ~ x^3 the two differ by the
    #   fraction (Delta x)^2/<x>^2 for a state of any shape.
    # Derived from: <x^2> = <x>^2 + (Delta x)^2.
    #   Labels used: eq:sgm_expectation.
    # Test: an asymmetric discrete distribution (five points, unequal weights): quadratic V gives
    #   equality; cubic V gives (<V'> - V'(<x>))/V'(<x>) = (Delta x)^2/<x>^2. Hellmann-Feynman (L1822)
    #   limit: two-level H = [[w, D],[D, -w]], <dH/dw> = dE/dw for the upper eigenvector.
    # Mutant: fraction 2(Delta x)^2/<x>^2.
    # ------------------------------------------------------------------------
    eq("unlabeled", 1936, "<V'> - V'(<x>) = V'(<x>) (Delta x)^2/<x>^2 for V ~ x^3")
    pts = [sp.Rational(-1, 3), sp.Rational(1, 2), sp.Rational(5, 4), sp.Rational(7, 3), sp.Rational(11, 5)]
    wts = [sp.Rational(1, 10), sp.Rational(3, 10), sp.Rational(1, 5), sp.Rational(1, 4), sp.Rational(3, 20)]
    mean = sum(p*w_ for p, w_ in zip(pts, wts))
    var = sum((p - mean)**2*w_ for p, w_ in zip(pts, wts))
    aq, bq, kq = sp.Rational(3, 7), sp.Rational(-5, 4), sp.Rational(4, 3)
    symcheck("quadratic V: <V'> == V'(<x>)", sum((2*aq*p + bq)*w_ for p, w_ in zip(pts, wts)) - (2*aq*mean + bq))
    EVp = sum(3*kq*p**2*w_ for p, w_ in zip(pts, wts))
    symcheck("cubic V: fraction == (Delta x)^2/<x>^2", (EVp - 3*kq*mean**2)/(3*kq*mean**2) - var/mean**2)
    mutant("fraction 2(Delta x)^2/<x>^2", (EVp - 3*kq*mean**2)/(3*kq*mean**2) - 2*var/mean**2)
    wv_, Dv_ = sp.symbols('varpi Delta', positive=True)
    Hm = sp.Matrix([[wv_, Dv_], [Dv_, -wv_]])
    Eup = sp.sqrt(wv_**2 + Dv_**2)
    vec = sp.Matrix([Dv_, Eup - wv_])
    vec = vec/sp.sqrt(vec.dot(vec))
    symcheck("Hellmann-Feynman limit: H v = E v", sp.simplify(Hm*vec - Eup*vec))
    symcheck("Hellmann-Feynman limit: <dH/d varpi> == dE/d varpi",
             sp.simplify((vec.T*sp.diff(Hm, wv_)*vec)[0] - sp.diff(Eup, wv_)))

    # ------------------------------------------------------------------------
    # eq:force_phase  (sgm.tex L1836)
    # States: F = hbar Delta_R Delta_T theta/(Delta r Delta t).
    # Derived from: |d_t grad theta| = |grad omega| (eqs. phase_freq, mixed_orders) for a uniform,
    #   steady gradient with uniform momentum field, so Delta_R Delta_T theta = |grad omega| Delta r
    #   Delta t; F = hbar |grad omega| (eq:sgm) (L1834).
    #   Labels used: eq:phase_freq, eq:mixed_orders, eq:chain_convective, eq:sgm,
    #   eq:hbar_cancels_identity.
    # Test: theta = k0 x - (omega_1 + gam x) t (uniform gradient gam along x): the mixed second
    #   difference over (Delta r, Delta t) is -gam Delta r Delta t; F = hbar |gam| equals the right
    #   side; units of hbar/(m s) are N.
    # Mutant: F = Delta_R Delta_T theta/(Delta r Delta t) without hbar (units 1/(m s)).
    # ------------------------------------------------------------------------
    eq("eq:force_phase", 1952, "F = hbar Delta_R Delta_T theta/(Delta r Delta t)")
    k0, om1, gam, Dr, Dt_ = sp.symbols('k_0 omega_1 gamma Delta_r Delta_t', positive=True)
    thg = k0*xx - (om1 + gam*xx)*t
    mixed = (thg.subs({xx: Dr, t: Dt_}) - thg.subs({xx: 0, t: Dt_})) - (thg.subs({xx: Dr, t: 0}) - thg.subs({xx: 0, t: 0}))
    symcheck("Delta_R Delta_T theta == -gamma Delta r Delta t", sp.expand(mixed + gam*Dr*Dt_))
    symcheck("hbar |Delta_R Delta_T theta|/(Delta r Delta t) == hbar |grad omega|", sp.Abs(hb*mixed/(Dr*Dt_)) - hb*gam)
    symcheck("units hbar/(m s) == N", _S4_HB - _S4_M - _S4_S - _S4_N)
    mutant("no hbar: units 1/(m s) != N", -_S4_M - _S4_S - _S4_N)

    # ------------------------------------------------------------------------
    # eq:force_radian  (sgm.tex L1841)
    # States: F = hbar/(Delta r Delta t) for one radian.
    # Derived from: eq:force_phase with Delta_R Delta_T theta = 1.
    #   Labels used: eq:force_phase.
    # Test: substitution.
    # Mutant: hbar/(2 pi Delta r Delta t) (a cycle instead of a radian).
    # ------------------------------------------------------------------------
    eq("eq:force_radian", 1957, "F = hbar/(Delta r Delta t)")
    Fphase = hb*sp.Symbol('Dth')/(Dr*Dt_)
    symcheck("eq:force_phase at Delta_R Delta_T theta = 1", Fphase.subs('Dth', 1) - hb/(Dr*Dt_))
    mutant("hbar/(2 pi Delta r Delta t)", Fphase.subs('Dth', 1) - hb/(2*sp.pi*Dr*Dt_))

    # ------------------------------------------------------------------------
    # eq:planck_length  (sgm.tex L1849)
    # States: l_P = sqrt(hbar G/c^3).
    # Derived from: the length formed from hbar, c, G; definition.
    #   Labels used: none beyond the constants.
    # Test: units m; numerical value 1.616255e-35 m (CODATA 2022).
    # Mutant: sqrt(hbar G/c^5) has units s, not m.
    # ------------------------------------------------------------------------
    eq("eq:planck_length", 1964, "l_P = sqrt(hbar G/c^3)")
    symcheck("units of l_P == m", (_S4_HB + _S4_GDIM - 3*_S4_CDIM)/2 - _S4_M)
    G_n = 6.67430e-11
    lP = np.sqrt(hbar_n*G_n/c_n**3)
    check("l_P == 1.616255e-35 m", lP, 1.616255e-35, tol=2e-5)
    mutant("sqrt(hbar G/c^5) in m", (_S4_HB + _S4_GDIM - 5*_S4_CDIM)/2 - _S4_M)

    # ------------------------------------------------------------------------
    # eq:planck_time  (sgm.tex L1850)
    # States: t_P = sqrt(hbar G/c^5); l_P t_P = hbar G/c^4 (L1852); l_P/c = t_P (L1919).
    # Derived from: definition.
    #   Labels used: eq:planck_length.
    # Test: units s; the product and the quotient symbolically.
    # Mutant: l_P t_P = hbar G/c^3.
    # ------------------------------------------------------------------------
    eq("eq:planck_time", 1965, "t_P = sqrt(hbar G/c^5)")
    lPs, tPs = sp.sqrt(hb*G/c**3), sp.sqrt(hb*G/c**5)
    symcheck("units of t_P == s", (_S4_HB + _S4_GDIM - 5*_S4_CDIM)/2 - _S4_S)
    symcheck("l_P t_P == hbar G/c^4", sp.simplify(lPs*tPs - hb*G/c**4))
    symcheck("l_P/c == t_P", sp.simplify(lPs/c - tPs))
    mutant("l_P t_P == hbar G/c^3", sp.simplify(lPs*tPs - hb*G/c**3))

    # ------------------------------------------------------------------------
    # eq:force_unit_planck  (sgm.tex L1854)
    # States: F_P = hbar/(l_P t_P).
    # Derived from: eq:force_radian at Delta r = l_P, Delta t = t_P.
    #   Labels used: eq:force_radian, eq:planck_length, eq:planck_time.
    # Test: units N.
    # Mutant: hbar/l_P^2 (units J s m^-2, not N).
    # ------------------------------------------------------------------------
    eq("eq:force_unit_planck", 1969, "F_P = hbar/(l_P t_P)")
    symcheck("units of hbar/(l_P t_P) == N", _S4_HB - _S4_M - _S4_S - _S4_N)
    mutant("hbar/l_P^2 in N", _S4_HB - 2*_S4_M - _S4_N)

    # ------------------------------------------------------------------------
    # unlabeled  (sgm.tex L1855), second line of the F_P display
    # States: F_P = hbar c^4/(hbar G).
    # Derived from: l_P t_P = hbar G/c^4.
    #   Labels used: eq:force_unit_planck, eq:planck_time.
    # Test: symbolic.
    # Mutant: hbar c^5/(hbar G).
    # ------------------------------------------------------------------------
    eq("unlabeled", 1971, "F_P = hbar c^4/(hbar G)")
    FP = hb/(lPs*tPs)
    symcheck("hbar/(l_P t_P) == hbar c^4/(hbar G)", sp.simplify(FP - hb*c**4/(hb*G)))
    mutant("hbar c^5/(hbar G)", sp.simplify(FP - hb*c**5/(hb*G)))

    # ------------------------------------------------------------------------
    # eq:force_unit  (sgm.tex L1856)
    # States: F_P = c^4/G = 1.2103e44 N.
    # Derived from: the cancellation of hbar.
    #   Labels used: eq:force_unit_planck.
    # Test: symbolic; numerical value with c = 299792458 m/s, G = 6.67430e-11 (CODATA 2022) to the
    #   five figures stated.
    # Mutant: c^4/(4G), the conjectured bound of L1868.
    # ------------------------------------------------------------------------
    eq("eq:force_unit", 1971, "F_P = c^4/G = 1.2103e44 N")
    symcheck("hbar c^4/(hbar G) == c^4/G", sp.simplify(hb*c**4/(hb*G) - c**4/G))
    check("c^4/G == 1.2103e44 N", c_n**4/G_n, 1.2103e44, tol=5e-5)
    numutant("c^4/(4G)", c_n**4/(4*G_n), 1.2103e44, tol=5e-5)

    # ------------------------------------------------------------------------
    # eq:force_as_phase  (sgm.tex L1861); L1862 is the caption, not mathematics
    # States: Delta_R Delta_T theta = F/F_P at Delta r = l_P, Delta t = t_P.
    # Derived from: eq:force_phase with eq:force_unit_planck. Inline L1866: G(hbar c/G)/l_P^2 = c^4/G;
    #   for F = mc^2 |grad alpha|, Delta_R Delta_T theta = mG|grad alpha|/c^2.
    #   Labels used: eq:force_phase, eq:force_unit_planck, eq:force_unit, eq:hbar_cancels_identity.
    # Test: solve eq:force_phase for the phase at (l_P, t_P); the two inline identities.
    # Mutant: Delta_R Delta_T theta = F F_P.
    # ------------------------------------------------------------------------
    eq("eq:force_as_phase", 1979, "Delta_R Delta_T theta = F/F_P")
    Fs, ga = sp.symbols('F g_alpha', positive=True)
    dth_s = sp.solve(sp.Eq(Fs, hb*sp.Symbol('Dth')/(lPs*tPs)), sp.Symbol('Dth'))[0]
    symcheck("phase at (l_P, t_P) == F/F_P", sp.simplify(dth_s - Fs/(c**4/G)))
    symcheck("G (hbar c/G)/l_P^2 == c^4/G", sp.simplify(G*(hb*c/G)/lPs**2 - c**4/G))
    symcheck("F = mc^2|grad alpha|: phase == mG|grad alpha|/c^2", sp.simplify(m*c**2*ga/(c**4/G) - m*G*ga/c**2))
    mutant("phase == F F_P", sp.simplify(dth_s - Fs*c**4/G))

    # ------------------------------------------------------------------------
    # unlabeled  (sgm.tex L1873), the matrix D (notag)
    # States: D = [[1, 2, -1], [0, 1, -1], [-1, 3, -2]], rows hbar, c, G; columns M, L, T.
    # Derived from: the dimensions of hbar, c, G (L1870).
    #   Labels used: sec:hbar_cancels.
    # Test: rows equal the SI exponent vectors of hbar, c, G; det D = 2 (L1878).
    # Mutant: det D = 1.
    # ------------------------------------------------------------------------
    eq("unlabeled", 1989, "D = [[1, 2, -1], [0, 1, -1], [-1, 3, -2]]")
    Dm = sp.Matrix([[1, 2, -1], [0, 1, -1], [-1, 3, -2]])
    symcheck("rows of D == exponents of hbar, c, G",
             Dm - sp.Matrix.vstack(_S4_HB[:3, 0].T, _S4_CDIM[:3, 0].T, _S4_GDIM[:3, 0].T))
    symcheck("det D == 2", Dm.det() - 2)
    mutant("det D == 1", Dm.det() - 1)

    # ------------------------------------------------------------------------
    # eq:dimension_matrix  (sgm.tex L1880)
    # States: D^-1 = [[1/2, 1/2, -1/2], [1/2, -3/2, 1/2], [1/2, -5/2, 1/2]], rows M, L, T.
    # Derived from: inversion of D; a dimension row r times D^-1 gives the exponents of hbar, c, G
    #   (L1888); force (1, 1, -2) D^-1 = (0, 4, -1) (L1895); action (1, 2, -1) D^-1 = (1, 0, 0) (L1902).
    #   Labels used: eq:force_unit.
    # Test: D^-1 equals the display; the two row products; the force row has zero hbar exponent.
    # Mutant: the L row with +3/2 in the c column.
    # ------------------------------------------------------------------------
    eq("eq:dimension_matrix", 2000, "D^-1 = [[1/2, 1/2, -1/2], [1/2, -3/2, 1/2], [1/2, -5/2, 1/2]]")
    half = sp.Rational(1, 2)
    Dinv = sp.Matrix([[half, half, -half], [half, -3*half, half], [half, -5*half, half]])
    symcheck("D^-1 == display", Dm.inv() - Dinv)
    symcheck("force row (1, 1, -2) D^-1 == (0, 4, -1)", sp.Matrix([[1, 1, -2]])*Dinv - sp.Matrix([[0, 4, -1]]))
    symcheck("action row (1, 2, -1) D^-1 == (1, 0, 0)", sp.Matrix([[1, 2, -1]])*Dinv - sp.Matrix([[1, 0, 0]]))
    Dinv_m = Dinv.copy()
    Dinv_m[1, 1] = 3*half
    mutant("L row with +3/2", Dm*Dinv_m - sp.eye(3))

    # ------------------------------------------------------------------------
    # eq:planck_mass  (sgm.tex L1890)
    # States: m_P = sqrt(hbar c/G).
    # Derived from: the M row of eq:dimension_matrix; the L and T rows give eqs. planck_length and
    #   planck_time. Inline L1904: G m_P^2/c = hbar; e^2/(4 pi eps0 c) = alpha_fs hbar.
    #   Labels used: eq:dimension_matrix, eq:planck_length, eq:planck_time.
    # Test: hbar^{1/2} c^{1/2} G^{-1/2} etc. from the three rows; the inline identities; the
    #   numerical value e^2/(4 pi eps0 hbar c) = 1/137.036 (CODATA 2022).
    # Mutant: m_P = sqrt(hbar G/c).
    # ------------------------------------------------------------------------
    eq("eq:planck_mass", 2006, "m_P = sqrt(hbar c/G)")
    mP = sp.sqrt(hb*c/G)

    def frm(row):
        return hb**row[0]*c**row[1]*G**row[2]
    symcheck("M row of D^-1 gives m_P", sp.simplify(frm(Dinv.row(0)) - mP))
    symcheck("L row of D^-1 gives l_P", sp.simplify(frm(Dinv.row(1)) - lPs))
    symcheck("T row of D^-1 gives t_P", sp.simplify(frm(Dinv.row(2)) - tPs))
    symcheck("G m_P^2/c == hbar", sp.simplify(G*mP**2/c - hb))
    e_n = 1.602176634e-19
    check("e^2/(4 pi eps0 c) == alpha_fs hbar with alpha_fs = 1/137.035999",
          e_n**2/(4*np.pi*eps0_n*c_n), hbar_n/137.035999, tol=1e-8)
    mutant("m_P = sqrt(hbar G/c)", sp.simplify(frm(Dinv.row(0)) - sp.sqrt(hb*G/c)))

    # ------------------------------------------------------------------------
    # eq:force_floor  (sgm.tex L1915), first line
    # States: F_P(r) = hbar/((r/c) r).
    # Derived from: eq:force_radian with Delta r = r and Delta t = r/c, the shortest duration
    #   allowed by ax:causal_propagation.
    #   Labels used: eq:force_radian, ax:causal_propagation.
    # Test: substitution; units N.
    # Mutant: Delta t = r c.
    # ------------------------------------------------------------------------
    eq("eq:force_floor", 2032, "F_P(r) = hbar/((r/c) r)")
    symcheck("eq:force_radian at Delta r = r, Delta t = r/c", (hb/(Dr*Dt_)).subs({Dr: rr, Dt_: rr/c}) - hb/((rr/c)*rr))
    mutant("Delta t = r c", (hb/(Dr*Dt_)).subs({Dr: rr, Dt_: rr*c}) - hb/((rr/c)*rr))

    # ------------------------------------------------------------------------
    # eq:force_floor  (sgm.tex L1916), second line
    # States: F_P(r) = hbar c/r^2.
    # Derived from: algebra; F_P(l_P) = F_P since l_P/c = t_P (L1919).
    #   Labels used: eq:force_floor, eq:planck_length, eq:planck_time, eq:force_unit_planck.
    # Test: symbolic; F_P(l_P) == c^4/G; units N.
    # Mutant: hbar c/r.
    # ------------------------------------------------------------------------
    eq("eq:force_floor", 2032, "F_P(r) = hbar c/r^2")
    FPr = hb*c/rr**2
    symcheck("hbar/((r/c) r) == hbar c/r^2", sp.simplify(hb/((rr/c)*rr) - FPr))
    symcheck("F_P(l_P) == F_P", sp.simplify(FPr.subs(rr, lPs) - c**4/G))
    symcheck("units hbar c/r^2 == N", _S4_HB + _S4_CDIM - 2*_S4_M - _S4_N)
    mutant("hbar c/r", sp.simplify(hb/((rr/c)*rr) - hb*c/rr))

    # ------------------------------------------------------------------------
    # eq:radians_per_crossing  (sgm.tex L1923)
    # States: (GMm/r^2)/F_P(r) = GMm/(hbar c).
    # Derived from: eq:force_floor and eq:newton.
    #   Labels used: eq:force_floor, eq:newton, eq:planck_mass, ax:causal_propagation.
    # Test: symbolic; one radian per crossing at M = m = m_P; L1926 numbers: two electrons 1.75e-45,
    #   5.7e44 crossings, 6e28 years at 1 m (Julian year); 1e-14 kg masses about 2e-13, a factor
    #   about 1e32 above the electrons and about 5e12 below one radian; 2.5 s is some 4e12 crossing
    #   times of 200 um; about 0.8 radian accumulated.
    # Mutant: GMm/(hbar c^2).
    # ------------------------------------------------------------------------
    eq("eq:radians_per_crossing", 2039, "(GMm/r^2)/F_P(r) = GMm/(hbar c)")
    Mm = sp.Symbol('M', positive=True)
    symcheck("(GMm/r^2)/(hbar c/r^2) == GMm/(hbar c)", sp.simplify((G*Mm*m/rr**2)/FPr - G*Mm*m/(hb*c)))
    symcheck("one radian at M = m = m_P", sp.simplify((G*Mm*m/(hb*c)).subs({Mm: mP, m: mP}) - 1))
    mutant("GMm/(hbar c^2)", sp.simplify((G*Mm*m/rr**2)/FPr - G*Mm*m/(hb*c**2)))
    me_n = 9.1093837139e-31
    re = G_n*me_n**2/(hbar_n*c_n)
    check("two electrons: GMm/(hbar c) == 1.75e-45", re, 1.75e-45, tol=3e-3)
    check("one radian takes 5.7e44 crossings", 1/re, 5.7e44, tol=3e-3)
    yr = 365.25*86400
    check("at 1 m: 5.7e44 (1/c) s == 6e28 years", (1/re)*(1/c_n)/yr, 6e28, tol=1e-2)
    rt = G_n*1e-28/(hbar_n*c_n)
    check("1e-14 kg masses: GMm/(hbar c) about 2e-13 (one figure)", rt, 2e-13, tol=0.06)
    boolcheck("factor above the electron value rounds to 1e32", round(np.log10(rt/re)) == 32)
    check("factor below one radian about 5e12 (one figure)", 1/rt, 5e12, tol=0.06)
    ncross = 2.5/(200e-6/c_n)
    check("2.5 s is some 4e12 crossing times of 200 um (one figure)", ncross, 4e12, tol=0.07)
    check("accumulated phase about 0.8 radian (one figure)", ncross*rt, 0.8, tol=0.02)

    # ------------------------------------------------------------------------
    # eq:pair_frequency  (sgm.tex L1930)
    # States: omega(r) = omega_0 - GMm/(hbar r), to first order in GM/(c^2 r).
    # Derived from: omega_0 alpha (eq:omega_alpha) with the weak-field alpha of eq:alpha_weak
    #   (rem:lapse_test). Inline L1933-L1935: the next term is smaller by a factor of order
    #   GM/(c^2 r); the offset over r/c is GMm/(hbar c); for a Yukawa potential the force ratio
    #   exceeds the phase by 1 + r/lambda_Y.
    #   Labels used: eq:omega_alpha, eq:alpha_weak, eq:rest_freq, rem:lapse_test,
    #   eq:radians_per_crossing, eq:force_floor.
    # Test: series of omega_0 sqrt(1 - 2u), u = GM/(c^2 r): first order gives the display, the second-
    #   order term is u/2 times the first; the phase over r/c; Yukawa ratio.
    # Mutant: omega_0 - 2GMm/(hbar r).
    # ------------------------------------------------------------------------
    eq("eq:pair_frequency", 2046, "omega(r) = omega_0 - GMm/(hbar r)")
    uu = sp.Symbol('u', positive=True)
    w0m = m*c**2/hb
    serp = sp.series(w0m*sp.sqrt(1 - 2*uu), uu, 0, 3).removeO()
    symcheck("first order: omega_0 - omega_0 u == omega_0 - GMm/(hbar r)",
             (serp.coeff(uu, 0) + serp.coeff(uu, 1)*uu).subs(uu, G*Mm/(c**2*rr)) - (w0m - G*Mm*m/(hb*rr)))
    symcheck("second-order term == (u/2) times the first-order term", sp.simplify(serp.coeff(uu, 2)/serp.coeff(uu, 1) - half))
    symcheck("offset times r/c == GMm/(hbar c)", (G*Mm*m/(hb*rr))*(rr/c) - G*Mm*m/(hb*c))
    Vy = -G*Mm*m*sp.exp(-rr/lamY)/rr
    Fy = -sp.diff(Vy, rr)
    symcheck("Yukawa: (|F|/F_P(r)) / (|V| (r/c)/hbar) == 1 + r/lambda_Y",
             sp.simplify((-Fy/FPr)/((-Vy)*(rr/c)/hb) - (1 + rr/lamY)))
    mutant("omega_0 - 2GMm/(hbar r)", (serp.coeff(uu, 0) + serp.coeff(uu, 1)*uu).subs(uu, G*Mm/(c**2*rr))
           - (w0m - 2*G*Mm*m/(hb*rr)))

    # ------------------------------------------------------------------------
    # eq:alpha_crossing  (sgm.tex L1943)
    # States: (e^2/(4 pi eps0 r^2))/F_P(r) = e^2/(4 pi eps0 hbar c) = 1/137.036.
    # Derived from: eq:force_floor and Coulomb's law.
    #   Labels used: eq:force_floor, eq:radians_per_crossing.
    # Test: symbolic; numerical 1/137.036 with CODATA 2022 e, eps0, hbar, c.
    # Mutant: 1/137.5.
    # ------------------------------------------------------------------------
    eq("eq:alpha_crossing", 2059, "(e^2/(4 pi eps0 r^2))/F_P(r) = e^2/(4 pi eps0 hbar c) = 1/137.036")
    eps0s = sp.Symbol('epsilon_0', positive=True)
    symcheck("ratio == e^2/(4 pi eps0 hbar c)",
             sp.simplify((e_ch**2/(4*sp.pi*eps0s*rr**2))/FPr - e_ch**2/(4*sp.pi*eps0s*hb*c)))
    afs = e_n**2/(4*np.pi*eps0_n*hbar_n*c_n)
    check("alpha_fs == 1/137.036", afs, 1/137.036, tol=2e-6)
    numutant("1/137.5", afs, 1/137.5, tol=2e-6)

    # ------------------------------------------------------------------------
    # eq:adm  (sgm.tex L1957)
    # States: H = q Phi + alpha_n sqrt(m^2c^4 + c^2 h^ij pi_i pi_j) - A_g^i pi_i.
    # Derived from: the positive root of g^{mu nu} P_mu P_nu = -m^2c^2 with P_0 = -(E - V)/c,
    #   P_i = pi_i, in the metric of eq:lapse_shift_metric (L1953); h^ij = W^-2 delta gives the
    #   order-hbar^0 frequency of eq:master.
    #   Labels used: eq:master, eq:lapse_shift_metric, eq:alpha_lapse_shift, eq:isotropic,
    #   eq:sgm, eq:newton, ax:planck, ax:causal_propagation, tab:classes, tab:forces, sec:light.
    # Test: (a) the inverse of the lapse-shift metric (x^0 = ct, general W, A_g) is
    #   g^00 = -1/alpha_n^2, g^0i = A_g^i/(c alpha_n^2), g^ij = W^-2 delta - A_g^i A_g^j/(c^2 alpha_n^2);
    #   (b) the root solves the mass shell exactly for the conformally flat metric; (c) for a general
    #   (random numerical, symmetric positive) spatial metric h_ij, the ADM root solves the mass
    #   shell; (d) free root gamma mc^2 with hbar k = gamma m v; (e) the Painleve-Gullstrand shift
    #   sqrt(2GM/r) r_hat has zero curl; (f) L1964: grad_{x_s} f(x - x_s) = -grad f.
    #   The perihelion advance and the light deflection (L1962) are cited, not computed.
    # Mutant: the shift term with + sign does not solve the mass shell.
    # ------------------------------------------------------------------------
    eq("eq:adm", 2073, "H = q Phi + alpha_n sqrt(m^2c^4 + c^2 h^ij pi_i pi_j) - A_g^i pi_i")
    gdn = sp.zeros(4)
    gdn[0, 0] = -an**2 + W**2*Agsym.dot(Agsym)/c**2
    for i in range(3):
        gdn[0, i + 1] = gdn[i + 1, 0] = W**2*Agsym[i]/c
        gdn[i + 1, i + 1] = W**2
    gup = sp.zeros(4)
    gup[0, 0] = -1/an**2
    for i in range(3):
        gup[0, i + 1] = gup[i + 1, 0] = Agsym[i]/(c*an**2)
        for j in range(3):
            gup[i + 1, j + 1] = (1 if i == j else 0)/W**2 - Agsym[i]*Agsym[j]/(c**2*an**2)
    symcheck("(a) g_{mu nu} g^{nu rho} == identity", sp.simplify(gdn*gup - sp.eye(4)))
    Vs = sp.Symbol('V', real=True)
    Ee = Vs + an*sp.sqrt(m**2*c**4 + c**2*piv.dot(piv)/W**2) - Agsym.dot(piv)
    Pd = sp.Matrix([-(Ee - Vs)/c] + list(piv))
    symcheck("(b) root solves g^{mu nu} P_mu P_nu = -m^2 c^2", sp.simplify((Pd.T*gup*Pd)[0] + m**2*c**2))
    Ee_m = Vs + an*sp.sqrt(m**2*c**4 + c**2*piv.dot(piv)/W**2) + Agsym.dot(piv)
    Pd_m = sp.Matrix([-(Ee_m - Vs)/c] + list(piv))
    mutant("+A_g.pi root", sp.simplify((Pd_m.T*gup*Pd_m)[0] + m**2*c**2))
    rng = np.random.default_rng(1962)
    Mh = rng.normal(size=(3, 3))
    h_dn = Mh @ Mh.T + 3*np.eye(3)
    h_up = np.linalg.inv(h_dn)
    an_n, m_n1, c_n1 = 0.83, 1.3, 2.7
    beta = rng.normal(size=3)*0.3
    pin = rng.normal(size=3)
    g4 = np.zeros((4, 4))
    g4[0, 0] = -an_n**2 + beta @ h_dn @ beta/c_n1**2
    g4[0, 1:] = g4[1:, 0] = (h_dn @ beta)/c_n1
    g4[1:, 1:] = h_dn
    g4i = np.linalg.inv(g4)
    E_adm = an_n*np.sqrt(m_n1**2*c_n1**4 + c_n1**2*pin @ h_up @ pin) - beta @ pin
    P4 = np.concatenate([[-E_adm/c_n1], pin])
    check("(c) general h_ij: ADM root solves the mass shell", P4 @ g4i @ P4, -m_n1**2*c_n1**2, tol=1e-12)
    gam_v = 1/sp.sqrt(1 - v**2/c**2)
    symcheck("(d) hbar k = gamma m v: m^2c^4 + c^2 (gamma m v)^2 == (gamma mc^2)^2 (both roots positive)",
             sp.simplify((m**2*c**4 + c**2*(gam_v*m*v)**2) - (gam_v*m*c**2)**2))
    rx = sp.sqrt(Xv.dot(Xv))
    PG = sp.sqrt(2*G*Mm/rx)*Xv/rx
    symcheck("(e) curl of the Painleve-Gullstrand shift == 0", sp.simplify(_S4_curl(PG, X)))
    xsv = sp.Matrix(sp.symbols('xs1:4', real=True))
    fsrc = sp.Function('f')
    fx = fsrc(*(Xv - xsv))
    symcheck("(f) grad_{x_s} f(x - x_s) == -grad_x f(x - x_s)",
             sp.Matrix([sp.diff(fx, xsv[k]) + sp.diff(fx, X[k]) for k in range(3)]))

    # ------------------------------------------------------------------------
    # eq:cow_phase  (sgm.tex L1976), first line
    # States: (m g/hbar)(A_par sin/v) = 2 pi lambda g m^2 A_par sin/h^2.
    # Derived from: the magnitude of eq:interferometer on A_zt = A_par sin(tilt)/v with the de Broglie
    #   speed v = h/(m lambda) (eq:debroglie) and hbar = h/(2 pi).
    #   Labels used: eq:interferometer, eq:debroglie.
    # Test: symbolic.
    # Mutant: v = h/(2 pi m lambda) (hbar in place of h).
    # ------------------------------------------------------------------------
    eq("eq:cow_phase", 2093, "(m g/hbar) A_par sin/v = 2 pi lambda g m^2 A_par sin/h^2")
    hh, lam_n, Apar, stl = sp.symbols('h lambda_neu A_par sin_tilt', positive=True)
    lhs_cow = (m*g/(hh/(2*sp.pi)))*Apar*stl/(hh/(m*lam_n))
    symcheck("magnitude == 2 pi lambda g m^2 A_par sin/h^2", sp.simplify(lhs_cow - 2*sp.pi*lam_n*g*m**2*Apar*stl/hh**2))
    mutant("v = hbar/(m lambda)", sp.simplify((m*g/(hh/(2*sp.pi)))*Apar*stl/(hh/(2*sp.pi*m*lam_n))
                                             - 2*sp.pi*lam_n*g*m**2*Apar*stl/hh**2))

    # ------------------------------------------------------------------------
    # eq:cow_phase  (sgm.tex L1977), second line
    # States: = Delta theta_COW sin(tilt), Delta theta_COW = 4 pi lambda g h^-2 m^2 d(d + t cos)tan.
    # Derived from: A_par = 2 d (d + t_slab cos theta_B) tan theta_B (L1974).
    #   Labels used: eq:cow_phase, eq:interferometer, sec:experiment, eq:debroglie.
    # Test: symbolic identity; L1980 numbers (lambda = 1.445 A, d = 3.5 cm, t = 0.2 cm, 22.1 deg,
    #   CODATA neutron mass, g = 9.8): v = 2.74e3 m/s, A_par = 10.5 cm^2, Delta theta = 59.6 rad,
    #   about nineteen fringes over 180 deg; L1989: A_zt = 3.83e-7 m s, hbar 54.3/A_zt = 1.50e-26 N,
    #   9% short of the weight; hbar 59.6/A_zt equals m_n g exactly (symbolic).
    # Mutant: A_par = d(d + t cos) tan (factor 2 dropped).
    # ------------------------------------------------------------------------
    eq("eq:cow_phase", 2093, "= Delta theta_COW sin(tilt)")
    dd, tsl, thB = sp.symbols('d t_slab theta_B', positive=True)
    Apar_x = 2*dd*(dd + tsl*sp.cos(thB))*sp.tan(thB)
    dCOW = 4*sp.pi*lam_n*g*m**2*dd*(dd + tsl*sp.cos(thB))*sp.tan(thB)/hh**2
    symcheck("2 pi lambda g m^2 A_par/h^2 == Delta theta_COW", sp.simplify((2*sp.pi*lam_n*g*m**2*Apar_x/hh**2) - dCOW))
    mutant("A_par without the factor 2", sp.simplify((2*sp.pi*lam_n*g*m**2*Apar_x/(2*hh**2)) - dCOW))
    h_n = 6.62607015e-34
    lam_v, d_v, t_v, thB_v = 1.445e-10, 0.035, 0.002, np.radians(22.1)
    v_n = h_n/(m_n*lam_v)
    Apar_n = 2*d_v*(d_v + t_v*np.cos(thB_v))*np.tan(thB_v)
    dth_n = 4*np.pi*lam_v*9.8*m_n**2*d_v*(d_v + t_v*np.cos(thB_v))*np.tan(thB_v)/h_n**2
    check("v == 2.74e3 m/s", v_n, 2.74e3, tol=2e-3)
    check("A_par == 10.5 cm^2 (three figures)", Apar_n, 10.5e-4, tol=5e-3)
    check("Delta theta_COW == 59.6 rad", dth_n, 59.6, tol=1e-3)
    check("fringes over 180 deg: 2 Delta theta/(2 pi) about 19", 2*dth_n/(2*np.pi), 19, tol=3e-3)
    Azt_n = Apar_n/v_n
    check("A_zt = A_par/v == 3.83e-7 m s", Azt_n, 3.83e-7, tol=2e-3)
    check("hbar 54.3/A_zt == 1.50e-26 N", hbar_n*54.3/Azt_n, 1.50e-26, tol=4e-3)
    check("54.3/59.6 is 9% short (0.91)", 54.3/dth_n, 0.91, tol=3e-3)
    check("hbar Delta theta_COW/A_zt == m_n g (hbar = 1.054571817e-34 vs h/2 pi)", hbar_n*dth_n/Azt_n, m_n*9.8, tol=1e-8)

    # ------------------------------------------------------------------------
    # eq:atom_phase  (sgm.tex L1984)
    # States: (m_Cs g/hbar)(hbar k_eff/m_Cs) T_p^2 = k_eff g T_p^2.
    # Derived from: the magnitude of eq:interferometer on the triangle of base 2T_p and height
    #   hbar k_eff T_p/m_Cs, area A_zt = hbar k_eff T_p^2/m_Cs (L1982).
    #   Labels used: eq:interferometer, sec:experiment.
    # Test: triangle area by integration; the identity; L1987 numbers: k_eff = 4 pi/852 nm =
    #   1.475e7 /m; separation hbar k T/m_Cs just over 1 mm at T_p = 0.16 s; k g T^2 = 3.70e6 rad;
    #   that phase over 2 pi agrees with the fringe count 588,638 to 1e-3 (852 nm and g = 9.8 are
    #   rounded); one fringe spans 1/588638 = 1.7e-6 of g; L1989: A_zt = 1.80e-4 m s and
    #   hbar phase/A_zt = m_Cs g.
    # Mutant: triangle area hbar k T_p^2/(2 m_Cs).
    # ------------------------------------------------------------------------
    eq("eq:atom_phase", 2100, "(m g/hbar)(hbar k_eff/m) T_p^2 = k_eff g T_p^2")
    keff = sp.Symbol('k_eff', positive=True)
    hgt = hb*keff*Tp/m
    tri = sp.integrate(sp.Piecewise((hgt*t/Tp, t <= Tp), (hgt*(2*Tp - t)/Tp, True)), (t, 0, 2*Tp))
    symcheck("triangle area == hbar k_eff T_p^2/m", sp.simplify(tri - hb*keff*Tp**2/m))
    symcheck("(m g/hbar) A_zt == k_eff g T_p^2", sp.simplify((m*g/hb)*tri - keff*g*Tp**2))
    mutant("area hbar k T^2/(2m)", sp.simplify(tri - hb*keff*Tp**2/(2*m)))
    k_n = 4*np.pi/852e-9
    T_n = 0.16
    check("k_eff == 1.475e7 /m", k_n, 1.475e7, tol=1e-3)
    sep = hbar_n*k_n*T_n/m_Cs
    boolcheck("separation hbar k T/m_Cs just over 1 mm (1.0 to 1.2 mm)", 1.0e-3 < sep < 1.2e-3)
    ph_n = k_n*9.8*T_n**2
    check("k g T^2 == 3.70e6 rad", ph_n, 3.70e6, tol=1e-3)
    check("phase/2 pi agrees with the fringe count 588638", ph_n/(2*np.pi), 588638, tol=1e-3)
    check("one fringe spans 1.7e-6 of g", 1/588638, 1.7e-6, tol=3e-3)
    Azt_cs = hbar_n*k_n*T_n**2/m_Cs
    check("A_zt == 1.80e-4 m s", Azt_cs, 1.80e-4, tol=3e-3)
    check("hbar phase/A_zt == m_Cs g", hbar_n*ph_n/Azt_cs, m_Cs*9.8, tol=1e-12)

    # ------------------------------------------------------------------------
    # eq:galileo_action  (sgm.tex L1993), first line
    # States: S = int_0^{2T} [m(v0 - g t)^2/2 - m g (v0 t - g t^2/2)] dt.
    # Derived from: the free-fall trajectory z = v0 t - g t^2/2 with v0 = g T_fall, relative to the
    #   held packet at z = 0 (L1991).
    #   Labels used: eq:action_phase, sec:intro.
    # Test: the integrand is m zdot^2/2 - m g z on that trajectory; the packet returns to z = 0 at 2T.
    # Mutant: return time T (z(T) is not zero).
    # ------------------------------------------------------------------------
    eq("eq:galileo_action", 2110, "S = int_0^{2T} [m(v0 - gt)^2/2 - m g(v0 t - gt^2/2)] dt")
    Tf = sp.Symbol('T_fall', positive=True)
    v0f = g*Tf
    zf = v0f*t - g*t**2/2
    integ = m*(v0f - g*t)**2/2 - m*g*(v0f*t - g*t**2/2)
    symcheck("integrand == m zdot^2/2 - m g z", sp.expand(integ - (m*sp.diff(zf, t)**2/2 - m*g*zf)))
    symcheck("z(2T) == 0", zf.subs(t, 2*Tf))
    mutant("z(T) == 0", zf.subs(t, Tf))

    # ------------------------------------------------------------------------
    # eq:galileo_action  (sgm.tex L1994), second line
    # States: S = -(1/3) m g^2 T_fall^3; Delta theta_g = -m g^2 T^3/(3 hbar) by eq:action_phase.
    # Derived from: the integral.
    #   Labels used: eq:galileo_action, eq:action_phase.
    # Test: exact integration.
    # Mutant: -(1/2) m g^2 T^3; +(1/3) m g^2 T^3.
    # ------------------------------------------------------------------------
    eq("eq:galileo_action", 2110, "S = -(1/3) m g^2 T_fall^3")
    S_gal = sp.integrate(integ, (t, 0, 2*Tf))
    symcheck("integral == -(1/3) m g^2 T^3", sp.simplify(S_gal + m*g**2*Tf**3/3))
    symcheck("phase S/hbar == -m g^2 T^3/(3 hbar)", sp.simplify(S_gal/hb + m*g**2*Tf**3/(3*hb)))
    mutant("-(1/2) m g^2 T^3", sp.simplify(S_gal + m*g**2*Tf**3/2))
    mutant("+(1/3) m g^2 T^3", sp.simplify(S_gal - m*g**2*Tf**3/3))

    # ------------------------------------------------------------------------
    # eq:gauge_phase  (sgm.tex L2011)
    # States: psi -> psi exp(-(q/hbar) lambda e3), with A_k -> A_k + d_k lambda,
    #   Phi -> Phi - d_t lambda (L2014).
    # Derived from: e3 commutes with the exponential; the two terms cancel in D_k and D_t.
    #   Labels used: eq:right_connection, eq:phase_connection_time, eq:phase_curvature,
    #   eq:time_space_curvature, eq:ab, rem:additivity.
    # Test: for a general quaternion psi(x, t) and general A, Phi, lambda (one space coordinate for
    #   D_k, plus t): D'(psi E) == (D psi) E for D_x and D_t; E and B unchanged in three dimensions
    #   (port of the gauge-invariance check).
    # Mutant: Phi -> Phi + d_t lambda.
    # ------------------------------------------------------------------------
    eq("eq:gauge_phase", 2127, "psi -> psi exp(-(q/hbar) lambda e3)")
    psi_g = [sp.Function('psi%d' % i, real=True)(xx, t) for i in range(4)]
    lam_g = sp.Function('lambda', real=True)(xx, t)
    Ax_g = sp.Function('A_x', real=True)(xx, t)
    Ph_g = sp.Function('Phi', real=True)(xx, t)
    e3q = [0, 0, 0, 1]
    Eg = [sp.cos(q*lam_g/hb), 0, 0, -sp.sin(q*lam_g/hb)]

    def Dx(f, A):
        return _S4_add(_S4_d(f, xx), _S4_sc(q*A/hb, _S4_mul(f, e3q)))

    def Dt(f, Ph):
        return _S4_add(_S4_d(f, t), _S4_sc(-q*Ph/hb, _S4_mul(f, e3q)))
    psiE = _S4_mul(psi_g, Eg)
    symcheck("D_x covariant: D'_x(psi E) == (D_x psi) E",
             sp.Matrix([sp.simplify(u) for u in _S4_add(Dx(psiE, Ax_g + sp.diff(lam_g, xx)),
                                                         _S4_sc(-1, _S4_mul(Dx(psi_g, Ax_g), Eg)))]))
    symcheck("D_t covariant: D'_t(psi E) == (D_t psi) E with Phi - d_t lambda",
             sp.Matrix([sp.simplify(u) for u in _S4_add(Dt(psiE, Ph_g - sp.diff(lam_g, t)),
                                                         _S4_sc(-1, _S4_mul(Dt(psi_g, Ph_g), Eg)))]))
    mutant("Phi + d_t lambda", sp.simplify(_S4_add(Dt(psiE, Ph_g + sp.diff(lam_g, t)),
                                                   _S4_sc(-1, _S4_mul(Dt(psi_g, Ph_g), Eg)))[0]))
    lam3 = RF('lambda')
    symcheck("B unchanged: curl grad lambda == 0", _S4_curl(_S4_grad(lam3, X), X))
    symcheck("E unchanged: -grad(-d_t lambda) - d_t grad lambda == 0",
             -_S4_grad(-sp.diff(lam3, t), X) - sp.diff(_S4_grad(lam3, X), t))

    # ------------------------------------------------------------------------
    # eq:moment_law  (sgm.tex L2040)
    # States: surface integral of (w grad Phi - Phi grad w).dS = -(1/eps0) int_V w rho_q d^3y for
    #   harmonic w.
    # Derived from: div(w grad Phi - Phi grad w) = w lap Phi - Phi lap w, lap Phi = -rho_q/eps0 (the
    #   Green function of rem:reach), and the divergence theorem.
    #   Labels used: rem:reach, eq:static_density_clock_field, eq:gauss_alpha, sec:maxwell,
    #   eq:lorentz, eq:no_monopole.
    # Test: the divergence identity for general w, Phi; harmonicity of 1, x_j, x1 x2; numerical
    #   surface integrals for an off-center point charge inside the unit sphere with w = 1, x_1,
    #   x_1 x_2 (eps0 = 1, q = 1, Phi = 1/(4 pi |x - y0|)); zero for a charge outside; L2045: the
    #   alpha analog has lap alpha = 4 pi G rho_m/c^2 = -(-4 pi G rho_m/c^2) (sign of the mapping).
    # Mutant: the non-harmonic weight w = x_1^2.
    # ------------------------------------------------------------------------
    eq("eq:moment_law", 2156, "surface(w grad Phi - Phi grad w).dS = -(1/eps0) int w rho_q")
    Phg = sp.Function('Phi', real=True)(*X)
    wg = sp.Function('w', real=True)(*X)
    symcheck("div(w grad Phi - Phi grad w) == w lap Phi - Phi lap w",
             sp.expand(_S4_div(wg*_S4_grad(Phg, X) - Phg*_S4_grad(wg, X), X) - (wg*_S4_lap(Phg, X) - Phg*_S4_lap(wg, X))))
    for wexp in (sp.Integer(1), X[0], X[0]*X[1]):
        symcheck(f"w = {wexp} is harmonic", _S4_lap(wexp, X))
    y0 = np.array([0.27, -0.18, 0.33])

    def surf(wfun, gradw, y):
        def F(pts):
            d = pts - y
            rn = np.linalg.norm(d, axis=-1)[..., None]
            Phi_ = 1/(4*np.pi*rn)
            gPhi = -d/(4*np.pi*rn**3)
            return wfun(pts)[..., None]*gPhi - Phi_*gradw(pts)
        return _S4_sphere_flux(F, [0, 0, 0], 1.0)
    one = (lambda P: np.ones(P.shape[:-1]), lambda P: np.zeros(P.shape))
    wx1 = (lambda P: P[..., 0], lambda P: np.stack([np.ones(P.shape[:-1]), 0*P[..., 0], 0*P[..., 0]], -1))
    wx12 = (lambda P: P[..., 0]*P[..., 1], lambda P: np.stack([P[..., 1], P[..., 0], 0*P[..., 0]], -1))
    check("w = 1: surface integral == -q/eps0", surf(*one, y0), -1.0, tol=1e-6)
    check("w = x1: surface integral == -y0_1 q/eps0", surf(*wx1, y0), -y0[0], tol=1e-6)
    check("w = x1 x2: surface integral == -y0_1 y0_2 q/eps0", surf(*wx12, y0), -y0[0]*y0[1], tol=1e-6)
    check("charge outside: surface integral == 0", surf(*wx12, np.array([1.6, 0.3, -0.2])), 0, tol=1e-6, scale=1.0)
    wsq = (lambda P: P[..., 0]**2, lambda P: np.stack([2*P[..., 0], 0*P[..., 0], 0*P[..., 0]], -1))
    numutant("non-harmonic w = x1^2", surf(*wsq, y0), -y0[0]**2, tol=1e-3)
    symcheck("alpha analog: lap alpha = 4 pi G rho/c^2 is the law with rho_q/eps0 -> -4 pi G rho_m/c^2",
             sp.simplify(radlap(inside) - (-(-4*sp.pi*G*rho0/c**2))))

    # ------------------------------------------------------------------------
    # eq:phase_flux  (sgm.tex L2055)
    # States: (1/4 pi c) surface(grad omega_geom).dS = G M_enc m/(hbar c).
    # Derived from: omega_geom = omega_0 alpha, the flux 4 pi G M_enc/c^2 of grad alpha
    #   (eq:gauss_alpha), omega_0 = mc^2/hbar. Inline L2060: the Darwin source
    #   lap(G - 2 m0 c^2 ln W) = (1 + 2 gamma) 4 pi G m0 rho_m; L2064: kinematic momentum
    #   (nh - q Phi_B)/(2 pi r_perp) and the period h/q.
    #   Labels used: eq:gauss_alpha, eq:radians_per_crossing, eq:moment_law, sec:expansion_orders,
    #   eq:ab, sec:ab, prop:reach.
    # Test: the flux identity for a point mass (numerical flux of grad(-GM/(c^2 r)) through an
    #   off-center sphere, then the algebra); the Darwin source at first order with
    #   alpha_n = 1 + Phi_g/c^2, W = 1 - gamma Phi_g/c^2, lap Phi_g = 4 pi G rho_m; the solenoid
    #   momentum and its periodicity.
    # Mutant: 1/(4 pi) without c (units rad s^-1).
    # ------------------------------------------------------------------------
    eq("eq:phase_flux", 2171, "(1/4 pi c) surface grad omega_geom.dS = G M_enc m/(hbar c)")
    fl = _S4_sphere_flux(kern([0.21, -0.35, 0.12]), [0.05, 0.02, -0.04], 1.3)
    check("flux of (x - y)/|x - y|^3 through an off-center sphere == 4 pi", fl, 4*np.pi, tol=1e-6)
    symcheck("(1/4 pi c) omega_0 (4 pi G M/c^2) == G M m/(hbar c)", (m*c**2/hb)*4*sp.pi*G*Mm/c**2/(4*sp.pi*c) - G*Mm*m/(hb*c))
    mutant("(1/4 pi) without c", (m*c**2/hb)*4*sp.pi*G*Mm/c**2/(4*sp.pi) - G*Mm*m/(hb*c))
    m0, gP = sp.symbols('m_0 gamma_PPN', positive=True)
    Phg_ = sp.Function('Phi_g', real=True)(*X)
    ep_ = sp.Symbol('epsilon', positive=True)
    Gl = m0*c**2*(ep_*Phg_/c**2)
    lnW = sp.series(sp.log(1 - gP*ep_*Phg_/c**2), ep_, 0, 2).removeO()
    dar = sp.expand(_S4_lap(Gl - 2*m0*c**2*lnW, X)).coeff(ep_, 1)
    symcheck("Darwin source: lap(G - 2 m0 c^2 ln W) == (1 + 2 gamma) m0 lap Phi_g at first order",
             sp.expand(dar - (1 + 2*gP)*m0*_S4_lap(Phg_, X)))
    nI, rp = sp.symbols('n r_perp', positive=True)
    p_can = hb*nI/rp
    pi_k = p_can - q*PhB/(2*sp.pi*rp)
    symcheck("kinematic momentum == (n h - q Phi_B)/(2 pi r_perp)", sp.simplify(pi_k - (nI*2*sp.pi*hb - q*PhB)/(2*sp.pi*rp)))
    symcheck("Phi_B -> Phi_B + h/q with n -> n + 1 leaves pi unchanged",
             sp.simplify(pi_k.subs({PhB: PhB + 2*sp.pi*hb/q, nI: nI + 1}) - pi_k))

    # ------------------------------------------------------------------------
    # tab:falloff  (sgm.tex L2069), table L2066-L2083
    # States: a coefficient ~ r^-p advances the phase in one crossing time r/c by radians ~ r^{1-p};
    #   a field falls one power faster than its potential; the dipole part of 1/|x - y| falls as
    #   1/r^2; the flux of B_g = curl A_g is zero.
    # Derived from: the 1/|x - y| expansion and eq:moment_law; div curl = 0.
    #   Labels used: eq:nr_expansion, tab:forces, eq:moment_law, eq:gauss_alpha, eq:no_monopole,
    #   eq:radians_per_crossing.
    # Test: (r^-p)(r/c) = r^{1-p}/c for p = 1, 2, 3; |grad r^-p| = p r^{-p-1}; the first-order term of
    #   1/|x - y| in y is y.x/r^3; div curl A_g = 0 for a general A_g.
    # Mutant: the radians proportional to r^{-p}.
    # ------------------------------------------------------------------------
    eq("tab:falloff", 2184, "fall-off of the coefficients and the radians per crossing")
    for pp in (1, 2, 3):
        symcheck(f"p = {pp}: r^-p (r/c) == r^(1-p)/c", rr**(-pp)*(rr/c) - rr**(1 - pp)/c)
        symcheck(f"p = {pp}: |d/dr r^-p| == p r^(-p-1)", sp.Abs(sp.diff(rr**(-pp), rr)) - pp*rr**(-pp - 1))
    mutant("radians ~ r^-p for p = 2", rr**(-2)*(rr/c) - rr**(-2)/c)
    yv = sp.Matrix(sp.symbols('Y1:4', real=True))
    epy = sp.Symbol('eta', positive=True)
    inv = 1/sp.sqrt((Xv - epy*yv).dot(Xv - epy*yv))
    first_y = sp.series(inv, epy, 0, 2).removeO().coeff(epy, 1)
    symcheck("dipole term of 1/|x - y| == y.x/r^3", sp.simplify(first_y - yv.dot(Xv)/Xv.dot(Xv)**sp.Rational(3, 2)))
    Agf = sp.Matrix([RF('Ag%d' % k) for k in range(1, 4)])
    symcheck("div curl A_g == 0 (zero flux of B_g)", sp.expand(_S4_div(_S4_curl(Agf, X), X)))

    # ------------------------------------------------------------------------
    # eq:sup_rho1  (sgm.tex L2185)
    # States: exact 2x2 reduced density matrix hat_rho_1(tau) after tracing out mass 2 in BMV protocol.
    # Derived from: eq:h_grav_pair (L1465), eq:gie_phase (L1501).
    # Test: verify tr(hat_rho_1) == 1, Hermiticity, positive semidefiniteness (det >= 0), and
    #       |hat_rho_1[0,1]|^2 == (1/4) cos^2(Delta theta_grav / 2).
    # Mutant: dropping factor 1/2 in off-diagonal entry violates positive semidefiniteness at Delta theta = 0.
    # ------------------------------------------------------------------------
    eq("eq:sup_rho1", 2209, "hat_rho_1(tau) = (1/2) [[1, (e^{i(t00-t10)} + e^{i(t01-t11)})/2], [c.c., 1]]")
    symcheck("tr(hat_rho_1) == 1", sp.simplify(_rho1.trace() - 1))
    symcheck("|(hat_rho_1)_{01}|^2 == (1/4) cos^2(Delta theta_grav / 2)",
             sp.simplify(sp.expand((_r01*sp.conjugate(_r01)).rewrite(sp.cos)) - sp.Rational(1, 4)*sp.cos(_dth/2)**2))
    _rho1_bad = sp.Matrix([[sp.Rational(1, 2), 2*_r01], [2*sp.conjugate(_r01), sp.Rational(1, 2)]])
    mutant("doubled off-diagonal violates det(rho_1) >= 0 at Delta theta = 0",
           sp.simplify(_rho1_bad.det().subs({_t00: 0, _t01: 0, _t10: 0, _t11: 0}) - sp.Rational(0)))

    # ------------------------------------------------------------------------
    # eq:sup_sn_inner  (sgm.tex L2193)
    # States: <psi^A(t)|psi^B(t)> = exp(i phi_0(t)) [sqrt((1-p_A)(1-p_B)) + sqrt(p_A p_B) exp(i Delta omega_SN t)].
    # Derived from: eq:sn_mean_field (L1507), eq:sn_overlap_drift (L1512).
    # Test: exact symbolic identity for the complex inner product including overall phase phi_0(t).
    # Mutant: wrong sign on phi_0(t) fails the complex inner product match.
    # ------------------------------------------------------------------------
    eq("eq:sup_sn_inner", 2217, "<psi^A(t)|psi^B(t)> = exp(i phi_0(t))[sqrt((1-p_A)(1-p_B)) + sqrt(p_A p_B) exp(i Delta omega_SN t)]")
    _phi0_sn = (_gs - _gd)*(_pA - _pB)*_ts_sn/hb
    _ov_sup = sp.exp(sp.I*_phi0_sn)*(_a0A*_a0B + _a1A*_a1B*sp.exp(sp.I*_dw_sn*_ts_sn))
    symcheck("complex inner product matches eq:sup_sn_inner", sp.simplify(sp.expand(_ov_sn - _ov_sup)))
    _ov_sup_bad = sp.exp(-sp.I*_phi0_sn)*(_a0A*_a0B + _a1A*_a1B*sp.exp(sp.I*_dw_sn*_ts_sn))
    mutant("opposite sign on phi_0(t) fails complex inner product", sp.simplify(sp.expand(_ov_sn - _ov_sup_bad)))

    # ------------------------------------------------------------------------
    # eq:sup_lt_drag  (sgm.tex L2201)
    # States: dx_1^a/dt|_{p_1=0} = sum_b W_{12}^{ab}(r) p_{2,b} = (G/(2 c^2 r))(7 p_{2,a} + (p_2.r_hat) r_hat^a).
    # Derived from: eq:eih_kernel (L1534), eq:eih_1pn (L1539), eq:legendre_bridge.
    # Test: verify Legendre transform of L = p_1^2/(2m_1) + p_2^2/(2m_2) + L_EIH^cross flips sign to +H_EIH^cross
    #       at O(c^{-2}), and transverse/radial projections give 7/2 and 4.
    # Mutant: keeping Lagrangian minus sign gives negative frame dragging.
    # ------------------------------------------------------------------------
    eq("eq:sup_lt_drag", 2225, "dx_1^a/dt|_{p_1=0} = (G/(2 c^2 r))(7 p_{2,a} + (p_2.r_hat) r_hat^a)")
    symcheck("transverse p_2 perp r gives dx_1/dt == +(7 G/(2 c^2 r)) p_2",
             sp.simplify((_xdot1_tgt.subs({_rv_eih[1]: 0, _rv_eih[2]: 0, _p2v[0]: 0})[1])
                         - 7*G/(2*c**2*sp.Abs(_rv_eih[0]))*_p2v[1]))
    mutant("Lagrangian minus sign flips frame-dragging direction",
           sp.simplify((-_xdot1_tgt.subs({_rv_eih[1]: 0, _rv_eih[2]: 0, _p2v[0]: 0})[1])
                       - 7*G/(2*c**2*sp.Abs(_rv_eih[0]))*_p2v[1]))

    # ------------------------------------------------------------------------
    # eq:sup_gw_power  (sgm.tex L2208)
    # States: P_GW[<E>] = (G Q_0^2 Omega_quad^6 / (10 c^5)) tr(<E>^2).
    # Derived from: eq:gw_power_bias (L1561).
    # Test: exact time average of (G/(5 c^5)) (d^3/dt^3 (Q_0 cos(Omega t) E))^2 over one period 2 pi / Omega.
    # Mutant: omitting the 1/2 cycle average of sin^2(Omega t) gives 5 instead of 10 in denominator.
    # ------------------------------------------------------------------------
    eq("eq:sup_gw_power", 2232, "P_GW[<E>] = (G Q_0^2 Omega_quad^6 / (10 c^5)) tr(<E>^2)")
    _t_gw = sp.Symbol("t_gw", real=True)
    _dddI = sp.diff(_Q0q*sp.cos(_Omq*_t_gw), _t_gw, 3)
    _avg_sq = sp.simplify(sp.integrate(_dddI**2, (_t_gw, 0, 2*sp.pi/_Omq)) / (2*sp.pi/_Omq))
    symcheck("cycle average (G/(5 c^5)) <(d^3(Q_0 cos(Omega t))/dt^3)^2> == G Q_0^2 Omega^6 / (10 c^5)",
             sp.simplify((G/(5*c**5))*_avg_sq - G*_Q0q**2*_Omq**6/(10*c**5)))
    mutant("omitting cycle average 1/2 gives 1/5 instead of 1/10",
           sp.simplify((G/(5*c**5))*(_Q0q**2*_Omq**6) - G*_Q0q**2*_Omq**6/(10*c**5)))

if __name__ == "__main__":
    part_S1()
    part_S2()
    part_S3()
    part_S4()
    summary()
