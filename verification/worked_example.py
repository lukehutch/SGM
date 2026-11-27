"""Numerical test of the level theorems of sgm_wavefn.tex: Tables 2, 3, 4, 5 and 6.

The paper prints no numerical tables; the tables named here are the ones this script
prints. Theorems 1, 2 and 4 below are the paper's thm:levels_B, thm:levels_V and
thm:levels_grav.

Units m = c = 1, p_z = 0, spin sector s = +1 except in Part F. Run: python3 worked_example.py
(numpy and scipy; about twenty minutes on one processor for Parts A-E, and nine more
for Part F, three for Part G). Every number in Tables 2-6 and in the count of Part G is printed.

Part A (Table 2, Theorems 1 and 2): flat spacetime, rho = 3x/sqrt(9 + x^2), p_y = 0,
V = 0 or V = 0.2 cos x. The reference level is the zero in E of the (n+1)th
eigenvalue (n counted from 0) of the reduced operator
    K(E) = Pi^dagger (E - V + 1)^(-1) Pi - (E - V - 1),   Pi = p - i rho,
on N points of the periodic interval [-L, L), p applied by the discrete Fourier
transform. Its convergence in N and L is printed.

Part B (Table 3, Theorem 4): a static metric with a varying lapse,
alpha_n = 1 - 0.2 exp(-x^2/0.81), scale factor W = 1/alpha_n, V = 0, uniform qB = 1.3,
p_y = 0. The reference levels are the eigenvalues of the 2x2 block
    [[alpha_n,  C Pi^dagger C], [C Pi C,  -alpha_n]],   C = (alpha_n/W)^(1/2),
on a Fourier grid, keeping the eigenvectors with negligible weight near the box edge;
the change of each level for a wider box (L = 7) and for 1.5 times as many points
is printed.
The level followed is the computed level nearest E = 1.5, and n is the integer of the
full rule at that level.

Part C (Table 4, Theorem 4): the scale factor of Part B with the lapse set to 1,
alpha_n = 1, W = 1/(1 - 0.2 exp(-x^2/0.81)), computed in the same way.

Part D (Table 5, Theorem 4): the spectra of Part B, the levels nearest E = 1.2, 1.5, 1.8
and 2.1, with the error of the full rule divided by hbar^2 and that of the root rule
divided by hbar.

Part E: every solution of the two rules in Tables 2-5 recomputed with twice as many
quadrature nodes; the largest change of a level and of Theta_spin is printed.

Part F (Table 6, Theorems 1 and 2): the setting of Part A with p_y = 0.5, so that
rho = 3x/sqrt(9 + x^2) - 0.5, and the spin sector s = -1, Pi_{-1} = p + i rho. On the
periodic interval K(E) has states at the edge, where rho jumps; they are excluded by
their eigenvectors' weight near the edge (the lowest of them is printed), followed by
the convergence checks of Part A and the quadrature check of Part E for these levels.

Part G (Section 5, Theorem 4, the one-to-one count): the spectra of Part B on the whole
interval J = [1.2, 2.1]. Every computed level in J and every solution of the full rule in
J are listed; the program checks that each solution has exactly one level within hbar/4,
that these levels are distinct and are all the levels in J, and prints the largest error.

The full rule is  oint xi dx = 2 pi hbar (n + 1/2) + (1/2) s hbar Theta_spin  on the
orbit xi^2 + rho^2 = W^2 (cE^2 - 1), cE = (E - V)/alpha_n, with
    Theta_spin = -oint (rho (ln Ups)' + qB) dx/xi,   Ups = 1/(W (cE + 1)).
The root rule is  oint xi dx = 2 pi hbar (n + 1/2)  on xi^2 + rho^2 - s hbar qB = W^2 (cE^2 - 1).
"""
import numpy as np
from scipy.optimize import brentq


M_QUAD = 200                       # nodes of the midpoint rule; Part E doubles it
calls = []                         # every solution of the rules, rechecked in Part E


def oint(fx, E, h=None, xs=np.linspace(-12, 12, 24001)):
    """Area oint xi dx of the curve xi^2 = fx(x, E) if h is None, else oint h(x) dx/xi.
    The substitution x = mid - half cos t removes the square-root endpoints and makes
    both integrands periodic in t and, for the analytic data of these examples, analytic,
    so the midpoint rule in t converges exponentially; it never evaluates at a turning
    point. Part E checks the convergence."""
    M = M_QUAD
    idx = np.where(fx(xs, E) > 0)[0]
    assert np.all(np.diff(idx) == 1), "the orbit is not one closed curve"
    x1 = brentq(lambda x: fx(x, E), xs[idx[0]] - 0.01, xs[idx[0]])
    x2 = brentq(lambda x: fx(x, E), xs[idx[-1]], xs[idx[-1]] + 0.01)
    mid, half = (x1 + x2) / 2, (x2 - x1) / 2
    t = (np.arange(M) + 0.5) * np.pi / M
    x = mid - half * np.cos(t)
    xi = np.sqrt(fx(x, E))
    return 2 * np.pi / M * np.sum((xi if h is None else h(x, E) / xi) * half * np.sin(t))


def fourier(N, L, hb):
    x = np.linspace(-L, L, N, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(N, d=x[1] - x[0])
    P = np.fft.ifft(hb * k[:, None] * np.fft.fft(np.eye(N), axis=0), axis=0)
    return x, P


def rules(rho, qB, V, Vp, al, alp, W, Wp, hb, n=None, E_ref=None, s=1, lo=1.0 + 1e-9, hi=3.0):
    """Solutions of the full rule and of the root rule with the integer n; if n is None,
    n is the integer of the full rule at the energy E_ref."""
    cE = lambda x, E: (E - V(x)) / al(x)
    f0 = lambda x, E: W(x)**2 * (cE(x, E)**2 - 1) - rho(x)**2
    fr = lambda x, E: f0(x, E) + s * hb * qB(x)

    def lnUps_p(x, E):
        e = cE(x, E)
        ep = -(Vp(x) + e * alp(x)) / al(x)
        return -Wp(x) / W(x) - ep / (e + 1)
    theta = lambda E: -oint(f0, E, lambda x, E: rho(x) * lnUps_p(x, E) + qB(x))
    if n is None:
        n = int(round((oint(f0, E_ref) - 0.5 * s * hb * theta(E_ref)) / (2 * np.pi * hb) - 0.5))
    full = brentq(lambda E: oint(f0, E) - 0.5 * s * hb * theta(E) - 2 * np.pi * hb * (n + 0.5), lo, hi, xtol=1e-14)
    root = brentq(lambda E: oint(fr, E) - 2 * np.pi * hb * (n + 0.5), lo, hi, xtol=1e-14)
    calls.append(((rho, qB, V, Vp, al, alp, W, Wp, hb, n), dict(s=s, lo=lo, hi=hi), full, root, theta(full)))
    return full, root, theta, n


# ---------------------------------------------------------------- Part A
rhoA = lambda x: 3 * x / np.sqrt(9 + x**2)
qBA = lambda x: 27 / (9 + x**2)**1.5
one = lambda x: 1.0 + 0 * x
zero = lambda x: 0.0 * x


def level_K(n, hb, V, N=512, L=8.0):
    x, P = fourier(N, L, hb)
    Pi = P - 1j * np.diag(rhoA(x))

    def k_n(E):
        K = Pi.conj().T @ np.diag(1 / (E - V(x) + 1)) @ Pi - np.diag(E - V(x) - 1)
        return np.linalg.eigvalsh(0.5 * (K + K.conj().T))[n]
    return brentq(k_n, 1.43, 2.03, xtol=1e-13)


print("Table 2: rho = 3x/sqrt(9+x^2), n hbar = 0.8, s = +1")
print(f"{'V':>12} {'hbar':>5} {'E':>12} {'full - E':>11} {'root - E':>11} {'Theta_spin':>11}")
for A0 in (0.0, 0.2):
    V = lambda x, A0=A0: A0 * np.cos(x)
    Vp = lambda x, A0=A0: -A0 * np.sin(x)
    for hb in (0.16, 0.08, 0.04):
        n = int(round(0.8 / hb))
        E = level_K(n, hb, V)
        full, root, theta, n = rules(rhoA, qBA, V, Vp, one, zero, one, zero, hb, n, lo=1.4, hi=2.1)
        print(f"{('0' if A0 == 0 else '0.2 cos x'):>12} {hb:5.2f} {E:12.7f} {full - E:11.2e} {root - E:11.2e} {theta(E):11.6f}")
for hb in (0.16, 0.04):
    n = int(round(0.8 / hb))
    V = lambda x: 0.2 * np.cos(x)
    E0 = level_K(n, hb, V)
    dev = max(abs(level_K(n, hb, V, N=N, L=L) - E0) for N, L in ((512, 9.0), (1024, 8.0), (1024, 9.0)))
    print(f"convergence, V = 0.2 cos x, hbar = {hb}: largest change for L = 9 and N = 1024: {dev:.1e}")

# ---------------------------------------------------------------- Part B
qB0, eps, Lb = 1.3, 0.2, 0.9
alB = lambda x: 1 - eps * np.exp(-x**2 / Lb**2)
alpB = lambda x: 2 * x / Lb**2 * eps * np.exp(-x**2 / Lb**2)
WB = lambda x: 1 / alB(x)
WpB = lambda x: -alpB(x) / alB(x)**2
rhoB = lambda x: qB0 * x
qBB = lambda x: qB0 + 0 * x


def levels_block(hb, N=1024, L=6.0, al=alB, W=WB):
    x, P = fourier(N, L, hb)
    C = np.diag(np.sqrt(al(x) / W(x)))
    Q = C @ (P + 1j * np.diag(rhoB(x))) @ C      # C Pi^dagger C, Pi = p - i rho
    H = np.block([[np.diag(al(x)), Q], [Q.conj().T, -np.diag(al(x))]])
    ev, vec = np.linalg.eigh(H)
    edge = np.abs(x) > L - 1.5
    keep = [(e, v) for e, v in zip(ev, vec.T) if 0.8 < e < 2.2
            and (np.abs(v[:N])**2 + np.abs(v[N:])**2)[edge].sum() < 1e-8]
    return np.array([e for e, _ in keep])


print("\nTable 3: alpha_n = 1 - 0.2 exp(-x^2/0.81), W = 1/alpha_n, qB = 1.3, s = +1, level nearest E = 1.5")
print(f"{'hbar':>5} {'n':>4} {'E':>12} {'full - E':>11} {'root - E':>11} {'Theta_spin':>11}")
errs = []
lvB, altB = {}, {}                            # the spectra, kept for Part D
for hb in (0.04, 0.02, 0.01):
    lv = lvB[hb] = levels_block(hb, N=1024 if hb > 0.015 else 2048)
    E = lv[np.argmin(abs(lv - 1.5))]          # the computed level nearest 1.5
    full, root, theta, n = rules(rhoB, qBB, zero, zero, alB, alpB, WB, WpB, hb, E_ref=E)
    errs.append((full - E, root - E))
    print(f"{hb:5.2f} {n:4d} {E:12.7f} {full - E:11.2e} {root - E:11.2e} {theta(E):11.6f}")
for i in range(len(errs) - 1):
    print(f"error ratios on halving hbar: full {errs[i][0] / errs[i + 1][0]:.2f}, root {errs[i][1] / errs[i + 1][1]:.2f}")
for hb in (0.04, 0.02, 0.01):
    N0 = 1024 if hb > 0.015 else 2048
    lv = levels_block(hb, N=N0)
    E0 = lv[np.argmin(abs(lv - 1.5))]
    altB[hb] = (levels_block(hb, N=N0, L=7.0), levels_block(hb, N=N0 * 3 // 2))
    dev = max(abs(lv[np.argmin(abs(lv - 1.5))] - E0) for lv in altB[hb])
    print(f"convergence, hbar = {hb}: largest change for L = 7 or N = {N0 * 3 // 2}: {dev:.1e}")

# ---------------------------------------------------------------- Part C
alC = one
WC = lambda x: 1 / alB(x)          # the scale factor of Part B with the lapse set to 1
WpC = lambda x: -alpB(x) / alB(x)**2

print("\nTable 4: alpha_n = 1, W = 1/(1 - 0.2 exp(-x^2/0.81)), qB = 1.3, s = +1, level nearest E = 1.5")
print(f"{'hbar':>5} {'n':>4} {'E':>12} {'full - E':>11} {'root - E':>11} {'Theta_spin':>11}")
errs = []
for hb in (0.04, 0.02, 0.01):
    N0 = 1024 if hb > 0.015 else 2048
    lv = levels_block(hb, N=N0, al=alC, W=WC)
    E = lv[np.argmin(abs(lv - 1.5))]
    full, root, theta, n = rules(rhoB, qBB, zero, zero, alC, zero, WC, WpC, hb, E_ref=E)
    errs.append((full - E, root - E))
    print(f"{hb:5.2f} {n:4d} {E:12.7f} {full - E:11.2e} {root - E:11.2e} {theta(E):11.6f}")
    dev = max(abs(lv[np.argmin(abs(lv - 1.5))] - E)
              for lv in (levels_block(hb, N=N0, L=7.0, al=alC, W=WC),
                         levels_block(hb, N=N0 * 3 // 2, al=alC, W=WC)))
    print(f"convergence, hbar = {hb}: largest change for L = 7 or N = {N0 * 3 // 2}: {dev:.1e}")
for i in range(len(errs) - 1):
    print(f"error ratios on halving hbar: full {errs[i][0] / errs[i + 1][0]:.2f}, root {errs[i][1] / errs[i + 1][1]:.2f}")

# ---------------------------------------------------------------- Part D
print("\nTable 5: the metric of Table 3, levels nearest E = 1.2, 1.5, 1.8, 2.1: full error / hbar^2, root error / hbar")
print(f"{'near':>5} {'hbar':>5} {'n':>4} {'E':>12} {'full/hbar^2':>12} {'root/hbar':>10} {'change':>8}")
for Et in (1.2, 1.5, 1.8, 2.1):
    for hb in (0.04, 0.02, 0.01):
        lv = lvB[hb]
        E = lv[np.argmin(abs(lv - Et))]
        full, root, theta, n = rules(rhoB, qBB, zero, zero, alB, alpB, WB, WpB, hb, E_ref=E)
        dev = max(abs(a[np.argmin(abs(a - Et))] - E) for a in altB[hb])
        print(f"{Et:5.1f} {hb:5.2f} {n:4d} {E:12.7f} {(full - E) / hb**2:12.4f} {(root - E) / hb:10.4f} {dev:8.1e}")

# ---------------------------------------------------------------- Part E
done, M_QUAD = list(calls), 2 * M_QUAD
dfull = droot = dtheta = 0.0
for args, kw, full, root, th in done:
    full2, root2, theta2, _ = rules(*args, **kw)
    dfull, droot = max(dfull, abs(full2 - full)), max(droot, abs(root2 - root))
    dtheta = max(dtheta, abs(theta2(full) - th))
print(f"\nquadrature, all {len(done)} rule solutions of Tables 2-5 with {M_QUAD} nodes instead of {M_QUAD // 2}:"
      f" largest change of a level {max(dfull, droot):.1e}, of Theta_spin {dtheta:.1e}")

# ---------------------------------------------------------------- Part F
py = 0.5                                       # p_y != 0 and the spin sector s = -1
rhoF = lambda x: rhoA(x) - py
M_QUAD, calls = M_QUAD // 2, []                # undo the doubling of Part E


def level_KF(n, hb, V, N=512, L=8.0):
    """The zero in E of the (n+1)th eigenvalue of K(E) for s = -1, Pi_{-1} = p + i rho,
    counting only eigenvectors with weight below 1/2 within 1.5 of the edge of the periodic
    box, where rho jumps; the states at the edge are excluded by that weight."""
    x, P = fourier(N, L, hb)
    Pi = P + 1j * np.diag(rhoF(x))
    edge = np.abs(x) > L - 1.5

    def k_n(E):
        K = Pi.conj().T @ np.diag(1 / (E - V(x) + 1)) @ Pi - np.diag(E - V(x) - 1)
        ev, vec = np.linalg.eigh(0.5 * (K + K.conj().T))
        return ev[(np.abs(vec)**2)[edge].sum(axis=0) < 0.5][n]
    return brentq(k_n, 1.3, 2.3, xtol=1e-13)


print("\nTable 6: rho = 3x/sqrt(9+x^2) - 0.5, n hbar = 0.8, s = -1")
print(f"{'V':>12} {'hbar':>5} {'E':>12} {'full - E':>11} {'root - E':>11} {'Theta_spin':>11}")
for A0 in (0.0, 0.2):
    V = lambda x, A0=A0: A0 * np.cos(x)
    Vp = lambda x, A0=A0: -A0 * np.sin(x)
    for hb in (0.16, 0.08, 0.04):
        n = int(round(0.8 / hb))
        E = level_KF(n, hb, V)
        full, root, theta, n = rules(rhoF, qBA, V, Vp, one, zero, one, zero, hb, n, s=-1, lo=1.3, hi=2.3)
        print(f"{('0' if A0 == 0 else '0.2 cos x'):>12} {hb:5.2f} {E:12.7f} {full - E:11.2e} {root - E:11.2e} {theta(E):11.6f}")
for hb in (0.16, 0.08, 0.04):
    x, P = fourier(512, 8.0, hb)
    Pi = P + 1j * np.diag(rhoF(x))
    v = np.linalg.eigh(Pi.conj().T @ Pi)[1][:, 0]
    print(f"hbar = {hb}: weight of the lowest state of Pi^dagger Pi within 1.5 of the edge: {(np.abs(v)**2)[np.abs(x) > 6.5].sum():.6f}")
for hb in (0.16, 0.04):
    n = int(round(0.8 / hb))
    V = lambda x: 0.2 * np.cos(x)
    E0 = level_KF(n, hb, V)
    dev = max(abs(level_KF(n, hb, V, N=N, L=L) - E0) for N, L in ((512, 9.0), (1024, 8.0), (1024, 9.0)))
    print(f"convergence, V = 0.2 cos x, hbar = {hb}: largest change for L = 9 and N = 1024: {dev:.1e}")
done, M_QUAD = list(calls), 2 * M_QUAD
dlev = max(max(abs(f2 - full), abs(r2 - root)) for (args, kw, full, root, _), (f2, r2, _, _) in
           ((c, rules(*c[0], **c[1])) for c in done))
print(f"quadrature, the {len(done)} rule solutions of Table 6 with {M_QUAD} nodes instead of {M_QUAD // 2}:"
      f" largest change of a level {dlev:.1e}")

# ---------------------------------------------------------------- Part G
M_QUAD //= 2                                   # undo the doubling of Part F
J = (1.2, 2.1)
f0G = lambda x, E: WB(x)**2 * ((E / alB(x))**2 - 1) - rhoB(x)**2


def lnUpsG(x, E):
    e = E / alB(x)
    return -WpB(x) / WB(x) + e * alpB(x) / alB(x) / (e + 1)


print(f"\nAll levels in J = [{J[0]}, {J[1]}], metric of Table 3: every level against every solution of the full rule")
print(f"{'hbar':>5} {'levels':>7} {'solutions':>10} {'n':>9} {'one level within hbar/4':>24} {'max|err|/hbar^2':>16}")
for hb in (0.04, 0.02, 0.01):
    lv = lvB[hb]
    lvJ = lv[(lv >= J[0]) & (lv <= J[1])]
    S = lambda E: (oint(f0G, E) + 0.5 * hb * oint(f0G, E, lambda x, E: rhoB(x) * lnUpsG(x, E) + qBB(x))) / (2 * np.pi * hb) - 0.5
    n0, n1 = int(np.ceil(S(J[0]))), int(np.floor(S(J[1])))
    sol = np.array([brentq(lambda E: S(E) - n, 1.0 + 1e-9, 2.19, xtol=1e-14) for n in range(n0, n1 + 1)])
    near = np.array([lv[np.argmin(abs(lv - e))] for e in sol])
    one_each = all((abs(lv - e) < 0.25 * hb).sum() == 1 for e in sol) and len(set(near)) == len(sol) and np.isin(lvJ, near).all()
    print(f"{hb:5.2f} {len(lvJ):7d} {len(sol):10d} {f'{n0}-{n1}':>9} {str(one_each):>24} {abs(near - sol).max() / hb**2:16.4f}"
          f"   least spacing of the solutions / hbar {np.diff(sol).min() / hb:.3f}")
