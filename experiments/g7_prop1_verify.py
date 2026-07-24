"""Monte-Carlo verification of Proposition 1 (paper/sections/app_proofs.tex).

Checks the three closed forms against simulation, including the cells where the
hypotheses bite:

  (eq:prop1)  affine class F = {c0 + c1*ybar + c2*g} available to RAW:
                  dMSE_F = kappa^2 [Var(ybar)Var(g) + Cov(ybar,g)^2]
              -- needs joint GAUSSIANITY (Isserlis); understated under t.

  (eq:prop1b) affine class R = {c0 + ybar + c2*g} reachable by RevIN, extra cost:
                  dMSE_R - dMSE_F = (a-1+kappa*E[g])^2 Var(ybar)(1-rho^2)
              -- needs vanishing third cross-moments, NOT Gaussianity;
                 needs g centred, does NOT need ybar centred.

  (eq:prop1c) unconstrained backbone: RAW attains Bayes (0), RevIN is confined to
              {ybar + phi(g)} and floors at
                  [(a-1+kappa*E[g])^2 + kappa^2 Var(g)] Var(ybar)(1-rho^2)
              -- so the RevIN-minus-RAW gap GROWS with capacity.

Target DGP throughout: y = a*ybar + b*g + kappa*ybar*g + eta.

Usage:  uv run python -m experiments.g7_prop1_verify
"""

import numpy as np

N = 2_000_000
NOISE_SD = 0.1
RTOL = 0.02  # MC tolerance at this N; the identities are exact


def draw(rng, rho, vy, vg, my=0.0, mg=0.0, dist="gauss", df=8):
    """(ybar, g) with the requested first two moments; 'ell_t' = elliptical t_df."""
    cov = rho * np.sqrt(vy * vg)
    z = rng.multivariate_normal([0.0, 0.0], [[vy, cov], [cov, vg]], size=N)
    if dist == "ell_t":                       # same covariance, heavier tails
        w = np.sqrt(df / rng.chisquare(df, size=N))[:, None]
        z = z * w / np.sqrt(df / (df - 2.0))
    return z[:, 0] + my, z[:, 1] + mg


def excesses(rng, a, b, kappa, rho, vy, vg, my=0.0, mg=0.0, dist="gauss"):
    """Empirical excess risk over Bayes for the four classes of Proposition 1."""
    yb, g = draw(rng, rho, vy, vg, my, mg, dist)
    y = a * yb + b * g + kappa * yb * g + rng.normal(0.0, NOISE_SD, N)
    sig2 = NOISE_SD ** 2

    X = np.column_stack([np.ones(N), yb, g])                    # F: RAW, affine
    f_aff = np.mean((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2) - sig2

    r = y - yb                                                  # R: RevIN, affine
    Xr = np.column_stack([np.ones(N), g])
    r_aff = np.mean((r - Xr @ np.linalg.lstsq(Xr, r, rcond=None)[0]) ** 2) - sig2

    # unconstrained: RAW reaches Bayes; RevIN is confined to {ybar + phi(g)} and its
    # optimum is phi*(g) = E[y - ybar | g] = (a-1+kappa*g) E[ybar|g] + b*g, which is
    # QUADRATIC in g. We evaluate phi* analytically rather than estimating it: an
    # equal-count binning estimator is badly biased here because the slope of r in g
    # is kappa*E[ybar], which is large whenever ybar is not centred, and the bias
    # would be mistaken for a failure of the closed form.
    cov = rho * np.sqrt(vy * vg)
    e_yb_given_g = my + (cov / vg) * (g - mg)
    phi_star = (a - 1.0 + kappa * g) * e_yb_given_g + b * g
    r_flex = np.mean((r - phi_star) ** 2) - sig2
    return f_aff, r_aff, r_flex


def closed_forms(a, b, kappa, rho, vy, vg, mg=0.0):
    cov = rho * np.sqrt(vy * vg)
    d_f = kappa ** 2 * (vy * vg + cov ** 2)
    shift = (a - 1.0 + kappa * mg) ** 2
    d_r = d_f + shift * vy * (1.0 - rho ** 2)
    d_flex = (shift + kappa ** 2 * vg) * vy * (1.0 - rho ** 2)
    return d_f, d_r, d_flex


# (a, b, kappa, rho, var_ybar, var_g, mean_ybar, mean_g, dist)
GRID = [
    (0.7, 1.2, 0.5, 0.4, 2.0, 3.0, 0.0, 0.0, "gauss"),
    (1.0, -0.5, 0.9, -0.6, 1.5, 0.8, 0.0, 0.0, "gauss"),   # a=1: eq:prop1b is 0
    (0.2, 1.0, 0.0, 0.6, 1.5, 0.7, 0.0, 0.0, "gauss"),     # kappa=0: no interaction
    (1.3, 0.5, 0.8, -0.3, 2.0, 1.0, 0.0, 0.0, "gauss"),
    (0.4, 0.2, 0.6, 0.0, 1.0, 2.0, 0.0, 0.0, "gauss"),     # rho=0
    (0.6, 1.0, 0.3, 0.5, 1.0, 1.0, 100.0, 0.0, "gauss"),   # ybar not centred: OK
    (0.6, 1.0, 0.3, 0.5, 1.0, 1.0, 0.0, 5.0, "gauss"),     # g not centred: needs mg
    (0.7, 1.2, 0.5, 0.4, 2.0, 3.0, 0.0, 0.0, "ell_t"),     # heavy tails: (i) fails
]


def main():
    rng = np.random.default_rng(0)
    hdr = ("a", "kap", "rho", "E[g]", "dist", "F mc", "F cf",
           "R mc", "R cf", "flex mc", "flex cf")
    print("{:>5}{:>5}{:>6}{:>6}{:>7} | {:>8}{:>8} | {:>8}{:>8} | {:>8}{:>8}"
          .format(*hdr))
    gauss_ok = True
    for a, b, k, rho, vy, vg, my, mg, dist in GRID:
        mc = excesses(rng, a, b, k, rho, vy, vg, my, mg, dist)
        cf = closed_forms(a, b, k, rho, vy, vg, mg)
        print("{:>5}{:>5}{:>6}{:>6}{:>7} | {:8.4f}{:8.4f} | {:8.4f}{:8.4f} | "
              "{:8.4f}{:8.4f}".format(a, k, rho, mg, dist,
                                      mc[0], cf[0], mc[1], cf[1], mc[2], cf[2]))
        if dist == "gauss":
            for got, want, name in zip(mc, cf, ("eq:prop1", "eq:prop1b", "eq:prop1c")):
                if abs(got - want) > RTOL * max(abs(want), 1e-3):
                    gauss_ok = False
                    print(f"    MISMATCH {name}: mc={got:.5f} closed form={want:.5f}")
    assert gauss_ok, "a closed form disagreed with Monte Carlo on a Gaussian cell"

    # the two documented hypothesis failures, asserted so they cannot rot silently
    mc_t = excesses(rng, 0.7, 1.2, 0.5, 0.4, 2.0, 3.0, dist="ell_t")
    cf_t = closed_forms(0.7, 1.2, 0.5, 0.4, 2.0, 3.0)
    assert mc_t[0] > 1.2 * cf_t[0], "eq:prop1 should UNDERSTATE under heavy tails"
    assert abs((mc_t[1] - mc_t[0]) - (cf_t[1] - cf_t[0])) < 0.05 * cf_t[1], \
        "eq:prop1b should survive heavy tails (it needs no Isserlis)"

    # capacity: the RevIN-minus-RAW gap grows, by exactly kappa^2 Var(g) Var(ybar)(1-rho^2)
    a, k, rho, vy, vg = 1.0, 0.9, -0.6, 1.5, 0.8
    d_f, d_r, d_flex = closed_forms(a, 0.0, k, rho, vy, vg)
    assert abs((d_flex - 0.0) - (d_r - d_f) - k ** 2 * vg * vy * (1 - rho ** 2)) < 1e-12
    print("\nOK: eq:prop1/1b/1c reproduce; Gaussianity is load-bearing only for "
          "eq:prop1; the RevIN-minus-RAW gap grows with capacity.")


if __name__ == "__main__":
    main()
