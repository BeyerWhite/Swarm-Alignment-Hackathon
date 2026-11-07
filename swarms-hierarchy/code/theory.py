"""Reference implementations of the hierarchy-formation models.

Bonabeau (as in Lacasa & Luque 2005, physics/0511105):
    P_ij = 1 / (1 + exp(eta * (h_j - h_i)))      i attacks j, i wins w.p. P_ij
    winner += Q (=1), loser -= F
    after each step, h <- (1 - mu) h
    rho = probability an agent initiates a contest in a step
Stauffer variant: eta replaced by the global order parameter sigma(t).
Ben-Naim & Redner 2005 (cond-mat/0503451): fitter always wins, decline at rate r.
"""
import numpy as np


def win_prob(h, eta):
    d = h[None, :] - h[:, None]            # d[i, j] = h_j - h_i
    return 1.0 / (1.0 + np.exp(np.clip(eta * d, -50, 50)))


def sigma_order(h, eta):
    """Std of P_ij over ordered pairs i != j (Sousa-Stauffer order parameter)."""
    n = len(h)
    P = win_prob(h, eta)
    return float(P[~np.eye(n, dtype=bool)].std())


def rho_c_bonabeau(mu, eta, F=1.0, N=None):
    """Pitchfork point of the egalitarian state (Lacasa & Luque eq. 8/9)."""
    fac = 1.0 if N is None else (N - 1) / N
    return 4 * mu * fac / (eta * (1 - mu) * (1 + F))


def mean_field_map(rho, mu, eta, F=1.0, N=2, T=4000, h0=None, rng=None):
    """Deterministic N-agent mean-field iteration (Lacasa & Luque eq. 3)."""
    rng = rng or np.random.default_rng(0)
    h = rng.normal(0, 1e-3, N) if h0 is None else np.array(h0, float)
    for _ in range(T):
        P = win_prob(h, eta)
        np.fill_diagonal(P, 0.0)
        gain = (P * (1 + F) - F)
        np.fill_diagonal(gain, 0.0)
        h = (1 - rho) * (1 - mu) * h + rho * (1 - mu) * (h + gain.sum(1) / (N - 1))
    return h


def simulate_bonabeau(N, rho, mu, eta, F=1.0, Q=1.0, T=2000, rng=None, h0=None,
                      stauffer=False, sigma_warmup=0, record_every=0):
    """Stochastic mean-field (well-mixed) Bonabeau / Stauffer simulation.

    Returns final h and (optionally) a time series of sigma.
    """
    rng = rng or np.random.default_rng()
    h = np.zeros(N) if h0 is None else np.array(h0, float)
    sig = 0.0
    trace = []
    for t in range(T):
        for i in rng.permutation(N):
            if rng.random() >= rho:
                continue
            j = rng.integers(N - 1)
            j = j + (j >= i)
            if stauffer:
                e = 1.0 if t < sigma_warmup else sig
            else:
                e = eta
            p = 1.0 / (1.0 + np.exp(np.clip(e * (h[j] - h[i]), -50, 50)))
            if rng.random() < p:
                h[i] += Q; h[j] -= F
            else:
                h[j] += Q; h[i] -= F
        h *= (1 - mu)
        if stauffer:  # sigma(t+1) = std of P_ij evaluated with the sigma used at step t
            sig = sigma_order(h, 1.0 if t < sigma_warmup else sig)
        if record_every and t % record_every == 0:
            trace.append(sig if stauffer else sigma_order(h, eta))
    return h, np.array(trace)


def simulate_bnr(N, r, T=2000, rng=None):
    """Ben-Naim & Redner: random pair meets at rate 1 (per agent pair-event), fitter wins (ties random),
    each agent declines by 1 at rate r (floor 0). Discrete-time Gillespie-like approximation:
    per unit time, N/2 contests and Poisson(r) declines per agent."""
    rng = rng or np.random.default_rng()
    k = np.zeros(N, int)
    for _ in range(T):
        for _ in range(N // 2):
            i, j = rng.choice(N, 2, replace=False)
            if k[i] > k[j] or (k[i] == k[j] and rng.random() < 0.5):
                k[i] += 1
            else:
                k[j] += 1
        dec = rng.poisson(r, N)
        k = np.maximum(k - dec, 0)
    return k


if __name__ == "__main__":
    # Sanity check against Lacasa & Luque reference case F=1, eta=1, mu=0.1
    mu, eta, F = 0.1, 1.0, 1.0
    for N in (2, 10):
        rc = rho_c_bonabeau(mu, eta, F, N)
        print(f"N={N}: analytic rho_c={rc:.4f}")
        for rho in np.round(np.linspace(0.5 * rc, min(1, 2.5 * rc), 9), 4):
            h = mean_field_map(rho, mu, eta, F, N)
            print(f"   rho={rho:.3f}  mean-field spread={h.max()-h.min():8.4f}  sigma={sigma_order(h, eta):.3f}")
