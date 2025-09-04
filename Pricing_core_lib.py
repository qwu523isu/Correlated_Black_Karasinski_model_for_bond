import numpy as np
from scipy.optimize import minimize


# ============================================
# Calibration Step 1: Risk-Free Short Rate
# ============================================
def calibrate_risk_free(maturities, zc_market, r0=0.02):
    def objective(params):
        alpha, sigma = params
        model_prices = [bond_price_bk(alpha, sigma, r0, T) for T in maturities]
        return np.sum((np.array(model_prices) - zc_market)**2)
    
    res = minimize(objective, [0.1, 0.01], bounds=[(0.01,1), (0.001,0.1)])
    return res.x  # [alpha_r, sigma_r]

# ============================================
# Black-Karasinski Simulation (log form)
# ============================================
def simulate_bk(alpha, sigma, r0, T, N, n_paths=200, dt=1/252):
    """
    Simulate one-factor BK log process:
    d(ln X_t) = -alpha * ln X_t dt + sigma dW
    """
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, N))
    X = np.full((n_paths, N+1), r0)
    for t in range(N):
        drift = -alpha * np.log(X[:, t]) * dt
        diffusion = sigma * dW[:, t]
        X[:, t+1] = np.exp(np.log(X[:, t]) + drift + diffusion)
    return X

# ============================================
# Pricing ZC bond with BK (for calibration)
# ============================================
def bond_price_bk(alpha, sigma, r0, T, n_paths=200, dt=1/252):
    N = int(T/dt)
    r_paths = simulate_bk(alpha, sigma, r0, T, N, n_paths, dt)
    disc = np.exp(-np.sum(r_paths[:, :-1]*dt, axis=1))
    return disc.mean()