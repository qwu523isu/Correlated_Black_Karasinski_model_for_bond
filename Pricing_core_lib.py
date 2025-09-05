import numpy as np
from scipy.optimize import minimize

# ============================================
# Solve for time-dependent theta_r(t) so BK fits ZC curve exactly
# ============================================
def bootstrap_theta_r(maturities, zc_market, alpha_r, sigma_r, r0):
    theta_r = {}
    for T, P_mkt in zip(maturities, zc_market):
        theta_r[T] = np.log(-np.log(P_mkt)/alpha_r)  # placeholder form
    return theta_r

# ============================================
# Solve for time-dependent theta_lambda(t) so BK fits survival curve exactly.
# ============================================
def bootstrap_theta_lambda(maturities, cds_survival, alpha_l, sigma_l, lambda0):
    theta_l = {}
    for T, Q_mkt in zip(maturities, cds_survival):
        theta_l[T] = np.log(-np.log(Q_mkt)/alpha_l)  # placeholder
    return theta_l

# ============================================
# Calibration Risk-Free Short Rate
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
# Pricing ZC bond with BK for calibration
# ============================================
def bond_price_bk(alpha, sigma, r0, T, n_paths=200, dt=1/252):
    N = int(T/dt)
    r_paths = simulate_bk(alpha, sigma, r0, T, N, n_paths, dt)
    disc = np.exp(-np.sum(r_paths[:, :-1]*dt, axis=1))
    return disc.mean()

# ============================================
# Simulate BK dynamics for lambda_t
# ============================================
def simulate_bk_lambda(alpha, sigma, lambda0, T, N, n_paths=200, dt=1/252):
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, N))
    X = np.full((n_paths, N+1), lambda0)
    for t in range(N):
        drift = -alpha * np.log(X[:, t]) * dt
        diffusion = sigma * dW[:, t]
        X[:, t+1] = np.exp(np.log(X[:, t]) + drift + diffusion)
    return X

# ============================================
# Compute survival probability under BK
# ============================================
def survival_prob_bk(alpha, sigma, lambda0, T, n_paths=200, dt=1/252):
    N = int(T/dt)
    lam_paths = simulate_bk_lambda(alpha, sigma, lambda0, T, N, n_paths, dt)
    surv = np.exp(-np.sum(lam_paths[:, :-1]*dt, axis=1))
    return surv.mean()

# ============================================
# Calibration function for credit intensity
# ============================================
def calibrate_default_intensity(maturities, cds_survival, lambda0=0.05):
    def objective(params):
        alpha, sigma = params
        model_surv = [survival_prob_bk(alpha, sigma, lambda0, T) for T in maturities]
        return np.sum((np.array(model_surv) - cds_survival)**2)
    
    res = minimize(objective, [0.2, 0.05], bounds=[(0.01,1), (0.001,0.2)])
    return res.x  # [alpha_lambda, sigma_lambda]

# ============================================
# Correlated BK simulation
# ============================================
def simulate_correlated_bk(
        alpha_r, sigma_r, r0,
        alpha_l, sigma_l, lambda0,
        rho, T, N, n_paths=200, dt=1/52,
        theta_r=None, theta_l=None, seed=None):
    rng = np.random.default_rng(seed)

    # Correlated Brownian increments
    dW_r = rng.normal(0, np.sqrt(dt), (n_paths, N))
    dW_l_indep = rng.normal(0, np.sqrt(dt), (n_paths, N))
    dW_l = rho * dW_r + np.sqrt(1 - rho**2) * dW_l_indep

    # Initialize log states
    ln_r = np.full((n_paths, N+1), np.log(r0))
    ln_l = np.full((n_paths, N+1), np.log(lambda0))

    # Convert theta dicts into arrays for interpolation
    t_r = np.array(list(theta_r.keys()), dtype=float)
    v_r = np.array(list(theta_r.values()), dtype=float)
    t_l = np.array(list(theta_l.keys()), dtype=float)
    v_l = np.array(list(theta_l.values()), dtype=float)

    # Simulation loop
    for t in range(N):
        current_time = (t+1) * dt

        theta_r_val = np.interp(current_time, t_r, v_r)
        theta_l_val = np.interp(current_time, t_l, v_l)

        ln_r[:, t+1] = ln_r[:, t] + alpha_r * (theta_r_val - ln_r[:, t]) * dt + sigma_r * dW_r[:, t]
        ln_l[:, t+1] = ln_l[:, t] + alpha_l * (theta_l_val - ln_l[:, t]) * dt + sigma_l * dW_l[:, t]

    r_paths = np.exp(ln_r)
    l_paths = np.exp(ln_l)

    return r_paths, l_paths

# ============================================
# HY non coupon bond pricing
# Price a risky zero-coupon (non-coupon) bond under correlated BK model.
# If theta_r and theta_l are provided, uses full BK with time-dependent drifts.
# ============================================   
def HY_non_coupon_bond_price(
        alpha_r, sigma_r, r0,
        alpha_l, sigma_l, lambda0,
        rho, T, theta_r=None, theta_l=None,
        n_paths=200, dt=1/52, seed=None):
    N = int(T/dt)
    r_paths, l_paths = simulate_correlated_bk(
        alpha_r, sigma_r, r0,
        alpha_l, sigma_l, lambda0,
        rho, T, N, n_paths, dt,
        theta_r=theta_r, theta_l=theta_l,
        seed=seed
    )
    disc = np.exp(-np.sum((r_paths[:, :-1] + l_paths[:, :-1]) * dt, axis=1))
    return disc.mean()


# ============================================
# HY coupon bond pricing
# ============================================    
def HY_coupon_bond_price(
        alpha_r, sigma_r, r0,
        alpha_l, sigma_l, lambda0,
        rho, T,
        theta_r, theta_l,
        par=100, coupon_rate=0.08, recovery=0.40,
        n_paths=5000, dt=1/12, seed=None):
    N = int(T/dt)

    # Simulate correlated paths with theta curves
    r_paths, l_paths = simulate_correlated_bk(
        alpha_r, sigma_r, r0,
        alpha_l, sigma_l, lambda0,
        rho, T, N, n_paths, dt,
        theta_r=theta_r, theta_l=theta_l, seed=seed
    )

    # Discount factors
    df = np.ones((n_paths, N+1))
    for t in range(N):
        df[:, t+1] = df[:, t] * np.exp(-r_paths[:, t]*dt)

    # Simulate default times
    rng = np.random.default_rng(seed)
    U = rng.random((n_paths, N))
    default_step = np.full(n_paths, N+1, dtype=int)
    for t in range(N):
        default_now = (U[:, t] < 1 - np.exp(-l_paths[:, t]*dt)) & (default_step == N+1)
        default_step[default_now] = t

    # Coupon schedule (semiannual)
    coupon_amount = par * coupon_rate / 2
    coupon_steps = np.arange(int(0.5/dt), N+1, int(0.5/dt))

    pv = np.zeros(n_paths)

    for step in coupon_steps:
        cf = coupon_amount + (par if step == N else 0.0)
        alive = default_step > step
        pv += cf * df[:, step] * alive

    # Add recovery payoff
    has_default = default_step <= N
    pv[has_default] += recovery * par * df[has_default, default_step[has_default]]

    return pv.mean()
 
# ============================================
# Callable HY coupon bond pricing
# ============================================      
def HY_callable_bond_price(
        alpha_r, sigma_r, r0,
        alpha_l, sigma_l, lambda0,
        rho, T,
        theta_r, theta_l,
        par=100, coupon_rate=0.08, recovery=0.40,
        call_time=2.0, call_price=100.0,
        n_paths=5000, dt=1/12, seed=None):
    N = int(T/dt)
    call_step = int(call_time/dt)

    # Simulate correlated paths with theta curves
    r_paths, l_paths = simulate_correlated_bk(
        alpha_r, sigma_r, r0,
        alpha_l, sigma_l, lambda0,
        rho, T, N, n_paths, dt,
        theta_r=theta_r, theta_l=theta_l, seed=seed
    )

    # Discount factors
    df = np.ones((n_paths, N+1))
    for t in range(N):
        df[:, t+1] = df[:, t] * np.exp(-r_paths[:, t]*dt)

    # Simulate defaults
    rng = np.random.default_rng(seed)
    U = rng.random((n_paths, N))
    default_step = np.full(n_paths, N+1, dtype=int)
    for t in range(N):
        default_now = (U[:, t] < 1 - np.exp(-l_paths[:, t]*dt)) & (default_step == N+1)
        default_step[default_now] = t

    # Non-callable value
    coupon_amount = par * coupon_rate / 2
    coupon_steps = np.arange(int(0.5/dt), N+1, int(0.5/dt))

    pv_nc = np.zeros(n_paths)
    for step in coupon_steps:
        cf = coupon_amount + (par if step == N else 0.0)
        alive = default_step > step
        pv_nc += cf * df[:, step] * alive

    has_default = default_step <= N
    pv_nc[has_default] += recovery * par * df[has_default, default_step[has_default]]

    price_nc0 = pv_nc.mean()

    # Value after call date
    value_after_call0 = np.zeros(n_paths)
    for step in coupon_steps:
        if step > call_step:
            cf = coupon_amount + (par if step == N else 0.0)
            alive = default_step > step
            value_after_call0 += cf * df[:, step] * alive

    default_after_call = (default_step > call_step) & (default_step <= N)
    value_after_call0[default_after_call] += recovery * par * df[default_after_call, default_step[default_after_call]]

    price_at_call = np.zeros(n_paths)
    alive_call = default_step > call_step
    price_at_call[alive_call] = value_after_call0[alive_call] / df[alive_call, call_step]

    # Issuer option
    call_payoff = np.maximum(price_at_call - call_price, 0.0)
    call_price0 = np.mean(call_payoff * df[:, call_step])

    # Callable value = NC - option
    price_callable0 = price_nc0 - call_price0
    return price_callable0, price_nc0, call_price0
