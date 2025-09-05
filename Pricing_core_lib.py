import numpy as np
from scipy.optimize import minimize

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
        rho, T, N, n_paths=200, dt=1/52):
    rng = np.random.default_rng(42)
    # correlated Brownian increments
    dw_r = rng.normal(0, np.sqrt(dt), (n_paths, N))
    dw_l_indep = rng.normal(0, np.sqrt(dt), (n_paths, N))
    dw_l = rho * dw_r + np.sqrt(1-rho**2) * dw_l_indep
    
    r = np.full((n_paths, N+1), r0)
    l = np.full((n_paths, N+1), lambda0)

    for t in range(N):
        r[:, t+1] = np.exp(np.log(r[:,t]) - alpha_r*np.log(r[:, t])*dt + sigma_r*dw_r[:, t])
        l[:, t+1] = np.exp(np.log(l[:, t]) - alpha_l*np.log(l[:, t])*dt + sigma_l*dw_l[:, t])

    return r, l

# ============================================
# HY non coupon bond pricing
# ============================================   
def HY_non_coupon_bond_price(
        alpha_r, sigma_r, r0,
        alpha_l, sigma_l, lambda0,
        rho, T, n_paths=200, dt=1/52):
    N = int(T/dt)
    r_paths, l_paths = simulate_correlated_bk(
        alpha_r, sigma_r, r0,
        alpha_l, sigma_l, lambda0,
        rho, T, N, n_paths, dt
    )
    disc = np.exp(-np.sum((r_paths[:, :-1] + l_paths[:, :-1]) * dt, axis=1))
    return disc.mean()

# ============================================
# HY coupon bond pricing
# ============================================    
def HY_coupon_bond_price(
        alpha_r, sigma_r, r0,
        alpha_l, sigma_l, lambda0,
        rho, T, par=100, coupon_rate=0.08, recovery=0.40,
        n_paths=500, dt=1/12, seed=None):
        
    N = int(T/dt)
    rng = np.random.default_rng(seed)
    
    # --- Generate correlated shocks ---
    dW_r = rng.normal(0, np.sqrt(dt), (n_paths, N))
    dW_l_indep = rng.normal(0, np.sqrt(dt), (n_paths, N))
    dW_l = rho * dW_r + np.sqrt(1-rho**2) * dW_l_indep
    
    # --- Initialize state variables ---
    ln_r = np.full((n_paths, N+1), np.log(r0))
    ln_l = np.full((n_paths, N+1), np.log(lambda0))
    
    # --- Simulate BK dynamics ---
    for t in range(N):
        ln_r[:, t+1] = ln_r[:, t] - alpha_r*ln_r[:, t]*dt + sigma_r*dW_r[:, t]
        ln_l[:, t+1] = ln_l[:, t] - alpha_l*ln_l[:, t]*dt + sigma_l*dW_l[:, t]
    
    r_paths = np.exp(ln_r)
    l_paths = np.exp(ln_l)
    
    # --- Build discount factors ---
    df = np.ones((n_paths, N+1))
    for t in range(N):
        df[:, t+1] = df[:, t] * np.exp(-r_paths[:, t]*dt)
    
    # --- Simulate default times ---
    U = rng.random((n_paths, N))
    default_step = np.full(n_paths, N+1, dtype=int)  # sentinel = no default
    for t in range(N):
        default_now = (U[:, t] < 1 - np.exp(-l_paths[:, t]*dt)) & (default_step == N+1)
        default_step[default_now] = t
    
    # --- Price cash-flows ---
    coupon_amount = par * coupon_rate / 2   # semiannual
    coupon_steps = np.arange(int(0.5/dt), N+1, int(0.5/dt))  # semiannual schedule
    
    pv = np.zeros(n_paths)
    
    for step in coupon_steps:
        cf = coupon_amount + (par if step == N else 0.0)
        alive = default_step > step
        pv += cf * df[:, step] * alive
    
    # --- Add recovery if default occurs before maturity ---
    has_default = default_step <= N
    pv[has_default] += recovery * par * df[has_default, default_step[has_default]]
    
    return pv.mean()
 
# ============================================
# Callable HY coupon bond pricing
# ============================================      
def HY_callable_bond_price(
        alpha_r, sigma_r, r0,
        alpha_l, sigma_l, lambda0,
        rho, T, par=100, coupon_rate=0.08, recovery=0.40,
        call_time=2.0, call_price=100.0,
        n_paths=5000, dt=1/12, seed=None):
    
    N = int(T/dt)
    call_step = int(call_time/dt)
    rng = np.random.default_rng(seed)
    
    # --- Correlated shocks ---
    dW_r = rng.normal(0, np.sqrt(dt), (n_paths, N))
    dW_l_indep = rng.normal(0, np.sqrt(dt), (n_paths, N))
    dW_l = rho * dW_r + np.sqrt(1-rho**2) * dW_l_indep
    
    # --- Simulate BK processes ---
    ln_r = np.full((n_paths, N+1), np.log(r0))
    ln_l = np.full((n_paths, N+1), np.log(lambda0))
    
    for t in range(N):
        ln_r[:, t+1] = ln_r[:, t] - alpha_r*ln_r[:, t]*dt + sigma_r*dW_r[:, t]
        ln_l[:, t+1] = ln_l[:, t] - alpha_l*ln_l[:, t]*dt + sigma_l*dW_l[:, t]
    
    r_paths = np.exp(ln_r)
    l_paths = np.exp(ln_l)
    
    # --- Discount factors ---
    df = np.ones((n_paths, N+1))
    for t in range(N):
        df[:, t+1] = df[:, t] * np.exp(-r_paths[:, t]*dt)
    
    # --- Default simulation ---
    U = rng.random((n_paths, N))
    default_step = np.full(n_paths, N+1, dtype=int)
    for t in range(N):
        default_now = (U[:, t] < 1 - np.exp(-l_paths[:, t]*dt)) & (default_step == N+1)
        default_step[default_now] = t
    
    # --- Cashflows without call ---
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
    
    # --- Call option valuation ---
    # Future cashflows after call (discounted to t=0)
    value_after_call0 = np.zeros(n_paths)
    for step in coupon_steps:
        if step > call_step:
            cf = coupon_amount + (par if step == N else 0.0)
            alive = default_step > step
            value_after_call0 += cf * df[:, step] * alive
    
    # Add recovery if default occurs after call date
    default_after_call = (default_step > call_step) & (default_step <= N)
    value_after_call0[default_after_call] += recovery * par * df[default_after_call, default_step[default_after_call]]
    
    # Bond value at call date (t = call_time)
    price_at_call = np.zeros(n_paths)
    alive_call = default_step > call_step
    price_at_call[alive_call] = value_after_call0[alive_call] / df[alive_call, call_step]
    
    # Issuer's call payoff = max(BondValue_at_call - CallPrice, 0)
    call_payoff = np.maximum(price_at_call - call_price, 0.0)
    
    # Discount back to t=0
    call_price0 = np.mean(call_payoff * df[:, call_step])
    
    # --- Final callable bond price ---
    price_callable0 = price_nc0 - call_price0
    return price_callable0, price_nc0, call_price0


    
