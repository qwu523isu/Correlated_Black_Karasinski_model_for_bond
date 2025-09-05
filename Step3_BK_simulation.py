import numpy as np
import pandas as pd
import os
from Pricing_core_lib import HY_non_coupon_bond_price, HY_coupon_bond_price, HY_callable_bond_price

# Output directory
output_dir = os.path.join(
    r"G:\My Drive\0_Advanced Finance Note\Related_Code\Correlated_BK_model\Output",
    "Step3_joint_model"
)
os.makedirs(output_dir, exist_ok=True)

# --- Load parameters from Step 1 & Step 2 ---
risk_free_params = pd.read_csv(
    r"G:\My Drive\0_Advanced Finance Note\Related_Code\Correlated_BK_model\Output\Step1_risk_free_rate_params\risk_free_params.csv"
)
theta_r_curve = pd.read_csv(
    r"G:\My Drive\0_Advanced Finance Note\Related_Code\Correlated_BK_model\Output\Step1_risk_free_rate_params\theta_r_curve.csv"
)

credit_params = pd.read_csv(
    r"G:\My Drive\0_Advanced Finance Note\Related_Code\Correlated_BK_model\Output\Step2_default_intensity_params\default_intensity_params.csv"
)
theta_l_curve = pd.read_csv(
    r"G:\My Drive\0_Advanced Finance Note\Related_Code\Correlated_BK_model\Output\Step2_default_intensity_params\theta_lambda_curve.csv"
)

alpha_r = risk_free_params.loc[risk_free_params["Parameter"]=="alpha_r","Value"].values[0]
sigma_r = risk_free_params.loc[risk_free_params["Parameter"]=="sigma_r","Value"].values[0]
alpha_l = credit_params.loc[credit_params["Parameter"]=="alpha_lambda","Value"].values[0]
sigma_l = credit_params.loc[credit_params["Parameter"]=="sigma_lambda","Value"].values[0]

# Convert theta curves to dicts {maturity: value}
theta_r = dict(zip(theta_r_curve["Maturity"], theta_r_curve["Theta_r"]))
theta_l = dict(zip(theta_l_curve["Maturity"], theta_l_curve["Theta_lambda"]))

# Initial levels
r0 = 0.02
lambda0 = 0.05
rho = -0.3   # correlation assumption or calibration

# --- Case Non Coupon HY bond ---
maturities = [1, 3, 5, 7]
results = []
for T in maturities:
    price = HY_non_coupon_bond_price(alpha_r, sigma_r, r0,
                                     alpha_l, sigma_l, lambda0,
                                     rho, T,
                                     theta_r=theta_r, theta_l=theta_l,
                                     n_paths=200, dt=1/52, seed=42)
    results.append((T, price))

df = pd.DataFrame(results, columns=["Maturity","RiskyBondPrice"])
df.to_csv(os.path.join(output_dir, "HY_non_coupon_bond_prices.csv"), index=False)
print(df)

# --- Case Coupon HY bond ---
price_coupon = HY_coupon_bond_price(alpha_r, sigma_r, r0,
                                    alpha_l, sigma_l, lambda0,
                                    rho, T=5,
                                    theta_r=theta_r, theta_l=theta_l,
                                    par=100, coupon_rate=0.08, recovery=0.40,
                                    n_paths=2000, dt=1/12, seed=42)

df_coupon = pd.DataFrame([{"Maturity": 5, "HY_CouponBondPrice": price_coupon}])
df_coupon.to_csv(os.path.join(output_dir, "HY_coupon_bond_price.csv"), index=False)
print("\n=== Coupon HY Bond ===")
print(df_coupon)

# --- Case Callable Coupon HY bond ---
price_callable, price_nc, option_value = HY_callable_bond_price(
    alpha_r, sigma_r, r0,
    alpha_l, sigma_l, lambda0,
    rho, T=5,
    theta_r=theta_r, theta_l=theta_l,
    par=100, coupon_rate=0.08, recovery=0.40,
    call_time=2.0, call_price=100.0,
    n_paths=2000, dt=1/12, seed=42
)

df_callable = pd.DataFrame([{
    "Maturity": 5,
    "NonCallable": price_nc,
    "CallOptionValue": option_value,
    "CallableBond": price_callable
}])
df_callable.to_csv(os.path.join(output_dir, "HY_callable_bond_price.csv"), index=False)
print("\n=== Callable HY Bond ===")
print(df_callable)