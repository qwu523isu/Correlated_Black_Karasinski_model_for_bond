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

# Load calibrated parameters
risk_free_params = pd.read_csv(
r"G:\My Drive\0_Advanced Finance Note\Related_Code\Correlated_BK_model\Output\Step1_risk_free_rate_params\risk_free_params.csv"
)

default_intensity_params = pd.read_csv(
r"G:\My Drive\0_Advanced Finance Note\Related_Code\Correlated_BK_model\Output\Step2_default_intensity_params\default_intensity_params.csv"
)

alpha_r = risk_free_params.loc[risk_free_params["Parameter"]=="alpha_r","Value"].values[0]
sigma_r = risk_free_params.loc[risk_free_params["Parameter"]=="sigma_r","Value"].values[0]
alpha_l = default_intensity_params.loc[default_intensity_params["Parameter"]=="alpha_lambda","Value"].values[0]
sigma_l = default_intensity_params.loc[default_intensity_params["Parameter"]=="sigma_lambda","Value"].values[0]

# Initial levels
r0 = 0.02
lambda0 = 0.05
rho = -0.3

# Case1: HY non coupon bond pricing
maturities = [1, 3, 5, 7]
results = []
for T in maturities:
    price = HY_non_coupon_bond_price(alpha_r, sigma_r, r0,
                                     alpha_l, sigma_l, lambda0,
                                     rho, T, n_paths=200)
    results.append((T, price))

HY_non_coupon_bond_price_df = pd.DataFrame(results, columns=["Maturity","HYNonCouponBondPrice"])
output_file = os.path.join(output_dir, "HY_non_coupon_bond_prices.csv")
HY_non_coupon_bond_price_df.to_csv(output_file, index=False)

# Case2: HY coupon bond pricing
# Example: 5Y HY bond, 8% coupon, 40% recovery
HY_coupon_bond_price = HY_coupon_bond_price(alpha_r, sigma_r, r0,
                             alpha_l, sigma_l, lambda0,
                             rho, T=5,
                             par=100, coupon_rate=0.08, recovery=0.40,
                             n_paths=500, dt=1/12, seed=42)

print(f"5Y HY coupon bond price = {price:.4f}")

price_callable, price_nc, call_option_value = HY_callable_bond_price(
    alpha_r, sigma_r, r0,
    alpha_l, sigma_l, lambda0,
    rho, T=5,
    par=100, coupon_rate=0.08, recovery=0.40,
    call_time=2.0, call_price=100.0,
    n_paths=500, dt=1/12, seed=42
)

print(f"Non-callable bond price = {price_nc:.4f}")
print(f"Call option value (issuer) = {call_option_value:.4f}")
print(f"Callable bond price (investor) = {price_callable:.4f}")


