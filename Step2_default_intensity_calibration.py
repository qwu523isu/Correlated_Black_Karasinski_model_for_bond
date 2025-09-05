import numpy as np
import pandas as pd
import os
from Pricing_core_lib import calibrate_default_intensity, bootstrap_theta_lambda

output_dir = os.path.join(
    r"G:\My Drive\0_Advanced Finance Note\Related_Code\Correlated_BK_model\Output",
    "Step2_default_intensity_params"
)
os.makedirs(output_dir, exist_ok=True)

np.random.seed(42)
maturities_cds = np.array([1, 3, 5, 7]) # years
cds_survival_probability = np.array([0.95, 0.82, 0.70, 0.58])  # survival probs

# 1. Calibrate alpha, sigma
alpha_l, sigma_l = calibrate_default_intensity(
    maturities_cds, cds_survival_probability, lambda0=0.05
)
print("Default Intensity Calibration:")
print(f"alpha_lambda = {alpha_l:.4f}, sigma_lambda = {sigma_l:.4f}")

# 2. Bootstrap theta_lambda(t) for exact survival fit
theta_l = bootstrap_theta_lambda(maturities_cds, cds_survival_probability,
                                 alpha_l, sigma_l, lambda0=0.05)
print("\nBootstrapped theta_lambda(t):")
for T, th in theta_l.items():
    print(f"Maturity {T}y: theta_lambda = {th:.4f}")

# 3. Save results
params_df = pd.DataFrame({
    "Parameter": ["alpha_lambda", "sigma_lambda"],
    "Value": [alpha_l, sigma_l]
})
params_df.to_csv(os.path.join(output_dir, "default_intensity_params.csv"), index=False)

theta_df = pd.DataFrame({
    "Maturity": list(theta_l.keys()),
    "Theta_lambda": list(theta_l.values())
})
theta_df.to_csv(os.path.join(output_dir, "theta_lambda_curve.csv"), index=False)

print(f"\nParameters saved to {output_dir}")

