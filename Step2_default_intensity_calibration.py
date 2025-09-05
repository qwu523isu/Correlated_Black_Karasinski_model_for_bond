import numpy as np
import pandas as pd
import os
from Pricing_core_lib import calibrate_default_intensity

output_dir = os.path.join(
    r"G:\My Drive\0_Advanced Finance Note\Related_Code\Correlated_BK_model\Output",
    "Step2_default_intensity_params"
)
os.makedirs(output_dir, exist_ok=True)

np.random.seed(42)
maturities_cds = np.array([1, 3, 5, 7]) # years
cds_survival_probability = np.array([0.95, 0.82, 0.70, 0.58])  # survival probs

alpha_l, sigma_l = calibrate_default_intensity(maturities_cds, cds_survival_probability, lambda0=0.5)
print("Default Intensity Calibration:")
print("alpha_lambda = %.4f, sigma_lambda = %.4f" % (alpha_l, sigma_l))

output_file = os.path.join(output_dir, "default_intensity_params.csv")
params_df = pd.DataFrame({
    "Parameter": ["alpha_lambda", "sigma_lambda"],
    "Value": [alpha_l, sigma_l]
})
params_df.to_csv(output_file, index=False)

