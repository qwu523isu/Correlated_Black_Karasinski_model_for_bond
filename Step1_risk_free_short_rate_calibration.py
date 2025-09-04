import numpy as np
import pandas as pd
from Pricing_core_lib import calibrate_risk_free

# Step 1. Market zero-coupon bond curve (toy data)
np.random.seed(42)
maturities_rf = np.array([1, 3, 5, 7, 10])
zc_market = np.array([0.98, 0.92, 0.86, 0.81, 0.76])
    
alpha_r, sigma_r = calibrate_risk_free(maturities_rf, zc_market, r0=0.02)
print("Risk-Free Calibration:")
print("alpha_r = %.4f, sigma_r = %.4f" % (alpha_r, sigma_r))

parameters_df = pd.DataFrame({
    "Parameter": ["alpha_r", "sigma_r"],
    "Value": [alpha_r, sigma_r]
})

parameters_df.to_csv("risk_free_rate_params.csv", index=False)


