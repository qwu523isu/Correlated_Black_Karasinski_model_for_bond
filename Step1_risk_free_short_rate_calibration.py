import numpy as np
import pandas as pd
import os
from Pricing_core_lib import calibrate_risk_free

output_dir = os.path.join(
    r"G:\My Drive\0_Advanced Finance Note\Related_Code\Correlated_BK_model\Output",
    "Step1_risk_free_rate_params"
)

os.makedirs(output_dir, exist_ok=True)

# Step 1. Market zero-coupon bond curve (toy data)
np.random.seed(42)
maturities_rf = np.array([1, 3, 5, 7, 10]) 
# bond maturities in years, zero-coupon bonds (ZCBs) with maturities of 1y, 3y, 5y, 7y, and 10y.
zc_market = np.array([0.98, 0.92, 0.86, 0.81, 0.76])
# market prices of the zero-coupon bonds today.Each number is the discount factor (present value of $1 received at maturity).
# At 1 year: bond price = 0.98 → means the market is discounting $1 due in 1 year to 98 cents today.
# At 10 years: bond price = 0.76 → means $1 due in 10 years is worth 76 cents today.
    
alpha_r, sigma_r = calibrate_risk_free(maturities_rf, zc_market, r0=0.02)
print("Risk-Free Rate Calibration:")
print("alpha_r = %.4f, sigma_r = %.4f" % (alpha_r, sigma_r))

output_file = os.path.join(output_dir, "risk_free_params.csv")
parameters_df = pd.DataFrame({
    "Parameter": ["alpha_r", "sigma_r"],
    "Value": [alpha_r, sigma_r]
})
parameters_df.to_csv(output_file, index=False)





