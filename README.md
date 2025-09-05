# Correlated Black–Karasinski Model for HY Bond Pricing

This repository implements a **Correlated Black–Karasinski (BK) framework** for modeling interest rates and default intensities, and for pricing high-yield (HY) bonds.  

The workflow is organized into **three steps**:  

---

## 🔹 Step 1 — Risk-Free Short Rate Calibration
- **Model:**
  \[
  d(\ln r_t) = \kappa_r\big(\theta_r(t) - \ln r_t\big)\,dt + \sigma_r\,dW_t^r
  \]
- **Inputs:** Treasury/OIS zero-coupon bond curve.  
- **Calibration:**
  - Estimate global dynamics \((\alpha_r, \sigma_r)\).  
  - Bootstrap time-dependent drift \(\theta_r(t)\) to fit the ZC curve exactly.  
- **Outputs:**
  - `risk_free_params.csv` — calibrated \(\alpha_r, \sigma_r\)  
  - `theta_r_curve.csv` — bootstrapped \(\theta_r(t)\)  
- 📂 Saved in `Output/Step1_risk_free_rate_params/`

---

## 🔹 Step 2 — Default Intensity Calibration
- **Model:**
  \[
  d(\ln \lambda_t) = \kappa_\lambda\big(\theta_\lambda(t) - \ln \lambda_t\big)\,dt + \sigma_\lambda\,dW_t^\lambda
  \]
- **Inputs:** CDS survival curve.  
- **Calibration:**
  - Estimate global dynamics \((\alpha_\lambda, \sigma_\lambda)\).  
  - Bootstrap \(\theta_\lambda(t)\) to fit CDS survival probabilities exactly.  
- **Outputs:**
  - `default_intensity_params.csv` — calibrated \(\alpha_\lambda, \sigma_\lambda\)  
  - `theta_lambda_curve.csv` — bootstrapped \(\theta_\lambda(t)\)  
- 📂 Saved in `Output/Step2_default_intensity_params/`

---

## 🔹 Step 3 — Joint Correlated BK Simulation & Pricing
- **Model:**
  \[
  \begin{aligned}
  d(\ln r_t) &= \kappa_r(\theta_r(t) - \ln r_t)\,dt + \sigma_r\,dW_t^r \\
  d(\ln \lambda_t) &= \kappa_\lambda(\theta_\lambda(t) - \ln \lambda_t)\,dt + \sigma_\lambda\,dW_t^\lambda \\
  dW_t^r\, dW_t^\lambda &= \rho\,dt
  \end{aligned}
  \]
- **Simulation Engine:**  
  - Implemented in `simulate_correlated_bk`.  
  - Uses interpolation for \(\theta_r(t), \theta_\lambda(t)\).  
- **Pricing Functions:**
  - `HY_non_coupon_bond_price` — risky ZC HY bonds  
  - `HY_coupon_bond_price` — coupon-paying HY bonds (default + recovery)  
  - `HY_callable_bond_price` — callable HY bonds (NC – issuer’s call option)  
- **Outputs:**
  - `HY_non_coupon_bond_prices.csv` — risky ZC HY bond prices across maturities  
  - `HY_coupon_bond_price.csv` — example coupon HY bond price  
  - `HY_callable_bond_price.csv` — decomposition of NC value, option value, and callable bond value  
- 📂 Saved in `Output/Step3_joint_model/`

---

## 🔹 Repository Structure
```
Correlated_BK_model/
│
├── Pricing_core_lib.py               # Core functions (calibration, simulation, pricing)
│
├── Step1_risk_free_short_rate_calibration.py
│── Step2_default_intensity_calibration.py
│── Step3_BK_simulation.py
│
└── Output/
    ├── Step1_risk_free_rate_params/
    │   ├── risk_free_params.csv
    │   └── theta_r_curve.csv
    ├── Step2_default_intensity_params/
    │   ├── default_intensity_params.csv
    │   └── theta_lambda_curve.csv
    └── Step3_joint_model/
        ├── HY_non_coupon_bond_prices.csv
        ├── HY_coupon_bond_price.csv
        └── HY_callable_bond_price.csv
```

---

## 🔹 Usage
1. **Run Step 1** to calibrate the short rate.  
   ```bash
   python Step1_risk_free_short_rate_calibration.py
   ```
2. **Run Step 2** to calibrate the default intensity.  
   ```bash
   python Step2_default_intensity_calibration.py
   ```
3. **Run Step 3** to simulate joint dynamics and price HY bonds.  
   ```bash
   python Step3_BK_simulation.py
   ```

---

## 🔹 Notes
- All calibration is **curve-consistent** via bootstrapped \(\theta(t)\).  
- Monte Carlo paths are simulated with correlation \(\rho\) between interest rates and intensity.  
- Extendable to more products: CDS options, convertible bonds, etc.  
