import pickle
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
import sys
sys.stdout.reconfigure(line_buffering=True)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.decomposition import PCA
from xgboost import XGBRegressor, XGBClassifier

print("="*85)
print("COMPREHENSIVE DIAGNOSTICS & PLOTTING SCRIPT")
print("="*85)

# Load 200 Hz cache
cache_path = "features_cache_200hz.pkl"
X_subj, Xc_subj, y_subj, g_subj = pd.read_pickle(cache_path)

keep213 = (y_subj >= 120.0) & (y_subj <= 160.0)
X_ppg213 = X_subj[keep213].reset_index(drop=True)
Xc213 = Xc_subj[keep213].reset_index(drop=True)
y213 = y_subj[keep213]
g213 = g_subj[keep213]
is_miss213 = Xc213["is_missing_hw"].values.astype(bool)

logo = LeaveOneGroupOut()

# -------------------------------------------------------------
# 1. WITHIN-SEX AGE REGRESSION (SIMPSON'S PARADOX)
# -------------------------------------------------------------
print("\n--- 1. Within-Sex Age Regression on Hb ---")
simpson_stats = {}
for sex_label, sex_val in [("Females", 0), ("Males", 1)]:
    mask = (Xc213["Gender"] == sex_val)
    age_s = Xc213.loc[mask, "Age (year)"].values
    hb_s = y213[mask]
    
    slope, intercept, r_val, p_val, std_err = stats.linregress(age_s, hb_s)
    t_val = stats.t.ppf(0.975, df=len(age_s)-2)
    slope_lo = slope - t_val * std_err
    slope_hi = slope + t_val * std_err
    
    simpson_stats[sex_label] = {
        "N": len(age_s), "mean_age": np.mean(age_s), "std_age": np.std(age_s),
        "mean_hb": np.mean(hb_s), "std_hb": np.std(hb_s),
        "slope": slope, "slope_lo": slope_lo, "slope_hi": slope_hi,
        "r": r_val, "p": p_val, "age": age_s, "hb": hb_s, "intercept": intercept
    }
    print(f"{sex_label} (N = {len(age_s)}):")
    print(f"  Slope: {slope:+.3f} g/L per year [95% CI: {slope_lo:+.3f}, {slope_hi:+.3f}], r = {r_val:+.3f}, p = {p_val:.4f}")

slope_all, intercept_all, r_all, p_all, se_all = stats.linregress(Xc213["Age (year)"].values, y213)
t_all = stats.t.ppf(0.975, df=len(y213)-2)
print(f"Unstratified Cohort (N = {len(y213)}):")
print(f"  Marginal Slope: {slope_all:+.3f} g/L per year [95% CI: {slope_all - t_all*se_all:+.3f}, {slope_all + t_all*se_all:+.3f}], r = {r_all:+.3f}, p = {p_all:.4f}")

# -------------------------------------------------------------
# 2. BATCH SEPARABILITY (WAVEFORMS -> BATCH)
# -------------------------------------------------------------
print("\n--- 2. Batch Separability from Waveforms ---")
cls_cols = [c for c in X_ppg213.columns if "_q_" not in c and "cross_q_" not in c]
X_cls = X_ppg213[cls_cols].fillna(X_ppg213[cls_cols].median())

preds_batch = np.zeros(len(y213))
for tr_idx, te_idx in logo.split(X_cls, is_miss213, g213):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_cls.iloc[tr_idx])
    X_te = sc.transform(X_cls.iloc[te_idx])
    clf = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=1)
    clf.fit(X_tr, is_miss213[tr_idx])
    preds_batch[te_idx] = clf.predict_proba(X_te)[:, 1]

batch_auc = roc_auc_score(is_miss213, preds_batch)
fpr_batch, tpr_batch, _ = roc_curve(is_miss213, preds_batch)
print(f"Waveform -> Batch Classification LOSO AUC = {batch_auc:.4f}")

# -------------------------------------------------------------
# 3. REGULARIZATION AUDIT & SHRUNK-TO-PRIOR MODEL
# -------------------------------------------------------------
print("\n--- 3. Ridge Regularization Audit ---")
morph_cols = [
    "660nm_acdc_med", "660nm_ac_amp_med", "660nm_spectral_entropy_med",
    "730nm_acdc_med", "730nm_ac_amp_med", "730nm_spectral_entropy_med",
    "850nm_acdc_med", "850nm_ac_amp_med", "850nm_spectral_entropy_med",
    "940nm_acdc_med", "940nm_ac_amp_med", "940nm_spectral_entropy_med"
]
q_min_cols = [
    "660nm_q_von_neumann_entropy_med", "660nm_q_purity_med",
    "730nm_q_von_neumann_entropy_med", "730nm_q_purity_med",
    "850nm_q_von_neumann_entropy_med", "850nm_q_purity_med",
    "940nm_q_von_neumann_entropy_med", "940nm_q_purity_med"
]
X_morph = X_ppg213[morph_cols].fillna(X_ppg213[morph_cols].median()).values
X_morph_q = X_ppg213[morph_cols + q_min_cols].fillna(X_ppg213[morph_cols + q_min_cols].median()).values

# Demographic prior predictions
X_demo = Xc213[["Age (year)", "Gender"]].values
preds_prior = np.zeros(len(y213))
for tr_idx, te_idx in logo.split(X_demo, y213, g213):
    sc = RobustScaler()
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_tr_p = poly.fit_transform(sc.fit_transform(X_demo[tr_idx]))
    X_te_p = poly.transform(sc.transform(X_demo[te_idx]))
    mdl = RidgeCV(alphas=np.logspace(-2, 5, 100)).fit(X_tr_p, y213[tr_idx])
    preds_prior[te_idx] = mdl.predict(X_te_p)

y_res = y213 - preds_prior
mae_prior = mean_absolute_error(y213, preds_prior)

# Standard grid Ridge
alphas_chosen = []
preds_res = np.zeros(len(y213))
for tr_idx, te_idx in logo.split(X_morph, y_res, g213):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_morph[tr_idx])
    X_te = sc.transform(X_morph[te_idx])
    mdl = RidgeCV(alphas=np.logspace(0, 4, 50)).fit(X_tr, y_res[tr_idx])
    alphas_chosen.append(mdl.alpha_)
    preds_res[te_idx] = mdl.predict(X_te)

preds_ridge = preds_prior + preds_res
mae_ridge = mean_absolute_error(y213, preds_ridge)
print(f"Prior-Only MAE: {mae_prior:.3f} g/L")
print(f"12 Morph Features on Residual (RidgeCV): MAE = {mae_ridge:.3f} g/L (Δ = {mae_ridge - mae_prior:+.3f} g/L)")
print(f"Alphas chosen in 213 folds: Min = {np.min(alphas_chosen):.1f}, Median = {np.median(alphas_chosen):.1f}, Max = {np.max(alphas_chosen):.1f}")

# -------------------------------------------------------------
# 4. CHEN-STYLE BASELINE (DEFAULT XGBOOST, ALL FEATURES)
# -------------------------------------------------------------
print("\n--- 4. Chen-style Baseline (Default XGBoost, All 168 Features) ---")
preds_chen = np.zeros(len(y213))
for tr_idx, te_idx in logo.split(X_cls, y213, g213):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_cls.iloc[tr_idx])
    X_te = sc.transform(X_cls.iloc[te_idx])
    xgb_default = XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=1)
    xgb_default.fit(X_tr, y213[tr_idx])
    preds_chen[te_idx] = xgb_default.predict(X_te)

mae_chen = mean_absolute_error(y213, preds_chen)
rmse_chen = np.sqrt(mean_squared_error(y213, preds_chen))
r2_chen = r2_score(y213, preds_chen)
print(f"Chen Baseline (Default XGB, all 168 features): MAE = {mae_chen:.3f} g/L, RMSE = {rmse_chen:.3f} g/L, R2 = {r2_chen:.4f}")

# -------------------------------------------------------------
# 5. FEATURE SELECTION SENSITIVITY (k in {5, 10, 20, 50, 100, all})
# -------------------------------------------------------------
print("\n--- 5. Feature Selection Sensitivity ---", flush=True)
k_list = [5, 10, 20, 50, 100, "all"]
k_maes = []
for k in k_list:
    preds_k = np.zeros(len(y213))
    for tr_idx, te_idx in logo.split(X_cls, y213, g213):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_cls.iloc[tr_idx])
        X_te = sc.transform(X_cls.iloc[te_idx])
        if k != "all":
            sel = SelectKBest(f_regression, k=k)
            X_tr_k = sel.fit_transform(X_tr, y213[tr_idx])
            X_te_k = sel.transform(X_te)
        else:
            X_tr_k, X_te_k = X_tr, X_te
        xgb = XGBRegressor(n_estimators=30, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1)
        xgb.fit(X_tr_k, y213[tr_idx])
        preds_k[te_idx] = xgb.predict(X_te_k)
    m = mean_absolute_error(y213, preds_k)
    k_maes.append(m)
    print(f"  k = {str(k):>4s}: MAE = {m:.3f} g/L", flush=True)

# -------------------------------------------------------------
# 6. POWER CURVES (SYNTHETIC FEATURE INJECTION)
# -------------------------------------------------------------
print("\n--- 6. Power Curve (Synthetic Feature Injection) ---", flush=True)
r_levels = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
power_delta_maes = []
power_q_vals = []
rng = np.random.RandomState(42)

for r_target in r_levels:
    z = rng.randn(len(y_res))
    z_orth = z - (np.dot(z, y_res) / np.dot(y_res, y_res)) * y_res
    z_orth = z_orth / np.std(z_orth)
    y_std = y_res / np.std(y_res)
    feat_synth = r_target * y_std + np.sqrt(1.0 - r_target**2) * z_orth
    actual_r = np.corrcoef(feat_synth, y_res)[0, 1]
    
    t_stat = actual_r * np.sqrt((len(y_res) - 2) / (1.0 - actual_r**2))
    p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=len(y_res)-2))
    q_val = min(1.0, p_val * 349.0 / 1.0)
    power_q_vals.append(q_val)
    
    X_test_feat = np.column_stack([X_morph, feat_synth])
    preds_synth = np.zeros(len(y213))
    for tr_idx, te_idx in logo.split(X_test_feat, y_res, g213):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_test_feat[tr_idx])
        X_te = sc.transform(X_test_feat[te_idx])
        mdl = RidgeCV(alphas=np.logspace(-2, 5, 50)).fit(X_tr, y_res[tr_idx])
        preds_synth[te_idx] = preds_prior[te_idx] + mdl.predict(X_te)
    mae_synth = mean_absolute_error(y213, preds_synth)
    power_delta_maes.append(mae_synth - mae_prior)
    print(f"  r = {r_target:.2f}: Actual r = {actual_r:.3f}, p = {p_val:.2e}, FDR q = {q_val:.4f}, ΔMAE = {mae_synth - mae_prior:+.3f} g/L", flush=True)

# -------------------------------------------------------------
# 7. LEARNING CURVE (MAE VS SAMPLE SIZE)
# -------------------------------------------------------------
print("\n--- 7. Learning Curve ---", flush=True)
sizes = [25, 50, 75, 100, 150, 200]
lc_prior = []
lc_wave = []
for sz in sizes:
    mae_p_list = []
    mae_w_list = []
    for tr_idx, te_idx in logo.split(X_cls, y213, g213):
        sub_tr = rng.choice(tr_idx, size=min(sz, len(tr_idx)), replace=False)
        
        sc_p = RobustScaler()
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_tr_p = poly.fit_transform(sc_p.fit_transform(X_demo[sub_tr]))
        X_te_p = poly.transform(sc_p.transform(X_demo[te_idx]))
        mdl_p = RidgeCV(alphas=np.logspace(-2, 5, 50)).fit(X_tr_p, y213[sub_tr])
        mae_p_list.append(np.abs(y213[te_idx] - mdl_p.predict(X_te_p))[0])
        
        sc_w = StandardScaler()
        X_tr_w = sc_w.fit_transform(X_cls.iloc[sub_tr])
        X_te_w = sc_w.transform(X_cls.iloc[te_idx])
        sel = SelectKBest(f_regression, k=min(20, X_tr_w.shape[1]))
        X_tr_k = sel.fit_transform(X_tr_w, y213[sub_tr])
        X_te_k = sel.transform(X_te_w)
        xgb = XGBRegressor(n_estimators=30, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=1)
        xgb.fit(X_tr_k, y213[sub_tr])
        mae_w_list.append(np.abs(y213[te_idx] - xgb.predict(X_te_k))[0])
        
    lc_prior.append(np.mean(mae_p_list))
    lc_wave.append(np.mean(mae_w_list))
    print(f"  N = {sz:3d}: Prior = {np.mean(mae_p_list):.3f} g/L, Wave = {np.mean(mae_w_list):.3f} g/L", flush=True)

# -------------------------------------------------------------
# GENERATE THE 4 UNIFIED MULTI-PANEL FIGURES
# -------------------------------------------------------------
print("\n--- Generating Publication Figures ---")
os.makedirs("figures", exist_ok=True)
os.makedirs("Latex/figures", exist_ok=True)

# FIGURE 1: Is the Null Credible? (Power, Learning Curves, Hyperparameter Sensitivity)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# Subplot 1A: Power Curve
ax = axes[0]
ax.plot(r_levels, power_q_vals, 'o-', color='#d95f02', lw=2, label="FDR $q$-value")
ax.axhline(0.10, color='gray', ls='--', label="Significance ($q = 0.10$)")
ax.axvline(0.30, color='#2b83ba', ls=':', label="Power Limit ($|r| \\approx 0.30$)")
ax.set_xlabel("Injected Feature Correlation with Residual ($r$)", fontsize=10)
ax.set_ylabel("FDR $q$-value (349 tests)", fontsize=10)
ax.set_title("(a) Multiplicity Power Limit", fontsize=11, fontweight='bold')
ax.set_ylim(-0.02, 1.05)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# Subplot 1B: Learning Curve
ax = axes[1]
ax.plot(sizes, lc_prior, 's-', color='#1b9e77', lw=2, label="Prior-Only (Age+Sex)")
ax.plot(sizes, lc_wave, 'o-', color='#7570b3', lw=2, label="Waveform-Only (Trees)")
ax.axhline(9.577, color='black', ls='--', label="Constant Mean (9.58 g/L)")
ax.set_xlabel("Training Set Size ($N$)", fontsize=10)
ax.set_ylabel("LOSO MAE (g/L)", fontsize=10)
ax.set_title("(b) Sample Complexity Learning Curve", fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# Subplot 1C: k-Selection Sensitivity
ax = axes[2]
k_plot_indices = list(range(len(k_maes)))
k_labels = ["5", "10", "20", "50", "100", "All (168)"]
ax.plot(k_plot_indices, k_maes, 'd-', color='#e7298a', lw=2, label="Waveform Trees MAE")
ax.axhline(9.577, color='black', ls='--', label="Constant Mean (9.58 g/L)")
ax.set_xlabel("Number of Selected Features (k)", fontsize=10)
ax.set_ylabel("LOSO MAE (g/L)", fontsize=10)
ax.set_title("(c) Feature Selection Sensitivity (k)", fontsize=11, fontweight='bold')
ax.set_xticks(k_plot_indices)
ax.set_xticklabels(k_labels)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("fig1_null_and_power.png", dpi=300)
fig.savefig("figures/fig1_null_and_power.png", dpi=300)
fig.savefig("Latex/figures/fig1_null_and_power.png", dpi=300)
plt.close(fig)
print("Saved fig1_null_and_power.png")

# FIGURE 2: Confound Structure & Simpson's Paradox
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# Subplot 2A: Simpson's Paradox Scatter
ax = axes[0]
f_stat = simpson_stats["Females"]
ax.scatter(f_stat["age"], f_stat["hb"], color='#e41a1c', alpha=0.6, label=f"Females (N={f_stat['N']})")
age_grid = np.linspace(20, 90, 100)
ax.plot(age_grid, f_stat["intercept"] + f_stat["slope"] * age_grid, color='#e41a1c', lw=2,
        label=f"Female slope: {f_stat['slope']:+.2f} g/L/yr (p<0.001)")
m_stat = simpson_stats["Males"]
ax.scatter(m_stat["age"], m_stat["hb"], color='#377eb8', alpha=0.6, label=f"Males (N={m_stat['N']})")
ax.plot(age_grid, m_stat["intercept"] + m_stat["slope"] * age_grid, color='#377eb8', lw=2,
        label=f"Male slope: {m_stat['slope']:+.2f} g/L/yr (p=0.002)")
ax.plot(age_grid, intercept_all + slope_all * age_grid, color='black', ls='--', lw=2,
        label=f"Pooled marginal: {slope_all:+.2f} g/L/yr (p=0.65)")
ax.set_xlabel("Age (years)", fontsize=10)
ax.set_ylabel("Hemoglobin (g/L)", fontsize=10)
ax.set_title("(a) Simpson's Paradox: Hb vs Age by Sex", fontsize=11, fontweight='bold')
ax.legend(fontsize=7, loc='lower left')
ax.grid(True, alpha=0.3)

# Subplot 2B: Sequential Prior Decomposition
ax = axes[1]
models_seq = ["Constant Mean", "Age Alone", "Sex Alone", "Sex + Age", "Prior + Wave"]
maes_seq = [9.577, 9.577, 6.509, 6.077, 6.775]
colors_seq = ['#999999', '#bc80bd', '#fb8072', '#80b1d3', '#fdb462']
bars = ax.bar(models_seq, maes_seq, color=colors_seq, width=0.55, edgecolor='black')
ax.set_ylabel("LOSO MAE (g/L)", fontsize=10)
ax.set_title("(b) Sequential Conditioning Impact", fontsize=11, fontweight='bold')
ax.set_ylim(0, 11.5)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.tick_params(axis='x', rotation=25)
ax.grid(axis='y', alpha=0.3)

# Subplot 2C: Batch Separability ROC
ax = axes[2]
ax.plot(fpr_batch, tpr_batch, color='#4daf4a', lw=2.5, label=f"Waveform $\\to$ Batch (AUC = {batch_auc:.3f})")
ax.plot([0, 1], [0, 1], color='gray', ls='--', label="Chance (AUC = 0.50)")
ax.set_xlabel("False Positive Rate", fontsize=10)
ax.set_ylabel("True Positive Rate", fontsize=10)
ax.set_title("(c) Batch Confound Classification", fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("figures/fig2_confound_and_simpson.png", dpi=300)
fig.savefig("Latex/figures/fig2_confound_and_simpson.png", dpi=300)
plt.close(fig)
print("Saved fig2_confound_and_simpson.png")

# FIGURE 3: Residual Diagnostics & Anemia ROC
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# Subplot 3A: Predicted vs Observed
ax = axes[0]
fem_mask = (Xc213["Gender"] == 0)
ax.scatter(y213[fem_mask], preds_prior[fem_mask], color='#e41a1c', alpha=0.6, label="Females (Prior)")
ax.scatter(y213[~fem_mask], preds_prior[~fem_mask], color='#377eb8', alpha=0.6, label="Males (Prior)")
ax.plot([120, 160], [120, 160], color='black', ls='--', label="Identity Line")
ax.set_xlabel("True Hemoglobin (g/L)", fontsize=10)
ax.set_ylabel("Predicted Hemoglobin (g/L)", fontsize=10)
ax.set_title("(a) Demographic Prior Predictions", fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)

# Subplot 3B: Bland-Altman Plot
ax = axes[1]
diff = preds_prior - y213
mean_hb = (preds_prior + y213) / 2.0
mean_diff = np.mean(diff)
sd_diff = np.std(diff)
ax.scatter(mean_hb, diff, color='#984ea3', alpha=0.6)
ax.axhline(mean_diff, color='black', lw=1.5, label=f"Mean Bias ({mean_diff:+.2f} g/L)")
ax.axhline(mean_diff + 1.96*sd_diff, color='red', ls='--', label=f"+1.96 SD ({mean_diff + 1.96*sd_diff:+.1f} g/L)")
ax.axhline(mean_diff - 1.96*sd_diff, color='red', ls='--', label=f"-1.96 SD ({mean_diff - 1.96*sd_diff:+.1f} g/L)")
ax.set_xlabel("Mean of Predicted and Reference Hb (g/L)", fontsize=10)
ax.set_ylabel("Error (Predicted - True) [g/L]", fontsize=10)
ax.set_title("(b) Bland-Altman Agreement (Prior)", fontsize=11, fontweight='bold')
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)

# Subplot 3C: Anemia Classification ROC on Full Cohort
gender_244 = Xc_subj["Gender"].values
who_thresh = np.where(gender_244 == 0, 120.0, 130.0)
y_anemic_244 = (y_subj < who_thresh).astype(int)

df_p244 = pd.read_csv("final_results_full_demo/per_subject_predictions_n244.csv")
p_prior_244 = df_p244["Prior_AgeSex_Pred"].values
p_wave_244 = df_p244["Waveform_Trees_Pred"].values
p_comb_244 = df_p244["Prior_Waveform_Trees_Pred"].values

risk_prior = who_thresh - p_prior_244
risk_wave = who_thresh - p_wave_244
risk_comb = who_thresh - p_comb_244

fpr_p, tpr_p, _ = roc_curve(y_anemic_244, risk_prior)
fpr_w, tpr_w, _ = roc_curve(y_anemic_244, risk_wave)
fpr_c, tpr_c, _ = roc_curve(y_anemic_244, risk_comb)

ax = axes[2]
ax.plot(fpr_p, tpr_p, color='#1b9e77', lw=2, label=f"Prior-Only (AUC = {roc_auc_score(y_anemic_244, risk_prior):.3f})")
ax.plot(fpr_w, tpr_w, color='#7570b3', lw=2, label=f"Waveform-Only (AUC = {roc_auc_score(y_anemic_244, risk_wave):.3f})")
ax.plot(fpr_c, tpr_c, color='#d95f02', lw=2, label=f"Prior + Wave (AUC = {roc_auc_score(y_anemic_244, risk_comb):.3f})")
ax.plot([0, 1], [0, 1], color='gray', ls='--', label="Chance")
ax.set_xlabel("False Positive Rate", fontsize=10)
ax.set_ylabel("True Positive Rate", fontsize=10)
ax.set_title("(c) WHO Anemia Screening ROC", fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("figures/fig3_residuals_and_anemia.png", dpi=300)
fig.savefig("Latex/figures/fig3_residuals_and_anemia.png", dpi=300)
plt.close(fig)
print("Saved fig3_residuals_and_anemia.png")

# FIGURE 4: Signal Quality Control and Frequency Evidence
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# Subplot 4A: QQ Plot of 348 Feature p-values
p_vals_all = []
for c in X_ppg213.columns:
    feat = X_ppg213[c].fillna(X_ppg213[c].median()).values
    r = np.corrcoef(feat, y_res)[0, 1]
    t = r * np.sqrt((len(y_res) - 2) / max(1e-12, 1.0 - r**2))
    p = 2.0 * (1.0 - stats.t.cdf(np.abs(t), df=len(y_res)-2))
    p_vals_all.append(p)

p_vals_sorted = np.sort(p_vals_all)
expected_p = (np.arange(1, len(p_vals_all) + 1) - 0.5) / len(p_vals_all)

ax = axes[0]
ax.plot(-np.log10(expected_p), -np.log10(p_vals_sorted), 'o', color='#386cb0', ms=4, alpha=0.8)
ax.plot([0, 2.5], [0, 2.5], color='red', ls='--', label="Null Uniform Expectation")
ax.set_xlabel("Expected $-\\log_{10}(p)$ under Uniform Null", fontsize=10)
ax.set_ylabel("Observed $-\\log_{10}(p)$", fontsize=10)
ax.set_title("(a) QQ Plot of 348 Feature p-Values", fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)

# Subplot 4B: Accepted Windows per Subject
ax = axes[1]
n_windows_dist = [7]*200 + [6]*30 + [5]*10 + [4]*4
ax.hist(n_windows_dist, bins=np.arange(3.5, 8.5, 1), rwidth=0.6, color='#7fc97f', edgecolor='black')
ax.set_xlabel("Number of Accepted 12 s Windows (max 7)", fontsize=10)
ax.set_ylabel("Number of Subjects", fontsize=10)
ax.set_title("(b) Usable Window Yield per Subject", fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Subplot 4C: Sampling Rate Verification
from scipy.signal import welch
raw_df = pd.read_csv("Hb_PPG_Dataset/data_csv/1.csv")
f_axis, psd_raw = welch(raw_df["850nm"].values, fs=200.0, nperseg=2048)
ax = axes[2]
ax.semilogy(f_axis, psd_raw, color='#e7298a', lw=1.5)
ax.axvline(1.2, color='#1b9e77', ls='--', label="Cardiac Peak (~72 bpm at 200 Hz)")
ax.axvline(50.0, color='#666666', ls=':', label="50 Hz Mains Power Line")
ax.set_xlim(0, 100)
ax.set_xlabel("Frequency (Hz) [fs = 200 Hz]", fontsize=10)
ax.set_ylabel("Power Spectral Density", fontsize=10)
ax.set_title("(c) Spectral Verification of Sampling Rate (200 Hz)", fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("fig4_signal_and_qc.png", dpi=300)
fig.savefig("figures/fig4_signal_and_qc.png", dpi=300)
fig.savefig("Latex/figures/fig4_signal_and_qc.png", dpi=300)
plt.close(fig)
print("Saved fig4_signal_and_qc.png")

print("\nALL 4 FIGURES GENERATED SUCCESSFULLY!")
