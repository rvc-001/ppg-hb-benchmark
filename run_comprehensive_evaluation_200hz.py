"""
run_comprehensive_evaluation_200hz.py
===============================================================================
Comprehensive, rigorous evaluation incorporating user feedback:
1. Physical Sampling Rate fs = 200 Hz (Chen et al., Electronics 2023, 12, 1346).
2. The 77-subject elderly cohort audit (HW missing vs present) and pseudo-external test.
3. Full comparison table with missing rows: Constant Mean, Prior-Only, and Prior+Trees.
4. 10 random seeds across tree ensembles with paired bootstrap 95% CIs.
5. Pre-specified minimal quantum test (8 fixed features: S and purity per channel in Ridge/ElasticNet).
6. Zoomed waveform & autocorrelation diagnostics for Sub 119, Sub 140, Sub 112, Sub 1.
"""

import os, sys, warnings, time
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch, hilbert
from scipy.stats import pearsonr, skew, kurtosis, ttest_ind
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.preprocessing import RobustScaler, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV, ElasticNetCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import bp_quantum_features as bqf

warnings.filterwarnings('ignore')

BASE_DIR = os.environ.get("PPG_DATA_DIR", "Hb_PPG_Dataset")
MAT_DIR = os.path.join(BASE_DIR, "data_mat")
EXCEL_PATH = os.path.join(BASE_DIR, "subject information.xlsx")
CACHE_FILE = "features_cache_200hz.pkl"
RESULTS_DIR = os.path.join("final_results_full_demo")
PLOTS_DIR = os.path.join("final_plots_full_demo")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

FS = 200  # Physical sampling rate
WINDOW_SEC = 12
OVERLAP = 0.40
WINDOW_LEN = FS * WINDOW_SEC
CH_NAMES = ["660nm", "730nm", "850nm", "940nm"]
TARGET_COL, ID_COL = "Hemoglobin (g/L)", "ID"
HB_MIN, HB_MAX = 120.0, 160.0

def calculate_mard(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / y_true) * 100.0

def bootstrap_ci(diff_errors, n_boot=2000, ci=95, seed=42):
    rng = np.random.RandomState(seed)
    boot_means = [np.mean(rng.choice(diff_errors, size=len(diff_errors), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boot_means, (100 - ci) / 2.0)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2.0)
    return float(np.mean(diff_errors)), float(lo), float(hi)

# ─── PART 1: FEATURE EXTRACTION AT FS = 200 HZ ───────────────────────────────
def extract_all_features_200hz():
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached features from {CACHE_FILE}...", flush=True)
        return pd.read_pickle(CACHE_FILE)

    print("Extracting features at true physical sampling rate FS = 200 Hz...", flush=True)
    df_info = pd.read_excel(EXCEL_PATH)

    df_info["Gender"] = (df_info["Gender"].astype(str).str.lower().str.strip()
                         .map({"m": 1, "male": 1, "1": 1, "f": 0, "female": 0, "0": 0}).fillna(0))
    w = pd.to_numeric(df_info["Weight (kg)"], errors="coerce")
    h = pd.to_numeric(df_info["Height (cm)"], errors="coerce") / 100.0
    df_info["BMI (kg/m2)"] = w / (h ** 2 + 1e-9)
    df_info["is_missing_hw"] = w.isna().astype(int)

    cov_cols = ["Age (year)", "Gender", "Weight (kg)", "Height (cm)", "BMI (kg/m2)", "is_missing_hw"]
    for c in cov_cols + [TARGET_COL]:
        if c in df_info.columns:
            df_info[c] = pd.to_numeric(df_info[c], errors="coerce")

    feat_rows, cov_rows, targets, subjects = [], [], [], []

    for idx, row in df_info.iterrows():
        sid = int(row[ID_COL])
        mat_path = os.path.join(MAT_DIR, f"{sid}.mat")
        if not os.path.exists(mat_path):
            continue
        try:
            mat = sio.loadmat(mat_path)
            st = mat["PPGdata"][0, 0]
            raw = np.vstack([st[i].flatten().astype(np.float32) for i in range(len(st))])
        except Exception:
            continue

        if raw.shape[-1] < WINDOW_LEN:
            continue

        filt = bqf.butter_bandpass(raw, fs=FS, order=4)
        step = int(FS * WINDOW_SEC * (1 - OVERLAP))
        raw_wins = [raw[:, s:s + WINDOW_LEN].T for s in range(0, raw.shape[-1] - WINDOW_LEN + 1, step)]
        filt_wins = [filt[:, s:s + WINDOW_LEN].T for s in range(0, filt.shape[-1] - WINDOW_LEN + 1, step)]

        hb = float(row[TARGET_COL])
        n_total = len(raw_wins)
        usable_wins = []

        for r_w, f_w in zip(raw_wins, filt_wins):
            clean_ch0 = bqf.wavelet_denoise(f_w[:, 0])
            if np.ptp(clean_ch0) < 1e-4:
                continue

            p_test = bqf.find_systolic_peaks(clean_ch0, fs=FS)
            # Normal resting HR 40-150 bpm over 12s -> 8 to 30 beats
            if not (5 <= len(p_test) <= 35):
                continue

            hr_ac = bqf.hr_autocorr(clean_ch0, fs=FS)
            if hr_ac is not None:
                hr_pk = 60.0 / (np.mean(np.diff(p_test)) / FS)
                if abs(hr_pk - hr_ac) / (hr_ac + 1e-9) > 0.08:
                    continue  # Gate rejection

            clean_f = np.zeros_like(f_w)
            clean_f[:, 0] = clean_ch0
            for c in range(1, 4):
                clean_f[:, c] = bqf.wavelet_denoise(f_w[:, c])
            usable_wins.append((r_w, clean_f))

        n_kept = len(usable_wins)
        frac_rejected = float((n_total - n_kept) / (n_total + 1e-9))

        base_cov = {c: row.get(c, np.nan) for c in cov_cols}
        base_cov["n_windows_kept"] = float(n_kept)
        base_cov["frac_windows_rejected"] = frac_rejected

        for r_w, clean_f in usable_wins:
            d = {}
            filt_chs = [clean_f[:, c] for c in range(4)]
            raw_chs = [r_w[:, c] for c in range(4)]
            ac_dc = {}

            for c, nm in enumerate(CH_NAMES):
                raw_c, filt_c = raw_chs[c], filt_chs[c]
                dc = float(np.mean(raw_c)) + 1e-9
                ac = float(np.std(filt_c))
                ac_dc[nm] = ac / dc
                d[f"{nm}_dc"] = dc
                d[f"{nm}_ac_amp"] = ac
                d[f"{nm}_acdc"] = ac / dc
                d[f"{nm}_skew"] = float(skew(filt_c))
                d[f"{nm}_kurt"] = float(kurtosis(filt_c))
                vpg = np.gradient(filt_c)
                apg = np.gradient(vpg)
                d[f"{nm}_vpg_skew"] = float(skew(vpg))
                d[f"{nm}_apg_kurt"] = float(kurtosis(apg))

                f_psd, psd = welch(filt_c, fs=FS, nperseg=min(512, len(filt_c)))
                tp = np.sum(psd) + 1e-9
                nrm = psd / tp
                d[f"{nm}_spectral_entropy"] = float(-np.sum(nrm[nrm > 0] * np.log2(nrm[nrm > 0])))
                lf = np.trapezoid(psd[(f_psd >= 0.5) & (f_psd < 2.0)], f_psd[(f_psd >= 0.5) & (f_psd < 2.0)]) + 1e-12
                hf = np.trapezoid(psd[(f_psd >= 2.0) & (f_psd < 4.0)], f_psd[(f_psd >= 2.0) & (f_psd < 4.0)]) + 1e-12
                d[f"{nm}_hflf_ratio"] = float(hf / lf)

                analytic = hilbert(filt_c)
                env = np.abs(analytic)
                d[f"{nm}_env_mean"] = float(np.mean(env))
                d[f"{nm}_env_std"] = float(np.std(env))

            for i in range(4):
                for j in range(i + 1, 4):
                    ni, nj = CH_NAMES[i], CH_NAMES[j]
                    c_corr, _ = pearsonr(filt_chs[i], filt_chs[j])
                    d[f"corr_{ni}_{nj}"] = float(c_corr)
                    d[f"acdc_ratio_{ni}_{nj}"] = float(ac_dc[ni] / (ac_dc[nj] + 1e-9))

            q_multi = bqf.extract_multichannel_quantum_descriptors(
                clean_f.T, fs=FS, ch_names=CH_NAMES, method="peak_to_peak", is_prefiltered=True
            )
            if q_multi is not None:
                d.update(q_multi)
            else:
                for nm in CH_NAMES:
                    d[f"{nm}_q_von_neumann_entropy"] = 0.0
                    d[f"{nm}_q_purity"] = 0.0
                    d[f"{nm}_q_effective_rank"] = 0.0
                    d[f"{nm}_q_participation_ratio"] = 0.0
                    d[f"{nm}_q_self_fidelity"] = 0.0
                    d[f"{nm}_q_self_trace_distance"] = 0.0
                    d[f"{nm}_q_n_pulses"] = 0.0
                    for e in range(1, 6):
                        d[f"{nm}_q_eig_{e}"] = 0.0
                for i in range(4):
                    for j in range(i + 1, 4):
                        d[f"cross_q_fid_{CH_NAMES[i]}_{CH_NAMES[j]}"] = 0.0
                        d[f"cross_q_td_{CH_NAMES[i]}_{CH_NAMES[j]}"] = 0.0

            feat_rows.append(d)
            cov_rows.append(base_cov)
            targets.append(hb)
            subjects.append(sid)

    X_raw = pd.DataFrame(feat_rows)
    Xc_raw = pd.DataFrame(cov_rows)
    y_raw = np.array(targets, dtype=np.float32)
    g_raw = np.array(subjects, dtype=np.int32)

    df_all = pd.concat([X_raw, Xc_raw], axis=1)
    df_all["TARGET"] = y_raw
    df_all["GROUP"] = g_raw

    grp = df_all.groupby("GROUP")
    ppg_cols = X_raw.columns.tolist()
    cov_cols_final = [c for c in cov_cols if c in Xc_raw.columns] + ["n_windows_kept", "frac_windows_rejected"]

    med_df = grp[ppg_cols].median()
    std_df = grp[ppg_cols].std().fillna(0.0)
    iqr_df = grp[ppg_cols].quantile(0.75) - grp[ppg_cols].quantile(0.25)

    med_df.columns = [f"{c}_med" for c in ppg_cols]
    std_df.columns = [f"{c}_std" for c in ppg_cols]
    iqr_df.columns = [f"{c}_iqr" for c in ppg_cols]

    std_df = std_df.loc[:, std_df.std() > 1e-6]
    iqr_df = iqr_df.loc[:, iqr_df.std() > 1e-6]

    X_subj = pd.concat([med_df, std_df, iqr_df], axis=1).reset_index(drop=True)
    Xc_subj = grp[cov_cols_final].median().reset_index(drop=True)
    y_subj = grp["TARGET"].median().values
    g_subj = grp["TARGET"].median().index.values

    print(f"Extracted {len(y_subj)} subjects, {len(X_subj.columns)} window-derived PPG features at FS = 200 Hz.", flush=True)
    pd.to_pickle((X_subj, Xc_subj, y_subj, g_subj), CACHE_FILE)
    return X_subj, Xc_subj, y_subj, g_subj

# ─── PART 2: 77-SUBJECT BATCH AUDIT & PSEUDO-EXTERNAL VALIDATION ──────────────
def audit_77_subject_batch(Xc_subj, y_subj, X_subj):
    print("\n" + "="*80)
    print("PART 2: 77-SUBJECT ELDERLY BATCH AUDIT (HW MISSING vs PRESENT)")
    print("="*80)
    is_miss = Xc_subj["is_missing_hw"].values.astype(bool)
    n_miss = int(np.sum(is_miss))
    n_pres = int(np.sum(~is_miss))
    print(f"Cohort Split: Present HW = {n_pres} subjects, Missing HW = {n_miss} subjects.")

    # Comparisons
    age_pres, age_miss = Xc_subj.loc[~is_miss, "Age (year)"], Xc_subj.loc[is_miss, "Age (year)"]
    hb_pres, hb_miss = y_subj[~is_miss], y_subj[is_miss]
    
    # Pulse count / HR from 850nm channel
    hr_pres = (60.0 / 12.0) * X_subj.loc[~is_miss, "850nm_q_n_pulses_med"]
    hr_miss = (60.0 / 12.0) * X_subj.loc[is_miss, "850nm_q_n_pulses_med"]
    
    # Amplitude from 850nm channel
    amp_pres = X_subj.loc[~is_miss, "850nm_ac_amp_med"]
    amp_miss = X_subj.loc[is_miss, "850nm_ac_amp_med"]

    t_age, p_age = ttest_ind(age_pres, age_miss, nan_policy='omit')
    t_hb, p_hb = ttest_ind(hb_pres, hb_miss, nan_policy='omit')
    t_hr, p_hr = ttest_ind(hr_pres, hr_miss, nan_policy='omit')
    t_amp, p_amp = ttest_ind(amp_pres, amp_miss, nan_policy='omit')

    print(f"Age: Present = {np.mean(age_pres):.1f} ± {np.std(age_pres):.1f} yr | Missing = {np.mean(age_miss):.1f} ± {np.std(age_miss):.1f} yr (t={t_age:.2f}, p={p_age:.2e})")
    print(f"Hb:  Present = {np.mean(hb_pres):.1f} ± {np.std(hb_pres):.1f} g/L | Missing = {np.mean(hb_miss):.1f} ± {np.std(hb_miss):.1f} g/L (t={t_hb:.2f}, p={p_hb:.4f})")
    print(f"HR:  Present = {np.mean(hr_pres):.1f} ± {np.std(hr_pres):.1f} bpm | Missing = {np.mean(hr_miss):.1f} ± {np.std(hr_miss):.1f} bpm (t={t_hr:.2f}, p={p_hr:.4f})")
    print(f"Amp: Present = {np.mean(amp_pres):.1f} ± {np.std(amp_pres):.1f} | Missing = {np.mean(amp_miss):.1f} ± {np.std(amp_miss):.1f} (t={t_amp:.2f}, p={p_amp:.2e})")

    # Pseudo-external testing using Age + Sex prior
    print("\nPseudo-External Testing (Prior on Age + Sex only):")
    X_demo = Xc_subj[["Age (year)", "Gender"]].values
    sc = RobustScaler()
    poly = PolynomialFeatures(degree=2, include_bias=False)
    
    # Train on Present (Younger), Test on Missing (Elderly)
    X_tr, y_tr = poly.fit_transform(sc.fit_transform(X_demo[~is_miss])), y_subj[~is_miss]
    X_te, y_te = poly.transform(sc.transform(X_demo[is_miss])), y_subj[is_miss]
    mdl1 = RidgeCV(alphas=np.logspace(0, 4, 100)).fit(X_tr, y_tr)
    p_te1 = mdl1.predict(X_te)
    mae1 = mean_absolute_error(y_te, p_te1)
    r2_1 = r2_score(y_te, p_te1)
    print(f"  Train Younger (N={n_pres}) -> Test Elderly (N={n_miss}): MAE = {mae1:.2f} g/L, R2 = {r2_1:.3f}")

    # Train on Missing (Elderly), Test on Present (Younger)
    X_tr2, y_tr2 = poly.fit_transform(sc.fit_transform(X_demo[is_miss])), y_subj[is_miss]
    X_te2, y_te2 = poly.transform(sc.transform(X_demo[~is_miss])), y_subj[~is_miss]
    mdl2 = RidgeCV(alphas=np.logspace(0, 4, 100)).fit(X_tr2, y_tr2)
    p_te2 = mdl2.predict(X_te2)
    mae2 = mean_absolute_error(y_te2, p_te2)
    r2_2 = r2_score(y_te2, p_te2)
    print(f"  Train Elderly (N={n_miss}) -> Test Younger (N={n_pres}): MAE = {mae2:.2f} g/L, R2 = {r2_2:.3f}")

# ─── PART 3: ZOOMED DIAGNOSTICS FOR SUB 119, 140, 112, 1 ─────────────────────
def plot_zoomed_diagnostics():
    print("\n" + "="*80)
    print("PART 3: GENERATING ZOOMED DIAGNOSTIC PLOTS (SUB 119, 140, 112, 1)")
    print("="*80)
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    subjects = [1, 112, 119, 140]

    for row_idx, sid in enumerate(subjects):
        mat_path = os.path.join(MAT_DIR, f"{sid}.mat")
        mat = sio.loadmat(mat_path)
        st = mat["PPGdata"][0, 0]
        raw = st[0].flatten().astype(np.float32)
        filt = bqf.butter_bandpass(raw, fs=FS, order=4)
        
        # 12s window at 200 Hz
        sig_win = filt[:2400]
        clean = bqf.wavelet_denoise(sig_win)
        peaks = bqf.find_systolic_peaks(clean, fs=FS)
        t_sec = np.arange(len(clean)) / FS

        # Waveform & Peaks
        ax_w = axes[row_idx, 0]
        ax_w.plot(t_sec, clean, color='#1f77b4', lw=1.5, label='Filtered PPG (660nm)')
        if len(peaks) > 0:
            ax_w.plot(peaks / FS, clean[peaks], 'ro', markersize=6, label=f'Systolic Peaks (N={len(peaks)})')
        hr_pk = 60.0 / (np.mean(np.diff(peaks)) / FS) if len(peaks) > 1 else 0
        ax_w.set_title(f"Subject {sid}: Waveform Trace (Peak HR = {hr_pk:.1f} bpm)", fontsize=13, fontweight='bold')
        ax_w.set_xlabel("Time (s)", fontsize=11)
        ax_w.set_ylabel("Amplitude", fontsize=11)
        ax_w.legend(loc='upper right', fontsize=10)

        # Autocorrelation
        ax_a = axes[row_idx, 1]
        x = clean - np.mean(clean)
        ac = np.correlate(x, x, "full")[len(x)-1:]
        ac /= ac[0]
        lo, hi = int(0.4 * FS), int(1.5 * FS)
        lags_sec = np.arange(lo, hi) / FS
        ax_a.plot(lags_sec, ac[lo:hi], color='#2ca02c', lw=1.8, label='Normalized Autocorrelation')
        
        pks_ac, props = find_peaks(ac[lo:hi], height=0.3)
        ac_hr = bqf.hr_autocorr(clean, fs=FS)
        if len(pks_ac) > 0:
            ax_a.plot((lo + pks_ac) / FS, ac[lo + pks_ac], 'm^', markersize=8, label='AC Candidates (>=0.30)')
        
        status = "DISCORDANT"
        if ac_hr is not None:
            disc = abs(hr_pk - ac_hr) / ac_hr
            status = f"ACCEPTED (Δ={disc*100:.1f}%)" if disc <= 0.08 else f"REJECTED (Δ={disc*100:.1f}%)"
        else:
            status = "UNVERIFIED"

        ax_a.set_title(f"Subject {sid}: Autocorr (Ac-HR = {ac_hr if ac_hr is not None else 'None'}) -> [{status}]",
                       fontsize=13, fontweight='bold')
        ax_a.set_xlabel("Lag (s)", fontsize=11)
        ax_a.set_ylabel("Autocorrelation", fontsize=11)
        ax_a.axhline(0.3, color='gray', ls='--', alpha=0.7, label='Height Threshold (0.30)')
        ax_a.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "zoomed_qc_diagnostics_200hz.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved zoomed diagnostics to {out_path}", flush=True)

# ─── PART 4: LOSO-CV EVALUATION WITH MISSING ROWS & 10 SEEDS ──────────────────
def run_full_loso_evaluation(X_subj, Xc_subj, y_subj, g_subj):
    print("\n" + "="*80)
    print("PART 4: DEFINITIVE BASELINE COMPARISON (10 SEEDS & PAIRED BOOTSTRAP CIS)")
    print("="*80)

    keep = (y_subj >= HB_MIN) & (y_subj <= HB_MAX)
    X_ppg = X_subj[keep].reset_index(drop=True)
    Xc = Xc_subj[keep].reset_index(drop=True)
    y = y_subj[keep]
    g = g_subj[keep]
    n_sub = len(y)
    print(f"Cohort in clinical operating range (120-160 g/L): N = {n_sub} subjects.")

    # 1. Constant Mean Predictor
    preds_const = np.zeros(n_sub)
    for i in range(n_sub):
        preds_const[i] = np.mean(np.delete(y, i))
    mae_const = mean_absolute_error(y, preds_const)
    rmse_const = np.sqrt(mean_squared_error(y, preds_const))
    r2_const = r2_score(y, preds_const)
    mard_const = calculate_mard(y, preds_const)

    # 2. Prior-Only Models (LOSO)
    cov_cols_all = ["Age (year)", "Gender", "Weight (kg)", "Height (cm)", "BMI (kg/m2)"]
    cov_cols_demo = ["Age (year)", "Gender"]
    Xc_all_np = Xc[cov_cols_all].values.astype(np.float64)
    Xc_demo_np = Xc[cov_cols_demo].values.astype(np.float64)

    logo = LeaveOneGroupOut()

    # Prior (Age + Sex only)
    preds_prior_demo = np.zeros(n_sub)
    for tr_idx, te_idx in logo.split(Xc_demo_np, y, g):
        sc = RobustScaler()
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_tr = poly.fit_transform(sc.fit_transform(Xc_demo_np[tr_idx]))
        X_te = poly.transform(sc.transform(Xc_demo_np[te_idx]))
        mdl = RidgeCV(alphas=np.logspace(0, 4, 100)).fit(X_tr, y[tr_idx])
        preds_prior_demo[te_idx] = mdl.predict(X_te)

    # Prior (All Covariates with In-Fold Median Imputation)
    preds_prior_all = np.zeros(n_sub)
    for tr_idx, te_idx in logo.split(Xc_all_np, y, g):
        X_tr = Xc_all_np[tr_idx]
        X_te = Xc_all_np[te_idx]
        tr_med = np.nanmedian(X_tr, axis=0)
        X_tr = np.where(np.isnan(X_tr), tr_med, X_tr)
        X_te = np.where(np.isnan(X_te), tr_med, X_te)

        sc = RobustScaler()
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_tr_p = poly.fit_transform(sc.fit_transform(X_tr))
        X_te_p = poly.transform(sc.transform(X_te))
        mdl = RidgeCV(alphas=np.logspace(0, 4, 100)).fit(X_tr_p, y[tr_idx])
        preds_prior_all[te_idx] = mdl.predict(X_te_p)

    errors_prior_all = np.abs(y - preds_prior_all)

    # Feature partitions
    q_cols = [c for c in X_ppg.columns if "_q_" in c or "cross_q_" in c]
    npulse_cols = [c for c in q_cols if "n_pulses" in c]
    pure_q_cols = [c for c in q_cols if "n_pulses" not in c]
    cls_cols = [c for c in X_ppg.columns if c not in q_cols]

    X_cls = X_ppg[cls_cols].values.astype(np.float64)
    X_cls_npulse = X_ppg[cls_cols + npulse_cols].values.astype(np.float64)
    X_full_q = X_ppg[cls_cols + npulse_cols + pure_q_cols].values.astype(np.float64)

    feature_configs = {
        "trees_cls": X_cls,
        "trees_cls+npulses": X_cls_npulse,
        "trees_q": X_full_q
    }

    seeds = [42 + s for s in range(10)]
    all_summary = []

    # Add Constant Mean row
    all_summary.append({
        "Model": "Constant Mean (LOSO)",
        "Features": "None",
        "Mode": "Baseline",
        "MAE (g/L)": f"{mae_const:.3f}",
        "RMSE (g/L)": f"{rmse_const:.3f}",
        "R2 Score": f"{r2_const:.4f}",
        "MARD (%)": f"{mard_const:.2f}%",
        "ΔMAE vs Prior (95% CI)": "N/A"
    })

    # Add Prior-Only (Age+Sex) row
    mae_pd = mean_absolute_error(y, preds_prior_demo)
    rmse_pd = np.sqrt(mean_squared_error(y, preds_prior_demo))
    r2_pd = r2_score(y, preds_prior_demo)
    mard_pd = calculate_mard(y, preds_prior_demo)
    d_mean, d_lo, d_hi = bootstrap_ci(np.abs(y - preds_prior_demo) - errors_prior_all)
    all_summary.append({
        "Model": "Prior-Only (Age + Sex)",
        "Features": "Age, Sex (deg-2 poly)",
        "Mode": "Prior Baseline",
        "MAE (g/L)": f"{mae_pd:.3f}",
        "RMSE (g/L)": f"{rmse_pd:.3f}",
        "R2 Score": f"{r2_pd:.4f}",
        "MARD (%)": f"{mard_pd:.2f}%",
        "ΔMAE vs Prior (95% CI)": f"{d_mean:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}]"
    })

    # Add Prior-Only (All Covariates) row
    mae_pa = mean_absolute_error(y, preds_prior_all)
    rmse_pa = np.sqrt(mean_squared_error(y, preds_prior_all))
    r2_pa = r2_score(y, preds_prior_all)
    mard_pa = calculate_mard(y, preds_prior_all)
    all_summary.append({
        "Model": "Prior-Only (All Covariates)",
        "Features": "Age, Sex, W, H, BMI (deg-2 poly)",
        "Mode": "Prior Baseline",
        "MAE (g/L)": f"{mae_pa:.3f}",
        "RMSE (g/L)": f"{rmse_pa:.3f}",
        "R2 Score": f"{r2_pa:.4f}",
        "MARD (%)": f"{mard_pa:.2f}%",
        "ΔMAE vs Prior (95% CI)": "Reference [0.000]"
    })

    # Run Tree Ensembles across 10 Seeds (optimized with precomputed fold selections)
    print("\nRunning Tree Ensembles across 10 Seeds (42-51)...", flush=True)
    
    # Precompute Stage-1 Prior and Residual targets ONCE across all folds
    print("  Precomputing Stage-1 Prior (RidgeCV) across all 220 folds...", flush=True)
    cov_oof_all = np.zeros(n_sub)
    cov_pred_all = np.zeros(n_sub)
    y_res_all_tr = []

    for tr_idx, te_idx in logo.split(Xc_all_np, y, g):
        Xc_tr, Xc_te = Xc_all_np[tr_idx], Xc_all_np[te_idx]
        tr_cov_med = np.nanmedian(Xc_tr, axis=0)
        Xc_tr = np.where(np.isnan(Xc_tr), tr_cov_med, Xc_tr)
        Xc_te = np.where(np.isnan(Xc_te), tr_cov_med, Xc_te)

        sc_c = RobustScaler()
        poly = PolynomialFeatures(degree=2, include_bias=False)
        Xc_tr_p = poly.fit_transform(sc_c.fit_transform(Xc_tr))
        Xc_te_p = poly.transform(sc_c.transform(Xc_te))

        cov_mdl = RidgeCV(alphas=np.logspace(0, 4, 100)).fit(Xc_tr_p, y[tr_idx])
        cov_pred_all[te_idx] = cov_mdl.predict(Xc_te_p)
        oof_tr = cross_val_predict(cov_mdl, Xc_tr_p, y[tr_idx], cv=5)
        y_res_all_tr.append(y[tr_idx] - oof_tr)

    for name, X_feat in feature_configs.items():
        print(f"\n  Evaluating Configuration [{name}] (D = {X_feat.shape[1]}) across 10 seeds...", flush=True)
        
        # Precompute feature selection once per fold
        print(f"    Precomputing fold feature selection (Top 50 via MI)...", flush=True)
        fold_data = []
        for f_idx, (tr_idx, te_idx) in enumerate(logo.split(X_feat, y, g)):
            X_tr, X_te = X_feat[tr_idx], X_feat[te_idx]
            tr_med = np.nanmedian(X_tr, axis=0)
            X_tr = np.where(np.isnan(X_tr), tr_med, X_tr)
            X_te = np.where(np.isnan(X_te), tr_med, X_te)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            k_sel = min(50, X_tr_s.shape[1])
            selector = SelectKBest(mutual_info_regression, k=k_sel)
            X_tr_sel = selector.fit_transform(X_tr_s, y[tr_idx])
            X_te_sel = selector.transform(X_te_s)
            fold_data.append((X_tr_sel, X_te_sel, tr_idx, te_idx, y[tr_idx], y_res_all_tr[f_idx]))

        seed_maes_w, seed_rmses_w, seed_r2s_w, seed_mards_w = [], [], [], []
        seed_maes_cp, seed_rmses_cp, seed_r2s_cp, seed_mards_cp = [], [], [], []
        all_diff_errors_cp = []

        for s_idx, seed in enumerate(seeds):
            preds_wave = np.zeros(n_sub)
            preds_cov_trees = np.zeros(n_sub)

            for f_idx, (X_tr_sel, X_te_sel, tr_idx, te_idx, y_tr, y_res_tr) in enumerate(fold_data):
                # Waveform trees
                xgb_w = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.04,
                                     subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
                cat_w = CatBoostRegressor(iterations=200, depth=4, learning_rate=0.04,
                                          l2_leaf_reg=3, random_seed=seed, thread_count=-1, verbose=0)
                xgb_w.fit(X_tr_sel, y_tr)
                cat_w.fit(X_tr_sel, y_tr)
                preds_wave[te_idx] = 0.5 * (xgb_w.predict(X_te_sel) + cat_w.predict(X_te_sel))

                # Residual trees
                xgb_res = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.04,
                                       subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
                cat_res = CatBoostRegressor(iterations=200, depth=4, learning_rate=0.04,
                                            l2_leaf_reg=3, random_seed=seed, thread_count=-1, verbose=0)
                xgb_res.fit(X_tr_sel, y_res_tr)
                cat_res.fit(X_tr_sel, y_res_tr)
                preds_cov_trees[te_idx] = cov_pred_all[te_idx] + 0.5 * (xgb_res.predict(X_te_sel) + cat_res.predict(X_te_sel))

            seed_maes_w.append(mean_absolute_error(y, preds_wave))
            seed_rmses_w.append(np.sqrt(mean_squared_error(y, preds_wave)))
            seed_r2s_w.append(r2_score(y, preds_wave))
            seed_mards_w.append(calculate_mard(y, preds_wave))

            seed_maes_cp.append(mean_absolute_error(y, preds_cov_trees))
            seed_rmses_cp.append(np.sqrt(mean_squared_error(y, preds_cov_trees)))
            seed_r2s_cp.append(r2_score(y, preds_cov_trees))
            seed_mards_cp.append(calculate_mard(y, preds_cov_trees))
            all_diff_errors_cp.append(np.abs(y - preds_cov_trees) - errors_prior_all)
            print(f"    Seed {seed} ({s_idx+1}/10): Wave MAE = {seed_maes_w[-1]:.3f}, Prior+Trees MAE = {seed_maes_cp[-1]:.3f}", flush=True)

        # Average paired differences across seeds for bootstrap CI
        mean_diff_errors = np.mean(all_diff_errors_cp, axis=0)
        d_mean, d_lo, d_hi = bootstrap_ci(mean_diff_errors)

        # Append Waveform-Only row
        all_summary.append({
            "Model": f"{name} (Waveform-Only, 10 seeds)",
            "Features": f"Top 50 of {X_feat.shape[1]} (MI)",
            "Mode": "Waveform-Only",
            "MAE (g/L)": f"{np.mean(seed_maes_w):.3f} ± {np.std(seed_maes_w):.3f}",
            "RMSE (g/L)": f"{np.mean(seed_rmses_w):.3f} ± {np.std(seed_rmses_w):.3f}",
            "R2 Score": f"{np.mean(seed_r2s_w):.4f} ± {np.std(seed_r2s_w):.4f}",
            "MARD (%)": f"{np.mean(seed_mards_w):.2f}% ± {np.std(seed_mards_w):.2f}%",
            "ΔMAE vs Prior (95% CI)": "N/A (No prior)"
        })

        # Append Prior + Trees row
        all_summary.append({
            "Model": f"Prior + {name} (10 seeds)",
            "Features": f"Prior + Top 50 of {X_feat.shape[1]}",
            "Mode": "Prior + Waveform",
            "MAE (g/L)": f"{np.mean(seed_maes_cp):.3f} ± {np.std(seed_maes_cp):.3f}",
            "RMSE (g/L)": f"{np.mean(seed_rmses_cp):.3f} ± {np.std(seed_rmses_cp):.3f}",
            "R2 Score": f"{np.mean(seed_r2s_cp):.4f} ± {np.std(seed_r2s_cp):.4f}",
            "MARD (%)": f"{np.mean(seed_mards_cp):.2f}% ± {np.std(seed_mards_cp):.2f}%",
            "ΔMAE vs Prior (95% CI)": f"{d_mean:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}]"
        })

    # ─── PART 5: PRE-SPECIFIED MINIMAL QUANTUM TEST (RIDGE/ELASTIC-NET) ───────────
    print("\n" + "="*80)
    print("PART 5: PRE-SPECIFIED MINIMAL QUANTUM TEST (RIDGE RESIDUAL, NO SELECTION)")
    print("="*80)
    # Fixed 8 quantum features: median S and purity per channel
    q_minimal_cols = [
        "660nm_q_von_neumann_entropy_med", "660nm_q_purity_med",
        "730nm_q_von_neumann_entropy_med", "730nm_q_purity_med",
        "850nm_q_von_neumann_entropy_med", "850nm_q_purity_med",
        "940nm_q_von_neumann_entropy_med", "940nm_q_purity_med"
    ]
    # Fixed baseline features: 12 classical features (acdc, ac_amp, spectral_entropy across 4 channels)
    base_morph_cols = [
        "660nm_acdc_med", "660nm_ac_amp_med", "660nm_spectral_entropy_med",
        "730nm_acdc_med", "730nm_ac_amp_med", "730nm_spectral_entropy_med",
        "850nm_acdc_med", "850nm_ac_amp_med", "850nm_spectral_entropy_med",
        "940nm_acdc_med", "940nm_ac_amp_med", "940nm_spectral_entropy_med"
    ]

    X_base = X_ppg[base_morph_cols].values
    X_q_min = X_ppg[base_morph_cols + q_minimal_cols].values

    preds_base_ridge = np.zeros(n_sub)
    preds_q_ridge = np.zeros(n_sub)

    for tr_idx, te_idx in logo.split(X_base, y, g):
        # Stage 1 prior
        Xc_tr, Xc_te = Xc_all_np[tr_idx], Xc_all_np[te_idx]
        tr_cov_med = np.nanmedian(Xc_tr, axis=0)
        Xc_tr = np.where(np.isnan(Xc_tr), tr_cov_med, Xc_tr)
        Xc_te = np.where(np.isnan(Xc_te), tr_cov_med, Xc_te)

        sc_c = RobustScaler()
        poly = PolynomialFeatures(degree=2, include_bias=False)
        Xc_tr_p = poly.fit_transform(sc_c.fit_transform(Xc_tr))
        Xc_te_p = poly.transform(sc_c.transform(Xc_te))
        cov_mdl = RidgeCV(alphas=np.logspace(0, 4, 100)).fit(Xc_tr_p, y[tr_idx])
        y_res_tr = y[tr_idx] - cross_val_predict(cov_mdl, Xc_tr_p, y[tr_idx], cv=5)
        cov_pred_te = cov_mdl.predict(Xc_te_p)

        # Baseline Ridge Residual (12 features)
        sc_b = StandardScaler()
        Xb_tr = sc_b.fit_transform(X_base[tr_idx])
        Xb_te = sc_b.transform(X_base[te_idx])
        mdl_b = RidgeCV(alphas=np.logspace(-2, 4, 100)).fit(Xb_tr, y_res_tr)
        preds_base_ridge[te_idx] = cov_pred_te + mdl_b.predict(Xb_te)

        # Quantum Ridge Residual (12 baseline + 8 quantum = 20 features)
        sc_q = StandardScaler()
        Xq_tr = sc_q.fit_transform(X_q_min[tr_idx])
        Xq_te = sc_q.transform(X_q_min[te_idx])
        mdl_q = RidgeCV(alphas=np.logspace(-2, 4, 100)).fit(Xq_tr, y_res_tr)
        preds_q_ridge[te_idx] = cov_pred_te + mdl_q.predict(Xq_te)

    mae_base_r = mean_absolute_error(y, preds_base_ridge)
    rmse_base_r = np.sqrt(mean_squared_error(y, preds_base_ridge))
    r2_base_r = r2_score(y, preds_base_ridge)
    mard_base_r = calculate_mard(y, preds_base_ridge)
    d_mean_b, d_lo_b, d_hi_b = bootstrap_ci(np.abs(y - preds_base_ridge) - errors_prior_all)

    all_summary.append({
        "Model": "Prior + 12 Baseline Morph (Ridge)",
        "Features": "AC/DC, AC Amp, SpecEnt (RidgeCV)",
        "Mode": "Linear Residual",
        "MAE (g/L)": f"{mae_base_r:.3f}",
        "RMSE (g/L)": f"{rmse_base_r:.3f}",
        "R2 Score": f"{r2_base_r:.4f}",
        "MARD (%)": f"{mard_base_r:.2f}%",
        "ΔMAE vs Prior (95% CI)": f"{d_mean_b:+.3f} [{d_lo_b:+.3f}, {d_hi_b:+.3f}]"
    })

    mae_q_r = mean_absolute_error(y, preds_q_ridge)
    rmse_q_r = np.sqrt(mean_squared_error(y, preds_q_ridge))
    r2_q_r = r2_score(y, preds_q_ridge)
    mard_q_r = calculate_mard(y, preds_q_ridge)
    d_mean_q, d_lo_q, d_hi_q = bootstrap_ci(np.abs(y - preds_q_ridge) - errors_prior_all)

    # Paired test specifically between Minimal Quantum and Baseline Morph
    dq_vs_base_mean, dq_vs_base_lo, dq_vs_base_hi = bootstrap_ci(np.abs(y - preds_q_ridge) - np.abs(y - preds_base_ridge))

    all_summary.append({
        "Model": "Prior + 12 Baseline + 8 Minimal Q (Ridge)",
        "Features": "Baseline + S & Purity (RidgeCV)",
        "Mode": "Linear Residual",
        "MAE (g/L)": f"{mae_q_r:.3f}",
        "RMSE (g/L)": f"{rmse_q_r:.3f}",
        "R2 Score": f"{r2_q_r:.4f}",
        "MARD (%)": f"{mard_q_r:.2f}%",
        "ΔMAE vs Prior (95% CI)": f"{d_mean_q:+.3f} [{d_lo_q:+.3f}, {d_hi_q:+.3f}]"
    })

    df_final = pd.DataFrame(all_summary)
    print("\n" + "="*80)
    print("FINAL CONSOLIDATED BENCHMARK TABLE")
    print("="*80)
    print(df_final.to_string(index=False))
    print("="*80)
    print(f"\nPaired Test: Minimal Q vs Baseline Morph in Ridge:")
    print(f"  ΔMAE (Q - Base) = {dq_vs_base_mean:+.3f} g/L (95% CI: [{dq_vs_base_lo:+.3f}, {dq_vs_base_hi:+.3f}])")

    csv_out = os.path.join(RESULTS_DIR, "consolidated_benchmark_200hz.csv")
    df_final.to_csv(csv_out, index=False)
    print(f"Saved full table to {csv_out}", flush=True)
    return df_final

def main():
    X_subj, Xc_subj, y_subj, g_subj = extract_all_features_200hz()
    audit_77_subject_batch(Xc_subj, y_subj, X_subj)
    plot_zoomed_diagnostics()
    run_full_loso_evaluation(X_subj, Xc_subj, y_subj, g_subj)

if __name__ == "__main__":
    main()
