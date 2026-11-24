import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch, detrend, medfilt
from scipy.stats import skew, kurtosis, iqr

FS         = 100
WINDOW_SEC = 10
WINDOW_LEN = FS * WINDOW_SEC
OVERLAP    = 0.50
STEP       = int(WINDOW_LEN * (1 - OVERLAP))
MIN_SQI    = 0.35


# ─── Signal utilities ─────────────────────────────────────────────────────────
def butter_bandpass(sig, lo=0.5, hi=8.0, fs=FS, order=4):
    sig_clean = medfilt(detrend(sig), kernel_size=3)
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig_clean)


def signal_quality_index(sig, fs=FS):
    """Autocorrelation-based SQI: periodicity × amplitude consistency."""
    peaks, _ = find_peaks(sig, distance=int(0.30 * fs))
    if len(peaks) < 3:
        return 0.0
    pp = np.diff(peaks) / fs
    if len(pp) < 2 or np.mean(pp) <= 0:
        return 0.0
    lag = int(np.mean(pp) * fs)
    if lag <= 0 or lag >= len(sig) - 1:
        return 0.0
    sig_z = sig - sig.mean()
    denom = np.linalg.norm(sig_z[:-lag]) * np.linalg.norm(sig_z[lag:])
    ac = float(np.dot(sig_z[:-lag], sig_z[lag:]) / denom) if denom > 1e-9 else 0.0
    periodicity  = max(0.0, np.nan_to_num(ac))
    amps = sig[peaks]
    consistency  = max(0.0, 1.0 - min(np.std(amps) / (abs(np.mean(amps)) + 1e-6), 1.0))
    return float(0.6 * periodicity + 0.4 * consistency)


def is_usable_window(sig, fs=FS):
    if np.ptp(sig) < 1e-3 or np.isnan(sig).any():
        return False
        
    # Reject segments with excessive flatlines (clipping/saturation)
    diffs = np.diff(sig)
    if np.sum(diffs == 0) > 0.05 * len(sig): 
        return False
        
    # Reject segments with sudden unphysiological spikes (artifacts)
    if np.max(np.abs(diffs)) > 0.75 * np.ptp(sig):
        return False

    peaks, _ = find_peaks(sig, distance=int(0.30 * fs))
    dur_min = len(sig) / fs / 60.0
    rate = len(peaks) / dur_min if dur_min > 0 else 0
    return (35 <= rate <= 200) and signal_quality_index(sig, fs) >= MIN_SQI


def _safe_mean(arr):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if len(a) else np.nan


# ─── Feature groups ──────────────────────────────────────────────────────────
def _sdptg_features(sig, troughs, fs=FS):
    """Second derivative of PPG (SDPTG) wave indices a–e + reflection index."""
    d2 = np.gradient(np.gradient(sig))
    keys = ["sdptg_b_a", "sdptg_c_a", "sdptg_d_a", "sdptg_e_a", "reflection_index",
            "ai_sdptg"]
    if len(troughs) < 3:
        return {k: np.nan for k in keys}

    ba, ca, da, ea, ri = [], [], [], [], []
    for i in range(len(troughs) - 1):
        seg  = d2[troughs[i]:troughs[i + 1]]
        orig = sig[troughs[i]:troughs[i + 1]]
        if len(seg) < 5:
            continue
        a_idx = int(np.argmax(seg[:max(len(seg) // 3, 1)]))
        a_val = seg[a_idx]
        if a_val == 0 or not np.isfinite(a_val):
            continue
        rest = seg[a_idx:]
        if len(rest) < 4:
            continue
        b_idx = a_idx + int(np.argmin(rest[:max(len(rest) // 4, 1)]))
        b_val = seg[b_idx] if b_idx < len(seg) else np.nan
        tail  = seg[b_idx - a_idx:]
        c_val = np.max(tail) if len(tail) else np.nan
        d_val = np.min(tail) if len(tail) else np.nan
        e_val = seg[-1]
        ba.append(b_val / a_val)
        ca.append(c_val / a_val)
        da.append(d_val / a_val)
        ea.append(e_val / a_val)
        if np.ptp(orig) > 0:
            ri.append((orig[-1] - orig.min()) / np.ptp(orig))

    b_a = _safe_mean(ba)
    return {
        "sdptg_b_a": b_a,
        "sdptg_c_a": _safe_mean(ca),
        "sdptg_d_a": _safe_mean(da),
        "sdptg_e_a": _safe_mean(ea),
        "reflection_index": _safe_mean(ri),
        "ai_sdptg": (-b_a) if np.isfinite(b_a) else np.nan,  # Augmentation Index proxy
    }


def _respiratory_features(sig, peaks, pp_intervals, fs=FS):
    """Respiratory modulation of PPG: RIIV, RIAV, RIFV power ratios."""
    out = {"riiv_power_ratio": np.nan, "rifv_power_ratio": np.nan, "riav_power_ratio": np.nan}
    if len(peaks) < 6:
        return out

    amps     = sig[peaks]
    baseline = np.array([sig[max(p - int(0.1 * fs), 0):p + int(0.1 * fs)].mean()
                         for p in peaks])
    times    = peaks / fs

    def resp_ratio(series, t):
        if len(series) < 6 or np.std(series) < 1e-9:
            return np.nan
        tu = np.linspace(t.min(), t.max(), len(series))
        si = np.interp(tu, t, series) - series.mean()
        f, psd = welch(si, fs=1.0 / max(np.mean(np.diff(tu)), 1e-9),
                       nperseg=min(len(si), 16))
        rb = (f >= 0.15) & (f <= 0.45)
        return float(psd[rb].sum() / (psd.sum() + 1e-9))

    out["riiv_power_ratio"] = resp_ratio(amps, times)
    out["riav_power_ratio"] = resp_ratio(baseline, times)
    if len(pp_intervals) >= 6:
        out["rifv_power_ratio"] = resp_ratio(pp_intervals, times[1:len(pp_intervals) + 1])
    return out


def _spectral_features(sig, fs=FS):
    """
    Welch PSD features across 5 physiological bands:
      VLF 0.003–0.04 Hz  (very-low-freq vasomotion)
      LF  0.04–0.15 Hz   (sympathetic)
      HF  0.15–0.40 Hz   (parasympathetic / respiratory)
      HF2 0.40–4.0  Hz   (PPG harmonics)
      TOT 0.003–4.0 Hz

    Also centroid, bandwidth, spectral entropy.
    """
    f, psd = welch(sig, fs=fs, nperseg=min(512, len(sig)))
    total  = psd.sum() + 1e-9
    psd_n  = psd / total

    def band_power(lo, hi):
        mask = (f >= lo) & (f < hi)
        return float(psd[mask].sum())

    vlf_p  = band_power(0.003, 0.04)
    lf_p   = band_power(0.04,  0.15)
    hf_p   = band_power(0.15,  0.40)
    hf2_p  = band_power(0.40,  4.0)
    tot_p  = band_power(0.003, 4.0) + 1e-9

    centroid  = float(np.sum(f * psd_n))
    bw        = float(np.sqrt(np.sum(((f - centroid) ** 2) * psd_n)))
    spec_entr = float(-np.sum(psd_n[psd_n > 0] * np.log(psd_n[psd_n > 0] + 1e-12)))

    return {
        "psd_vlf": vlf_p, "psd_lf": lf_p, "psd_hf": hf_p, "psd_hf2": hf2_p,
        "psd_lf_hf_ratio": lf_p / (hf_p + 1e-9),
        "spectral_centroid": centroid, "spectral_bandwidth": bw,
        "spectral_entropy": spec_entr,
    }


def _morphology_features(sig, filt, peaks, troughs, pp_intervals, fs=FS):
    """
    Pulse-level morphology: crest time, rise/fall time, diastolic area,
    systolic area, dicrotic notch proxy, pulse width (1/3 and 2/3 amp),
    diastolic-to-systolic area ratio, perfusion index.
    """
    ac   = np.ptp(filt)
    dc   = np.mean(sig)

    if len(peaks) < 2 or len(troughs) < 2:
        return {k: np.nan for k in [
            "crest_time", "fall_time", "systolic_area", "diastolic_area",
            "dia_sys_area_ratio", "pulse_width_33", "pulse_width_66",
            "dicrotic_notch_rel", "perfusion_index",
        ]}

    crest_times, fall_times = [], []
    sys_areas,  dia_areas   = [], []
    pw33, pw66              = [], []
    notch_rels              = []

    for k, peak in enumerate(peaks):
        # find preceding trough (onset) and following trough (end)
        pre = troughs[troughs < peak]
        post = troughs[troughs > peak]
        if len(pre) == 0 or len(post) == 0:
            continue
        onset = pre[-1]
        end   = post[0]

        seg_sys = filt[onset:peak]
        seg_dia = filt[peak:end]

        if len(seg_sys) < 2 or len(seg_dia) < 2:
            continue

        crest_times.append((peak - onset) / fs)
        fall_times.append((end - peak) / fs)

        sys_areas.append(float(np.trapz(np.abs(seg_sys))))
        dia_areas.append(float(np.trapz(np.abs(seg_dia))))

        # Pulse width at 1/3 and 2/3 amplitude
        amp    = filt[peak] - filt[onset]
        thresh33 = filt[onset] + amp / 3.0
        thresh66 = filt[onset] + 2.0 * amp / 3.0
        full_seg = filt[onset:end]
        above33  = np.where(full_seg >= thresh33)[0]
        above66  = np.where(full_seg >= thresh66)[0]
        if len(above33) > 1:
            pw33.append((above33[-1] - above33[0]) / fs)
        if len(above66) > 1:
            pw66.append((above66[-1] - above66[0]) / fs)

        # Dicrotic notch: local minimum in diastolic segment (after 40% of diastole)
        if len(seg_dia) > 4:
            search_start = max(0, len(seg_dia) // 5)
            search_end   = min(len(seg_dia), 3 * len(seg_dia) // 4)
            sub = seg_dia[search_start:search_end]
            if len(sub) > 2:
                notch_val  = float(np.min(sub))
                notch_rel  = (filt[peak] - notch_val) / (amp + 1e-9)
                notch_rels.append(notch_rel)

    dia_a  = _safe_mean(dia_areas)
    sys_a  = _safe_mean(sys_areas)
    ds_r   = (dia_a / (sys_a + 1e-9)) if (np.isfinite(dia_a) and np.isfinite(sys_a)) else np.nan

    return {
        "crest_time":        _safe_mean(crest_times),
        "fall_time":         _safe_mean(fall_times),
        "systolic_area":     sys_a,
        "diastolic_area":    dia_a,
        "dia_sys_area_ratio": ds_r,
        "pulse_width_33":    _safe_mean(pw33),
        "pulse_width_66":    _safe_mean(pw66),
        "dicrotic_notch_rel": _safe_mean(notch_rels),
        "perfusion_index":   (ac / (abs(dc) + 1e-6)) * 100.0,
    }


def _variability_features(pp_intervals):
    """HRV-inspired variability features on inter-pulse intervals."""
    if len(pp_intervals) < 3:
        return {k: np.nan for k in ["sdnn_pp", "rmssd_pp", "pnn20_pp", "cv_pp"]}
    diff = np.diff(pp_intervals)
    return {
        "sdnn_pp":  float(np.std(pp_intervals, ddof=1)),
        "rmssd_pp": float(np.sqrt(np.mean(diff ** 2))),
        "pnn20_pp": float(np.mean(np.abs(diff) > 0.020)),  # >20 ms equivalent
        "cv_pp":    float(np.std(pp_intervals) / (np.mean(pp_intervals) + 1e-9)),
    }


# ─── Public API ──────────────────────────────────────────────────────────────
def extract_ppg_window_features(raw_win, fs=FS):
    """
    Extract ~50 PPG-derived features from one window of raw PPG samples.
    Returns a dict (no ART signal touched at all).
    Returns None if the window fails SQI or has < 3 detectable peaks.
    """
    if not is_usable_window(raw_win, fs):
        return None

    filt = butter_bandpass(raw_win, fs=fs)
    ac   = np.ptp(filt)
    dc   = np.mean(raw_win)

    peaks,   _ = find_peaks(filt, distance=int(0.30 * fs), prominence=0.05 * ac)
    troughs, _ = find_peaks(-filt, distance=int(0.30 * fs))
    if len(peaks) < 3 or len(troughs) < 3:
        return None

    pp_intervals = np.diff(peaks) / fs
    if len(pp_intervals) < 2:
        return None

    amps = filt[peaks]
    mean_hr = 60.0 / np.mean(pp_intervals)

    # Rise times (onset-to-peak)
    rise_times = []
    for p in peaks:
        pre = troughs[troughs < p]
        if len(pre):
            rise_times.append((p - pre[-1]) / fs)
    rise_times = np.array(rise_times) if rise_times else np.array([np.nan])

    feats = {
        # Amplitude / DC
        "ac_amp":           ac,
        "dc_level":         dc,
        "ac_dc_ratio":      ac / (abs(dc) + 1e-6),
        # Pulse amplitude stats
        "mean_pulse_amp":   float(np.mean(amps)),
        "std_pulse_amp":    float(np.std(amps)),
        "cv_pulse_amp":     float(np.std(amps) / (abs(np.mean(amps)) + 1e-6)),
        "max_pulse_amp":    float(np.max(amps)),
        "min_pulse_amp":    float(np.min(amps)),
        # Pulse rate
        "mean_pp_interval": float(np.mean(pp_intervals)),
        "std_pp_interval":  float(np.std(pp_intervals)),
        "pulse_rate_bpm":   mean_hr,
        # Rise time
        "mean_rise_time":   float(np.nanmean(rise_times)),
        "std_rise_time":    float(np.nanstd(rise_times)),
        # Shape
        "skewness":         float(skew(filt)),
        "kurtosis":         float(kurtosis(filt)),
        "signal_iqr":       float(iqr(filt)),
        # Signal quality
        "sqi":              signal_quality_index(filt, fs=fs),
        # NOTE: n_peaks here is the count of PPG peaks, distinct from n_art_peaks which is used as a label quality gate.
        "n_peaks":          float(len(peaks)),
    }

    feats.update(_variability_features(pp_intervals))
    feats.update(_spectral_features(filt, fs=fs))
    feats.update(_sdptg_features(filt, troughs, fs=fs))
    feats.update(_respiratory_features(filt, peaks, pp_intervals, fs=fs))
    feats.update(_morphology_features(raw_win, filt, peaks, troughs, pp_intervals, fs=fs))

    return feats


def extract_sbp_dbp_from_window(art_win, fs=FS):
    """
    Extract SBP (systolic peak median) and DBP (diastolic trough median)
    from one window of the raw arterial pressure signal.

    Returns (sbp, dbp, n_art_peaks) where n_art_peaks is the number of
    detected ART systolic peaks used to compute the median.
    Returns (np.nan, np.nan, 0) if the window is unusable.
    """
    if art_win is None or len(art_win) < fs * 2 or np.isnan(art_win).any():
        return np.nan, np.nan, 0
        
    # Check for ART artifacts (e.g. flushing or line draws causing huge sudden jumps)
    if np.max(np.abs(np.diff(art_win))) > 60: 
        return np.nan, np.nan, 0

    art = art_win[(art_win > 20) & (art_win < 300)]
    if len(art) < fs * 2:
        return np.nan, np.nan, 0

    nyq = 0.5 * fs
    b, a = butter(2, [0.5 / nyq, 15.0 / nyq], btype="band")
    filt = filtfilt(b, a, art)

    peaks,   _ = find_peaks(filt,  distance=int(0.4 * fs), prominence=3)
    troughs, _ = find_peaks(-filt, distance=int(0.4 * fs), prominence=3)

    if len(peaks) < 2 or len(troughs) < 2:
        return np.nan, np.nan, 0

    # INTENTIONAL: We find peak locations on the smoothed/filtered signal (filt)
    # to avoid noise/artifact triggering false peaks, but we read the absolute 
    # amplitude from the raw, unfiltered signal (art) at those indices. 
    # This prevents the bandpass filter's ringing from biasing the true mmHg SBP/DBP.
    sbp = float(np.median(art[peaks]))
    dbp = float(np.median(art[troughs]))

    if not (40 < dbp < 200 and dbp < sbp < 300 and (sbp - dbp) > 5):
        return np.nan, np.nan, 0

    return sbp, dbp, int(len(peaks))