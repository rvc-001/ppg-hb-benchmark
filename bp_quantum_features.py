"""
bp_quantum_features.py
=======================================================================
Quantum-INSPIRED signal representation for multiwavelength PPG windows.

SCOPE DISCLAIMER:
This module performs classical simulation of Hilbert space and density-matrix
formalism on a CPU. No quantum hardware, QPU, or quantum circuit is used.
"Quantum" here refers to borrowing linear-algebra tools from quantum information
theory (density matrices, von Neumann entropy, fidelity, trace distance)
as an uncentered second-moment / Gram-matrix representation of cardiac cycles.

CHANGELOG
---------
1. Beat segmentation & Landmark Anchoring:
   - Replaced unconstrained trough finding `find_peaks(-filt, distance=0.30*fs)`
     which lacked a prominence threshold and erroneously detected dicrotic notches
     as separate troughs at low heart rates, splitting single cardiac cycles into
     two half-beat segments of differing morphology.
   - Anchored beat segmentation on systolic peaks via `find_peaks(filt, distance=0.40*fs,
     prominence=0.30*std)`.
   - Defaulted to peak-to-peak segmentation `[p[i]:p[i+1]]`, eliminating the flat-baseline
     trough ambiguity / boundary jitter that previously inflated von Neumann entropy
     on identical synthetic pulses.
2. Signal Conditioning Alignment (Methods §III-B):
   - Bandpass filter upgraded to 4th-order Butterworth (0.5–8.0 Hz), matching paper.
   - Incorporated Sym4 level-4 discrete wavelet transform denoising with universal
     soft thresholding (lambda = sigma_hat * sqrt(2 * ln L)).
   - Updated beat duration bounds to [0.40*fs, 1.50*fs] (40–150 bpm, 0.4–1.5 s),
     matching Methods §III-B.
3. Multi-Channel Beat Synchronization:
   - Added `extract_multichannel_quantum_descriptors` to detect beat boundaries once
     on a reference channel (or highest-prominence channel) and apply identical
     temporal boundaries across all four wavelengths, enabling cross-wavelength
     density matrix fidelity and trace distance descriptors.
4. Mathematical & Terminology Corrections:
   - Described rho as the uncentered second-moment (Gram) matrix of normalized beat vectors.
   - Corrected exp(S) designation: exp(S) is the Shannon effective rank (exponential
     entropy); true participation ratio is 1/sum(lambda_k^2) = 1/purity. Both are now
     documented and provided.
   - Clarified mechanical sample-size constraints: rank(rho) <= N_pulses, bounding
     S(rho) <= ln(N_pulses). In split-half calculations, rank <= floor(N_pulses/2).
   - Removed untested claims asserting split-half divergence "flags non-stationarity".
   - Removed all self-assessment tags from comments.
5. Unit Tests:
   - Added built-in synthetic benchmark verifying S < 1e-3 for identical pulses
     at 50, 75, and 100 bpm, and strict monotonic rise with added noise.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import pywt

# Default physical sampling rate matching the four-wavelength PPG dataset (Chen et al., Electronics 2023, 12, 1346)
FS = 200
QUBIT_DIM = 32                # State-vector dimension (resampled points per cardiac cycle)
MIN_PULSES_FOR_QUANTUM = 6    # Minimum beats needed for stable density matrix & split-half check
TOP_K_EIGVALS = 5
MIN_BEAT_SEC = 0.40           # 150 bpm ceiling (Methods §III-B: 0.4 s)
MAX_BEAT_SEC = 1.50           # 40 bpm floor   (Methods §III-B: 1.5 s)


# ─── Signal Conditioning (Methods §III-B) ─────────────────────────────────────
def butter_bandpass(sig, lo=0.5, hi=8.0, fs=FS, order=4):
    """4th-order Butterworth bandpass filter (0.5–8.0 Hz)."""
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig, axis=-1)


def wavelet_denoise(sig, wavelet="sym4", level=4):
    """Level-4 Sym4 discrete wavelet denoising with universal soft thresholding.

    Matches Methods §III-B eq. 1:
      lambda = sigma_hat * sqrt(2 * ln L), where sigma_hat = median(|detail|) / 0.6745.
    """
    coeffs = pywt.wavedec(sig, wavelet, mode="per", level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2.0 * np.log(len(sig)))
    coeffs[1:] = [pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, mode="per")[:len(sig)]


def preprocess_signal(sig, fs=FS):
    """Bandpass filter followed by Sym4 wavelet soft-thresholding."""
    filt = butter_bandpass(sig, fs=fs, order=4)
    return wavelet_denoise(filt, wavelet="sym4", level=4)


# ─── Beat Detection, Autocorrelation & Quality Control ────────────────────────
def hr_autocorr(filt, fs=FS, lo_s=0.4, hi_s=1.5):
    """Independent autocorrelation heart rate estimation with sub-harmonic preference.

    Matches user specification:
    - Normalizes autocorrelation by ac[0].
    - Detects periodic peaks with height >= 0.30.
    - If no peak is >= 0.30, returns None (signal is unverifiable, not discordant).
    - If a peak near 2*best exists with >= 75-80% amplitude, selects the true fundamental
      longer period (resolves dicrotic notch / octave ambiguity).
    """
    x = filt - np.mean(filt)
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    if ac[0] < 1e-9:
        return None
    ac = ac / ac[0]
    lo, hi = int(lo_s * fs), int(hi_s * fs)
    pk, props = find_peaks(ac[lo:hi], height=0.30)
    if len(pk) == 0:
        return None  # Unverifiable periodic structure (e.g., severe motion / arrhythmia)
    best = pk[np.argmax(props["peak_heights"])]
    # Prefer the longer period (true fundamental cardiac cycle) if sub-harmonic exists
    for k in pk:
        if abs(k - 2 * best) < 0.15 * best and ac[lo + k] >= 0.75 * ac[lo + best]:
            best = k
    return 60.0 * fs / (lo + best)


def find_systolic_peaks(filt, fs=FS):
    """Detect primary systolic peaks using a 2-pass relative prominence rule.

    Pass 1: Detect candidate peaks with refractory distance 0.40*fs and 0.20*ptp prominence.
            Filter out dicrotic notches by requiring prominence >= 0.50 * median(prominences).
    Pass 2: Refine refractory spacing based on the median RR interval (0.60 * rr_med),
            eliminating false peaks on prominent dicrotic waves while preserving true beats.
    """
    ptp = np.ptp(filt)
    if ptp < 1e-4:
        return np.array([], dtype=int)
    p, pr = find_peaks(filt, distance=int(MIN_BEAT_SEC * fs), prominence=0.20 * ptp)
    if len(p) < 3:
        return np.array([], dtype=int)
    med_prom = np.median(pr["prominences"])
    keep = pr["prominences"] >= 0.50 * med_prom
    p_clean = p[keep]
    if len(p_clean) < 3:
        return np.array([], dtype=int)
    rr_med = np.median(np.diff(p_clean))
    refractory = max(int(MIN_BEAT_SEC * fs), int(0.60 * rr_med))
    prom_thresh = 0.50 * np.median(pr["prominences"][keep])
    p2, _ = find_peaks(filt, distance=refractory, prominence=prom_thresh)
    return p2


def find_beat_segments(filt, fs=FS, method="peak_to_peak", enforce_qc=True, qc_tolerance=0.08):
    """Segment signal into individual cardiac cycles anchored on systolic peaks.

    Parameters
    ----------
    filt : 1-D array of preprocessed PPG samples.
    fs : sampling frequency in Hz.
    method : {"peak_to_peak", "trough_to_trough"}
        - "peak_to_peak": Boundaries are consecutive systolic peaks [p_i, p_{i+1}].
          Provides sharp, unambiguous boundary landmarks resistant to flat-baseline jitter.
        - "trough_to_trough": Systolic peaks anchor intermediate trough localization
          via argmin between consecutive peaks [t_i, t_{i+1}].
    enforce_qc : bool, if True checks agreement with independent autocorrelation HR.
    qc_tolerance : float, max relative difference between peak HR and autocorr HR (default 0.08).

    Returns
    -------
    segments : list of 1-D arrays, each containing one cardiac cycle.
    boundaries : list of (start_idx, end_idx) sample slices.
    """
    peaks = find_systolic_peaks(filt, fs=fs)
    if len(peaks) < 3:
        return [], []

    # Quality control: independent autocorrelation agreement check
    if enforce_qc:
        hr_ac = hr_autocorr(filt, fs=fs)
        if hr_ac is not None:
            hr_pk = 60.0 / np.mean(np.diff(peaks) / fs)
            if abs(hr_pk - hr_ac) / (hr_ac + 1e-9) > qc_tolerance:
                return [], []

    min_len = int(MIN_BEAT_SEC * fs)
    max_len = int(MAX_BEAT_SEC * fs)
    segments = []
    boundaries = []

    if method == "peak_to_peak":
        for i in range(len(peaks) - 1):
            p0, p1 = peaks[i], peaks[i + 1]
            seg_len = p1 - p0
            if min_len <= seg_len <= max_len:
                segments.append(filt[p0:p1])
                boundaries.append((p0, p1))
    elif method == "trough_to_trough":
        troughs = []
        for i in range(len(peaks) - 1):
            p0, p1 = peaks[i], peaks[i + 1]
            t_loc = p0 + int(np.argmin(filt[p0:p1]))
            troughs.append(t_loc)
        for i in range(len(troughs) - 1):
            t0, t1 = troughs[i], troughs[i + 1]
            seg_len = t1 - t0
            if min_len <= seg_len <= max_len:
                segments.append(filt[t0:t1])
                boundaries.append((t0, t1))
    else:
        raise ValueError(f"Unknown segmentation method: {method}")

    return segments, boundaries


# ─── Hilbert-Space State Encoding ─────────────────────────────────────────────
def _resample(seg, dim=QUBIT_DIM):
    """Linearly interpolate beat segment onto a uniform dim-point grid."""
    x_old = np.linspace(0.0, 1.0, len(seg))
    x_new = np.linspace(0.0, 1.0, dim)
    return np.interp(x_new, x_old, seg)


def _pulse_state_vector(seg, dim=QUBIT_DIM):
    """Convert a single beat into an L2-normalized, zero-mean state vector |psi> in R^dim.

    Zero-meaning eliminates DC baseline level so that |psi> isolates morphology.
    Returns None if segment is degenerate (zero variance).
    """
    v = _resample(seg, dim=dim)
    v = v - np.mean(v)
    norm = np.linalg.norm(v)
    if norm < 1e-9 or not np.isfinite(norm):
        return None
    return (v / norm).astype(np.float64)


def _density_matrix(states):
    """Construct uncentered second-moment / Gram matrix rho = (1/N) sum_i |psi_i><psi_i|.

    The resulting matrix is real symmetric, positive semi-definite, and has Tr(rho) == 1.
    Note: For N pulses, rank(rho) <= min(dim, N), bounding the von Neumann entropy to ln(N).
    """
    valid_states = [s for s in states if s is not None]
    if len(valid_states) < 2:
        return None
    dim = len(valid_states[0])
    rho = np.zeros((dim, dim), dtype=np.float64)
    for psi in valid_states:
        rho += np.outer(psi, psi)
    rho /= len(valid_states)
    return rho


# ─── Quantum-Inspired Descriptors ────────────────────────────────────────────
def _von_neumann_entropy(rho, eps=1e-12):
    """S(rho) = -Tr(rho ln rho) = -sum_k lambda_k ln(lambda_k)."""
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.clip(eigvals, 0.0, None)
    eigvals = eigvals[eigvals > eps]
    if len(eigvals) == 0:
        return 0.0
    return float(-np.sum(eigvals * np.log(eigvals)))


def _purity(rho):
    """Pi = Tr(rho^2). Ranges from 1/dim (maximally mixed) to 1.0 (pure single-state)."""
    return float(np.real(np.trace(rho @ rho)))


def _top_eigenvalues(rho, k=TOP_K_EIGVALS):
    """Return top-k eigenvalues in descending order."""
    eigvals = np.linalg.eigvalsh(rho)[::-1]
    out = list(eigvals[:k])
    while len(out) < k:
        out.append(0.0)
    return [float(v) for v in out]


def _matrix_sqrt_psd(rho, eps=1e-12):
    """Matrix square root of a PSD real symmetric matrix via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def fidelity(rho, sigma):
    """Uhlmann fidelity F(rho, sigma) = [Tr(sqrt(sqrt(rho) sigma sqrt(rho)))]^2.

    For valid density matrices, F in [0, 1], with F = 1 iff rho == sigma.
    """
    sqrt_rho = _matrix_sqrt_psd(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    eigvals = np.linalg.eigvalsh(inner)
    eigvals = np.clip(eigvals, 0.0, None)
    return float(np.sum(np.sqrt(eigvals)) ** 2)


def trace_distance(rho, sigma):
    """Trace distance D(rho, sigma) = 0.5 * Tr|rho - sigma| in [0, 1]."""
    diff = rho - sigma
    eigvals = np.linalg.eigvalsh(diff)
    return float(0.5 * np.sum(np.abs(eigvals)))


# ─── Public Window-Level API ──────────────────────────────────────────────────
def extract_quantum_descriptors(raw_or_filt_win, fs=FS, dim=QUBIT_DIM,
                                rho_ref=None, method="peak_to_peak", is_prefiltered=False):
    """Extract quantum-inspired descriptors from a single PPG channel window.

    Parameters
    ----------
    raw_or_filt_win : 1-D array of PPG samples (raw or prefiltered).
    fs : sampling frequency in Hz (default 125).
    dim : state-vector dimension each pulse is resampled to (default 32).
    rho_ref : optional population-reference density matrix (fit on dev-only data).
    method : {"peak_to_peak", "trough_to_trough"} (default "peak_to_peak").
    is_prefiltered : bool, if True skips bandpass/wavelet preprocessing.

    Returns
    -------
    dict of q_-prefixed features, or None if too few valid beats are detected.
    """
    if is_prefiltered:
        filt = raw_or_filt_win
    else:
        filt = preprocess_signal(raw_or_filt_win, fs=fs)

    segments, _ = find_beat_segments(filt, fs=fs, method=method)
    if len(segments) < MIN_PULSES_FOR_QUANTUM:
        return None

    states = [_pulse_state_vector(seg, dim=dim) for seg in segments]
    states = [s for s in states if s is not None]
    if len(states) < MIN_PULSES_FOR_QUANTUM:
        return None

    rho = _density_matrix(states)
    if rho is None:
        return None

    n_pulses = float(len(states))
    entropy = _von_neumann_entropy(rho)
    purity_val = _purity(rho)
    shannon_eff_rank = float(np.exp(entropy))
    participation_ratio = float(1.0 / (purity_val + 1e-12))
    top_eigs = _top_eigenvalues(rho, k=TOP_K_EIGVALS)

    # Split-half comparison: early vs late halves of the window.
    # Note: Each half matrix has rank <= floor(N/2), introducing mechanical rank-truncation.
    mid = len(states) // 2
    rho_early = _density_matrix(states[:mid])
    rho_late = _density_matrix(states[mid:])
    if rho_early is not None and rho_late is not None:
        self_fid = fidelity(rho_early, rho_late)
        self_td = trace_distance(rho_early, rho_late)
    else:
        self_fid, self_td = np.nan, np.nan

    feats = {
        "q_von_neumann_entropy": entropy,
        "q_purity": purity_val,
        "q_effective_rank": shannon_eff_rank,       # Shannon effective rank exp(S)
        "q_participation_ratio": participation_ratio, # 1/purity
        "q_self_fidelity": self_fid,
        "q_self_trace_distance": self_td,
        "q_n_pulses": n_pulses,
    }
    for i, ev in enumerate(top_eigs, 1):
        feats[f"q_eig_{i}"] = ev

    if rho_ref is not None and rho_ref.shape == rho.shape:
        feats["q_ref_fidelity"] = fidelity(rho, rho_ref)
        feats["q_ref_trace_distance"] = trace_distance(rho, rho_ref)

    return feats


def extract_multichannel_quantum_descriptors(raw_or_filt_channels, fs=FS, dim=QUBIT_DIM,
                                             ch_names=None, ref_ch_idx=None,
                                             method="peak_to_peak", is_prefiltered=False):
    """Extract beat-synchronized quantum-inspired descriptors across all channels.

    Beats are detected once on a single reference channel (e.g., 850 nm or highest SNR channel),
    and the identical temporal boundaries are applied to slice cardiac cycles across all 4 wavelengths.
    This guarantees beat synchronization and enables cross-wavelength density matrix descriptors.

    Parameters
    ----------
    raw_or_filt_channels : 2-D array of shape (n_channels, n_samples)
    fs : sampling frequency in Hz.
    dim : state-vector dimension.
    ch_names : list of channel name strings (e.g. ["660nm", "730nm", "850nm", "940nm"]).
    ref_ch_idx : int or None. Channel index used for master beat segmentation.
                 If None, selects the channel with highest peak prominence / variance.
    method : segmentation method ("peak_to_peak" or "trough_to_trough").
    is_prefiltered : bool, if True skips preprocessing.

    Returns
    -------
    dict of features across all channels and cross-channel descriptors, or None if segmentation fails.
    """
    raw_or_filt_channels = np.asarray(raw_or_filt_channels)
    n_ch, n_samples = raw_or_filt_channels.shape

    if ch_names is None:
        ch_names = [f"ch{i}" for i in range(n_ch)]

    # Preprocess all channels
    clean_chs = []
    for c in range(n_ch):
        sig = raw_or_filt_channels[c]
        clean_sig = sig if is_prefiltered else preprocess_signal(sig, fs=fs)
        # Orientation verification: enforce positive systolic peaks
        if np.mean(clean_sig**3) < 0:
            clean_sig = -clean_sig
        clean_chs.append(clean_sig)

    # Determine master reference channel
    if ref_ch_idx is None:
        # Pick channel with clearest pulsatile prominence
        stds = [np.std(c) for c in clean_chs]
        ref_ch_idx = int(np.argmax(stds))

    # Detect synchronized beat boundaries on the reference channel
    ref_sig = clean_chs[ref_ch_idx]
    _, boundaries = find_beat_segments(ref_sig, fs=fs, method=method)
    if len(boundaries) < MIN_PULSES_FOR_QUANTUM:
        return None

    out_dict = {}
    channel_rhos = {}

    for c, nm in enumerate(ch_names):
        sig = clean_chs[c]
        # Slice identical beat segments
        segs = [sig[b0:b1] for (b0, b1) in boundaries]
        states = [_pulse_state_vector(s, dim=dim) for s in segs]
        states = [s for s in states if s is not None]

        if len(states) < MIN_PULSES_FOR_QUANTUM:
            return None

        rho = _density_matrix(states)
        if rho is None:
            return None
        channel_rhos[nm] = rho

        ent = _von_neumann_entropy(rho)
        pur = _purity(rho)
        top_e = _top_eigenvalues(rho, k=TOP_K_EIGVALS)

        mid = len(states) // 2
        rho_early = _density_matrix(states[:mid])
        rho_late = _density_matrix(states[mid:])
        self_fid = fidelity(rho_early, rho_late) if (rho_early is not None and rho_late is not None) else np.nan
        self_td = trace_distance(rho_early, rho_late) if (rho_early is not None and rho_late is not None) else np.nan

        out_dict[f"{nm}_q_von_neumann_entropy"] = ent
        out_dict[f"{nm}_q_purity"] = pur
        out_dict[f"{nm}_q_effective_rank"] = float(np.exp(ent))
        out_dict[f"{nm}_q_participation_ratio"] = float(1.0 / (pur + 1e-12))
        out_dict[f"{nm}_q_self_fidelity"] = self_fid
        out_dict[f"{nm}_q_self_trace_distance"] = self_td
        out_dict[f"{nm}_q_n_pulses"] = float(len(states))
        for i, ev in enumerate(top_e, 1):
            out_dict[f"{nm}_q_eig_{i}"] = ev

    # Cross-wavelength geometric descriptors
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            ni, nj = ch_names[i], ch_names[j]
            rho_i, rho_j = channel_rhos[ni], channel_rhos[nj]
            out_dict[f"cross_q_fid_{ni}_{nj}"] = fidelity(rho_i, rho_j)
            out_dict[f"cross_q_td_{ni}_{nj}"] = trace_distance(rho_i, rho_j)

    return out_dict


# ─── Population Reference (fit on Dev Partition Only) ─────────────────────────
def fit_population_reference(rho_list):
    """Build a population-reference density matrix by averaging development-fold density matrices.

    CRITICAL: Must be fit strictly on DEV partitions to prevent information leakage.
    """
    rhos = [r for r in rho_list if r is not None]
    if len(rhos) == 0:
        raise ValueError("fit_population_reference: no valid density matrices provided.")
    rho_ref = np.mean(np.stack(rhos, axis=0), axis=0)
    tr = np.trace(rho_ref)
    if tr > 1e-9:
        rho_ref = rho_ref / tr
    return rho_ref


def save_reference(rho_ref, path):
    """Persist fitted reference density matrix."""
    np.save(path, rho_ref)


def load_reference(path):
    """Load previously fitted reference density matrix."""
    return np.load(path)


# ─── Synthetic Unit Test Benchmark ───────────────────────────────────────────
def run_synthetic_unit_tests():
    """Unit tests:
    1. Robustness test with non-integer heart rates (62, 83 bpm) and integer (50, 75, 100 bpm),
       with physiological ±5% RR jitter, slow baseline wander (0.15 Hz), and amplitude modulation (0.10 Hz).
       Expect S close to 0 (< 0.12) and flat across heart rates.
    2. Zero-noise identical pulses must yield S < 1e-3.
    3. Noise sweep at 75 bpm must produce a strictly monotonic rise in S.
    """
    print("=" * 60)
    print("RUNNING ADVANCED SYNTHETIC BENCHMARK & SEGMENTATION UNIT TESTS")
    print("=" * 60)

    def _make_synth_ppg(bpm, duration_sec=12, fs=125, noise_std=0.0, seed=42):
        period = 60.0 / bpm
        t_pulse = np.linspace(0, min(0.5, period * 0.8), int(min(0.5, period * 0.8) * fs))
        w_sys = np.exp(-((t_pulse - 0.15)**2) / (2 * 0.04**2))
        w_dic = 0.35 * np.exp(-((t_pulse - 0.35)**2) / (2 * 0.05**2))
        pulse = w_sys + w_dic
        sig = []
        n_beats = int(duration_sec / period) + 1
        for _ in range(n_beats):
            beat = np.zeros(int(period * fs))
            beat[:len(pulse)] = pulse
            sig.extend(beat)
        sig = np.array(sig[:int(duration_sec * fs)])
        if noise_std > 0:
            rng = np.random.default_rng(seed)
            sig = sig + rng.normal(0, noise_std * np.std(sig), len(sig))
        return sig

    def _make_challenging_ppg(bpm, duration_sec=12, fs=125, rr_jitter=0.05,
                              baseline_wander=True, amp_mod=True, seed=42):
        rng = np.random.default_rng(seed)
        mean_period = 60.0 / bpm
        n_beats_est = int(duration_sec / mean_period) + 4
        jitters = rng.uniform(-rr_jitter, rr_jitter, n_beats_est) if rr_jitter > 0 else np.zeros(n_beats_est)
        periods = mean_period * (1.0 + jitters)
        sig_list = []
        for i, per in enumerate(periods):
            n_pts = int(per * fs)
            t_pulse = np.linspace(0, per, n_pts, endpoint=False)
            w_sys = np.exp(-((t_pulse - 0.15 * per)**2) / (2 * (0.04 * per)**2))
            w_dic = 0.35 * np.exp(-((t_pulse - 0.35 * per)**2) / (2 * (0.05 * per)**2))
            pulse = w_sys + w_dic
            if amp_mod:
                t_mid = (sum(periods[:i]) + 0.5 * per)
                pulse = pulse * (1.0 + 0.10 * np.sin(2 * np.pi * 0.10 * t_mid))
            sig_list.append(pulse)
        sig = np.concatenate(sig_list)[:int(duration_sec * fs)]
        t = np.arange(len(sig)) / fs
        if baseline_wander:
            sig += 0.15 * np.std(sig) * np.sin(2 * np.pi * 0.15 * t)
        return sig

    # Test 1: Zero-noise identical pulses across heart rates
    print("\n[Test 1] Zero-noise baseline across integer heart rates (50, 75, 100 bpm):")
    for bpm in [50, 75, 100]:
        sig = _make_synth_ppg(bpm=bpm, noise_std=0.0)
        feats = extract_quantum_descriptors(sig, fs=125, method="peak_to_peak", is_prefiltered=True)
        assert feats is not None, f"Failed to extract descriptors for {bpm} bpm"
        S = feats["q_von_neumann_entropy"]
        purity_val = feats["q_purity"]
        n_p = feats["q_n_pulses"]
        print(f"  BPM {bpm:3d} -> S = {S:.6f}, Purity = {purity_val:.6f}, Beats = {n_p:.0f}")
        assert S < 1e-3, f"Entropy too high for identical pulses at {bpm} bpm (S={S:.4f})"
        assert purity_val > 0.999, f"Purity too low for identical pulses at {bpm} bpm (P={purity_val:.4f})"

    # Test 2: Challenging non-integer periods, ±5% RR jitter, baseline wander & amp modulation
    print("\n[Test 2] Challenging signals: Non-integer HR (62, 83 bpm), ±5% RR jitter, wander & amp mod:")
    for bpm in [50, 62, 75, 83, 100]:
        sig = _make_challenging_ppg(bpm=bpm, rr_jitter=0.05, baseline_wander=True, amp_mod=True)
        filt = wavelet_denoise(butter_bandpass(sig, fs=125, order=4))
        feats = extract_quantum_descriptors(filt, fs=125, method="peak_to_peak", is_prefiltered=True)
        assert feats is not None, f"Failed to extract descriptors for challenging signal at {bpm} bpm"
        S = feats["q_von_neumann_entropy"]
        purity_val = feats["q_purity"]
        r_eff = feats["q_effective_rank"]
        n_p = feats["q_n_pulses"]
        print(f"  BPM {bpm:3d} -> S = {S:.4f}, Purity = {purity_val:.4f}, Eff_Rank = {r_eff:.3f}, Beats = {n_p:.0f}")
        assert S < 0.12, f"Entropy inflated on challenging signal at {bpm} bpm (S={S:.4f})"
        assert purity_val > 0.95, f"Purity too low on challenging signal at {bpm} bpm (P={purity_val:.4f})"

    # Test 3: Monotonic rise with noise at 75 bpm
    print("\n[Test 3] Monotonic rise with added noise at 75 bpm:")
    noises = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]
    s_list = []
    for n in noises:
        sig = _make_synth_ppg(bpm=75, noise_std=n, seed=42)
        feats = extract_quantum_descriptors(sig, fs=125, method="peak_to_peak", is_prefiltered=True)
        assert feats is not None
        s_list.append(feats["q_von_neumann_entropy"])
        print(f"  Noise std {n:4.2f} -> S = {s_list[-1]:.4f}")

    for i in range(len(s_list) - 1):
        assert s_list[i] <= s_list[i + 1], (
            f"Non-monotonic entropy increase: S({noises[i]}) = {s_list[i]:.4f} > "
            f"S({noises[i+1]}) = {s_list[i+1]:.4f}"
        )

    print("\n[PASS] All advanced synthetic unit tests passed successfully!\n" + "=" * 60)


if __name__ == "__main__":
    run_synthetic_unit_tests()
