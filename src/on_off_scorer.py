"""
Production ON/OFF Running-State Scorer
=======================================
Given a raw EMI spectrum (frequency array + amplitude array), outputs the
predicted generator running state (ON / OFF) together with per-feature
values, the composite weighted score, and a flat-line flag.

Usage
-----
    from on_off_scorer import OnOffScorer

    scorer = OnOffScorer()                        # default threshold 0.35
    result = scorer.score(freqs_hz, amplitudes_uv)

    result['on_status']       # True / False
    result['score']           # float in [0, 1]
    result['is_flat_line']    # True / False
    result['features']        # dict of feature name → value
    result['votes']           # dict of feature name → 0 or 1

Parameters are customizable via constructor kwargs; see OnOffScorer.__init__.
"""
from __future__ import annotations

import warnings

import numpy as np

_EPS = 1e-10

_EXPECTED_N_BINS = 8000
_EXPECTED_FREQ_MIN_HZ = 30_000
_EXPECTED_FREQ_MAX_HZ = 100_000_000
_MIN_BAND_POINTS = 3

BAND_DEFS = [
    ('B1', 30_000, 300_000),
    ('B2', 300_000, 3_000_000),
    ('B3', 3_000_000, 30_000_000),
    ('B4', 30_000_000, 100_000_000),
]

_DEFAULT_FEATURES = [
    {
        'name': 'ssn_B1',
        'direction': 'higher',
        'weight': 0.945,
    },
    {
        'name': 'lr_B2_B4',
        'direction': 'higher',
        'weight': 0.839,
    },
    {
        'name': 'spectral_spread',
        'direction': 'lower',
        'weight': 0.811,
    },
    {
        'name': 'rolloff_norm_B3',
        'direction': 'lower',
        'weight': 0.738,
    },
    {
        'name': 'curvature_B2',
        'direction': 'higher',
        'weight': 0.719,
    },
]

_DEFAULT_MEDIANS = {
    'ssn_B1': -0.1009,
    'lr_B2_B4': 1.61,
    'spectral_spread': 724.79,
    'rolloff_norm_B3': 0.6635,
    'curvature_B2': 0.2033,
}

_DEFAULT_THRESHOLD = 0.35

_DEFAULT_FLAT_THRESHOLD = 0.40


def _compute_band_power(freqs: np.ndarray, pwr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (freqs >= lo) & (freqs < hi)
    return pwr[mask], freqs[mask], mask


def validate_spectrum(freqs: np.ndarray, amplitudes_uv: np.ndarray) -> list[dict]:
    """Check that a spectrum matches the expected format from training data.

    The default medians and weights were calibrated on 8 000-bin spectrums
    spanning 30 kHz – 100 MHz.  Spectrums with a different length or frequency
    range will produce features on a different scale, making the default
    thresholds unreliable — especially ``spectral_spread`` which is computed
    on bin indices rather than Hz.

    Returns a list of warning dicts (empty if all checks pass).  Each dict
    has keys ``code`` (machine-readable) and ``message`` (human-readable).
    """
    issues: list[dict] = []
    n = len(freqs)

    if n != _EXPECTED_N_BINS:
        issues.append({
            'code': 'bin_count_mismatch',
            'message': (
                f'Expected {_EXPECTED_N_BINS} frequency bins but got {n}. '
                f'The spectral_spread median (724.79) was calibrated on '
                f'{_EXPECTED_N_BINS}-bin spectrums and will not be '
                f'comparable at a different resolution.'
            ),
        })

    if n > 0:
        f_min, f_max = float(freqs.min()), float(freqs.max())
        if f_min > _EXPECTED_FREQ_MIN_HZ * 1.5 or f_max < _EXPECTED_FREQ_MAX_HZ * 0.75:
            issues.append({
                'code': 'freq_range_mismatch',
                'message': (
                    f'Expected frequency range ~{_EXPECTED_FREQ_MIN_HZ/1e3:.0f} kHz – '
                    f'{_EXPECTED_FREQ_MAX_HZ/1e6:.0f} MHz but got '
                    f'{f_min/1e3:.1f} kHz – {f_max/1e6:.1f} MHz. '
                    f'Band power features may be incomplete.'
                ),
            })

    for bn, lo, hi in BAND_DEFS:
        n_in_band = int(((freqs >= lo) & (freqs < hi)).sum())
        if n_in_band < _MIN_BAND_POINTS:
            issues.append({
                'code': f'sparse_band_{bn}',
                'message': (
                    f'Band {bn} ({lo/1e3:.0f} kHz – {hi/1e6:.0f} MHz) has '
                    f'only {n_in_band} point(s); need ≥ {_MIN_BAND_POINTS} '
                    f'for stable slope/curvature fits.'
                ),
            })

    if len(freqs) != len(amplitudes_uv):
        issues.append({
            'code': 'length_mismatch',
            'message': (
                f'freqs has {len(freqs)} elements but amplitudes has '
                f'{len(amplitudes_uv)}. Arrays must be the same length.'
            ),
        })

    if np.any(amplitudes_uv < 0):
        issues.append({
            'code': 'negative_amplitudes',
            'message': 'Amplitude array contains negative values.',
        })

    return issues


def compute_features(freqs: np.ndarray, amplitudes_uv: np.ndarray) -> dict:
    """Compute the five scoring features plus flat-line metric from a raw spectrum.

    Parameters
    ----------
    freqs : array of shape (N,)
        Frequency values in Hz.
    amplitudes_uv : array of shape (N,)
        Amplitude values in µV.

    Returns
    -------
    dict with keys: ssn_B1, lr_B2_B4, spectral_spread, rolloff_norm_B3,
    curvature_B2, spectral_flatness_global.
    """
    freqs = np.asarray(freqs, dtype=float)
    amplitudes_uv = np.asarray(amplitudes_uv, dtype=float)
    pwr = amplitudes_uv ** 2
    total = pwr.sum() + _EPS

    band_pwr = {}
    band_logf = {}
    band_logp = {}
    band_freqs = {}
    for bn, lo, hi in BAND_DEFS:
        bp, bf, _ = _compute_band_power(freqs, pwr, lo, hi)
        band_pwr[bn] = bp
        band_freqs[bn] = bf
        band_logf[bn] = np.log(bf + _EPS)
        band_logp[bn] = np.log(bp + _EPS)

    # --- ssn_B1: normalised spectral slope in B1 ---
    lf_b1, lp_b1 = band_logf['B1'], band_logp['B1']
    if len(lf_b1) > 1:
        slope_b1, _ = np.polyfit(lf_b1, lp_b1, 1)
    else:
        slope_b1 = 0.0
    total_band_powers = {bn: band_pwr[bn].sum() for bn in ('B1', 'B2', 'B3', 'B4')}
    geo_mean = np.mean([np.log(total_band_powers[bn] + _EPS) for bn in ('B1', 'B2', 'B3', 'B4')])
    ssn_b1 = slope_b1 / (abs(geo_mean) + _EPS)

    # --- lr_B2_B4: log-ratio of B2 to B4 band power ---
    lr_b2_b4 = np.log(total_band_powers['B2'] + _EPS) - np.log(total_band_powers['B4'] + _EPS)

    # --- spectral_spread ---
    normp = pwr / total
    freq_idx = np.arange(len(pwr), dtype=float)
    centroid = np.sum(freq_idx * normp)
    spectral_spread = float(np.sqrt(np.sum((freq_idx - centroid) ** 2 * normp)))

    # --- rolloff_norm_B3: normalised 85%-energy rolloff in B3 ---
    bp_b3 = band_pwr['B3']
    bf_b3 = band_freqs['B3']
    lo_b3, hi_b3 = 3_000_000, 30_000_000
    if len(bp_b3) > 0 and bp_b3.sum() > 0:
        cumsum = np.cumsum(bp_b3)
        idx = int(np.searchsorted(cumsum, 0.85 * bp_b3.sum()))
        idx = min(idx, len(bf_b3) - 1)
        rolloff_norm_b3 = float((bf_b3[idx] - lo_b3) / (hi_b3 - lo_b3 + _EPS))
    else:
        rolloff_norm_b3 = 0.5

    # --- curvature_B2: 2nd-order polynomial coeff in log-log B2 ---
    lf_b2, lp_b2 = band_logf['B2'], band_logp['B2']
    if len(lf_b2) > 2:
        c2, _, _ = np.polyfit(lf_b2, lp_b2, 2)
        curvature_b2 = float(c2)
    else:
        curvature_b2 = 0.0

    # --- spectral_flatness_global: geometric / arithmetic mean of power ---
    log_mean = np.mean(np.log(pwr + _EPS))
    spectral_flatness_global = float(np.exp(log_mean) / (pwr.mean() + _EPS))

    return {
        'ssn_B1': float(ssn_b1),
        'lr_B2_B4': float(lr_b2_b4),
        'spectral_spread': spectral_spread,
        'rolloff_norm_B3': rolloff_norm_b3,
        'curvature_B2': curvature_b2,
        'spectral_flatness_global': spectral_flatness_global,
    }


class OnOffScorer:
    """Weighted rule-based ON/OFF scorer for EMI spectrums.

    Parameters
    ----------
    threshold : float
        Decision boundary on the normalised weighted score [0, 1].
        Scores >= threshold are classified ON.  Default 0.35.
    flat_threshold : float
        Spectral-flatness value above which the spectrum is flagged as
        a flat line.  Flat-line spectrums are inherently ambiguous for
        shape-based features.  Default 0.40.
    medians : dict[str, float] | None
        Per-feature median thresholds.  If None, uses the global defaults
        learned from the 272-spectrum training set.  Override with
        site-specific medians once enough labeled data is available.
    feature_params : list[dict] | None
        List of dicts with keys ``name``, ``direction``, ``weight``.
        If None, uses the default 5-feature system.
    strict_validation : bool
        If True (default), raise ``ValueError`` on fatal spectrum issues
        (length mismatch, empty arrays) and emit ``warnings.warn`` for
        non-fatal issues (bin count, frequency range).  If False,
        validation warnings are still collected in the result dict but
        no warnings are emitted.
    """

    def __init__(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        flat_threshold: float = _DEFAULT_FLAT_THRESHOLD,
        medians: dict[str, float] | None = None,
        feature_params: list[dict] | None = None,
        strict_validation: bool = True,
    ):
        self.threshold = threshold
        self.flat_threshold = flat_threshold
        self.medians = dict(medians) if medians is not None else dict(_DEFAULT_MEDIANS)
        self.feature_params = list(feature_params) if feature_params is not None else list(_DEFAULT_FEATURES)
        self._max_score = sum(fp['weight'] for fp in self.feature_params)
        self.strict_validation = strict_validation

    def score(self, freqs: np.ndarray, amplitudes_uv: np.ndarray) -> dict:
        """Score a single spectrum.

        Parameters
        ----------
        freqs : array of shape (N,)
            Frequency values in Hz.
        amplitudes_uv : array of shape (N,)
            Amplitude values in µV.

        Returns
        -------
        dict with keys:
            on_status       : bool   — True if predicted ON
            score           : float  — normalised weighted score in [0, 1]
            is_flat_line    : bool   — True if spectrum is flagged as flat
            threshold_used  : float  — the decision threshold applied
            features        : dict   — feature name → computed value
            votes           : dict   — feature name → vote (0 or 1)
            warnings        : list[dict] — validation issues (empty if clean)
        """
        freqs = np.asarray(freqs, dtype=float)
        amplitudes_uv = np.asarray(amplitudes_uv, dtype=float)

        validation_warnings = validate_spectrum(freqs, amplitudes_uv)

        fatal_codes = {'length_mismatch'}
        fatal = [w for w in validation_warnings if w['code'] in fatal_codes]
        if fatal and self.strict_validation:
            raise ValueError(fatal[0]['message'])

        if self.strict_validation:
            for w in validation_warnings:
                if w['code'] not in fatal_codes:
                    warnings.warn(w['message'], stacklevel=2)

        feat_values = compute_features(freqs, amplitudes_uv)
        is_flat = feat_values['spectral_flatness_global'] >= self.flat_threshold

        votes = {}
        weighted_sum = 0.0
        for fp in self.feature_params:
            name = fp['name']
            val = feat_values[name]
            median = self.medians[name]
            if fp['direction'] == 'higher':
                vote = 1 if val >= median else 0
            else:
                vote = 1 if val < median else 0
            if np.isnan(val):
                vote = 0
            votes[name] = vote
            weighted_sum += vote * fp['weight']

        normalised_score = weighted_sum / (self._max_score + _EPS)
        on_status = bool(normalised_score >= self.threshold)

        return {
            'on_status': on_status,
            'score': round(float(normalised_score), 6),
            'is_flat_line': is_flat,
            'threshold_used': self.threshold,
            'features': {k: v for k, v in feat_values.items() if k != 'spectral_flatness_global'},
            'votes': votes,
            'spectral_flatness': feat_values['spectral_flatness_global'],
            'n_bins': len(freqs),
            'warnings': validation_warnings,
        }

    def score_batch(self, spectrums: list[tuple[np.ndarray, np.ndarray]]) -> list[dict]:
        """Score multiple spectrums.

        Parameters
        ----------
        spectrums : list of (freqs, amplitudes_uv) tuples.

        Returns
        -------
        list of result dicts (same format as ``score``).
        """
        return [self.score(f, a) for f, a in spectrums]
