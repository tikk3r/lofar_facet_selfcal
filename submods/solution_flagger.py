#!/usr/bin/env python3

__author__ = "Roland Timmerman, Jurjen de Jong (jurjendejong@strw.leidenuniv.nl)"

from argparse import ArgumentParser, Namespace
from typing import Union, Sequence, Literal

import numpy as np
import pandas as pd
import tables
from losoto.h5parm import h5parm
from scipy.stats import circstd
from scipy.optimize import leastsq
from losoto.lib_operations import reorderAxes

def make_utf8(inp) -> str:
    """
    Convert input to UTF-8 string.

    Args:
        inp: String or bytes input.

    Returns:
        UTF-8 encoded string.
    """
    try:
        inp = inp.decode('utf8')
        return inp
    except (UnicodeDecodeError, AttributeError):
        return inp


def sigmoid(val: Union[float, np.ndarray],
            threshold: float = 0,
            sigma: float = 1.0,
            inverted: bool = False
            ) -> Union[float, np.ndarray]:
    """
    Apply sigmoid.

    Args:
        val: Input value or array.
        threshold: Midpoint (threshold) of the sigmoid in degrees.
        sigma: Controls the steepness of the sigmoid.
        inverted: If True, invert the sigmoid (monotonically decreasing).

    Returns:
        Sigmoid-transformed value(s).
    """
    sign = 1 if inverted else -1
    return 1 / (1 + np.exp(sign * (val - threshold) / sigma))


def parabola(x: Union[float, np.ndarray],
             p: Sequence[float] | None = None,
             a: float | None = None,
             b: float = 0.0,
             c: float = 0.0,
             ) -> Union[float, np.ndarray]:
    """
    Evaluate a quadratic function a*x^2 + b*x + c.

    Either provide coefficients as a sequence `p = (a, b, c)`
    or pass `a`, `b`, and `c` explicitly.

    Args:
        x: Input value(s).
        p: Sequence of coefficients [a, b, c].
        a: Quadratic coefficient (used if `p` is not given).
        b: Linear coefficient (default: 0.0).
        c: Constant term (default: 0.0).

    Returns:
        Evaluated parabola value(s).
    """
    if p is not None:
        a_, b_, c_ = p
    elif a is not None:
        a_, b_, c_ = a, b, c
    else:
        raise ValueError("Provide either p=(a, b, c) or at least a (with optional b, c).")

    return a_ * np.asarray(x) ** 2 + b_ * np.asarray(x) + c_


def get_chi(y: Union[float, np.ndarray],
            model_y: Union[float, np.ndarray]
            ) -> Union[float, np.ndarray]:
    """
    Compute wrapped residuals between observed and model values.

    The difference (y - model_y) is wrapped into the range [-π, π],
    ensuring residuals are bounded within this interval.

    Args:
        y: Observed value(s).
        model_y: Model-predicted value(s).

    Returns:
        Residuals wrapped to the interval [-π, π].
    """
    return normalise_phases(y - model_y)


def residuals(p: tuple[float, float, float],
              y: Union[float, np.ndarray],
              x: Union[float, np.ndarray]
              ) -> Union[float, np.ndarray]:
    """
    Get residuals between observed values and parabola model,
    wrapped into the range [-π, π].

    Args:
        p: Sequence of coefficients [a, b, c].
        y: Observed value(s).
        x: Input value(s).

    Returns:
        Residuals wrapped to [-π, π].
    """
    model_y = parabola(x=x, p=p)
    return get_chi(y, model_y)


def residuals_brute(a: float,
                    y: Union[float, np.ndarray],
                    x: Union[float, np.ndarray],
                    c: float = 0.0):
    """
    Get residuals for quadratic model with fixed constant term,
    wrapped into the range [-π, π].

    Args:
        a: Quadratic coefficient or array of coefficients.
        y: Observed value(s).
        x: Input value(s).
        c: Constant term.

    Returns:
        Sum of squared residuals for each coefficient in `a`.
    """
    model_y = parabola(x=x, a=a, c=c)
    chi = get_chi(y, model_y)
    return np.sum(chi**2, axis=1)


def subtract_parabola(phases: np.ndarray, multiplier: float = 1.0):
    """
    Subtract best-fit parabola from phase solutions.

    Args:
        phases: 2D array of phase values (time × interval).

    Returns:
        2D array of phases with parabola removed, wrapped to [-π, π].
    """
    sample_range = np.linspace(-5e-4,5e-4,512) * max(1.0, multiplier)
    new_phases = np.zeros(phases.shape)
    for idx, time_interval in enumerate(phases.T):
        p_0 = [0,0,0]

        if np.sum(~np.isnan(time_interval)) > 5:
            x = np.arange(-len(time_interval)//2,len(time_interval)//2)

            #Initial estimate of parameters
            p_0[2] = np.mean(time_interval[max(len(time_interval)//2-10, 0):len(time_interval)//2+10])
            chi2 = residuals_brute(sample_range[:,None], time_interval[None,~np.isnan(time_interval)], x[None,~np.isnan(time_interval)], p_0[2])
            p_0[0] = sample_range[np.argmin(chi2)]

            #Get proper fitted estimate
            p, cov = leastsq(residuals, p_0, args=(time_interval[~np.isnan(time_interval)], x[~np.isnan(time_interval)]))
            model_phases=p[0]*x**2 + p[1]*x + p[0]
            new_phases[:, idx] = time_interval - model_phases
        else:
            new_phases[:, idx] = time_interval
    norm_new_phases = normalise_phases(new_phases)

    return norm_new_phases


def calc_wraps(phase_solutions: np.ndarray,
               axes: list[str]
               ) -> float:
    """
    Calculate maximum number of phase wraps around 2pi from phase calibration solutions

    Args:
        phase_solutions: Phase calibration solutions
        axes: Axes names

    Returns: Number of phase wraps
    """

    # Get diffs
    phase_freq_diff = normalise_phases(np.diff(phase_solutions, axis=axes.index('freq')))

    # Get phase wrapping score
    freqsum = np.nansum(phase_freq_diff / (2*np.pi), axis=axes.index('freq'))
    wrap_count = np.max(np.abs(freqsum))

    return wrap_count


def normalise_phases(phase_sols: np.ndarray) -> np.ndarray:
    """
    Normalise phase solutions to the interval [-π, π].

    This ensures all phase values are wrapped into a consistent range,
    which is useful for comparing or averaging phases.

    Args:
        phase_sols: Array of phase solutions in radians.

    Returns:
        Array of phase solutions wrapped to [-π, π].
    """
    return (phase_sols - np.pi) % (2 * np.pi) - np.pi


def get_phase_noise_statistic(phase_sols: np.ndarray,
                              idx_ant: int,
                              freqs: np.ndarray,
                              wrap_count: float,
                              ) -> float:
    """
    Get phase noise statistic, which is the 90th percentile on the circular standard deviation of the phase solutions,
    after taking the 10% margin of the phase solutions, and subtracting parabola from the phase solutions.

    Args:
        phase_sols: Station referenced phase solutions
        idx_ant: Antenna index
        freqs: Frequency axis

    Returns: Phase score
    """

    phase_sols_sub = phase_sols[idx_ant]
    phase_freq_diff_sub = normalise_phases(phase_sols_sub[:-1, :] - phase_sols_sub[1:, :])

    # Slice with 10% margin on eiter side of the frequency band to discard for de-sloping
    slice_size = phase_freq_diff_sub.shape[0]//10

    delay_slope = np.nanmean(phase_freq_diff_sub[slice_size:-(slice_size+1)], axis=0)
    deslopvals = np.arange(-len(freqs) // 2, len(freqs) // 2)[:, None] * delay_slope[None, :]
    phases_desloped = normalise_phases(phase_sols_sub + deslopvals)
    phases_desloped = normalise_phases(phases_desloped - np.nanmean(phases_desloped[slice_size:-(slice_size+1)], axis=0))
    phases_desloped = subtract_parabola(phases_desloped, multiplier=wrap_count)
    phases_desloped = normalise_phases(phases_desloped - np.nanmean(phases_desloped[slice_size:-(slice_size+1)], axis=0))

    # Get phase score
    phase_noise = circstd(phases_desloped, axis=0, nan_policy='omit')
    #phase_score = np.nanpercentile(phase_noise, 90) / np.pi * 180 # Phase score based on 90th percentile of the phase noise
    phase_score = phase_noise / np.pi * 180 # Phase score based on 90th percentile of the phase noise

    return phase_score


def flag_on_phase_score(h5: str):
    """
    Get phase score, which is based on the circular standard deviation and wrap count

    Args:
        h5: h5parm solution file
    """

    with h5parm(h5, readonly=False) as hp:
        ss = hp.getSolset("sol000")
        st = ss.getSoltab("phase000")
        freqs = st.getAxisValues("freq")
        ants = st.getAxisValues("ant")
        phase_sols = st.getValues()[0]
        phase_weights = st.getValues(weight=True)[0]
        flagged_before = np.nansum(phase_weights)
        phase_sols = np.where(phase_weights > 0, phase_sols * phase_weights, np.nan)
        axes = st.getAxesNames()

        # Remove polarisation and direction axis
        had_pol = False
        if 'pol' in axes:
            phase_sols = np.take(phase_sols, 0, axis=axes.index('pol'))
            phase_weights = np.take(phase_weights, 0, axis=axes.index('pol'))
            had_pol = True

        had_dir = False
        if 'dir' in axes:
            phase_sols = np.take(phase_sols, 0, axis=axes.index('dir'))
            phase_weights = np.take(phase_weights, 0, axis=axes.index('dir'))
            had_dir = True

        # Reference solutions to first station
        ref_phase = np.take(phase_sols, [0], axis=axes.index('ant'))
        phase_sols -= ref_phase

        # Get wraps
        wrap_count = calc_wraps(phase_sols, axes)

        # De-slope
        old_axes_order = [a for a in axes if a not in ['pol', 'dir']]
        print(old_axes_order)
        phase_sols = reorderAxes(phase_sols, old_axes_order, ['ant', 'freq', 'time'])
        phase_weights = reorderAxes(phase_weights, old_axes_order, ['ant', 'freq', 'time'])

        for idx_ant, _ in enumerate(ants):
            time_noise = get_phase_noise_statistic(phase_sols, idx_ant, freqs, wrap_count)
            time_score = sigmoid(time_noise, 45, 10, True)
            bad_times = np.where(time_score < 0.5)
            phase_sols[idx_ant, :, bad_times, ...] = np.nan
            phase_weights[idx_ant, :, bad_times, ...] = 0

        phase_sols = reorderAxes(phase_sols, ['ant', 'freq', 'time'], old_axes_order)
        phase_weights = reorderAxes(phase_weights, ['ant', 'freq', 'time'], old_axes_order)
        if had_pol:
            phase_sols = phase_sols[..., np.newaxis]
            phase_weights = phase_weights[..., np.newaxis]
        if had_dir:
            phase_sols = phase_sols[..., np.newaxis]
            phase_weights = phase_weights[..., np.newaxis]
        st.setValues(phase_sols)
        st.setValues(phase_weights, weight=True)
        flagged_after = np.nansum(phase_weights)
        print(f"Change in flags after flagging: {(flagged_before - flagged_after) / flagged_after * 100}%")


def get_amp_score(h5: str) -> float:
    """
    Get amplitude score, which is based on the number of amplitude values above 1.5 and below 0.67, multiplied with
    (1-std), which is the standard deviation of the amplitude solutions.

    Args:
        h5: h5parm solution file

    Returns: Amplitude score between 0 and 1
    """

    with tables.open_file(h5) as H:
        amplitude_table = H.root.sol000.amplitude000
        axes = make_utf8(amplitude_table.val.attrs["AXES"]).split(',')

        # Remove polarisation axis and only select amplitude corrections where weights!=0
        if 'pol' in axes:
            weights = np.take(amplitude_table.weight[:], [0], axis=axes.index('pol'))!=0
            amplitude_sols = np.take(amplitude_table.val[:], [0], axis=axes.index('pol'))[weights]
        else:
            weights = amplitude_table.weight[:]!=0
            amplitude_sols = amplitude_table.val[:][weights]

    amplitude_std = np.std(amplitude_sols)

    # Score for large amplitude offsets
    amplitude_offset_score = 1 - len(amplitude_sols[(amplitude_sols > 1.5) | (amplitude_sols < 0.67)])/amplitude_sols.size

    # Final score
    amplitude_score = amplitude_offset_score * (1-amplitude_std)

    return amplitude_score
