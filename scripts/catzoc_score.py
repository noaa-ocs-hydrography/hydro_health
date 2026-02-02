# -*- coding: utf-8 -*-
"""
score.py

grice 20190725
V 0.0.1 20190725

Utilities for calculating metrics for scoring the quality of data.
"""
from datetime import date as _date
import math as _math


def catzoc(metadata: dict) -> int:
    """
    Return an enumeration representing the catzoc assocaited with the provided
    metrics.

    The enumeration (catzoc : value) is as follows:
        A1 : 1
        A2 : 2
        B  : 3
        C  : 4
        D  : 5
        U  : 6

    The provided metadata is expected to contain the following metadata values:

    """
    s = supersession(metadata)
    if s > 80:
        return 1
    elif s > 60:
        return 2
    elif s > 40:
        return 3
    elif s > 20:
        return 4
    else:
        return 5


def supersession(metadata: dict) -> float:
    """
    Return the superssion score as defined in Wyllie 2017 at US Hydro for the
    catzoc score.
    """

    required_entries = ['feat_detect', 'complete_coverage', 'horiz_uncert_fixed', 'vert_uncert_fixed',
                        'horiz_uncert_vari', 'vert_uncert_vari']

    for required_entry in required_entries:
        if required_entry not in metadata:
            survey_name = metadata['from_filename']
            raise ValueError(
                f'Metadata for survey "{survey_name}" does not contain an entry for "{required_entry}" and is thus not available to score')

    feat_score = _get_feature_detection(metadata)
    cov_score = _get_coverage(metadata)
    horz_score = _get_horizontal_uncertainty(metadata)
    vert_score = _get_vertical_uncertainty(metadata)
    score = min(feat_score, cov_score, horz_score, vert_score)
    if metadata['interpolated']:
        score -= 0.01
    return score


def _get_feature_detection(metadata: dict) -> float:
    """
    Determine the feature detection capability from the ability to detect
    features, detect the least depth, and the size of the feature.
    """
    least_depth = metadata['feat_detect'] and metadata['feat_least_depth']
    # size_okay = 'feat_size' in metadata and float(metadata['feat_size']) <= 2
    if metadata['feat_detect'] and least_depth:  # and size_okay:
        return 100
    else:
        return 60


def _get_coverage(metadata: dict) -> float:
    """
    Determine the coverage score and return.
    """
    if metadata['complete_coverage']:
        return 100
    else:
        return 60


def _get_horizontal_uncertainty(metadata: dict) -> float:
    """
    Determine the horizontal uncertainty score and return.
    """
    h_fix = float(metadata['horiz_uncert_fixed'])
    h_var = float(metadata['horiz_uncert_vari'])
    if h_fix <= 5 and h_var <= 0.05:
        s = 100
    elif h_fix <= 20:
        s = 80
    elif h_fix <= 50:
        s = 60
    elif h_fix <= 500:
        s = 40
    else:
        s = 20
    return s


def _get_vertical_uncertainty(metadata: dict) -> float:
    """
    Determine the vertical uncertainty score and return.
    """
    v_fix = float(metadata['vert_uncert_fixed'])
    v_var = float(metadata['vert_uncert_vari'])
    if v_fix <= 0.5 and v_var <= 0.01:
        s = 100
    elif v_fix <= 1 and v_var <= 0.02:
        s = 80
    elif v_fix <= 2 and v_var <= 0.05:
        s = 40
    else:
        s = 20
    return s


def decay(metadata: dict, date: _date, alpha: float = 0.022) -> float:
    """
    Return the decayed supersession_score.
    """
    sd = metadata['end_date' if 'end_date' in metadata else 'start_date']
    ss = float(metadata['supersession_score'])
    dt = date - sd
    days = dt.days + dt.seconds / (24 * 60 * 60)
    years = days / 365
    ds = ss * _math.exp(-alpha * years)
    if ds < 1:
        raise ValueError(f"Decay Score less than 1: end_date {sd}; supersession_score {ss}; date_delta {dt}, days {days}; years {years}; constant_alpha {alpha}")
    else:
        return ds
