"""
Dissimilarity module for the Prothon package.

Provides functions to compute local (per-feature) and global dissimilarity
between two ensemble representation matrices using Gaussian KDE and the Jensen–Shannon distance.
Also performs random sampling and applies the Mann–Whitney U test for statistical significance.
"""

import numpy as np
from scipy.stats import gaussian_kde, mannwhitneyu
from scipy.spatial.distance import jensenshannon

def random_sample(arr: np.ndarray, sample_size: int = 1000):
    """
    Randomly sample frames with replacement from an array.

    Parameters
    ----------
    arr : np.ndarray, shape=(n_frames, n_features)
    sample_size : int

    Returns
    -------
    sample : np.ndarray, shape=(sample_size, n_features)
    """
    n_frames = arr.shape[0]
    indices = np.random.randint(0, n_frames, sample_size)
    return arr[indices, :]

def estimate_pdf(arr: np.ndarray, x_min: float, x_max: float, x_num: int):
    """
    Estimate the probability density function for 1D data using Gaussian KDE.

    Parameters
    ----------
    arr : np.ndarray
        1D data.
    x_min : float
    x_max : float
    x_num : int

    Returns
    -------
    x : np.ndarray, shape=(x_num,)
        Discrete evaluation points.
    pdf : np.ndarray, shape=(x_num,)
        Estimated probability density values.
    """
    x = np.linspace(x_min, x_max, x_num)
    kde = gaussian_kde(arr, bw_method='silverman')
    pdf = kde(x)
    return x, pdf

def jsd_local(ensemble1: np.ndarray, ensemble2: np.ndarray, x_min: float, x_max: float, x_num: int):
    """
    Compute the local Jensen–Shannon distance (JSD) for each feature between two ensemble representations.

    Parameters
    ----------
    ensemble1, ensemble2 : np.ndarray, shape=(n_frames, n_features)
    x_min : float
    x_max : float
    x_num : int

    Returns
    -------
    jsd_vals : np.ndarray, shape=(n_features,)
    """
    n_features = ensemble1.shape[1]
    jsd_vals = np.zeros(n_features)
    for i in range(n_features):
        _, pdf1 = estimate_pdf(ensemble1[:, i], x_min, x_max, x_num)
        _, pdf2 = estimate_pdf(ensemble2[:, i], x_min, x_max, x_num)
        jsd = jensenshannon(pdf1, pdf2, base=2)
        jsd_vals[i] = 0.0 if np.isnan(jsd) or np.isinf(jsd) else jsd
    return jsd_vals

def dissimilarity(ref_rep: np.ndarray, rep: np.ndarray, x_min: float, x_max: float, x_num: int = 100, s_num: int = 5):
    """
    Calculate the dissimilarity between two ensemble representations.
    
    The function computes inter-ensemble and intra-ensemble JSDs via random sampling,
    applies the Mann–Whitney U test, and then calculates local and global dissimilarity.

    Parameters
    ----------
    ref_rep, rep : np.ndarray, shape=(n_frames, n_features)
    x_min : float
    x_max : float
    x_num : int, optional
    s_num : int, optional

    Returns
    -------
    global_diss : float
        Averaged (global) dissimilarity.
    local_diss : np.ndarray, shape=(n_features,)
        Dissimilarity per feature.
    p_value : float
        p-value from the Mann–Whitney U test.
    """
    inter_jsd = []
    intra_jsd = []
    for _ in range(s_num):
        sample1 = random_sample(ref_rep)
        for _ in range(s_num):
            sample2 = random_sample(rep)
            inter_jsd.append(jsd_local(sample1, sample2, x_min, x_max, x_num))
    inter_jsd = np.stack(inter_jsd, axis=0)
    for arr in [ref_rep, rep]:
        for i in range(s_num):
            for j in range(i+1, s_num):
                sample_i = random_sample(arr)
                sample_j = random_sample(arr)
                intra_jsd.append(jsd_local(sample_i, sample_j, x_min, x_max, x_num))
    intra_jsd = np.stack(intra_jsd, axis=0)
    _, p_value = mannwhitneyu(inter_jsd.flatten(), intra_jsd.flatten())
    local_diss = jsd_local(ref_rep, rep, x_min, x_max, x_num)
    local_diss[p_value >= 0.05] = 0.0
    global_diss = np.mean(local_diss)
    return global_diss, local_diss, p_value

