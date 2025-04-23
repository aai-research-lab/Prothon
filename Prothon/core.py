"""
Core module for the Prothon package

This module defines the Prothon class that computes the representation of
protein ensembles using different local structural properties and measures
the dissimilarity between ensembles using the Jensen–Shannon distance.
"""

import numpy as np
from mdtraj import load, load_dcd, compute_angles, compute_dihedrals, compute_distances, shrake_rupley
from itertools import combinations
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde, mannwhitneyu
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Prothon:
    """
    Class for representing and comparing protein ensembles.

    Parameters
    ----------
    traj_files : list or str
        A list/tuple of filenames or a glob pattern describing trajectory files.
    topology : str
        Path to the PDB file specifying the topology.
    output_file : str, optional
        Path to write output results.
    verbose : bool, optional
        If True, prints additional progress information.
    """

    def __init__(self, traj_files, topology, output_file=None, verbose=False):
        self.traj_files = traj_files
        self.topology = topology
        self.output_file = output_file
        self.verbose = verbose
        # representations will be computed later and stored as a list of numpy arrays
        self.ensembles = None

    def _load_ensemble(self, file):
        """Load a single trajectory using mdtraj (handle .dcd separately)."""
        if file.strip().lower().endswith('.dcd'):
            return load_dcd(file, self.topology)
        else:
            # attempt agnostic load for other formats
            return load(file, self.topology)

    def _compute_cbcn(self, traj):
        """
        Compute the C-beta contact number (CBCN) for a given mdtraj.Trajectory.

        Returns
        -------
        cbcn : np.ndarray, shape=(n_frames, n_residues)
            Array of CBCN values.
        """
        CONST = 50  # 1/nm
        r0 = 1      # cutoff in nm

        # Select C-beta atoms (note: Glycine lacks CB; adjust selection if needed)
        cb_indices = traj.topology.select("name == 'CB'")
        if self.verbose:
            print(f"Number of CB atoms: {len(cb_indices)}")
            
        # Define all valid pairs with sequence separation > 2
        cb_pairs = np.array(
            [(i, j) for (i, j) in combinations(cb_indices, 2)
             if abs(traj.topology.atom(i).residue.index - traj.topology.atom(j).residue.index) > 2]
        )

        # For each CB atom, calculate its contact number across frames
        cbcn = []
        for idx in cb_indices:
            # Select pairs that include the CB atom with index 'idx'
            pairs = cb_pairs[[idx in pair for pair in cb_pairs]]
            # Compute pairwise distances for these pairs
            dists = compute_distances(traj, pairs)
            # Compute contact number using a smooth cutoff
            contacts = np.sum(1.0 / (1 + np.exp(CONST * (dists - r0))), axis=1)
            cbcn.append(contacts)
        # Transpose so that rows correspond to frames and columns to residues
        return np.transpose(np.array(cbcn))

    def _compute_cacn(self, traj):
        """
        Compute the C-alpha contact number (CACN).

        Returns
        -------
        cacn : np.ndarray, shape=(n_frames, n_residues)
        """
        CONST = 50
        r0 = 1

        ca_indices = traj.topology.select("name == 'CA'")
        ca_pairs = np.array(
            [(i, j) for (i, j) in combinations(ca_indices, 2)
             if abs(traj.topology.atom(i).residue.index - traj.topology.atom(j).residue.index) > 2]
        )

        cacn = []
        for idx in ca_indices:
            pairs = ca_pairs[[idx in pair for pair in ca_pairs]]
            dists = compute_distances(traj, pairs)
            contacts = np.sum(1.0 / (1 + np.exp(CONST * (dists - r0))), axis=1)
            cacn.append(contacts)
        return np.transpose(np.array(cacn))

    def _compute_caba(self, traj):
        """
        Compute the virtual C-alpha - C-alpha bond angles (CABA).

        Returns
        -------
        caba : np.ndarray, shape=(n_frames, n_angles)
        """
        ca_indices = traj.topology.select("name == 'CA'")
        n_angles = len(ca_indices) - 2
        angle_indices = [(ca_indices[i], ca_indices[i+1], ca_indices[i+2]) for i in range(n_angles)]
        angles = compute_angles(traj, angle_indices)
        return angles

    def _compute_cata(self, traj):
        """
        Compute the virtual C-alpha - C-alpha torsion (dihedral) angles (CATA).

        Returns
        -------
        cata : np.ndarray, shape=(n_frames, n_dihedrals)
        """
        ca_indices = traj.topology.select("name == 'CA'")
        n_dihedrals = len(ca_indices) - 3
        dihedral_indices = [(ca_indices[i], ca_indices[i+1], ca_indices[i+2], ca_indices[i+3])
                            for i in range(n_dihedrals)]
        dihedrals = compute_dihedrals(traj, dihedral_indices)
        return dihedrals

    def _compute_sasa(self, traj):
        """
        Compute the solvent accessible surface area (SASA) per residue.

        Returns
        -------
        sasa : np.ndarray, shape=(n_frames, n_residues)
        """
        sasa = shrake_rupley(traj, mode='residue')
        return sasa

    def compute_ensemble_representation(self, measure='cbcn'):
        """
        Compute the representation of all ensembles using a local structural measure.

        Parameters
        ----------
        measure : str, {'cbcn', 'cacn', 'caba', 'cata', 'sasa'}
            The chosen local structural property.

        Returns
        -------
        ensembles : list of np.ndarray
            A list containing the representation arrays for each trajectory.
        """
        measure = measure.lower()
        representations = []
        if self.verbose:
            print(f"Representing {len(self.traj_files)} ensemble(s) using measure: {measure}")

        for file in self.traj_files:
            if self.verbose:
                print(f"Loading trajectory: {file}")
            traj = self._load_ensemble(file)
            if measure == 'cbcn':
                rep = self._compute_cbcn(traj)
            elif measure == 'cacn':
                rep = self._compute_cacn(traj)
            elif measure == 'caba':
                rep = self._compute_caba(traj)
            elif measure == 'cata':
                rep = self._compute_cata(traj)
            elif measure == 'sasa':
                rep = self._compute_sasa(traj)
            else:
                raise ValueError(f"Invalid measure: {measure}")
            representations.append(rep)
        self.ensembles = representations
        return representations

    def _random_sample(self, arr, sample_size=1000):
        """
        Randomly sample 'sample_size' frames (with replacement) from the array.

        Parameters
        ----------
        arr : np.ndarray, shape=(n_frames, n_features)
        sample_size : int, optional

        Returns
        -------
        sample : np.ndarray, shape=(sample_size, n_features)
        """
        n_frames = arr.shape[0]
        indices = np.random.randint(0, n_frames, sample_size)
        return arr[indices, :]

    def _estimate_pdf(self, arr, x_min, x_max, x_num):
        """
        Estimate probability density using Gaussian kernel density estimation.

        Parameters
        ----------
        arr : np.ndarray, shape=(n_samples, )
        x_min : float
        x_max : float
        x_num : int

        Returns
        -------
        x : np.ndarray, shape=(x_num,)
            Discrete points over the range.
        pdf : np.ndarray, shape=(x_num,)
            Estimated density values.
        """
        x = np.linspace(x_min, x_max, x_num)
        kde = gaussian_kde(arr, bw_method='silverman')
        pdf = kde(x)
        return x, pdf

    def _jsd_local(self, ensemble1, ensemble2, x_min, x_max, x_num):
        """
        Calculate the Jensen-Shannon distance between two ensemble representations
        for each local property.

        Parameters
        ----------
        ensemble1, ensemble2 : np.ndarray, shape=(n_frames, n_features)
        x_min : float
        x_max : float
        x_num : int

        Returns
        -------
        jsd : np.ndarray
            Jensen-Shannon distance for each feature (local dissimilarity).
        """
        n_features = ensemble1.shape[1]
        jsd_vals = np.zeros(n_features)
        for i in range(n_features):
            _, pdf1 = self._estimate_pdf(ensemble1[:, i], x_min, x_max, x_num)
            _, pdf2 = self._estimate_pdf(ensemble2[:, i], x_min, x_max, x_num)
            # Compute Jensen-Shannon distance (the function returns the square root of the divergence)
            jsd = jensenshannon(pdf1, pdf2, base=2)
            if np.isinf(jsd) or np.isnan(jsd):
                jsd = 0.0
            jsd_vals[i] = jsd
        return jsd_vals

    def dissimilarity(self, ensemble1, ensemble2, x_min, x_max, x_num=100, s_num=5):
        """
        Calculate the dissimilarity between two ensemble representations.

        Both local and global dissimilarity are computed, and a statistical
        test (Mann-Whitney U) is performed to mask non-significant local differences.

        Parameters
        ----------
        ensemble1, ensemble2 : np.ndarray, shape=(n_frames, n_features)
        x_min : float
            Minimum value for KDE.
        x_max : float
            Maximum value for KDE.
        x_num : int, optional
            Number of discrete points for density estimation.
        s_num : int, optional
            Number of random samples for statistical significance testing.

        Returns
        -------
        tuple of (global_dissimilarity, local_dissimilarity, p_value)
            global_dissimilarity : float (average over significant local differences)
            local_dissimilarity : np.ndarray of shape (n_features,)
            p_value : float from the Mann-Whitney U test.
        """
        # Generate random samples for intra- and inter-ensemble comparisons
        inter_jsd = []
        intra_jsd = []
        for i in range(s_num):
            sample1 = self._random_sample(ensemble1)
            for j in range(s_num):
                sample2 = self._random_sample(ensemble2)
                inter_jsd.append(self._jsd_local(sample1, sample2, x_min, x_max, x_num))
        inter_jsd = np.stack(inter_jsd, axis=0)

        # Intra-ensemble: compare within each ensemble
        for arr in [ensemble1, ensemble2]:
            for i in range(s_num):
                for j in range(i + 1, s_num):
                    sample_i = self._random_sample(arr)
                    sample_j = self._random_sample(arr)
                    intra_jsd.append(self._jsd_local(sample_i, sample_j, x_min, x_max, x_num))
        intra_jsd = np.stack(intra_jsd, axis=0)

        # Statistical significance test (Mann-Whitney U test)
        _, p_value = mannwhitneyu(inter_jsd.flatten(), intra_jsd.flatten())
        local_diss = self._jsd_local(ensemble1, ensemble2, x_min, x_max, x_num)
        # Only consider local differences with p < 0.05 as significant
        local_diss[p_value >= 0.05] = 0.0
        global_diss = np.mean(local_diss)
        return global_diss, local_diss, p_value

    def compare_ensembles(self, measure='cbcn', ref=0, x_num=100, s_num=5):
        """
        Compute and compare ensemble dissimilarity from a reference ensemble.

        Parameters
        ----------
        measure : str, optional
            The chosen local structural measure ('cbcn', 'cacn', 'caba', 'cata', or 'sasa').
        ref : int, optional
            The index of the reference ensemble. Default is 0.
        x_num : int, optional
            Number of discrete points for density estimation.
        s_num : int, optional
            Number of samples for statistical significance testing.

        Returns
        -------
        results : list of dict
            Each dictionary contains:
                - 'ensemble_index': index compared to ref (ref is skipped)
                - 'global_dissimilarity': average dissimilarity value,
                - 'local_dissimilarity': array of per–residue/local dissimilarity,
                - 'p_value': p-value from statistical testing.
        """
        # First, compute ensemble representations if not already done.
        reps = self.compute_ensemble_representation(measure)
        # Determine overall min and max from all ensembles if not provided
        overall_min = min(np.min(rep) for rep in reps)
        overall_max = max(np.max(rep) for rep in reps)
        if self.verbose:
            print(f"Using KDE range: x_min={overall_min}, x_max={overall_max}")

        results = []
        ref_rep = reps[ref]
        for i, rep in enumerate(reps):
            if i == ref:
                continue
            global_d, local_d, p_val = self.dissimilarity(ref_rep, rep, overall_min, overall_max, x_num, s_num)
            results.append({
                'ensemble_index': i,
                'global_dissimilarity': global_d,
                'local_dissimilarity': local_d,
                'p_value': p_val
            })
        return results

