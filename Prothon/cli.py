#!/usr/bin/env python
"""
Command-line interface for the Prothon package.

Example usage:
    prothon -traj "Q99.dcd,Q95.dcd,Q85.dcd,Q75.dcd" -top topology.pdb -m cbcn,sasa -o my_outputs -d pca,mds,tsne
"""

import argparse
import json
from Prothon import Prothon

def convert_results(results):
    """
    Recursively convert NumPy arrays in the results dictionary to lists for JSON serialization.
    """
    if isinstance(results, dict):
        return {k: convert_results(v) for k, v in results.items()}
    elif isinstance(results, list):
        return [convert_results(item) for item in results]
    elif hasattr(results, "tolist"):
        return results.tolist()
    else:
        return results

def main():
    parser = argparse.ArgumentParser(
        description="Prothon: Compare protein ensembles using local order parameters."
    )
    parser.add_argument(
        "-traj", "--trajectories", required=True,
        help="Comma-separated trajectory files, glob pattern, or list (e.g., \"Q99.dcd,Q95.dcd,Q85.dcd,Q75.dcd\")"
    )
    parser.add_argument(
        "-top", "--topology", required=True,
        help="Topology file in PDB format (e.g., topology.pdb)"
    )
    parser.add_argument(
        "-m", "--methods", default="cbcn",
        help="Comma-separated list of local structural measures (e.g., \"cbcn,sasa\")"
    )
    parser.add_argument(
        "-r", "--ref", type=int, default=0,
        help="Reference ensemble index (default: 0)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Root output directory. If not provided, each measure creates its own directory (e.g., cbcn_output)."
    )
    parser.add_argument(
        "-d", "--dimred", default="pca,mds,tsne",
        help="Comma-separated list of dimensionality reduction techniques (e.g., \"pca,mds,tsne\"). Use 'None' to disable."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Increase output verbosity."
    )
    args = parser.parse_args()

    dimred_val = None if args.dimred.strip().lower() == "none" else args.dimred
    prothon = Prothon(traj_files=args.trajectories, topology=args.topology,
                      output_dir=args.output, verbose=args.verbose)
    
    results = prothon.compare_ensembles(methods=args.methods, ref=args.ref, dimred=dimred_val)
    print(json.dumps(convert_results(results), indent=4))

if __name__ == "__main__":
    main()

