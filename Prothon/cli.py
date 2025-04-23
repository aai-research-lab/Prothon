#!/usr/bin/env python
"""
Command-line interface for the Prothon package.

Example usage:
    prothon -traj "traj1.dcd, traj2.dcd, traj3.dcd" -top top.pdb -m cbcn -o results.txt
"""

import argparse
import json
from Prothon import Prothon, load_trajectories

def main():
    parser = argparse.ArgumentParser(
        description="Prothon: Compare protein ensembles using local order parameters."
    )
    parser.add_argument(
        "-traj", "--trajectories", required=True,
        help="Comma-separated trajectory files, glob pattern, or list (e.g. \"traj1.dcd, traj2.dcd, traj3.dcd\")"
    )
    parser.add_argument(
        "-top", "--topology", required=True,
        help="Topology file in PDB format (e.g., top.pdb)"
    )
    parser.add_argument(
        "-m", "--method", default="cbcn", choices=["cbcn", "cacn", "caba", "cata", "sasa"],
        help="Local structural measure to use for ensemble representation (default: cbcn)"
    )
    parser.add_argument(
        "-r", "--ref", type=int, default=0,
        help="Reference ensemble index (default: 0)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output file to write the comparison results in JSON format."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Increase output verbosity."
    )
    args = parser.parse_args()

    # Process trajectories using the utility function:
    # (you can pass the same string to the Prothon class or pre‐process it)
    traj_input = args.trajectories.split(",") if "," in args.trajectories else args.trajectories

    # Create Prothon instance
    prothon = Prothon(traj_files=traj_input, topology=args.topology, verbose=args.verbose)

    # Compare ensembles using the chosen method and reference index.
    results = prothon.compare_ensembles(measure=args.method, ref=args.ref)

    # Print results to screen
    for res in results:
        print(f"Ensemble {res['ensemble_index']} vs Reference Ensemble:")
        print(f"  Global dissimilarity: {res['global_dissimilarity']:.4f}")
        print(f"  p-value: {res['p_value']}")
        print("  Local dissimilarity per residue/feature:")
        print(res['local_dissimilarity'])
        print("-" * 50)

    # Write to output file if provided
    if args.output:
        with open(args.output, "w") as fout:
            json.dump(results, fout, indent=4)
        if args.verbose:
            print(f"Results written to {args.output}")

if __name__ == "__main__":
    main()

