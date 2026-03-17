#!/usr/bin/env python3
"""
Extracts the 'gauss_mu' and 'gauss_chol' matrices from the Step 3 NPZ file
and exports them into a clean, human-readable JSON file optimized for 
a Python-based HPC environment.
"""

import argparse
import json
import os
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Extract HPC sampling matrices to JSON.")
    parser.add_argument("--npz", required=True, help="Path to input step3_mcmc.npz")
    parser.add_argument("--out", default="hpc_sampler.json", help="Output JSON file path")
    args = parser.parse_args()

    if not os.path.exists(args.npz):
        sys.exit(f"ERROR: File not found -> {args.npz}")

    print(f"Loading data from {args.npz}...")
    try:
        data = np.load(args.npz, allow_pickle=True)
        gauss_mu   = data['gauss_mu']
        gauss_chol = data['gauss_chol']
        
        # Try to grab parameter labels if they exist for human readability
        if 'param_labels' in data:
            labels = data['param_labels'].tolist()
        else:
            labels = [f"param_{i}" for i in range(len(gauss_mu))]
            
    except KeyError as e:
        sys.exit(f"ERROR: Required array missing from NPZ file: {e}")
    except Exception as e:
        sys.exit(f"ERROR: Could not read NPZ file: {e}")

    # Build a clean dictionary
    # .tolist() is used to convert numpy arrays into standard Python nested lists
    # which the 'json' module can serialize natively.
    hpc_data = {
        "description": "Fission Yield HPC Sampler Parameters (Gaussian Approx)",
        "n_params": len(gauss_mu),
        "param_labels": labels,
        "gauss_mu": gauss_mu.tolist(),
        "gauss_chol": gauss_chol.tolist()
    }

    print(f"Writing to {args.out}...")
    with open(args.out, 'w') as f:
        # indent=2 makes it beautifully formatted and human-readable
        json.dump(hpc_data, f, indent=2)

    print("\nExtraction Complete!")
    print(f"Output saved to: {args.out}")

if __name__ == "__main__":
    main()
