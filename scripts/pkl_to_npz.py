#!/usr/bin/env python
"""
Convert numpy 2.0 pkl files (saved with joblib) to numpy 1.0-compatible npz files.
This script should be run in a numpy 2.0 environment.

Usage:
    python convert_pkl_to_npz.py <input_pkl> <output_npz>
    python convert_pkl_to_npz.py <input_pkl>  # Output will be input_pkl.npz
    python convert_pkl_to_npz.py --batch <input_dir>  # Convert all .pkl files in directory
"""

import argparse
import joblib
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_pkl(pkl_path):
    """Load pickle file saved with joblib.dump()"""
    data = joblib.load(pkl_path)
    return data


def save_npz(data, npz_path):
    """Save data to npz format compatible with numpy 1.0+"""
    if isinstance(data, dict):
        # If data is a dictionary, save all items
        np.savez(npz_path, **data)
    else:
        # If data is a single array or other type, wrap it in a dictionary
        np.savez(npz_path, data=data)
    print(f"Saved to: {npz_path}")


def convert_single_file(input_pkl, output_npz=None):
    """Convert a single pkl file to npz"""
    input_pkl = Path(input_pkl)

    if not input_pkl.exists():
        raise FileNotFoundError(f"Input file not found: {input_pkl}")

    if output_npz is None:
        # Default output path: replace .pkl with .npz
        output_npz = input_pkl.with_suffix('.npz')
    else:
        output_npz = Path(output_npz)

    print(f"Loading: {input_pkl}")
    data = load_pkl(input_pkl)

    # Print data structure for debugging
    if isinstance(data, dict):
        print(f"Data contains {len(data)} keys: {list(data.keys())}")
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape} {value.dtype}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with {len(value)} keys")
            else:
                print(f"  {key}: {type(value)}")
    else:
        print(f"Data type: {type(data)}")

    save_npz(data, output_npz)
    return output_npz


def convert_batch(input_dir, output_dir=None):
    """Convert all .pkl files in a directory"""
    input_dir = Path(input_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    # Find all .pkl files
    pkl_files = list(input_dir.glob("*.pkl"))

    if len(pkl_files) == 0:
        print(f"No .pkl files found in {input_dir}")
        return

    print(f"Found {len(pkl_files)} .pkl files")

    # Setup output directory
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Convert each file
    for pkl_file in tqdm(pkl_files, desc="Converting"):
        try:
            output_npz = output_dir / pkl_file.with_suffix('.npz').name
            convert_single_file(pkl_file, output_npz)
        except Exception as e:
            print(f"Error converting {pkl_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert numpy 2.0 pkl files to numpy 1.0-compatible npz files"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input .pkl file or directory (with --batch)"
    )
    parser.add_argument(
        "output",
        type=str,
        nargs="?",
        default=None,
        help="Output .npz file or directory (optional)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Convert all .pkl files in input directory"
    )

    args = parser.parse_args()

    if args.batch:
        convert_batch(args.input, args.output)
    else:
        convert_single_file(args.input, args.output)

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
