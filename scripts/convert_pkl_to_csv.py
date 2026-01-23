#!/usr/bin/env python3
import pickle
import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def load_pkl_data(pkl_path):
    """Load the PKL file containing retargeted motion data."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def convert_pkl_to_csv_format(pkl_data, target_dofs=None):
    """
    Convert PKL data to CSV format matching the expected structure.
    
    CSV format:
    - Columns 0-2: root position (x, y, z)
    - Columns 3-6: root rotation quaternion (x, y, z, w format)
    - Columns 7+: DOF positions (joint angles)
    
    The loader will later convert quaternion from [x,y,z,w] to [w,x,y,z]
    
    Args:
        pkl_data: Dictionary containing motion data
        target_dofs: Target number of DOFs (25 or 29). If None, uses input DOFs.
    """
    
    n_frames = len(pkl_data['root_pos'])
    n_input_dofs = pkl_data['dof_pos'].shape[1]
    
    # Determine output DOFs
    if target_dofs is None:
        n_output_dofs = n_input_dofs
    else:
        n_output_dofs = target_dofs
    
    print(f"Input DOFs: {n_input_dofs}, Output DOFs: {n_output_dofs}")
    
    # Total columns: 3 (pos) + 4 (quat) + n_output_dofs
    total_cols = 3 + 4 + n_output_dofs
    
    # Initialize the output array
    csv_data = np.zeros((n_frames, total_cols))
    
    # Fill in root position (columns 0-2)
    csv_data[:, 0:3] = pkl_data['root_pos']
    
    # Fill in root rotation quaternion (columns 3-6)
    # PKL format has quaternion as [x, y, z, w] which matches expected format
    csv_data[:, 3:7] = pkl_data['root_rot']
    
    # Handle DOF positions based on input/output size
    if n_input_dofs == n_output_dofs:
        # Direct copy
        csv_data[:, 7:7+n_output_dofs] = pkl_data['dof_pos']
    elif n_input_dofs == 25 and n_output_dofs == 29:
        # Pad from 25 to 29 DOFs (add 4 zeros for missing joints)
        csv_data[:, 7:32] = pkl_data['dof_pos']
        # Additional 4 DOFs padded with zeros (already initialized)
        print("Padded 25 DOFs to 29 DOFs (added 4 zero-valued DOFs)")
    elif n_input_dofs == 29 and n_output_dofs == 25:
        # Truncate from 29 to 25 DOFs (remove last 4 joints)
        csv_data[:, 7:32] = pkl_data['dof_pos'][:, :25]
        print("Truncated 29 DOFs to 25 DOFs (removed last 4 DOFs)")
    elif n_input_dofs == 53 and n_output_dofs == 41:
        # G1 with inspired hands: 53 joints → 41 actuators
        # The robot has coupled finger joints, so we only extract actuated joints

        # Body DOFs: legs (12) + waist (3) + left arm (7) = 22 DOFs
        csv_data[:, 7:7+22] = pkl_data['dof_pos'][:, :22]

        # Left hand: select 6 actuated joints from 12 (indices 22-33)
        # Actuated: thumb_yaw(22), thumb_pitch(23), index(26), middle(28), ring(30), pinky(32)
        left_hand_indices = [22, 23, 26, 28, 30, 32]
        csv_data[:, 7+22:7+28] = pkl_data['dof_pos'][:, left_hand_indices]

        # Right arm: 7 DOFs (indices 34-40)
        csv_data[:, 7+28:7+35] = pkl_data['dof_pos'][:, 34:41]

        # Right hand: select 6 actuated joints from 12 (indices 41-52)
        # Actuated: thumb_yaw(41), thumb_pitch(42), index(45), middle(47), ring(49), pinky(51)
        right_hand_indices = [41, 42, 45, 47, 49, 51]
        csv_data[:, 7+35:7+41] = pkl_data['dof_pos'][:, right_hand_indices]

        print("Converted 53 DOFs to 41 actuators (G1 with inspired hands)")
    else:
        print(f"Warning: Unusual DOF conversion from {n_input_dofs} to {n_output_dofs}")
        # Copy what we can
        min_dofs = min(n_input_dofs, n_output_dofs)
        csv_data[:, 7:7+min_dofs] = pkl_data['dof_pos'][:, :min_dofs]
    
    return csv_data


def save_to_csv(csv_data, output_path):
    """Save the converted data to CSV file."""
    # Create DataFrame
    df = pd.DataFrame(csv_data)
    
    # Save to CSV without header and index to match the format
    df.to_csv(output_path, header=False, index=False)
    print(f"Saved converted data to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert PKL motion file to CSV format')
    parser.add_argument('pkl_path', type=str, help='Path to input PKL file')
    parser.add_argument('-o', '--output', type=str, help='Output CSV file path (optional)')
    parser.add_argument('-d', '--dofs', type=int, choices=[25, 29, 41],
                        help='Target number of DOFs (25=hi, 29=g1, 41=g1_inspired). If not specified, keeps original DOFs.')
    
    args = parser.parse_args()
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        # Generate output path based on input filename
        pkl_path = Path(args.pkl_path)
        output_path = pkl_path.stem + '_converted.csv'
    
    # Load PKL data
    print(f"Loading PKL file: {args.pkl_path}")
    pkl_data = load_pkl_data(args.pkl_path)
    
    # Print data info
    n_input_dofs = pkl_data['dof_pos'].shape[1]
    print(f"Loaded data with {len(pkl_data['root_pos'])} frames")
    print(f"FPS: {pkl_data.get('fps', 'N/A')}")
    print(f"Input DOFs: {n_input_dofs}")
    
    # Validate DOF compatibility
    if n_input_dofs not in [25, 29, 53]:
        print(f"Warning: Unusual number of input DOFs ({n_input_dofs}). Expected 25, 29, or 53.")
    
    # Convert to CSV format
    print("Converting to CSV format...")
    csv_data = convert_pkl_to_csv_format(pkl_data, target_dofs=args.dofs)
    
    # Save to CSV
    save_to_csv(csv_data, output_path)
    
    # Print summary
    print(f"\nConversion complete!")
    print(f"Input: {args.pkl_path}")
    print(f"Output: {output_path}")
    print(f"Shape: {csv_data.shape}")


if __name__ == "__main__":
    main()