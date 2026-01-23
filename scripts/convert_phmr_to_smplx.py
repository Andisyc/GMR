#!/usr/bin/env python3
import argparse
import numpy as np
import joblib
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import math


def convert_rotmat_to_axis_angle(rotmats):
    """
    Convert rotation matrices to axis-angle representation.
    
    Args:
        rotmats: (N, K, 3, 3) rotation matrices
        
    Returns:
        axis_angles: (N, K*3) axis-angle vectors
    """
    N, K, _, _ = rotmats.shape
    axis_angles = []
    
    for i in range(N):
        frame_aa = []
        for j in range(K):
            rot = R.from_matrix(rotmats[i, j])
            aa = rot.as_rotvec()
            frame_aa.append(aa)
        axis_angles.append(np.concatenate(frame_aa))
    
    return np.array(axis_angles)


def convert_phmr_to_smplx(input_path, output_path, person_id=None, fps=30.0, use_world_coords=True):
    """
    Convert PHMR estimation results to AMASS-compatible SMPL-X format.
    
    Args:
        input_path: Path to PHMR results.pkl file
        output_path: Path to save converted .npz file
        person_id: ID of person to extract (default: first person)
        fps: Frame rate for the output (default: 30.0)
        use_world_coords: Use world coordinates if available (default: True)
    """
    # Load PHMR data
    print(f"Loading PHMR data from {input_path}...")
    data = joblib.load(input_path)
    
    # Get person data
    people = data['people']
    if person_id is None:
        person_id = list(people.keys())[0]
        print(f"No person_id specified, using person {person_id}")
    
    if person_id not in people:
        raise ValueError(f"Person {person_id} not found. Available: {list(people.keys())}")
    
    person = people[person_id]
    
    # Extract SMPL-X parameters
    print("Extracting SMPL-X parameters...")
    
    # Check if we should use world coordinates
    if use_world_coords and 'smplx_world' in person:
        print("Using world coordinates...")
        world_data = person['smplx_world']
        
        # Get pose from world data (already in axis-angle format)
        poses_world = world_data['pose']  # (N, 165)
        N = poses_world.shape[0]
        
        # Extract components from the full pose
        root_orient_orig = poses_world[:, :3]  # First 3 values
        pose_body = poses_world[:, 3:66]  # Next 63 values (21 joints * 3)
        
        # Apply 90-degree rotation to fix orientation
        # X-axis rotation (correct rotation)
        # Create rotation of 90 degrees around X-axis
        x_rotation = R.from_euler('x', 90, degrees=True)
        
        # Z-axis rotation (commented out)
        # z_rotation = R.from_euler('z', 90, degrees=True)
        
        # Apply X-axis rotation to root orientation
        root_orient = []
        for i in range(N):
            # Convert current root orientation to rotation
            current_rot = R.from_rotvec(root_orient_orig[i])
            # Apply X-axis rotation: new_rot = x_rotation * current_rot
            new_rot = x_rotation * current_rot
            # Convert back to axis-angle
            root_orient.append(new_rot.as_rotvec())
        root_orient = np.array(root_orient)
        
        # Get translation and rotate it as well
        trans_orig = world_data['trans']  # (N, 3)
        # Apply X-axis rotation to translation
        trans = np.array([x_rotation.apply(t) for t in trans_orig])
        
        # Get betas from world data
        betas_10 = world_data['shape']  # (N, 10)
        
    else:
        # Original camera-space conversion
        print("Using camera coordinates...")
        # Get pose data - convert rotation matrices to axis-angle
        smplx_pose = person['smplx_pose']  # (N, 22, 3, 3)
        N = smplx_pose.shape[0]
        
        # The first rotation is global orientation (root)
        root_orient = convert_rotmat_to_axis_angle(smplx_pose[:, 0:1, :, :])  # (N, 3)
        
        # The next 21 rotations are body pose (21 joints * 3 = 63)
        pose_body = convert_rotmat_to_axis_angle(smplx_pose[:, 1:22, :, :])  # (N, 63)
        
        # Get translation
        trans = person['smplx_transl']  # (N, 3)
        
        # Get betas
        betas_10 = person['smplx_betas']  # (N, 10)
    # Use the mean across all frames for consistent body shape
    betas = np.zeros(16)
    betas[:10] = np.mean(betas_10, axis=0)
    
    # Create full poses array (165 dimensions as in AMASS)
    # 3 (root) + 63 (body) + 90 (hands) + 3 (jaw) + 6 (eyes) = 165
    poses = np.zeros((N, 165))
    poses[:, :3] = root_orient
    poses[:, 3:66] = pose_body
    # Extract or set hand/jaw/eye poses if using world coords
    if use_world_coords and 'smplx_world' in person:
        poses_world = world_data['pose']
        if poses_world.shape[1] >= 165:
            poses[:, 66:] = poses_world[:, 66:]  # Copy hand, jaw, eye poses if available
    
    # Create hand pose arrays
    pose_hand = np.zeros((N, 90))  # 30 joints * 3 for both hands
    
    # Jaw and eye poses
    pose_jaw = np.zeros((N, 3))
    pose_eye = np.zeros((N, 6))
    
    # Prepare output dictionary matching AMASS format
    output_dict = {
        'gender': 'female',  # Default to female as specified
        'surface_model_type': 'smplx',
        'mocap_frame_rate': fps,
        'mocap_time_length': N / fps,
        'trans': trans.astype(np.float64),
        'poses': poses.astype(np.float64),
        'betas': betas.astype(np.float64),
        'root_orient': root_orient.astype(np.float64),
        'pose_body': pose_body.astype(np.float64),
        'pose_hand': pose_hand.astype(np.float64),
        'pose_jaw': pose_jaw.astype(np.float64),
        'pose_eye': pose_eye.astype(np.float64),
        'num_betas': 16,
    }
    
    # Save as .npz file
    print(f"Saving converted data to {output_path}...")
    np.savez(output_path, **output_dict)
    
    print(f"Conversion complete!")
    print(f"  - Frames: {N}")
    print(f"  - Duration: {N/fps:.2f} seconds")
    print(f"  - Person ID: {person_id}")
    

def main():
    parser = argparse.ArgumentParser(description='Convert PHMR estimation results to AMASS-compatible SMPL-X format')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to PHMR results.pkl file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save converted .npz file')
    parser.add_argument('--person_id', type=int, default=None,
                        help='ID of person to extract (default: first person)')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Frame rate for the output (default: 30.0)')
    parser.add_argument('--use_world_coords', action='store_true', default=True,
                        help='Use world coordinates if available (default: True)')
    parser.add_argument('--use_camera_coords', action='store_true', 
                        help='Force use of camera coordinates instead of world')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert the data
    use_world = args.use_world_coords and not args.use_camera_coords
    convert_phmr_to_smplx(args.input, args.output, args.person_id, args.fps, use_world)


if __name__ == '__main__':
    main()