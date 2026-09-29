"""Locally repair the accepted G1 v013 motion without rerunning GMR."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LEFT = np.array([13, 14, 15, 16])
RIGHT = np.array([18, 19, 20, 21])
ARMS = np.r_[LEFT, RIGHT]
MIRROR = np.array([1.0, -1.0, -1.0, 1.0])


def ease(x):
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def source_torso_yaw(source):
    # SMPL-X body_pose joint indices: spine1=2, spine2=5, spine3=8.
    pose = source["pose_body"]
    rotations = [Rotation.from_rotvec(pose[:, 3*j:3*j+3]) for j in (2, 5, 8)]
    return (rotations[0] * rotations[1] * rotations[2]).as_euler("zyx")[:, 0]


def repair_mask(waist, target, fps):
    discrepancy = np.abs(np.angle(np.exp(1j * (waist - target))))
    jerk = np.r_[0.0, np.abs(np.diff(waist, n=2)), 0.0]
    marked = (discrepancy > np.deg2rad(8.0)) | (jerk > np.deg2rad(0.8))
    for start, end in [(59, 62), (74, 77), (90, 93), (100.3, 110.8), (139, 145)]:
        marked[round(start*fps):round(end*fps)+1] = True
    marked = binary_dilation(marked, iterations=round(0.7*fps))
    weights = np.zeros(len(waist))
    indices = np.flatnonzero(marked)
    for segment in np.split(indices, np.where(np.diff(indices) > 1)[0]+1):
        if not len(segment):
            continue
        fade = min(round(0.7*fps), max(1, len(segment)//2))
        weight = np.ones(len(segment))
        weight[:fade] = ease(np.arange(fade)/fade)
        weight[-fade:] = np.minimum(weight[-fade:], ease(np.arange(fade, 0, -1)/fade))
        weights[segment] = weight
    return marked, weights


def mirror_left_from_right(original, candidate, fps):
    start, end = round(100.3*fps), round(110.8*fps)
    fade = round(0.7*fps)
    for i in range(start, end+1):
        weight = min(ease((i-start)/fade), ease((round(109.7*fps)-i)/(2.7*fps)))
        mirror = original[i, RIGHT] * MIRROR
        candidate[i, LEFT] = (1.0-weight)*original[i, LEFT] + weight*mirror
    return start, end


def model_limits(model, indices):
    limits = []
    for index in indices:
        joint = int(model.dof_jntid[index+6])
        limits.append(model.jnt_range[joint])
    return np.asarray(limits)


def fix_arm_positions(original, candidate, root_pos, root_rot, marked, issue, model, fps):
    data = mj.MjData(model)
    body_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
                for name in ("left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand")]
    bounds = model_limits(model, ARMS)
    lo, hi = bounds.T

    changed = np.flatnonzero(marked)
    for count, i in enumerate(changed):
        base = np.r_[root_pos[i], np.roll(root_rot[i], 1), original[i]].copy()
        data.qpos[:] = base
        mj.mj_forward(model, data)
        desired_wrist = data.xpos[body_ids].copy()
        if issue[0] <= i <= issue[1]:
            # The right descent in v013 is synchronized with the source. Its
            # mirrored left-hand FK supplies a feasible low/front target.
            target_pose = base.copy()
            target_pose[7+LEFT] = candidate[i, LEFT]
            target_pose[7+12] = candidate[i, 12]
            data.qpos[:] = target_pose
            mj.mj_forward(model, data)
            desired_wrist[0] = data.xpos[body_ids[0]]
            # Keep both low hands in front with clearance; a perfect mirror
            # would put the rubber hands through each other near 106 s.
            spread = ease((i-103*fps)/(0.8*fps)) * ease((108.8*fps-i)/(1.0*fps))
            # Bring the left hand inward only at the end of the descent.
            settle = (ease((i-104.0*fps)/(0.9*fps))
                      * ease((106.4*fps-i)/(0.7*fps)))
            lateral = Rotation.from_quat(root_rot[i]).apply(
                [0.0, (0.13-0.085*settle)*spread, 0.0])
            desired_wrist[0] += lateral
        if 139*fps <= i <= 145*fps:
            # The source crosses its hands in front of the chest. With the
            # excessive waist twist removed, give both hands torso clearance.
            forward = ease((i-139*fps)/fps) * ease((145*fps-i)/fps)
            desired_wrist += Rotation.from_quat(root_rot[i]).apply(
                [0.09*forward, 0.0, 0.0])
            # At the crossing, stagger the hands front-to-back rather than
            # placing their collision geoms in the same space.
            stagger = ease((i-140.7*fps)/(0.6*fps)) * ease((142.3*fps-i)/(0.6*fps))
            desired_wrist[0] += Rotation.from_quat(root_rot[i]).apply(
                [0.08*stagger, 0.0, 0.0])
        base[7+12] = candidate[i, 12]
        desired = candidate[i, ARMS].copy()
        if np.abs(candidate[i, 12]-original[i, 12]) < 1e-5 and not (issue[0] <= i <= issue[1]):
            continue

        def residual(angles):
            data.qpos[:] = base
            data.qpos[7+ARMS] = angles
            mj.mj_forward(model, data)
            return np.r_[10.0*(data.xpos[body_ids]-desired_wrist).ravel(),
                         0.12*(angles-desired)]

        fit = least_squares(residual, np.clip(desired, lo+1e-7, hi-1e-7),
                            bounds=(lo, hi), max_nfev=24, ftol=2e-3)
        candidate[i, ARMS] = fit.x
        if count % 500 == 0:
            print(f"adjusted {count}/{len(changed)}", flush=True)
    return candidate


def smooth_changed_joints(original, candidate, mask, fps):
    # Short symmetric filtering removes IK frame noise; original values and
    # their first derivatives are retained at every repair boundary.
    weights = np.zeros(len(mask))
    ids = np.flatnonzero(mask)
    for segment in np.split(ids, np.where(np.diff(ids) > 1)[0]+1):
        if not len(segment):
            continue
        fade = min(round(0.6*fps), max(1, len(segment)//2))
        w = np.ones(len(segment))
        w[:fade] = ease(np.arange(fade)/fade)
        w[-fade:] = np.minimum(w[-fade:], ease(np.arange(fade, 0, -1)/fade))
        weights[segment] = w
    for j in [12, *ARMS]:
        filtered = gaussian_filter1d(candidate[:, j], sigma=2.5, mode="nearest")
        candidate[:, j] += weights*(filtered-candidate[:, j])
        candidate[~mask, j] = original[~mask, j]
    return candidate


def recompute_local(candidate, original, mask, xml):
    import torch
    from general_motion_retargeting.kinematics_model import KinematicsModel
    kin = KinematicsModel(str(xml), device="cpu")
    local = np.asarray(original["local_body_pos"]).copy()
    ids = np.flatnonzero(mask)
    for batch in np.array_split(ids, max(1, (len(ids)+255)//256)):
        if not len(batch):
            continue
        n = len(batch)
        pos, _ = kin.forward_kinematics(
            torch.zeros((n, 3)), torch.tensor([[0, 0, 0, 1.0]]).expand(n, -1),
            torch.from_numpy(candidate[batch]).float())
        local[batch] = pos.numpy()
    return local, kin.body_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=ROOT/"motion_smplx/eight_cut_1.npz")
    parser.add_argument("--xml", type=Path, default=ROOT/"assets/unitree_g1/g1_mocap_23dof.xml")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output must differ")
    with args.input.open("rb") as handle:
        motion = pickle.load(handle)
    fps = float(motion["fps"])
    original = np.asarray(motion["dof_pos"])
    if original.shape != (6752, 23) or fps != 30:
        raise ValueError("expected the 6752-frame, 30-fps native G1 23DoF motion")
    source = np.load(args.source)
    source_yaw = source_torso_yaw(source)
    if len(source_yaw) != len(original):
        raise ValueError("source and robot frame counts differ")
    model = mj.MjModel.from_xml_path(str(args.xml))
    candidate = original.copy()
    marked, weights = repair_mask(original[:, 12], source_yaw, fps)
    target_waist = gaussian_filter1d(source_yaw + 0.15*(original[:, 12]-source_yaw),
                                     sigma=5.0)
    candidate[:, 12] = original[:, 12]*(1.0-weights) + target_waist*weights
    issue = mirror_left_from_right(original, candidate, fps)
    marked[issue[0]:issue[1]+1] = True
    candidate = fix_arm_positions(original, candidate, motion["root_pos"],
                                  motion["root_rot"], marked, issue, model, fps)
    candidate = smooth_changed_joints(original, candidate, marked, fps)
    local, names = recompute_local(candidate, motion, marked, args.xml)
    if list(names) != list(motion["link_body_list"]):
        raise ValueError("body order changed")
    if not np.isfinite(candidate).all() or not np.array_equal(candidate[~marked], original[~marked]):
        raise ValueError("nonfinite motion or changes outside repair windows")
    if not np.array_equal(candidate[:, :12], original[:, :12]) or not np.array_equal(candidate[:, [17, 22]], original[:, [17, 22]]):
        raise ValueError("legs or wrist roll changed")
    limits = model_limits(model, np.arange(23))
    if np.any(candidate < limits[:, 0]-1e-5) or np.any(candidate > limits[:, 1]+1e-5):
        raise ValueError("joint limit violation")
    l, r = [names.index(s) for s in ("left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand")]
    low = slice(round(104*fps), round(107*fps)+1)
    wrist_gap = np.max(np.abs(local[low, l, 2]-local[low, r, 2]))
    waist_error = np.max(np.abs(candidate[round(140*fps):round(142*fps), 12]-source_yaw[round(140*fps):round(142*fps)]))
    vel = np.max(np.abs(np.diff(candidate[:, [12, *ARMS]], axis=0)*fps), axis=0)
    acc = np.max(np.abs(np.diff(candidate[:, [12, *ARMS]], n=2, axis=0)*fps**2), axis=0)
    print({"changed_frames": int(marked.sum()), "low_wrist_gap_m": float(wrist_gap),
           "waist_error_140_142_deg": float(np.degrees(waist_error)),
           "peak_speed_rad_s": vel.round(2).tolist(), "peak_acc_rad_s2": acc.round(1).tolist()}, flush=True)
    if wrist_gap > 0.08 or waist_error > np.deg2rad(10):
        raise ValueError("left-arm descent or waist target not achieved")
    if vel[0] > 1.0 or np.max(vel[1:]) > 6.2 or np.max(acc) > 75.0:
        raise ValueError("repair creates an abrupt joint speed or acceleration")
    result = dict(motion, dof_pos=candidate, local_body_pos=local)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(result, handle)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
