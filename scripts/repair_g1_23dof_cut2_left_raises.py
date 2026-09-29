"""Restore cut2's three alternating left-arm raises without changing v006 elsewhere."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
LEFT = np.arange(13, 17)
MIRROR = np.array([1.0, -1.0, -1.0, 1.0])
LEFT_BODIES = (
    "left_shoulder_pitch_link", "left_shoulder_roll_link",
    "left_shoulder_yaw_link", "left_elbow_link",
    "left_wrist_roll_rubber_hand",
)
# Each source-video left raise is paired with the following successful right raise.
# Entries are (left start, left peak, left end, right start, right peak, right end).
CYCLES = (
    (5.5, 8.53, 12.2, 16.27, 19.53, 24.8),
    (27.7, 30.53, 37.0, 39.87, 42.73, 47.8),
    (53.4, 58.10, 63.0, 65.90, 69.80, 74.43),
)


def ease(value: np.ndarray) -> np.ndarray:
    x = np.clip(value, 0.0, 1.0)
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def torso_local(data: mj.MjData, torso: int, body: int) -> np.ndarray:
    rotation = data.xmat[torso].reshape(3, 3)
    return rotation.T @ (data.xpos[body] - data.xpos[torso])


def restore_left_raises(
    motion: dict, reference: dict, model: mj.MjModel,
) -> tuple[np.ndarray, np.ndarray]:
    original = np.asarray(motion["dof_pos"])
    source = np.asarray(reference["dof_pos"])
    result = original.copy()
    changed = np.zeros(len(result), dtype=bool)
    fps = float(motion["fps"])
    joint_names = (
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint",
    )
    bounds = np.array([model.jnt_range[model.joint(name).id] for name in joint_names])
    lower, upper = bounds.T
    torso = model.body("torso_link").id
    left_wrist = model.body("left_wrist_roll_rubber_hand").id
    right_wrist = model.body("right_wrist_roll_rubber_hand").id
    left_elbow = model.body("left_elbow_link").id
    right_elbow = model.body("right_elbow_link").id
    hand_pairs = [(a, b) for a in range(model.ngeom) for b in range(model.ngeom)
                  if model.geom_bodyid[a] == left_wrist
                  and model.geom_bodyid[b] == right_wrist
                  and model.geom_contype[a] > 0 and model.geom_contype[b] > 0]
    source_data = mj.MjData(model)
    donor_data = mj.MjData(model)
    solve_data = mj.MjData(model)
    for data in (source_data, donor_data, solve_data):
        data.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)

    for left_start, left_peak, left_end, right_start, right_peak, right_end in CYCLES:
        start = round(left_start * fps)
        stop = round(left_end * fps)
        frames = np.arange(start, stop + 1)
        times = frames / fps
        donor_times = np.interp(
            times, [left_start, left_peak, left_end],
            [right_start, right_peak, right_end],
        )
        donor_frames = np.clip(np.rint(donor_times * fps).astype(int), 0, len(result) - 1)
        candidates = np.empty((len(frames), len(LEFT)))
        previous = original[start, LEFT].copy()
        for j, (frame, donor_frame) in enumerate(zip(frames, donor_frames)):
            source_data.qpos[7:] = source[frame]
            mj.mj_forward(model, source_data)
            wrist_target = torso_local(source_data, torso, left_wrist)

            donor_data.qpos[7:] = original[donor_frame]
            mj.mj_forward(model, donor_data)
            right_elbow_pos = torso_local(donor_data, torso, right_elbow)
            elbow_target = right_elbow_pos * np.array([1.0, -1.0, 1.0])
            donor = np.clip(original[donor_frame, 18:22] * MIRROR,
                            lower + 1e-5, upper - 1e-5)
            solve_data.qpos[7:] = original[frame]

            def residual(angles: np.ndarray) -> np.ndarray:
                solve_data.qpos[7 + LEFT] = angles
                mj.mj_forward(model, solve_data)
                wrist = torso_local(solve_data, torso, left_wrist)
                elbow = torso_local(solve_data, torso, left_elbow)
                hand_clearance = min(mj.mj_geomDistance(model, solve_data, a, b, 0.5, None)
                                     for a, b in hand_pairs)
                return np.r_[
                    25.0 * (wrist - wrist_target),
                    5.0 * (elbow - elbow_target),
                    0.8 * (angles - donor),
                    2.0 * (angles - previous),
                    5.0 * max(angles[1] - 1.9, 0.0),
                    120.0 * max(0.04 - hand_clearance, 0.0),
                ]

            fit = least_squares(
                residual, np.clip(previous, lower + 1e-5, upper - 1e-5), bounds=(lower, upper),
                max_nfev=35, ftol=2e-3, xtol=2e-3,
            )
            candidates[j] = fit.x
            previous = fit.x

        candidates = gaussian_filter1d(candidates, sigma=2.0, axis=0, mode="nearest")
        fade = ease((times - left_start) / 0.8) * ease((left_end - times) / 0.9)
        result[np.ix_(frames, LEFT)] = original[np.ix_(frames, LEFT)] + fade[:, None] * (
            candidates - original[frames][:, LEFT]
        )
        result[np.ix_(frames, LEFT)] = np.clip(result[np.ix_(frames, LEFT)], lower, upper)
        result[start, LEFT] = original[start, LEFT]
        result[stop, LEFT] = original[stop, LEFT]
        changed[frames[fade > 0.0]] = True
    return result, changed


def update_left_positions(
    motion: dict, solved: np.ndarray, changed: np.ndarray, model: mj.MjModel,
) -> np.ndarray:
    local = np.asarray(motion["local_body_pos"]).copy()
    ids = [(motion["link_body_list"].index(name), model.body(name).id)
           for name in LEFT_BODIES]
    data = mj.MjData(model)
    data.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
    for frame in np.flatnonzero(changed):
        data.qpos[7:] = solved[frame]
        mj.mj_forward(model, data)
        for output_id, model_id in ids:
            local[frame, output_id] = data.xpos[model_id]
    return local


def validate(
    motion: dict, reference: dict, solved: np.ndarray, local: np.ndarray,
    changed: np.ndarray, model: mj.MjModel,
) -> None:
    original = np.asarray(motion["dof_pos"])
    if not np.array_equal(solved[~changed], original[~changed]):
        raise ValueError("joint trajectory outside left-raise windows changed")
    preserved = np.setdiff1d(np.arange(23), LEFT)
    if not np.array_equal(solved[:, preserved], original[:, preserved]):
        raise ValueError("a non-left-arm joint changed")
    left_ids = [motion["link_body_list"].index(name) for name in LEFT_BODIES]
    other_ids = np.setdiff1d(np.arange(local.shape[1]), left_ids)
    if not np.array_equal(local[~changed], motion["local_body_pos"][~changed]):
        raise ValueError("body positions outside left-raise windows changed")
    if not np.array_equal(local[:, other_ids], motion["local_body_pos"][:, other_ids]):
        raise ValueError("a non-left-arm body position changed")

    names = motion["link_body_list"]
    torso = names.index("torso_link")
    wrist = names.index("left_wrist_roll_rubber_hand")
    new_height = local[:, wrist, 2] - local[:, torso, 2]
    source_local = np.asarray(reference["local_body_pos"])
    source_height = source_local[:, wrist, 2] - source_local[:, torso, 2]
    peaks = []
    for start, _, stop, *_ in CYCLES:
        frames = slice(round(start * motion["fps"]), round(stop * motion["fps"]) + 1)
        peak = float(np.max(new_height[frames]))
        target_peak = float(np.max(source_height[frames]))
        peaks.append(peak)
        if abs(peak - target_peak) > 0.08:
            raise ValueError(f"left wrist misses its source-video height in {start}-{stop}s")

    fps = float(motion["fps"])
    speed = float(np.max(np.abs(np.diff(solved[:, LEFT], axis=0) * fps)))
    acceleration = float(np.max(np.abs(np.diff(solved[:, LEFT], n=2, axis=0) * fps**2)))
    if speed > 6.0 or acceleration > 60.0:
        raise ValueError("left arm became too fast or abrupt")
    roll_limit = model.jnt_range[model.joint("left_shoulder_roll_joint").id, 1]
    if np.max(solved[changed, 14]) >= roll_limit - 0.10:
        raise ValueError("left shoulder still reaches its old joint limit")
    data = mj.MjData(model)
    left_hand = model.body("left_wrist_roll_rubber_hand").id
    other_bodies = [model.body(name).id for name in
                    ("right_wrist_roll_rubber_hand", "torso_link", "pelvis")]
    pairs = [(a, b) for a in range(model.ngeom) for b in range(model.ngeom)
             if model.geom_bodyid[a] == left_hand and model.geom_bodyid[b] in other_bodies
             and model.geom_contype[a] > 0 and model.geom_contype[b] > 0]
    minimum_clearance = np.inf
    for frame in np.flatnonzero(changed):
        data.qpos[:3] = motion["root_pos"][frame]
        data.qpos[3:7] = np.roll(motion["root_rot"][frame], 1)
        data.qpos[7:] = solved[frame]
        mj.mj_forward(model, data)
        minimum_clearance = min(
            minimum_clearance,
            *(mj.mj_geomDistance(model, data, a, b, 0.5, None) for a, b in pairs),
        )
    if minimum_clearance < -1e-3:
        raise ValueError(f"left hand penetrates another body: {minimum_clearance:.3f} m")
    print("left wrist peaks (m):", [round(value, 3) for value in peaks])
    print(f"left-arm peak speed: {speed:.2f} rad/s; acceleration: {acceleration:.1f} rad/s^2; "
          f"peak shoulder roll: {np.max(solved[changed, 14]):.2f} rad; "
          f"hand clearance: {minimum_clearance:.3f} m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--xml", type=Path, default=ROOT / "assets/unitree_g1/g1_mocap_23dof.xml")
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() in (args.input.resolve(), args.reference.resolve()):
        raise ValueError("output must be a new versioned PKL")
    with args.input.open("rb") as handle:
        motion = pickle.load(handle)
    with args.reference.open("rb") as handle:
        reference = pickle.load(handle)
    if np.asarray(motion["dof_pos"]).shape != (4412, 23) or float(motion["fps"]) != 30.0:
        raise ValueError("expected the full native G1 23DoF cut2 motion")
    model = mj.MjModel.from_xml_path(str(args.xml))
    solved, changed = restore_left_raises(motion, reference, model)
    local = update_left_positions(motion, solved, changed, model)
    validate(motion, reference, solved, local, changed, model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(dict(motion, dof_pos=solved, local_body_pos=local), handle)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
