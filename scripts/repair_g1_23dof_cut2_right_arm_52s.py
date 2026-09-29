"""Mirror cut2's right arm through the short, unwanted 52-second reach."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mujoco as mj
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEFT_ARM = np.arange(13, 17)
RIGHT_ARM = np.arange(18, 22)
MIRROR_SIGN = np.array([1.0, -1.0, -1.0, 1.0])
RIGHT_BODIES = (
    "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link",
    "right_wrist_roll_rubber_hand",
)


def ease(value: np.ndarray) -> np.ndarray:
    x = np.clip(value, 0.0, 1.0)
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def mirror_right_arm(motion: dict) -> tuple[np.ndarray, np.ndarray]:
    original = np.asarray(motion["dof_pos"])
    solved = original.copy()
    time = np.arange(len(original)) / float(motion["fps"])
    weight = ease((time - 51.9) / (52.25 - 51.9))
    weight *= ease((54.5 - time) / (54.5 - 54.1))
    changed = weight > 0.0
    target = original[changed][:, LEFT_ARM] * MIRROR_SIGN
    solved[np.ix_(changed, RIGHT_ARM)] += weight[changed, None] * (
        target - original[np.ix_(changed, RIGHT_ARM)]
    )
    return solved, changed


def update_right_body_positions(
    motion: dict, solved: np.ndarray, changed: np.ndarray, model: mj.MjModel,
) -> np.ndarray:
    positions = np.asarray(motion["local_body_pos"]).copy()
    body_ids = [(motion["link_body_list"].index(name), model.body(name).id)
                for name in RIGHT_BODIES]
    data = mj.MjData(model)
    data.qpos[:7] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    for frame in np.flatnonzero(changed):
        data.qpos[7:] = solved[frame]
        mj.mj_forward(model, data)
        for output_id, model_id in body_ids:
            positions[frame, output_id] = data.xpos[model_id]
    return positions


def validate(
    motion: dict, solved: np.ndarray, positions: np.ndarray,
    changed: np.ndarray, model: mj.MjModel,
) -> None:
    original = np.asarray(motion["dof_pos"])
    untouched_joints = np.setdiff1d(np.arange(original.shape[1]), RIGHT_ARM)
    if not np.array_equal(solved[~changed], original[~changed]):
        raise ValueError("joint trajectory changed outside the 52-second window")
    if not np.array_equal(solved[:, untouched_joints], original[:, untouched_joints]):
        raise ValueError("a joint other than the right shoulder/elbow changed")
    right_body_ids = [motion["link_body_list"].index(name) for name in RIGHT_BODIES]
    other_body_ids = np.setdiff1d(np.arange(positions.shape[1]), right_body_ids)
    if not np.array_equal(positions[~changed], motion["local_body_pos"][~changed]):
        raise ValueError("body positions changed outside the 52-second window")
    if not np.array_equal(positions[:, other_body_ids],
                          motion["local_body_pos"][:, other_body_ids]):
        raise ValueError("body positions outside the right arm changed")

    names = (
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
    )
    limits = np.array([model.jnt_range[model.joint(name).id] for name in names])
    if np.any(solved[changed][:, RIGHT_ARM] < limits[:, 0]) or np.any(
        solved[changed][:, RIGHT_ARM] > limits[:, 1]
    ):
        raise ValueError("right arm exceeds a joint limit")

    data = mj.MjData(model)
    torso = model.body("torso_link").id
    left_hand = model.body("left_wrist_roll_rubber_hand").id
    right_hand = model.body("right_wrist_roll_rubber_hand").id
    hand_geoms = [(a, b) for a in range(model.ngeom) for b in range(model.ngeom)
                  if model.geom_bodyid[a] == left_hand
                  and model.geom_bodyid[b] == right_hand
                  and model.geom_contype[a] > 0 and model.geom_contype[b] > 0]
    minimum_clearance = np.inf
    for frame in np.flatnonzero(changed):
        data.qpos[:3] = motion["root_pos"][frame]
        data.qpos[3:7] = np.roll(motion["root_rot"][frame], 1)
        data.qpos[7:] = solved[frame]
        mj.mj_forward(model, data)
        minimum_clearance = min(
            minimum_clearance,
            *(mj.mj_geomDistance(model, data, a, b, 0.5, None)
              for a, b in hand_geoms),
        )
        if frame == round(53.0 * motion["fps"]):
            rotation = data.xmat[torso].reshape(3, 3)
            left = rotation.T @ (data.xpos[left_hand] - data.xpos[torso])
            right = rotation.T @ (data.xpos[right_hand] - data.xpos[torso])
            symmetry_error = abs(left[1] + right[1])
    if minimum_clearance < -1e-3 or symmetry_error > 0.01:
        raise ValueError("hands are not separated and symmetric at 53 seconds")

    start, stop = round(51.5 * motion["fps"]), round(55.0 * motion["fps"])
    def motion_rates(values: np.ndarray) -> tuple[float, float]:
        arm = values[start:stop, RIGHT_ARM]
        fps = float(motion["fps"])
        return (float(np.max(np.abs(np.diff(arm, axis=0) * fps))),
                float(np.max(np.abs(np.diff(arm, n=2, axis=0) * fps**2))))

    old_speed, old_acceleration = motion_rates(original)
    speed, acceleration = motion_rates(solved)
    if speed > old_speed + 0.05 or acceleration > old_acceleration + 0.5:
        raise ValueError("right arm became faster or less smooth")
    print(f"53s lateral symmetry error: {symmetry_error:.3f} m; "
          f"minimum hand clearance: {minimum_clearance:.3f} m")
    print(f"right-arm peak speed: {old_speed:.2f} -> {speed:.2f} rad/s; "
          f"acceleration: {old_acceleration:.1f} -> {acceleration:.1f} rad/s^2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--xml", type=Path, default=ROOT / "assets/unitree_g1/g1_mocap_23dof.xml")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        raise ValueError("output must be a new versioned PKL")
    with args.input.open("rb") as handle:
        motion = pickle.load(handle)
    if np.asarray(motion["dof_pos"]).shape != (4412, 23) or float(motion["fps"]) != 30.0:
        raise ValueError("expected the native G1 23DoF cut2 motion")
    model = mj.MjModel.from_xml_path(str(args.xml))
    solved, changed = mirror_right_arm(motion)
    positions = update_right_body_positions(motion, solved, changed, model)
    validate(motion, solved, positions, changed, model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(dict(motion, dof_pos=solved, local_body_pos=positions), handle)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
