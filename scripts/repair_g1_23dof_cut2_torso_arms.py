"""Stabilize cut2's torso heading and keep repeated front-hand gestures open."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
ARM = np.array([13, 14, 15, 16, 18, 19, 20, 21])
FRONT_WINDOWS = (
    (83.8, 87.3), (96.0, 99.5), (108.4, 111.8),
    (120.9, 124.2), (132.9, 136.6), (145.5, 147.1),
)
UPPER_BODIES = (
    "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link",
    "left_shoulder_yaw_link", "left_elbow_link",
    "left_wrist_roll_rubber_hand",
    "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link",
    "right_wrist_roll_rubber_hand",
)


def ease(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def torso_yaw(data: mj.MjData, body_id: int) -> float:
    return float(Rotation.from_matrix(data.xmat[body_id].reshape(3, 3)).as_euler("xyz")[2])


def set_frame(data: mj.MjData, motion: dict, joints: np.ndarray, i: int) -> None:
    data.qpos[:3] = motion["root_pos"][i]
    data.qpos[3:7] = np.roll(motion["root_rot"][i], 1)
    data.qpos[7:] = joints[i]


def stabilize_torso_yaw(motion: dict, model: mj.MjModel) -> tuple[np.ndarray, float]:
    """Hold the world-facing torso direction while retaining root and legs."""
    original = np.asarray(motion["dof_pos"])
    result = original.copy()
    data = mj.MjData(model)
    torso = model.body("torso_link").id
    old_heading = np.empty(len(original))
    zero_waist_heading = np.empty(len(original))
    for i in range(len(original)):
        set_frame(data, motion, original, i)
        mj.mj_forward(model, data)
        old_heading[i] = torso_yaw(data, torso)
        data.qpos[7 + 12] = 0.0
        mj.mj_forward(model, data)
        zero_waist_heading[i] = torso_yaw(data, torso)

    fps = float(motion["fps"])
    neutral = np.r_[np.arange(round(4 * fps), round(10 * fps)),
                    np.arange(round(75 * fps), round(83 * fps))]
    heading = float(np.median(old_heading[neutral]))
    joint = model.joint("waist_yaw_joint")
    low, high = model.jnt_range[joint.id]
    desired = np.array([wrap(heading - angle) for angle in zero_waist_heading])
    result[:, 12] = np.clip(
        gaussian_filter1d(desired, sigma=5.0, mode="nearest"), low, high,
    )
    return result, heading


def front_window_frames(fps: float, length: int) -> list[tuple[int, int]]:
    return [(round(start * fps), min(length - 1, round(stop * fps)))
            for start, stop in FRONT_WINDOWS]


def repair_front_hand_cycles(
    motion: dict, model: mj.MjModel, stabilized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate and mirror both arms in the six repeated front-hand gestures."""
    result = stabilized.copy()
    changed = np.zeros(len(result), dtype=bool)
    fps = float(motion["fps"])
    bounds = np.array([
        model.jnt_range[model.joint(name).id] for name in (
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
        )
    ])
    low, high = bounds.T
    torso = model.body("torso_link").id
    wrists = [model.body(name).id for name in
              ("left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand")]
    elbows = [model.body(name).id for name in
              ("left_elbow_link", "right_elbow_link")]
    pairs = [(a, b) for a in range(model.ngeom) for b in range(model.ngeom)
             if model.geom_bodyid[a] == wrists[0] and model.geom_bodyid[b] == wrists[1]
             and model.geom_contype[a] > 0 and model.geom_contype[b] > 0]
    data = mj.MjData(model)

    for start, stop in front_window_frames(fps, len(result)):
        correction = np.zeros((stop - start + 1, len(ARM)))
        previous = stabilized[start, ARM].copy()
        for i in range(start, stop + 1):
            set_frame(data, motion, stabilized, i)
            mj.mj_forward(model, data)
            torso_pos = data.xpos[torso].copy()
            torso_rot = data.xmat[torso].reshape(3, 3).copy()

            def local(body_id: int) -> np.ndarray:
                return torso_rot.T @ (data.xpos[body_id] - torso_pos)

            wrist_old = np.array([local(body) for body in wrists])
            elbow_old = np.array([local(body) for body in elbows])
            width = wrist_old[0, 1] - wrist_old[1, 1]
            center = 0.5 * (wrist_old[0, 1] + wrist_old[1, 1])
            narrow = ease((0.30 - width) / 0.10)
            off_center = ease((abs(center) - 0.02) / 0.03)
            t = i / fps
            entry = ease((t - start / fps) / 0.7)
            exit_ = 1.0 if stop == len(result) - 1 else ease((stop / fps - t) / 0.7)
            strength = min(entry, exit_) * max(narrow, off_center)
            if strength < 1e-5:
                previous = stabilized[i, ARM].copy()
                continue

            half_width = max(0.12, 0.5 * width)
            wrist_mid = 0.5 * (wrist_old[0] + wrist_old[1])
            wrist_target = np.array([
                [wrist_mid[0], half_width, wrist_mid[2]],
                [wrist_mid[0], -half_width, wrist_mid[2]],
            ])
            elbow_half_width = max(0.13, 0.5 * (elbow_old[0, 1] - elbow_old[1, 1]))
            elbow_mid = 0.5 * (elbow_old[0] + elbow_old[1])
            elbow_target = np.array([
                [elbow_mid[0], elbow_half_width, elbow_mid[2]],
                [elbow_mid[0], -elbow_half_width, elbow_mid[2]],
            ])
            reference = stabilized[i, ARM]

            def residual(angles: np.ndarray) -> np.ndarray:
                data.qpos[7 + ARM] = angles
                mj.mj_forward(model, data)
                wrist_pos = np.array([local(body) for body in wrists])
                elbow_pos = np.array([local(body) for body in elbows])
                clearance = min(mj.mj_geomDistance(model, data, a, b, 0.5, None)
                                for a, b in pairs)
                return np.r_[
                    25.0 * (wrist_pos - wrist_target).ravel(),
                    8.0 * (elbow_pos - elbow_target).ravel(),
                    1.0 * (angles - reference),
                    0.5 * (angles - previous),
                    5.0 * (angles[3] - angles[7]),
                    120.0 * max(0.025 - clearance, 0.0),
                ]

            fit = least_squares(
                residual, np.clip(previous, low + 1e-5, high - 1e-5),
                bounds=(low, high), max_nfev=65, ftol=2e-3, xtol=2e-3,
            )
            previous = fit.x
            correction[i - start] = strength * (fit.x - reference)

        correction = gaussian_filter1d(correction, sigma=2.0, axis=0, mode="nearest")
        result[start:stop + 1, ARM] = np.clip(
            stabilized[start:stop + 1, ARM] + correction, low, high,
        )
        result[start, ARM] = stabilized[start, ARM]
        if stop < len(result) - 1:
            result[stop, ARM] = stabilized[stop, ARM]
        changed[start:stop + 1] = True

    # A single mesh-distance discontinuity can survive the Gaussian blend.
    # Interpolate that frame from its safe neighbors, which also reduces jerk.
    for i in np.flatnonzero(changed):
        set_frame(data, motion, result, i)
        mj.mj_forward(model, data)
        clearance = min(mj.mj_geomDistance(model, data, a, b, 0.5, None)
                        for a, b in pairs)
        if clearance >= -1e-3:
            continue
        if not (changed[i - 1] and changed[i + 1]):
            raise ValueError(f"hand penetration at front-window edge, frame {i}")
        proposal = 0.5 * (result[i - 1, ARM] + result[i + 1, ARM])
        data.qpos[7 + ARM] = proposal
        mj.mj_forward(model, data)
        projected_clearance = min(mj.mj_geomDistance(model, data, a, b, 0.5, None)
                                  for a, b in pairs)
        if projected_clearance < -1e-3:
            raise ValueError(f"hand penetration persists after interpolation, frame {i}")
        result[i, ARM] = proposal
    return result, changed


def recompute_upper_body(motion: dict, solved: np.ndarray, model: mj.MjModel) -> np.ndarray:
    local_pos = np.asarray(motion["local_body_pos"]).copy()
    indices = [(motion["link_body_list"].index(name), model.body(name).id)
               for name in UPPER_BODIES]
    data = mj.MjData(model)
    data.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
    for i in range(len(solved)):
        data.qpos[7:] = solved[i]
        mj.mj_forward(model, data)
        for output_id, model_id in indices:
            local_pos[i, output_id] = data.xpos[model_id]
    return local_pos


def validate(
    motion: dict, solved: np.ndarray, local_pos: np.ndarray, model: mj.MjModel,
    heading: float, changed: np.ndarray,
) -> None:
    original = np.asarray(motion["dof_pos"])
    preserved = list(range(12)) + [17, 22]
    if not np.array_equal(solved[:, preserved], original[:, preserved]):
        raise ValueError("legs or wrist rolls changed")
    if not np.array_equal(solved[~changed, :][:, ARM], original[~changed, :][:, ARM]):
        raise ValueError("arm trajectory outside front-hand windows changed")
    if not np.array_equal(local_pos[:, :15], motion["local_body_pos"][:, :15]):
        raise ValueError("lower-body positions changed")
    data = mj.MjData(model)
    torso = model.body("torso_link").id
    wrists = [model.body(name).id for name in
              ("left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand")]
    pairs = [(a, b) for a in range(model.ngeom) for b in range(model.ngeom)
             if model.geom_bodyid[a] == wrists[0] and model.geom_bodyid[b] == wrists[1]
             and model.geom_contype[a] > 0 and model.geom_contype[b] > 0]
    yaw_error = []
    widths = []
    clearance = np.inf
    for i in range(len(solved)):
        set_frame(data, motion, solved, i)
        mj.mj_forward(model, data)
        yaw_error.append(abs(wrap(torso_yaw(data, torso) - heading)))
        if changed[i]:
            R = data.xmat[torso].reshape(3, 3)
            left = R.T @ (data.xpos[wrists[0]] - data.xpos[torso])
            right = R.T @ (data.xpos[wrists[1]] - data.xpos[torso])
            widths.append((i, left[1] - right[1], left[1], right[1]))
            clearance = min(clearance, *(mj.mj_geomDistance(model, data, a, b, 0.5, None)
                                         for a, b in pairs))
    worst_yaw = np.degrees(max(yaw_error))
    print(f"torso heading target: {np.degrees(heading):.1f} deg; max deviation: {worst_yaw:.1f} deg")
    if worst_yaw > 5.0:
        raise ValueError("torso still turns away from its forward direction")
    prior_width = np.array([entry[1] for entry in widths])
    core = [item for item in widths if item[0] >= 75 * motion["fps"]
            and original[item[0], 21] > 0.55 and item[1] < 0.26]
    # Validate the observable front-hand result, not merely its IK objective.
    if core:
        minimum_width = min(item[1] for item in core)
        minimum_left = min(item[2] for item in core)
        maximum_right = max(item[3] for item in core)
    else:
        minimum_width = min(prior_width)
        minimum_left = min(item[2] for item in widths)
        maximum_right = max(item[3] for item in widths)
    print(f"front-hand min width: {minimum_width:.3f} m; left/right side limits: "
          f"{minimum_left:.3f}/{maximum_right:.3f} m; hand clearance: {clearance:.3f} m")
    if minimum_width < 0.20 or minimum_left < 0.05 or maximum_right > -0.05:
        raise ValueError("hands remain too close to the centerline")
    if clearance < -1e-3:
        raise ValueError("hands penetrate during front-hand gesture")
    fps = float(motion["fps"])
    # The source has an unrelated initial-frame spike; assess the repaired cycles.
    later = round(80 * fps)
    speed = np.max(np.abs(np.diff(solved[later:, [12, *ARM]], axis=0) * fps))
    acceleration = np.max(np.abs(np.diff(solved[later:, [12, *ARM]], n=2, axis=0) * fps**2))
    print(f"peak changed-joint speed: {speed:.2f} rad/s; acceleration: {acceleration:.1f} rad/s^2")
    if speed > 6.0 or acceleration > 60.0:
        raise ValueError("waist or arm trajectory changes too quickly")


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
        raise ValueError("expected the native G1 23DoF cut2 v004 motion")
    model = mj.MjModel.from_xml_path(str(args.xml))
    stabilized, heading = stabilize_torso_yaw(motion, model)
    solved, changed = repair_front_hand_cycles(motion, model, stabilized)
    local_pos = recompute_upper_body(motion, solved, model)
    validate(motion, solved, local_pos, model, heading, changed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(dict(motion, dof_pos=solved, local_body_pos=local_pos), handle)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
