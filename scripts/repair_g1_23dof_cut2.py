"""Keep cut2's left arm on one feasible G1 shoulder branch.

The source video has one initial left-arm descent; subsequent overhead gestures
belong to the right arm.  The original PKL supplies the initial left-hand path;
a mirrored lowered right arm and cut1 v028 supply low-hand references.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
LEFT = np.array([13, 14, 15, 16])
MIRROR = np.array([1.0, -1.0, -1.0, 1.0])
LEFT_BODIES = (
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_rubber_hand",
)

# Approximate source-video events in seconds.  0 = left hand raised, 1 = low.
# Only the opening gesture raises the left arm; later raises are right-arm only.
PHASE_KEYS = np.array([
    (0.0, 0), (2.0, 0), (3.8, 1), (147.1, 1),
], dtype=float)

MAX_SPEED = 6.0  # rad/s, stricter than GMR's 9.425-rad/s output limit
MAX_ACCEL = 60.0  # rad/s^2, matching the GMR output limit


def phase_weights(frame_count: int, fps: float) -> np.ndarray:
    """Interpolate video-labelled phases with zero-slope joins."""
    times = np.arange(frame_count) / fps
    out = np.empty(frame_count)
    for (start, value0), (stop, value1) in zip(PHASE_KEYS[:-1], PHASE_KEYS[1:]):
        mask = (times >= start) & (times < stop)
        u = np.clip((times[mask] - start) / (stop - start), 0.0, 1.0)
        u = u**3 * (10.0 - 15.0 * u + 6.0 * u**2)
        out[mask] = value0 + (value1 - value0) * u
    out[times >= PHASE_KEYS[-1, 0]] = PHASE_KEYS[-1, 1]
    return out


def model_joint_limits(model: mj.MjModel) -> tuple[np.ndarray, np.ndarray]:
    ranges = []
    for name in (
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint",
    ):
        ranges.append(model.jnt_range[model.joint(name).id])
    return np.array(ranges).T


def prepare_low_pose(motion: dict, model: mj.MjModel, reference: dict) -> np.ndarray:
    """Mirror only right-arm samples that are actually low, then fill gaps."""
    q = np.asarray(motion["dof_pos"])
    positions = np.asarray(motion["local_body_pos"])
    names = motion["link_body_list"]
    right_wrist = names.index("right_wrist_roll_rubber_hand")
    torso = names.index("torso_link")
    right_height = positions[:, right_wrist, 2] - positions[:, torso, 2]
    valid = (right_height < 0.12) & (q[:, 18] > -1.4)
    valid_ids = np.flatnonzero(valid)
    if len(valid_ids) < 30:
        raise ValueError("not enough low right-arm samples for cut2")

    mirrored = q[:, 18:22] * MIRROR
    ids = np.arange(len(q))
    low = np.column_stack([
        np.interp(ids, valid_ids, mirrored[valid_ids, j]) for j in range(4)
    ])
    # During long raised-right-arm spans, a linear interpolation between two
    # distant right poses is not a dependable low-arm reference.
    nearest_right = valid_ids[np.clip(np.searchsorted(valid_ids, ids), 0, len(valid_ids) - 1)]
    nearest_left = valid_ids[np.clip(np.searchsorted(valid_ids, ids) - 1, 0, len(valid_ids) - 1)]
    distance = np.minimum(np.abs(ids - nearest_left), np.abs(ids - nearest_right))
    fallback = np.clip((distance - 15.0) / 45.0, 0.0, 1.0)[:, None]
    reference_low = np.asarray(reference["dof_pos"])[round(105.0 * reference["fps"]), LEFT]
    low = low * (1.0 - fallback) + reference_low[None, :] * fallback
    low = gaussian_filter1d(low, sigma=5.0, axis=0, mode="nearest")
    lower, upper = model_joint_limits(model)
    return np.clip(low, lower + 1e-5, upper - 1e-5)


def build_left_target_trajectory(
    motion: dict, model: mj.MjModel, low_pose: np.ndarray, weights: np.ndarray,
) -> np.ndarray:
    """Preserve raised-hand FK and replace low-hand FK with a feasible pose."""
    q = np.asarray(motion["dof_pos"])
    positions = np.asarray(motion["local_body_pos"])
    names = motion["link_body_list"]
    left_wrist = names.index("left_wrist_roll_rubber_hand")
    torso = names.index("torso_link")
    raised = positions[:, left_wrist] - positions[:, torso]
    lowered = np.empty_like(raised)
    wrist_id = model.body("left_wrist_roll_rubber_hand").id
    torso_id = model.body("torso_link").id
    data = mj.MjData(model)
    data.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
    for i in range(len(q)):
        data.qpos[7:] = q[i]
        data.qpos[7 + LEFT] = low_pose[i]
        mj.mj_forward(model, data)
        lowered[i] = data.xpos[wrist_id] - data.xpos[torso_id]
    target = raised * (1 - weights[:, None]) + lowered * weights[:, None]
    return gaussian_filter1d(target, sigma=2.0, axis=0, mode="nearest")


def solve_left_arm_trajectory(
    motion: dict, model: mj.MjModel, target: np.ndarray,
    low_pose: np.ndarray, weights: np.ndarray,
) -> np.ndarray:
    """Find the opening branch backward from the low pose, then stay on it."""
    original = np.asarray(motion["dof_pos"])
    solved = original.copy()
    lower, upper = model_joint_limits(model)
    wrist_id = model.body("left_wrist_roll_rubber_hand").id
    torso_id = model.body("torso_link").id
    right_id = model.body("right_wrist_roll_rubber_hand").id
    hand_pairs = [
        (a, b)
        for a in range(model.ngeom)
        for b in range(model.ngeom)
        if model.geom_bodyid[a] == wrist_id
        and model.geom_bodyid[b] == right_id
        and model.geom_contype[a] > 0
        and model.geom_contype[b] > 0
    ]
    data = mj.MjData(model)
    data.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
    fps = float(motion["fps"])
    anchor = int(round(3.8 * fps))
    branch = low_pose.copy()

    # The first raised pose must inherit the branch of the first valid low
    # pose.  Forward IK from the retargeted first frame instead follows the
    # saturated shoulder and has to change branch during the descent.
    for i in range(anchor - 1, -1, -1):
        data.qpos[7:] = original[i]
        following = branch[i + 1]

        def opening_residual(angles: np.ndarray) -> np.ndarray:
            data.qpos[7 + LEFT] = angles
            mj.mj_forward(model, data)
            wrist = data.xpos[wrist_id] - data.xpos[torso_id]
            return np.r_[
                np.array([10.0, 10.0, 25.0]) * (wrist - target[i]),
                0.35 * (angles - following),
                0.08 * (angles - low_pose[i]),
                3.0 * max(angles[1] - 1.25, 0.0),
            ]

        branch[i] = least_squares(
            opening_residual, following, bounds=(lower, upper),
            max_nfev=50, ftol=2e-3, xtol=2e-3,
        ).x

    previous = branch[0].copy()
    previous_step = np.zeros(4)

    for i in range(len(original)):
        data.qpos[7:] = original[i]
        low_weight = weights[i]

        def residual(angles: np.ndarray) -> np.ndarray:
            data.qpos[7 + LEFT] = angles
            mj.mj_forward(model, data)
            wrist = data.xpos[wrist_id] - data.xpos[torso_id]
            # Keep the rubber hands and torso apart without distorting the
            # raised-hand path.  The final check uses actual MuJoCo geoms.
            hand_gap = np.linalg.norm(data.xpos[wrist_id] - data.xpos[right_id])
            torso_gap = np.linalg.norm(wrist)
            hand_clearance = (
                min(mj.mj_geomDistance(model, data, a, b, 0.5, None)
                    for a, b in hand_pairs)
                if low_weight > 0.0 else 0.5
            )
            return np.r_[
                np.array([10.0, 10.0, 25.0]) * (wrist - target[i]),
                0.80 * (angles - branch[i]),
                0.30 * (angles - previous),
                3.0 * max(angles[1] - 1.25, 0.0),
                12.0 * max(0.15 - hand_gap, 0.0),
                12.0 * max(0.13 - torso_gap, 0.0),
                120.0 * low_weight * max(0.06 - hand_clearance, 0.0),
            ]

        fit = least_squares(
            residual, previous, bounds=(lower, upper),
            max_nfev=25, ftol=2e-3, xtol=2e-3,
        )
        step = fit.x - previous
        step = np.clip(step, -MAX_SPEED / fps, MAX_SPEED / fps)
        step = np.clip(
            step,
            previous_step - MAX_ACCEL / fps**2,
            previous_step + MAX_ACCEL / fps**2,
        )
        current = np.clip(previous + step, lower, upper)
        solved[i, LEFT] = current
        previous_step = current - previous
        previous = current
    # Near a hard joint stop, clipping can remove velocity in one frame even
    # when the forward solve obeyed its acceleration cap.  Smooth that final
    # projection over a short, symmetric window before computing FK.
    solved[:, LEFT] = np.clip(
        gaussian_filter1d(solved[:, LEFT], sigma=3.0, axis=0, mode="nearest"),
        lower, upper,
    )
    return solved


def update_left_body_pos(
    motion: dict, solved: np.ndarray, model: mj.MjModel,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Refresh only left-arm links; all other PKL body samples stay unchanged."""
    local = np.asarray(motion["local_body_pos"]).copy()
    names = motion["link_body_list"]
    ids = [(names.index(name), model.body(name).id) for name in LEFT_BODIES]
    data = mj.MjData(model)
    data.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
    frames = range(len(solved)) if mask is None else np.flatnonzero(mask)
    for i in frames:
        data.qpos[7:] = solved[i]
        mj.mj_forward(model, data)
        for body_index, model_index in ids:
            local[i, body_index] = data.xpos[model_index]
    return local


def validate_motion(motion: dict, solved: np.ndarray, local: np.ndarray, model: mj.MjModel) -> None:
    original = np.asarray(motion["dof_pos"])
    if original.shape != (4412, 23) or float(motion["fps"]) != 30.0:
        raise ValueError("cut2 must have 4412 frames at 30 fps with 23 joints")
    if not np.isfinite(solved).all() or not np.isfinite(local).all():
        raise ValueError("non-finite motion sample")
    preserved = np.array([j for j in range(23) if j not in LEFT])
    if not np.array_equal(solved[:, preserved], original[:, preserved]):
        raise ValueError("a non-left-arm joint changed")
    names = motion["link_body_list"]
    preserved_bodies = [i for i, name in enumerate(names) if name not in LEFT_BODIES]
    if not np.array_equal(local[:, preserved_bodies], motion["local_body_pos"][:, preserved_bodies]):
        raise ValueError("a non-left-arm body position changed")
    lower, upper = model_joint_limits(model)
    if np.any(solved[:, LEFT] < lower - 1e-6) or np.any(solved[:, LEFT] > upper + 1e-6):
        raise ValueError("left-arm joint limit violation")
    speed = np.max(np.abs(np.diff(solved[:, LEFT], axis=0) * 30.0))
    acceleration = np.max(np.abs(np.diff(solved[:, LEFT], n=2, axis=0) * 30.0**2))
    torso = names.index("torso_link")
    left = names.index("left_wrist_roll_rubber_hand")
    right = names.index("right_wrist_roll_rubber_hand")
    low_times = (4, 13, 25, 39, 51, 65, 80, 90, 110, 130)
    gaps = [abs((local[round(t * 30), left, 2] - local[round(t * 30), torso, 2])
                 - (local[round(t * 30), right, 2] - local[round(t * 30), torso, 2]))
            for t in low_times]
    low = phase_weights(len(solved), float(motion["fps"])) > 0.95
    limit_fraction = np.mean(solved[low, 14] >= upper[1] - 0.01)
    after_descent = slice(round(4.0 * motion["fps"]), None)
    left_height = local[:, left, 2] - local[:, torso, 2]
    print(f"left roll at upper limit during low poses: {limit_fraction:.1%}; "
          f"peak speed: {speed:.2f} rad/s; peak acceleration: {acceleration:.1f} rad/s^2")
    print("low-pose left/right wrist height gaps (m):", np.round(gaps, 3).tolist())
    if limit_fraction > 0.05 or max(gaps) > 0.12:
        raise ValueError("left arm still misses a low pose or remains on the wrong branch")
    if np.max(left_height[after_descent]) > 0.15 or np.max(solved[:, 14]) >= upper[1] - 0.01:
        raise ValueError("left arm rises again or the shoulder reaches the wrong branch")
    if speed > MAX_SPEED + 1e-4 or acceleration > MAX_ACCEL + 1e-3:
        raise ValueError("left-arm motion exceeds speed or acceleration limit")

    left_id = model.body("left_wrist_roll_rubber_hand").id
    other_ids = [model.body(name).id for name in
                 ("right_wrist_roll_rubber_hand", "torso_link", "pelvis")]
    pairs = [(a, b) for a in range(model.ngeom) for b in range(model.ngeom)
             if model.geom_bodyid[a] == left_id
             and model.geom_bodyid[b] in other_ids
             and model.geom_contype[a] > 0 and model.geom_contype[b] > 0]
    data = mj.MjData(model)
    minimum_clearance = np.inf
    for i in range(len(solved)):
        data.qpos[:3] = motion["root_pos"][i]
        data.qpos[3:7] = np.roll(motion["root_rot"][i], 1)
        data.qpos[7:] = solved[i]
        mj.mj_forward(model, data)
        minimum_clearance = min(
            minimum_clearance,
            *(mj.mj_geomDistance(model, data, a, b, 0.5, None) for a, b in pairs),
        )
    print(f"minimum left-hand clearance: {minimum_clearance:.3f} m")
    if minimum_clearance < -1e-3:
        raise ValueError("left hand penetrates the right hand or torso")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--reference", type=Path,
        default=ROOT / "motion_pkl_23dof/cut1/eight_cut_1_v028.pkl",
    )
    parser.add_argument(
        "--xml", type=Path,
        default=ROOT / "assets/unitree_g1/g1_mocap_23dof.xml",
    )
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        raise ValueError("output must be a new versioned PKL and must not already exist")
    with args.input.open("rb") as handle:
        motion = pickle.load(handle)
    with args.reference.open("rb") as handle:
        reference = pickle.load(handle)
    original = np.asarray(motion["dof_pos"])
    if original.shape != (4412, 23) or float(motion["fps"]) != 30.0:
        raise ValueError("expected the complete native G1 23DoF cut2 motion")

    model = mj.MjModel.from_xml_path(str(args.xml))
    weights = phase_weights(len(original), float(motion["fps"]))
    low_pose = prepare_low_pose(motion, model, reference)
    target = build_left_target_trajectory(motion, model, low_pose, weights)
    solved = solve_left_arm_trajectory(motion, model, target, low_pose, weights)
    local = update_left_body_pos(motion, solved, model)
    validate_motion(motion, solved, local, model)

    result = dict(motion, dof_pos=solved, local_body_pos=local)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(result, handle)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
