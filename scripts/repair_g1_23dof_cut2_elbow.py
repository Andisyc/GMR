"""Keep cut2's repeated front-of-body gesture symmetric without folding the left elbow."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from repair_g1_23dof_cut2 import (
    LEFT, MAX_ACCEL, MAX_SPEED, MIRROR, ROOT,
    model_joint_limits, update_left_body_pos, validate_motion,
)


def ease(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def problem_windows(q: np.ndarray, fps: float) -> list[tuple[int, int, int, int]]:
    """Group recurring elbow-limit episodes and include room to join v003."""
    bad = ((q[:, 16] > 1.85) & (q[:, 16] - q[:, 21] > 0.5)
           & (np.arange(len(q)) >= round(75.0 * fps)))
    ids = np.flatnonzero(bad)
    if not len(ids):
        raise ValueError("no repeated elbow-fold episodes found")
    groups = np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1)
    pad = round(0.7 * fps)
    windows = [
        (max(0, int(group[0]) - pad), int(group[0]),
         int(group[-1]), min(len(q) - 1, int(group[-1]) + pad))
        for group in groups if len(group) >= round(0.25 * fps)
    ]
    if len(windows) < 5:
        raise ValueError("expected the repeated front-of-body gesture")
    return windows


def repair(motion: dict, model: mj.MjModel) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    original = np.asarray(motion["dof_pos"])
    fps = float(motion["fps"])
    windows = problem_windows(original, fps)
    result = original.copy()
    changed = np.zeros(len(original), dtype=bool)
    lower, upper = model_joint_limits(model)
    left_hand = model.body("left_wrist_roll_rubber_hand").id
    right_hand = model.body("right_wrist_roll_rubber_hand").id
    left_elbow = model.body("left_elbow_link").id
    right_elbow = model.body("right_elbow_link").id
    torso = model.body("torso_link").id
    pairs = [
        (a, b) for a in range(model.ngeom) for b in range(model.ngeom)
        if model.geom_bodyid[a] == left_hand and model.geom_bodyid[b] == right_hand
        and model.geom_contype[a] > 0 and model.geom_contype[b] > 0
    ]
    data = mj.MjData(model)
    data.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)

    for start, core_start, core_end, stop in windows:
        candidate = original[start:stop + 1, LEFT].copy()
        previous = original[start, LEFT].copy()
        for i in range(start, stop + 1):
            data.qpos[7:] = original[i]
            mirrored = np.clip(original[i, 18:22] * MIRROR, lower, upper)
            data.qpos[7 + LEFT] = mirrored
            mj.mj_forward(model, data)
            wrist_ref = (data.xpos[left_hand] - data.xpos[torso]).copy()
            right_elbow_ref = (data.xpos[right_elbow] - data.xpos[torso]).copy()
            elbow_ref = right_elbow_ref * np.array([1.0, -1.0, 1.0])
            mirror_gap = min(
                mj.mj_geomDistance(model, data, a, b, 0.5, None) for a, b in pairs
            )
            # A mirrored wrist overlaps the right hand during this gesture.
            # Move it only as far outward as the geometry requires.
            lateral = np.clip(0.05 - mirror_gap, 0.0, 0.15)
            wrist_target = wrist_ref + np.array([0.0, lateral, 0.0])

            def residual(angles: np.ndarray) -> np.ndarray:
                data.qpos[7 + LEFT] = angles
                mj.mj_forward(model, data)
                wrist = data.xpos[left_hand] - data.xpos[torso]
                elbow = data.xpos[left_elbow] - data.xpos[torso]
                gap = min(
                    mj.mj_geomDistance(model, data, a, b, 0.5, None) for a, b in pairs
                )
                return np.r_[
                    25.0 * (wrist - wrist_target),
                    2.0 * (angles - mirrored),
                    15.0 * (angles[3] - mirrored[3]),
                    6.0 * (elbow - elbow_ref),
                    0.5 * (angles - previous),
                    200.0 * max(0.05 - gap, 0.0),
                ]

            fit = least_squares(
                residual, np.clip(previous, lower + 1e-5, upper - 1e-5),
                bounds=(lower, upper), max_nfev=80, ftol=2e-3, xtol=2e-3,
            )
            previous = fit.x
            fade_in = 1.0 if core_start == start else ease((i - start) / (core_start - start))
            fade_out = 1.0 if core_end == stop else ease((stop - i) / (stop - core_end))
            strength = min(fade_in, fade_out)
            candidate[i - start] += strength * (fit.x - original[i, LEFT])

        correction = gaussian_filter1d(
            candidate - original[start:stop + 1, LEFT], sigma=2.0,
            axis=0, mode="nearest",
        )
        result[start:stop + 1, LEFT] = np.clip(
            original[start:stop + 1, LEFT] + correction, lower, upper,
        )
        # Exact joins make the preserved portion of v003 byte-identical.
        result[start, LEFT] = original[start, LEFT]
        if stop > core_end:
            result[stop, LEFT] = original[stop, LEFT]
        changed[start:stop + 1] = True
    return result, changed, windows


def validate_elbow_repair(
    original: dict, solved: np.ndarray, changed: np.ndarray,
    windows: list[tuple[int, int, int, int]], model: mj.MjModel,
) -> None:
    prior = np.asarray(original["dof_pos"])
    if not np.array_equal(solved[~changed], prior[~changed]):
        raise ValueError("frames outside elbow repair windows changed")
    if not np.array_equal(solved[:, [j for j in range(23) if j not in LEFT]],
                          prior[:, [j for j in range(23) if j not in LEFT]]):
        raise ValueError("a non-left-arm joint changed")
    core = np.zeros(len(solved), dtype=bool)
    for _, core_start, core_end, _ in windows:
        core[core_start:core_end + 1] = True
    elbow_error = np.max(np.abs(solved[core, 16] - solved[core, 21]))
    elbow_peak = np.max(solved[core, 16])
    speed = np.max(np.abs(np.diff(solved[:, LEFT], axis=0) * original["fps"]))
    acceleration = np.max(np.abs(np.diff(solved[:, LEFT], n=2, axis=0) * original["fps"]**2))
    print(f"repaired windows (s): {[(round(a/original['fps'], 2), round(d/original['fps'], 2)) for a, _, _, d in windows]}")
    print(f"core elbow difference: {elbow_error:.2f} rad; left elbow peak: {elbow_peak:.2f} rad; "
          f"peak speed: {speed:.2f} rad/s; peak acceleration: {acceleration:.1f} rad/s^2")
    if elbow_error > 0.35 or elbow_peak > 1.65:
        raise ValueError("left elbow still folds inward during symmetric gesture")
    if speed > MAX_SPEED + 1e-4 or acceleration > MAX_ACCEL + 1e-3:
        raise ValueError("left arm exceeds speed or acceleration limit")

    data = mj.MjData(model)
    left_hand = model.body("left_wrist_roll_rubber_hand").id
    right_hand = model.body("right_wrist_roll_rubber_hand").id
    pairs = [(a, b) for a in range(model.ngeom) for b in range(model.ngeom)
             if model.geom_bodyid[a] == left_hand and model.geom_bodyid[b] == right_hand
             and model.geom_contype[a] > 0 and model.geom_contype[b] > 0]
    minimum = np.inf
    for i in np.flatnonzero(changed):
        data.qpos[:3] = original["root_pos"][i]
        data.qpos[3:7] = np.roll(original["root_rot"][i], 1)
        data.qpos[7:] = solved[i]
        mj.mj_forward(model, data)
        minimum = min(minimum, *(mj.mj_geomDistance(model, data, a, b, 0.5, None)
                                 for a, b in pairs))
    print(f"minimum hand clearance in changed frames: {minimum:.3f} m")
    if minimum < -1e-3:
        raise ValueError("hands penetrate after elbow repair")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--xml", type=Path, default=ROOT / "assets/unitree_g1/g1_mocap_23dof.xml")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        raise ValueError("output must be a new, unused versioned PKL")
    with args.input.open("rb") as handle:
        motion = pickle.load(handle)
    if np.asarray(motion["dof_pos"]).shape != (4412, 23) or float(motion["fps"]) != 30.0:
        raise ValueError("expected the complete G1 23DoF cut2 v003 motion")
    model = mj.MjModel.from_xml_path(str(args.xml))
    solved, changed, windows = repair(motion, model)
    validate_elbow_repair(motion, solved, changed, windows, model)
    local = update_left_body_pos(motion, solved, model, changed)
    validate_motion(motion, solved, local, model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(dict(motion, dof_pos=solved, local_body_pos=local), handle)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
