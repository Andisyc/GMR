"""Repair eight_cut_1 with a synchronized, continuous arm descent."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


LEFT_ARM = np.array([13, 14, 15, 16])
RIGHT_ARM = np.array([18, 19, 20, 21])
NATURAL_LEFT = np.zeros(4)
NATURAL_RIGHT = np.zeros(4)

# Keep the left-arm repair from v001.  The right arm needs the same low-arm
# branch through each cycle; rejoining its old endpoint after three seconds
# would restore the unwanted shoulder rotation.
REPAIR_SEGMENTS = (
    ("left", 100.30, 103.30),
    ("left", 132.10, 135.10),
)
RIGHT_REPAIR_WINDOW = (52.80, 149.50)


def quintic_phase(length: int) -> np.ndarray:
    phase = np.linspace(0.0, 1.0, length)
    return phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2)


def interpolate_segment(
    dof_pos: np.ndarray,
    indices: np.ndarray,
    start_frame: int,
    end_frame: int,
    start_pose: np.ndarray | None = None,
    end_pose: np.ndarray | None = None,
) -> None:
    if end_frame <= start_frame:
        raise ValueError("repair segment must contain at least two frames")
    start = dof_pos[start_frame, indices].copy() if start_pose is None else start_pose
    end = dof_pos[end_frame, indices].copy() if end_pose is None else end_pose
    blend = quintic_phase(end_frame - start_frame + 1)[:, None]
    dof_pos[start_frame : end_frame + 1, indices] = (
        start[None, :] * (1.0 - blend) + end[None, :] * blend
    )


def repair_initial_pose(dof_pos: np.ndarray, fps: float) -> np.ndarray:
    """Keep naturally hanging arms, then join the first valid forward pose."""
    mask = np.zeros(len(dof_pos), dtype=bool)
    hold_end = int(round(20.0 * fps))
    join_end = int(round(24.0 * fps))
    for indices, neutral in (
        (LEFT_ARM, NATURAL_LEFT),
        (RIGHT_ARM, NATURAL_RIGHT),
    ):
        target = dof_pos[join_end, indices].copy()
        dof_pos[: hold_end + 1, indices] = neutral
        interpolate_segment(
            dof_pos,
            indices,
            hold_end,
            join_end,
            start_pose=neutral,
            end_pose=target,
        )
    mask[: join_end + 1] = True
    return mask


def repair_synchronized_descents(dof_pos: np.ndarray, fps: float) -> np.ndarray:
    mask = np.zeros(len(dof_pos), dtype=bool)
    arm_indices = {"left": LEFT_ARM, "right": RIGHT_ARM}
    for side, start_time, end_time in REPAIR_SEGMENTS:
        start, end = (int(round(t * fps)) for t in (start_time, end_time))
        indices = arm_indices[side]
        start_pose = dof_pos[start, indices].copy()
        end_pose = dof_pos[end, indices].copy()
        blend = quintic_phase(end - start + 1)[:, None]
        dof_pos[start : end + 1, indices] = (
            (1.0 - blend) * start_pose + blend * end_pose
        )
        mask[start : end + 1] = True
    return mask


def repair_right_arm_branch(dof_pos: np.ndarray, fps: float) -> np.ndarray:
    """Carry the right arm through the low branch until it naturally rejoins."""
    start, end = (int(round(t * fps)) for t in RIGHT_REPAIR_WINDOW)
    original = dof_pos[start : end + 1, RIGHT_ARM].copy()
    mirrored = dof_pos[start : end + 1, LEFT_ARM].copy()
    mirrored *= np.array([1.0, -1.0, -1.0, 1.0])

    # The source left arm takes the wrong branch in the fourth cycle.  Reuse
    # its successful earlier cycle there, then join the neighboring cycles.
    bad_start, bad_end = (int(round(t * fps)) for t in (100.30, 116.05))
    ref_start, ref_end = (int(round(t * fps)) for t in (67.70, 83.45))
    reference = dof_pos[ref_start : ref_end + 1, LEFT_ARM].copy()
    reference *= np.array([1.0, -1.0, -1.0, 1.0])
    phase = np.linspace(0.0, 1.0, bad_end - bad_start + 1)
    template = np.column_stack(
        [np.interp(phase, np.linspace(0.0, 1.0, len(reference)), reference[:, j])
         for j in range(4)]
    )
    segment = mirrored[bad_start - start : bad_end - start + 1]
    blend = quintic_phase(len(segment))[:, None]
    segment[:] = (
        template
        + (1.0 - blend) * (segment[0] - template[0])
        + blend * (segment[-1] - template[-1])
    )

    mirrored = gaussian_filter1d(mirrored, sigma=4.0, axis=0, mode="nearest")

    # Blend only where the two branches already nearly agree.
    fade = max(2, int(round(1.0 * fps)))
    weight = np.ones((len(mirrored), 1))
    weight[:fade] = quintic_phase(fade)[:, None]
    weight[-fade:] = quintic_phase(fade)[::-1, None]
    dof_pos[start : end + 1, RIGHT_ARM] = original * (1.0 - weight) + mirrored * weight
    mask = np.zeros(len(dof_pos), dtype=bool)
    mask[start : end + 1] = True
    return mask


def recompute_local_body_pos(dof_pos: np.ndarray, xml_path: Path) -> tuple[np.ndarray, list[str]]:
    import torch

    from general_motion_retargeting.kinematics_model import KinematicsModel

    kinematics = KinematicsModel(str(xml_path), device="cpu")
    root_pos = torch.zeros((len(dof_pos), 3), dtype=torch.float32)
    root_rot = torch.zeros((len(dof_pos), 4), dtype=torch.float32)
    root_rot[:, -1] = 1.0
    local_body_pos, _ = kinematics.forward_kinematics(
        root_pos,
        root_rot,
        torch.from_numpy(dof_pos).float(),
    )
    return local_body_pos.numpy(), kinematics.body_names


def validate_repair(
    original: dict,
    result: dict,
    changed_mask: np.ndarray,
) -> None:
    original_dof = np.asarray(original["dof_pos"])
    repaired_dof = np.asarray(result["dof_pos"])
    if repaired_dof.shape != (6752, 23):
        raise ValueError(f"unexpected G1 23DoF shape: {repaired_dof.shape}")
    if not np.isfinite(repaired_dof).all():
        raise ValueError("repaired motion contains non-finite joint values")
    if not np.array_equal(repaired_dof[~changed_mask], original_dof[~changed_mask]):
        raise ValueError("joint samples outside the repair windows changed")
    preserved_indices = np.array([*range(13), 17, 22])
    if not np.array_equal(
        repaired_dof[:, preserved_indices], original_dof[:, preserved_indices]
    ):
        raise ValueError("non-shoulder/elbow joint samples changed")
    for key in ("fps", "root_pos", "root_rot", "link_body_list"):
        before = original[key]
        after = result[key]
        if isinstance(before, np.ndarray):
            equal = np.array_equal(before, after)
        else:
            equal = before == after
        if not equal:
            raise ValueError(f"preserved field changed: {key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--xml",
        type=Path,
        default=ROOT / "assets/unitree_g1/g1_mocap_23dof.xml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("output must not overwrite the input motion")
    with args.input.open("rb") as handle:
        original = pickle.load(handle)

    fps = float(original["fps"])
    if fps <= 0:
        raise ValueError("motion fps must be positive")
    original_dof = np.asarray(original["dof_pos"])
    if original_dof.shape != (6752, 23):
        raise ValueError(f"expected (6752, 23) dof_pos, got {original_dof.shape}")

    repaired_dof = original_dof.copy()
    changed_mask = repair_initial_pose(repaired_dof, fps)
    changed_mask |= repair_synchronized_descents(repaired_dof, fps)
    changed_mask |= repair_right_arm_branch(repaired_dof, fps)
    local_body_pos, body_names = recompute_local_body_pos(repaired_dof, args.xml)

    result = dict(original)
    result["dof_pos"] = repaired_dof
    result["local_body_pos"] = local_body_pos
    if list(original["link_body_list"]) != list(body_names):
        raise ValueError("kinematics body order does not match the input PKL")

    validate_repair(original, result, changed_mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(result, handle)

    edited_speeds = []
    for side, prepare_time, end_time in REPAIR_SEGMENTS:
        indices = LEFT_ARM if side == "left" else RIGHT_ARM
        start = int(round(prepare_time * fps))
        end = int(round(end_time * fps))
        edited_speeds.append(
            np.max(np.abs(np.diff(repaired_dof[start : end + 1, indices], axis=0) * fps))
        )
    right_start, right_end = (int(round(t * fps)) for t in RIGHT_REPAIR_WINDOW)
    edited_speeds.append(
        np.max(np.abs(np.diff(repaired_dof[right_start : right_end + 1, RIGHT_ARM], axis=0) * fps))
    )
    print(
        {
            "saved": str(args.output),
            "changed_frames": int(changed_mask.sum()),
            "max_edited_arm_speed_rad_s": float(max(edited_speeds)),
        }
    )


if __name__ == "__main__":
    main()
