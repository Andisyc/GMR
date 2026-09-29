"""Cut four opening seconds from cut2 and settle its final second to standing."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--xml", type=Path, default=ROOT / "assets/unitree_g1/g1_mocap_23dof.xml")
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.input.resolve():
        raise ValueError("output must be a new versioned PKL")
    with args.input.open("rb") as handle:
        source = pickle.load(handle)
    fps = float(source["fps"])
    original = np.asarray(source["dof_pos"])
    if original.shape != (4412, 23) or fps != 30.0:
        raise ValueError("expected the native G1 23DoF cut2 v007 motion")

    cut = round(4.0 * fps)
    result = dict(source)
    for name in ("root_pos", "root_rot", "dof_pos", "local_body_pos"):
        result[name] = np.asarray(source[name])[cut:].copy()

    # The original is already upright at 144 s.  Use that same-sequence pose
    # so the end can settle before the source's final squat starts.
    anchor = round(144.0 * fps)
    start = round(145.0 * fps)
    settled = round(146.0 * fps)
    target_joints = original[anchor].copy()
    target_joints[13:23] = original[round(77.0 * fps), 13:23]
    target_root = np.asarray(source["root_pos"])[anchor].copy()
    target_rotation = np.asarray(source["root_rot"])[anchor].copy()
    joint_start = original[start].copy()
    root_start = np.asarray(source["root_pos"])[start].copy()
    rotation_start = np.asarray(source["root_rot"])[start].copy()
    joint_velocity = (original[start] - original[start - 1]) * fps
    root_velocity = (np.asarray(source["root_pos"])[start]
                     - np.asarray(source["root_pos"])[start - 1]) * fps
    previous_rotation = Rotation.from_quat(source["root_rot"][start - 1])
    initial_rotation = Rotation.from_quat(rotation_start)
    angular_velocity = (previous_rotation.inv() * initial_rotation).as_rotvec() * fps

    # Keep the nearby planted feet in place: the anchor and transition start
    # differ by only millimetres in horizontal root position.
    target_root[:2] = root_start[:2]
    target_rotation = rotation_start
    for frame in range(start, len(original)):
        output_frame = frame - cut
        u = min((frame - start) / (settled - start), 1.0)
        h00 = 2 * u**3 - 3 * u**2 + 1
        h10 = u**3 - 2 * u**2 + u
        h01 = -2 * u**3 + 3 * u**2
        duration = (settled - start) / fps
        result["dof_pos"][output_frame] = (
            h00 * joint_start + h10 * duration * joint_velocity + h01 * target_joints
        )
        result["root_pos"][output_frame] = (
            h00 * root_start + h10 * duration * root_velocity + h01 * target_root
        )
        result["root_rot"][output_frame] = (
            initial_rotation * Rotation.from_rotvec(h10 * duration * angular_velocity)
        ).as_quat()

    model = mj.MjModel.from_xml_path(str(args.xml))
    data = mj.MjData(model)
    body_ids = [model.body(name).id for name in source["link_body_list"]]
    data.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
    for frame in range(start - cut, len(result["dof_pos"])):
        data.qpos[7:] = result["dof_pos"][frame]
        mj.mj_forward(model, data)
        result["local_body_pos"][frame] = data.xpos[body_ids]

    before = slice(None, start - cut)
    for name in ("root_pos", "root_rot", "dof_pos", "local_body_pos"):
        if not np.array_equal(result[name][before], source[name][cut:start]):
            raise ValueError(f"preserved motion changed: {name}")
    if len(result["dof_pos"]) != len(original) - cut:
        raise ValueError("incorrect cropped length")
    if not np.allclose(result["dof_pos"][settled - cut:], target_joints):
        raise ValueError("final second is not settled")
    speed = np.max(np.abs(np.diff(result["dof_pos"][start - cut:], axis=0) * fps))
    print(f"frames: {len(original)} -> {len(result['dof_pos'])}; duration: {len(result['dof_pos']) / fps:.2f} s")
    print(f"end pose from original 144 s; final-second peak joint speed: {speed:.2f} rad/s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(result, handle)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
