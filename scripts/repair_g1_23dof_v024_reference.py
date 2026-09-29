"""Finish the left-arm descent by borrowing the matching earlier v024 cycle."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from repair_g1_23dof_v013 import ease, model_limits, recompute_local

ROOT = Path(__file__).resolve().parents[1]
LEFT = np.array([13, 14, 15, 16])
REFERENCE_SHIFT = round(16.7 * 30)  # 105 s matches the earlier 88.3 s cycle.


def hand_distance(model, data, pairs):
    return min(mj.mj_geomDistance(model, data, a, b, 0.4, None)
               for a, b in pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--xml", type=Path, default=ROOT / "assets/unitree_g1/g1_mocap_23dof.xml")
    parser.add_argument("--continue-raise", action="store_true",
                        help="extend the matching reference into the next left-arm raise")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output must differ")
    with args.input.open("rb") as f:
        motion = pickle.load(f)
    fps = float(motion["fps"])
    original = np.asarray(motion["dof_pos"])
    if original.shape != (6752, 23) or fps != 30:
        raise ValueError("expected the complete G1 23DoF trajectory")
    model = mj.MjModel.from_xml_path(str(args.xml))
    data = mj.MjData(model)
    hand_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "left_wrist_roll_rubber_hand")
    left_geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] == hand_body]
    other_bodies = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name) for name in
                    ("right_wrist_roll_rubber_hand", "pelvis", "torso_link")]
    other_geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] in other_bodies]
    pairs = [(a, b) for a in left_geoms for b in other_geoms]
    lo, hi = model_limits(model, LEFT).T
    result = original.copy()
    start, stop = ((106.5, 112.0) if args.continue_raise else (103.4, 108.4))
    begin, end = round(start*fps), round(stop*fps)
    frames = np.arange(begin, end+1)
    for i in frames:
        t = i/fps
        blend = (ease((t-start)/(1.2 if args.continue_raise else 1.4))
                 * ease((stop-t)/(1.5 if args.continue_raise else 1.4)))
        if blend == 0:
            continue
        base = np.r_[motion["root_pos"][i], np.roll(motion["root_rot"][i], 1), original[i]].copy()
        reference = original[i-REFERENCE_SHIFT, LEFT]
        preferred = original[i, LEFT] + blend*(reference-original[i, LEFT])
        data.qpos[:] = base
        data.qpos[7+LEFT] = preferred
        mj.mj_forward(model, data)
        desired = data.xpos[hand_body].copy()
        # The earlier cycle is the pose template. Only shift its late hand
        # slightly forward where the current right hand crosses its path.
        if args.continue_raise:
            advance = 0.09*ease((t-106.8)/0.8)*ease((111.3-t)/1.5)*blend
        else:
            advance = 0.08*ease((t-105.2)/1.1)*ease((108.4-t)/1.0)*blend
        desired += Rotation.from_quat(motion["root_rot"][i]).apply([advance, 0, 0])
        if advance > 1e-5 or hand_distance(model, data, pairs) < 0.035:
            def residual(q):
                data.qpos[:] = base
                data.qpos[7+LEFT] = q
                mj.mj_forward(model, data)
                clearance = np.array([mj.mj_geomDistance(model, data, a, b, 0.4, None)
                                      for a, b in pairs])
                return np.r_[12*(data.xpos[hand_body]-desired),
                             (np.array([0.16, 0.60, 0.30, 0.16]) if args.continue_raise
                              else 0.16)*(q-preferred),
                             35*np.minimum(clearance-0.035, 0)]
            fit = least_squares(residual, np.clip(preferred, lo+1e-7, hi-1e-7),
                                bounds=(lo, hi), max_nfev=40, ftol=1e-4)
            if args.continue_raise:
                # Match the original trajectory's velocity as the edit closes.
                fade_out = ease((stop-t)/1.5)
                result[i, LEFT] = preferred + fade_out*(fit.x-preferred)
            else:
                result[i, LEFT] = fit.x
        else:
            result[i, LEFT] = preferred
    for j in LEFT:
        filtered = gaussian_filter1d(result[:, j], sigma=4.0, mode="nearest")
        result[frames, j] = filtered[frames]
    result[begin, LEFT] = original[begin, LEFT]
    result[end, LEFT] = original[end, LEFT]
    mask = np.zeros(len(original), dtype=bool)
    mask[begin:end+1] = True
    local, names = recompute_local(result, motion, mask, args.xml)
    if list(names) != list(motion["link_body_list"]):
        raise ValueError("body order changed")
    if not np.isfinite(result).all() or not np.array_equal(result[~mask], original[~mask]):
        raise ValueError("invalid motion outside the repair window")
    if not np.array_equal(result[:, :13], original[:, :13]) or not np.array_equal(result[:, 17:], original[:, 17:]):
        raise ValueError("non-left-arm joint changed")
    if np.any(result[:, LEFT] < lo-1e-5) or np.any(result[:, LEFT] > hi+1e-5):
        raise ValueError("left joint limit violation")
    output = dict(motion, dof_pos=result, local_body_pos=local)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(output, f)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
