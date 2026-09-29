"""Balance the paired arm raise near 1:47 of the full v027 motion."""

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
ARMS = np.arange(13, 23)
LEFT = np.arange(13, 18)
RIGHT = np.arange(18, 23)
MIRROR = np.array([1, -1, -1, 1, -1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--xml', type=Path, default=ROOT / 'assets/unitree_g1/g1_mocap_23dof.xml')
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError('input and output must differ')
    with args.input.open('rb') as f:
        motion = pickle.load(f)
    original = np.asarray(motion['dof_pos'])
    fps = float(motion['fps'])
    if original.shape != (6752, 23) or fps != 30:
        raise ValueError('expected the complete G1 23DoF trajectory')

    model = mj.MjModel.from_xml_path(str(args.xml))
    data = mj.MjData(model)
    wrists = np.array([mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
                       for name in ('left_wrist_roll_rubber_hand', 'right_wrist_roll_rubber_hand')])
    hand_geoms = [[k for k in range(model.ngeom) if model.geom_bodyid[k] == b] for b in wrists]
    torso = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
             for name in ('pelvis', 'torso_link')]
    torso_geoms = [k for k in range(model.ngeom) if model.geom_bodyid[k] in torso]
    pairs = ([(a, b) for a in hand_geoms[0] for b in hand_geoms[1] + torso_geoms]
             + [(a, b) for a in hand_geoms[1] for b in torso_geoms])
    lo, hi = model_limits(model, ARMS).T
    start, stop = 105.2, 111.0
    begin, end = round(start * fps), round(stop * fps)
    frames = np.arange(begin, end + 1)
    candidate = original.copy()

    for i in frames:
        t = i / fps
        strength = ease((t - start) / 0.8) * ease((stop - t) / 2.2)
        if strength < 1e-8:
            continue
        base = np.r_[motion['root_pos'][i], np.roll(motion['root_rot'][i], 1), original[i]].copy()
        rotation = Rotation.from_quat(motion['root_rot'][i])
        data.qpos[:] = base
        mj.mj_forward(model, data)
        original_wrist = data.xpos[wrists].copy()
        local = rotation.inv().apply(original_wrist - motion['root_pos'][i])
        gap = local[0, 0] - local[1, 0]
        target = original_wrist.copy()
        # Share the forward asymmetry between both arms. A small symmetric
        # outward offset leaves room for the hand geoms as the arms cross.
        target[0] += rotation.apply([-0.55 * gap, 0.02, 0])
        target[1] += rotation.apply([0.20 * gap, -0.03, 0])
        preferred = original[i, ARMS].copy()
        preferred[:5] = 0.55 * preferred[:5] + 0.45 * (original[i, RIGHT] * MIRROR)

        def residual(angles):
            data.qpos[:] = base
            data.qpos[7 + ARMS] = angles
            mj.mj_forward(model, data)
            clearance = np.array([mj.mj_geomDistance(model, data, a, b, 0.4, None)
                                  for a, b in pairs])
            symmetry = angles[:5] - angles[5:] * MIRROR
            return np.r_[18 * (data.xpos[wrists] - target).ravel(),
                         0.24 * (angles - preferred),
                         0.16 * symmetry,
                         50 * np.minimum(clearance - 0.025, 0)]

        fit = least_squares(residual, np.clip(original[i, ARMS], lo + 1e-7, hi - 1e-7),
                            bounds=(lo, hi), max_nfev=65, ftol=2e-4)
        candidate[i, ARMS] = original[i, ARMS] + strength * (fit.x - original[i, ARMS])

    # Filter the correction itself, keeping every unedited frame byte-for-byte.
    correction = candidate[:, ARMS] - original[:, ARMS]
    smoothed = gaussian_filter1d(correction, sigma=3.0, axis=0, mode='constant')
    result = original.copy()
    for i in frames:
        t = i / fps
        envelope = ease((t - start) / 0.8) * ease((stop - t) / 2.2)
        result[i, ARMS] = original[i, ARMS] + envelope * smoothed[i]
    result[begin, ARMS] = original[begin, ARMS]
    result[end, ARMS] = original[end, ARMS]
    mask = np.zeros(len(original), dtype=bool)
    mask[frames] = True
    if not np.array_equal(result[~mask], original[~mask]) or not np.array_equal(result[:, :13], original[:, :13]):
        raise ValueError('motion outside the paired arm window changed')
    if np.any(result[:, ARMS] < lo - 1e-5) or np.any(result[:, ARMS] > hi + 1e-5):
        raise ValueError('arm joint limit violation')
    local, names = recompute_local(result, motion, mask, args.xml)
    if list(names) != list(motion['link_body_list']):
        raise ValueError('body order changed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as f:
        pickle.dump(dict(motion, dof_pos=result, local_body_pos=local), f)
    print(f'saved {args.output}')


if __name__ == '__main__':
    main()
