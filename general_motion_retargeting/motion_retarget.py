
import mink
import mujoco as mj
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool=True,
        output_fps: float=None,
    ) -> None:

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")
        
        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.use_separate_ik_offsets = ik_config.get("use_separate_ik_offsets", False)
        self.posture_cost = ik_config.get("posture_cost", 0.0)
        self.output_joint_velocity_limit = ik_config.get("output_joint_velocity_limit")
        self.output_joint_acceleration_limit = ik_config.get("output_joint_acceleration_limit")
        self.grounding_config = ik_config.get("grounding")
        self.wrist_target_clearance = ik_config.get("wrist_target_clearance")
        self._configure_soft_joint_limit(ik_config.get("soft_joint_limit"))
        self._configure_arm_branch_reseed(
            ik_config.get("arm_branch_reseeds", ik_config.get("arm_branch_reseed"))
        )
        self.output_fps = output_fps
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        if self.output_joint_velocity_limit is not None or self.output_joint_acceleration_limit is not None:
            if self.output_fps is None or self.output_fps <= 0:
                raise ValueError("output_fps must be positive when output motion limiting is enabled")
            actuated_joint_ids = np.array([
                self.model.actuator_trnid[i, 0]
                for i in range(self.model.nu)
            ], dtype=int)
            self.actuated_qpos_indices = self.model.jnt_qposadr[actuated_joint_ids].astype(int)
            self.actuated_qpos_lower = self.model.jnt_range[actuated_joint_ids, 0].copy()
            self.actuated_qpos_upper = self.model.jnt_range[actuated_joint_ids, 1].copy()
        else:
            self.actuated_qpos_indices = np.array([], dtype=int)
        self.previous_output_qpos = None
        self.previous_output_velocity = None

        self.max_iter = 10

        self.solver = solver
        self.damping = damping

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 
        self._configure_collision_avoidance(ik_config.get("collision_avoidance"))
            
        self.setup_retarget_configuration()
        if self.collision_geom_pairs:
            self.last_collision_safe_qpos = self.configuration.data.qpos.copy()

    def _configure_soft_joint_limit(self, config):
        self.soft_joint_limit_cost = None
        self.soft_joint_limit_qpos_indices = np.array([], dtype=int)
        self.soft_joint_limit_lower = np.array([], dtype=float)
        self.soft_joint_limit_upper = np.array([], dtype=float)

        if config is None:
            return

        margin_ratio = float(config["margin_ratio"])
        cost = float(config["cost"])
        joint_names = config["joint_names"]

        if not 0.0 < margin_ratio < 0.5:
            raise ValueError("soft_joint_limit.margin_ratio must be between 0 and 0.5")
        if cost <= 0.0:
            raise ValueError("soft_joint_limit.cost must be positive")
        if not isinstance(joint_names, list) or not joint_names:
            raise ValueError("soft_joint_limit.joint_names must be a non-empty list")
        if len(joint_names) != len(set(joint_names)):
            raise ValueError("soft_joint_limit.joint_names must not contain duplicates")

        dof_indices = []
        qpos_indices = []
        lower_bounds = []
        upper_bounds = []
        for joint_name in joint_names:
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Unknown soft joint limit joint: {joint_name}")
            if self.model.jnt_type[joint_id] != mj.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"Soft joint limit joint must be a hinge: {joint_name}")
            if not self.model.jnt_limited[joint_id]:
                raise ValueError(f"Soft joint limit joint has no range: {joint_name}")

            lower, upper = self.model.jnt_range[joint_id]
            margin = margin_ratio * (upper - lower)
            dof_indices.append(self.model.jnt_dofadr[joint_id])
            qpos_indices.append(self.model.jnt_qposadr[joint_id])
            lower_bounds.append(lower + margin)
            upper_bounds.append(upper - margin)

        self.soft_joint_limit_cost = np.zeros(self.model.nv)
        self.soft_joint_limit_cost[np.asarray(dof_indices, dtype=int)] = cost
        self.soft_joint_limit_qpos_indices = np.asarray(qpos_indices, dtype=int)
        self.soft_joint_limit_lower = np.asarray(lower_bounds)
        self.soft_joint_limit_upper = np.asarray(upper_bounds)

    def _configure_arm_branch_reseed(self, config):
        self.arm_branch_reseeds = []
        if config is None:
            return

        configs = config if isinstance(config, list) else [config]
        for arm_config in configs:
            joint_names = arm_config["joint_names"]
            seed_qpos = np.asarray(arm_config["seed_qpos"], dtype=float)
            if len(joint_names) != len(seed_qpos) or not joint_names:
                raise ValueError("arm_branch_reseed joint_names and seed_qpos must have the same non-zero length")

            qpos_indices = []
            for joint_name in joint_names:
                joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id < 0:
                    raise ValueError(f"Unknown arm branch reseed joint: {joint_name}")
                qpos_indices.append(self.model.jnt_qposadr[joint_id])

            robot_root_body_id = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_BODY, arm_config["robot_root_body"]
            )
            robot_wrist_body_id = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_BODY, arm_config["robot_wrist_body"]
            )
            if robot_root_body_id < 0 or robot_wrist_body_id < 0:
                raise ValueError("arm_branch_reseed robot body names must exist in the model")

            self.arm_branch_reseeds.append({
                **arm_config,
                "qpos_indices": np.asarray(qpos_indices, dtype=int),
                "seed_qpos": seed_qpos,
                "robot_root_body_id": robot_root_body_id,
                "robot_wrist_body_id": robot_wrist_body_id,
                "active": False,
            })

    def _configure_collision_avoidance(self, config):
        self.collision_geom_pairs = []
        self.collision_safety_distance = None
        self.collision_check_data = None
        if config is None:
            return

        def collision_geom_ids(body_names):
            geom_ids = []
            for body_name in body_names:
                body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body_name)
                if body_id < 0:
                    raise ValueError(f"Unknown collision avoidance body: {body_name}")
                geom_start = self.model.body_geomadr[body_id]
                geom_end = geom_start + self.model.body_geomnum[body_id]
                geom_ids.extend(
                    geom_id
                    for geom_id in range(geom_start, geom_end)
                    if self.model.geom_contype[geom_id] != 0
                    or self.model.geom_conaffinity[geom_id] != 0
                )
            if not geom_ids:
                raise ValueError(f"No collision geoms found for bodies: {body_names}")
            return geom_ids

        geom_pairs = [
            (collision_geom_ids(pair[0]), collision_geom_ids(pair[1]))
            for pair in config["body_pairs"]
        ]
        projection_joint_names = config.get("projection_joint_names")
        if projection_joint_names is None or len(projection_joint_names) != len(geom_pairs):
            raise ValueError(
                "collision_avoidance.projection_joint_names must match body_pairs"
            )
        self.collision_projection_groups = []
        for (geom_group_a, geom_group_b), joint_names in zip(
            geom_pairs, projection_joint_names
        ):
            qpos_indices = []
            for joint_name in joint_names:
                joint_id = mj.mj_name2id(
                    self.model, mj.mjtObj.mjOBJ_JOINT, joint_name
                )
                if joint_id < 0:
                    raise ValueError(
                        f"Unknown collision projection joint: {joint_name}"
                    )
                qpos_indices.append(self.model.jnt_qposadr[joint_id])
            self.collision_projection_groups.append({
                "geom_pairs": [
                    (geom_id_a, geom_id_b)
                    for geom_id_a in geom_group_a
                    for geom_id_b in geom_group_b
                    if geom_id_a != geom_id_b
                ],
                "qpos_indices": np.asarray(qpos_indices, dtype=int),
            })
        self.collision_geom_pairs = [
            (geom_id_a, geom_id_b)
            for geom_group_a, geom_group_b in geom_pairs
            for geom_id_a in geom_group_a
            for geom_id_b in geom_group_b
            if geom_id_a != geom_id_b
        ]
        self.collision_safety_distance = float(config["safety_distance"])
        self.collision_check_data = mj.MjData(self.model)
        self.ik_limits.append(
            mink.CollisionAvoidanceLimit(
                self.model,
                geom_pairs=geom_pairs,
                gain=float(config.get("gain", 0.85)),
                minimum_distance_from_collisions=float(config["minimum_distance"]),
                collision_detection_distance=float(config["detection_distance"]),
            )
        )
        

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
    
        self.tasks1 = []
        self.tasks2 = []
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
            self.rot_offsets1[body_name] = R.from_quat(
                rot_offset, scalar_first=True
            )
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.tasks1.append(task)
                self.task_errors1[task] = []
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
            self.rot_offsets2[body_name] = R.from_quat(
                rot_offset, scalar_first=True
            )
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.tasks2.append(task)
                self.task_errors2[task] = []

        self.posture_task = None
        if self.posture_cost > 0:
            self.posture_task = mink.PostureTask(self.model, cost=self.posture_cost)
            self.tasks1.append(self.posture_task)
            self.tasks2.append(self.posture_task)

        self.soft_joint_limit_task = None
        if self.soft_joint_limit_cost is not None:
            self.soft_joint_limit_task = mink.PostureTask(
                self.model, cost=self.soft_joint_limit_cost
            )
            self.tasks1.append(self.soft_joint_limit_task)
            self.tasks2.append(self.soft_joint_limit_task)

  
    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)

        human_data1 = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        if self.use_separate_ik_offsets:
            human_data2 = self.offset_human_data(human_data, self.pos_offsets2, self.rot_offsets2)
        else:
            human_data2 = human_data1
        if offset_to_ground:
            human_data1 = self.offset_human_data_to_ground(human_data1)
            if self.use_separate_ik_offsets:
                human_data2 = self.offset_human_data_to_ground(human_data2)
            else:
                human_data2 = human_data1
        self.target_human_data2 = human_data2
        self.scaled_human_data = human_data1
        if self.wrist_target_clearance is not None:
            self._apply_wrist_target_clearance(human_data2)

        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data1[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data2[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

    def _apply_wrist_target_clearance(self, human_data):
        config = self.wrist_target_clearance
        root_pos, root_quat = human_data[config["root_body"]]
        root_rotation = R.from_quat(root_quat, scalar_first=True)
        inverse_root_rotation = root_rotation.inv()
        wrist_keys = {
            "left": config["left_wrist_body"],
            "right": config["right_wrist_body"],
        }
        wrist_local = {
            side: inverse_root_rotation.apply(human_data[key][0] - root_pos)
            for side, key in wrist_keys.items()
        }

        for side, local_pos in wrist_local.items():
            if local_pos[2] <= float(config["low_height_max"]):
                local_pos[0] = max(local_pos[0], float(config["minimum_forward"]))
                minimum_lateral = float(config["minimum_lateral"])
                if side == "left":
                    local_pos[1] = max(local_pos[1], minimum_lateral)
                else:
                    local_pos[1] = min(local_pos[1], -minimum_lateral)

        wrist_distance = np.linalg.norm(wrist_local["left"] - wrist_local["right"])
        activation_distance = float(config["hand_hand_activation_distance"])
        full_distance = float(config["hand_hand_full_distance"])
        if wrist_distance < activation_distance:
            blend = np.clip(
                (activation_distance - wrist_distance)
                / (activation_distance - full_distance),
                0.0,
                1.0,
            )
            blend = blend * blend * (3.0 - 2.0 * blend)
            desired_separation = float(config["hand_hand_forward_separation"])
            current_separation = wrist_local["left"][0] - wrist_local["right"][0]
            correction = 0.5 * max(desired_separation - current_separation, 0.0) * blend
            wrist_local["left"][0] += correction
            wrist_local["right"][0] -= correction

        for side, key in wrist_keys.items():
            human_data[key][0] = root_pos + root_rotation.apply(wrist_local[side])
            
            
    def retarget(self, human_data, offset_to_ground=False):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)

        if self.posture_task is not None:
            posture_target = (
                self.previous_output_qpos
                if self.previous_output_qpos is not None
                else self.configuration.data.qpos.copy()
            )
            self.posture_task.set_target(posture_target)

        if self.soft_joint_limit_task is not None:
            self._update_soft_joint_limit_target()

        reseed_configs = [
            config
            for config in self.arm_branch_reseeds
            if self._should_try_arm_branch_reseed(config)
        ]
        self._solve_current_targets()

        for config in reseed_configs:
            self._try_arm_branch_reseed(config)

        qpos = self.configuration.data.qpos.copy()
        if self.output_joint_velocity_limit is not None or self.output_joint_acceleration_limit is not None:
            qpos = self._enforce_output_frame_limits(qpos)
            qpos[self.actuated_qpos_indices] = np.clip(
                qpos[self.actuated_qpos_indices],
                self.actuated_qpos_lower,
                self.actuated_qpos_upper,
            )
        if self.collision_geom_pairs:
            qpos = self._enforce_collision_safe_step(qpos)
        if self.previous_output_qpos is not None and self.previous_output_velocity is not None:
            self.previous_output_velocity = (
                qpos[self.actuated_qpos_indices]
                - self.previous_output_qpos[self.actuated_qpos_indices]
            ) * self.output_fps
        if (
            self.output_joint_velocity_limit is not None
            or self.output_joint_acceleration_limit is not None
            or self.collision_geom_pairs
        ):
            self.configuration.update(qpos)
        if (
            self.posture_task is not None
            or self.output_joint_velocity_limit is not None
            or self.output_joint_acceleration_limit is not None
        ):
            self.previous_output_qpos = qpos.copy()
        return qpos

    def _solve_current_targets(self):
        if self.use_ik_match_table1:
            # Solve the IK problem
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep
            vel1 = mink.solve_ik(
                self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel1 = mink.solve_ik(
                    self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1

        if self.use_ik_match_table2:
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                # Solve the IK problem with the second task
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel2, dt)
                
                next_error = self.error2()
                num_iter += 1

    def _should_try_arm_branch_reseed(self, config):
        if self.previous_output_qpos is None:
            return False

        human_root_pos = self.target_human_data2[config["human_root_body"]][0]
        human_wrist_pos = self.target_human_data2[config["human_wrist_body"]][0]
        target_wrist_height = human_wrist_pos[2] - human_root_pos[2]

        robot_root_pos = self.configuration.data.xpos[config["robot_root_body_id"]]
        robot_wrist_pos = self.configuration.data.xpos[config["robot_wrist_body_id"]]
        current_wrist_height = robot_wrist_pos[2] - robot_root_pos[2]

        if config["active"]:
            config["active"] = (
                target_wrist_height <= float(config["target_wrist_height_exit"])
                and current_wrist_height >= float(config["current_wrist_height_exit"])
            )
        elif (
            target_wrist_height <= float(config["target_wrist_height_max"])
            and current_wrist_height >= float(config["current_wrist_height_min"])
        ):
            config["active"] = True
        return config["active"]

    def _arm_branch_wrist_error(self, config):
        target = self.target_human_data2[config["human_wrist_body"]][0]
        actual = self.configuration.data.xpos[config["robot_wrist_body_id"]]
        return np.linalg.norm(actual - target)

    def _try_arm_branch_reseed(self, config):
        normal_qpos = self.configuration.data.qpos.copy()
        normal_wrist_error = self._arm_branch_wrist_error(config)

        seed_qpos = normal_qpos.copy()
        seed_qpos[config["qpos_indices"]] = config["seed_qpos"]
        self.configuration.update(seed_qpos)
        if self.posture_task is not None:
            self.posture_task.set_target(seed_qpos)
        if self.soft_joint_limit_task is not None:
            self._update_soft_joint_limit_target()

        self._solve_current_targets()
        candidate_wrist_error = self._arm_branch_wrist_error(config)
        if candidate_wrist_error + float(config["min_wrist_error_improvement"]) >= normal_wrist_error:
            self.configuration.update(normal_qpos)
            config["active"] = False

        if self.posture_task is not None:
            posture_target = (
                self.previous_output_qpos
                if self.previous_output_qpos is not None
                else self.configuration.data.qpos.copy()
            )
            self.posture_task.set_target(posture_target)

    def _update_soft_joint_limit_target(self):
        target = self.configuration.data.qpos.copy()
        indices = self.soft_joint_limit_qpos_indices
        target[indices] = np.clip(
            target[indices],
            self.soft_joint_limit_lower,
            self.soft_joint_limit_upper,
        )
        self.soft_joint_limit_task.set_target(target)

    def _minimum_collision_distance(self, qpos, geom_pairs=None):
        self.collision_check_data.qpos[:] = qpos
        mj.mj_forward(self.model, self.collision_check_data)
        from_to = np.empty(6)
        return min(
            mj.mj_geomDistance(
                self.model,
                self.collision_check_data,
                geom_id_a,
                geom_id_b,
                self.collision_safety_distance,
                from_to,
            )
            for geom_id_a, geom_id_b in (
                self.collision_geom_pairs if geom_pairs is None else geom_pairs
            )
        )

    def _enforce_collision_safe_step(self, qpos, commit=True):
        if self._minimum_collision_distance(qpos) >= self.collision_safety_distance:
            if commit:
                self.last_collision_safe_qpos = qpos.copy()
            return qpos

        safe_qpos = qpos.copy()
        for _ in range(3):
            changed = False
            for group in self.collision_projection_groups:
                geom_pairs = group["geom_pairs"]
                if self._minimum_collision_distance(safe_qpos, geom_pairs) >= self.collision_safety_distance:
                    continue
                indices = group["qpos_indices"]
                start_qpos = safe_qpos.copy()
                source_qpos = (
                    self.previous_output_qpos
                    if self.previous_output_qpos is not None
                    else self.last_collision_safe_qpos
                )
                start_qpos[indices] = source_qpos[indices]
                if self._minimum_collision_distance(start_qpos, geom_pairs) < self.collision_safety_distance:
                    start_qpos[indices] = self.last_collision_safe_qpos[indices]

                low = 0.0
                high = 1.0
                candidate_safe_qpos = start_qpos.copy()
                for _ in range(12):
                    alpha = 0.5 * (low + high)
                    candidate_qpos = safe_qpos.copy()
                    candidate_qpos[indices] = (
                        start_qpos[indices]
                        + alpha * (safe_qpos[indices] - start_qpos[indices])
                    )
                    if self._minimum_collision_distance(candidate_qpos, geom_pairs) >= self.collision_safety_distance:
                        low = alpha
                        candidate_safe_qpos = candidate_qpos
                    else:
                        high = alpha
                safe_qpos = candidate_safe_qpos
                changed = True
            if not changed or self._minimum_collision_distance(safe_qpos) >= self.collision_safety_distance:
                break

        if self._minimum_collision_distance(safe_qpos) < self.collision_safety_distance:
            safe_qpos = self.last_collision_safe_qpos.copy()
        if commit:
            self.last_collision_safe_qpos = safe_qpos.copy()
        return safe_qpos

    def _enforce_output_frame_limits(self, qpos):
        indices = self.actuated_qpos_indices
        qpos[indices] = np.clip(
            qpos[indices],
            self.actuated_qpos_lower,
            self.actuated_qpos_upper,
        )
        if self.previous_output_qpos is None:
            if self.output_joint_acceleration_limit is not None:
                self.previous_output_velocity = np.zeros(len(self.actuated_qpos_indices))
            return qpos

        velocity = (qpos[indices] - self.previous_output_qpos[indices]) * self.output_fps
        if self.output_joint_velocity_limit is not None:
            velocity = np.clip(
                velocity,
                -self.output_joint_velocity_limit,
                self.output_joint_velocity_limit,
            )
        if self.output_joint_acceleration_limit is not None:
            if self.previous_output_velocity is None:
                self.previous_output_velocity = np.zeros(len(indices))
            acceleration_limit = float(self.output_joint_acceleration_limit)
            timestep = 1.0 / self.output_fps
            distance_to_lower = np.maximum(
                self.previous_output_qpos[indices] - self.actuated_qpos_lower,
                0.0,
            )
            distance_to_upper = np.maximum(
                self.actuated_qpos_upper - self.previous_output_qpos[indices],
                0.0,
            )
            braking_term = (acceleration_limit * timestep) ** 2
            minimum_braking_velocity = (
                acceleration_limit * timestep
                - np.sqrt(braking_term + 2.0 * acceleration_limit * distance_to_lower)
            )
            maximum_braking_velocity = (
                -acceleration_limit * timestep
                + np.sqrt(braking_term + 2.0 * acceleration_limit * distance_to_upper)
            )
            velocity = np.clip(
                velocity,
                minimum_braking_velocity,
                maximum_braking_velocity,
            )
            max_velocity_delta = self.output_joint_acceleration_limit / self.output_fps
            velocity = self.previous_output_velocity + np.clip(
                velocity - self.previous_output_velocity,
                -max_velocity_delta,
                max_velocity_delta,
            )
            self.previous_output_velocity = velocity.copy()
        qpos[indices] = self.previous_output_qpos[indices] + velocity / self.output_fps
        return qpos


    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat
            
            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
            
            offset_human_data[body_name][0] = pos + global_pos_offset
           
        return offset_human_data
            
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data
