
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
        self._configure_soft_joint_limit(ik_config.get("soft_joint_limit"))
        self.output_fps = output_fps
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        if self.output_joint_velocity_limit is not None:
            if self.output_fps is None or self.output_fps <= 0:
                raise ValueError("output_fps must be positive when output joint velocity limiting is enabled")
            self.actuated_qpos_indices = np.array([
                self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]
                for i in range(self.model.nu)
            ], dtype=int)
        else:
            self.actuated_qpos_indices = np.array([], dtype=int)
        self.previous_output_qpos = None

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
            
        self.setup_retarget_configuration()

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
        self.scaled_human_data = human_data1

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
                
            
        qpos = self.configuration.data.qpos.copy()
        if self.output_joint_velocity_limit is not None:
            qpos = self._enforce_output_frame_limits(qpos)
            self.configuration.update(qpos)
        if self.posture_task is not None or self.output_joint_velocity_limit is not None:
            self.previous_output_qpos = qpos.copy()
        return qpos

    def _update_soft_joint_limit_target(self):
        target = self.configuration.data.qpos.copy()
        indices = self.soft_joint_limit_qpos_indices
        target[indices] = np.clip(
            target[indices],
            self.soft_joint_limit_lower,
            self.soft_joint_limit_upper,
        )
        self.soft_joint_limit_task.set_target(target)

    def _enforce_output_frame_limits(self, qpos):
        if self.previous_output_qpos is None or self.output_joint_velocity_limit is None:
            return qpos

        max_step = self.output_joint_velocity_limit / self.output_fps
        indices = self.actuated_qpos_indices
        delta = qpos[indices] - self.previous_output_qpos[indices]
        qpos[indices] = self.previous_output_qpos[indices] + np.clip(
            delta, -max_step, max_step
        )
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
