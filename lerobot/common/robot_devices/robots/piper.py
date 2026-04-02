"""
Teleoperation Agilex Piper with a PS5 controller
"""

import os
import time
import torch
import numpy as np
from dataclasses import replace

from lerobot.common.robot_devices.teleop.gamepad import SixAxisArmController_101
from lerobot.common.robot_devices.motors.utils import get_motor_names, make_motors_buses_from_configs
from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.robot_devices.robots.configs import PiperRobotConfig


class PiperRobot:
    def __init__(self, config: PiperRobotConfig | None = None, **kwargs):
        if config is None:
            config = PiperRobotConfig()

        self.config = replace(config, **kwargs)
        self.robot_type = self.config.type
        self.inference_time = self.config.inference_time

        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.piper_motors = make_motors_buses_from_configs(self.config.follower_arm)
        self.arm = self.piper_motors["main"]

        if not self.inference_time:
            self.teleop = SixAxisArmController_101()
        else:
            self.teleop = None

        self.logs = {}
        self.is_connected = False

        # Optional policy compatibility mode: expose a 6D state/action interface
        # to the policy by dropping joint_4 (0-based index 3) while keeping the
        # physical robot command 7D and holding joint_4 at a fixed value.
        self.policy_compact_joint4 = os.getenv(
            "LEROBOT_PIPER_POLICY_DROP_JOINT4", "0"
        ).strip().lower() in {"1", "true", "yes"}
        self.policy_disabled_joint_index = 3
        self._policy_joint_indices = [0, 1, 2, 4, 5, 6]
        self._fixed_joint4_value = None

        # policy limits
        self.policy_joint_delta_limit = 0.025
        self.policy_gripper_delta_limit = 0.005

        # teleop filters
        self.teleop_joint_deadbands = [0.01, 0.008, 0.008, 0.01, 0.003, 0.01]
        self.teleop_joint4_deadband = 0.003
        self.teleop_joint4_smoothing_alpha = 0.7
        self.teleop_target_alpha = [0.22, 0.14, 0.20, 0.20, 0.18, 0.20, 0.25]
        self.teleop_hysteresis = [0.006, 0.006, 0.006, 0.006, 0.008, 0.006, 0.002]
        self.teleop_settle_enter = [0.006, 0.0085, 0.010, 0.010, 0.007, 0.010, 0.004]
        self.teleop_settle_exit = [0.012, 0.0165, 0.018, 0.018, 0.014, 0.018, 0.007]
        self.teleop_follow_error_limit = [0.08, 0.05, 0.07, 0.07, 0.06, 0.05, 0.02]
        self.teleop_settle_frames = 3

        # states
        self._last_joint4_target = None
        self._teleop_filtered_target = None
        self._teleop_command_state = None
        self._teleop_settle_counter = [0] * 7
        self._teleop_settled = [False] * 7
        self._last_sent_teleop_target = None
        self._last_sent_policy_target = None

        # debug / reference mode
        self.teleop_clip_reference_mode = os.getenv(
            "LEROBOT_TELEOP_CLIP_REFERENCE", "last_sent"
        ).strip().lower()
        self.policy_clip_reference_mode = os.getenv(
            "LEROBOT_POLICY_CLIP_REFERENCE", "current"
        ).strip().lower()

        self.debug_teleop_clip = os.getenv(
            "LEROBOT_DEBUG_TELEOP_CLIP", "0"
        ).strip().lower() in {"1", "true", "yes"}
        self.debug_policy_clip = os.getenv(
            "LEROBOT_DEBUG_POLICY_CLIP", "0"
        ).strip().lower() in {"1", "true", "yes"}

        self._teleop_debug_counter = 0
        self._policy_debug_counter = 0

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        action_names = get_motor_names(self.piper_motors)
        state_names = get_motor_names(self.piper_motors)
        if self.policy_compact_joint4:
            action_names = [action_names[idx] for idx in self._policy_joint_indices]
            state_names = [state_names[idx] for idx in self._policy_joint_indices]
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def connect(self) -> None:
        """Connect piper and cameras."""
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "Piper is already connected. Do not run `robot.connect()` twice."
            )

        self.arm.connect(enable=True)
        print("piper connected")

        if self.teleop is not None and hasattr(self.teleop, "connect"):
            self.teleop.connect()

        for name in self.cameras:
            self.cameras[name].connect()
            print(f"camera {name} connected")

        print("All connected")
        self.is_connected = True
        self.run_calibration()

    def disconnect(self) -> None:
        """Move to home position, disable piper, and disconnect cameras."""
        if not self.inference_time and self.teleop is not None:
            self.teleop.stop()

        self.arm.safe_disconnect()
        print("piper disable after 5 seconds")
        time.sleep(5)
        self.arm.connect(enable=False)

        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self.is_connected = False

    def run_calibration(self):
        """Move piper to the home position."""
        if not self.is_connected:
            raise ConnectionError()

        self.arm.apply_calibration()

        self._last_joint4_target = None
        self._teleop_filtered_target = None
        self._teleop_command_state = None
        self._teleop_settle_counter = [0] * 7
        self._teleop_settled = [False] * 7
        self._last_sent_teleop_target = None
        self._last_sent_policy_target = None
        self._fixed_joint4_value = None

        if not self.inference_time and self.teleop is not None:
            self.teleop.reset()

    def _compact_state_for_policy(self, values: list[float]) -> list[float]:
        if not self.policy_compact_joint4:
            return list(values)
        return [values[idx] for idx in self._policy_joint_indices]

    def _get_fixed_joint4_value(self, full_state: list[float]) -> float:
        if self._fixed_joint4_value is None:
            self._fixed_joint4_value = float(full_state[self.policy_disabled_joint_index])
        return self._fixed_joint4_value

    def _expand_policy_action(self, policy_action: list[float], full_state: list[float]) -> list[float]:
        if not self.policy_compact_joint4:
            return list(policy_action)

        if len(policy_action) != len(self._policy_joint_indices):
            raise ValueError(
                f"Expected {len(self._policy_joint_indices)}-D compact policy action, got {len(policy_action)}."
            )

        fixed_joint4 = self._get_fixed_joint4_value(full_state)
        expanded_action = []
        compact_idx = 0
        for full_idx in range(len(full_state)):
            if full_idx == self.policy_disabled_joint_index:
                expanded_action.append(fixed_joint4)
            else:
                expanded_action.append(float(policy_action[compact_idx]))
                compact_idx += 1
        return expanded_action

    def _store_action_trace(
        self,
        prefix: str,
        raw_target_joints: list[float],
        reference_state: list[float],
        current_state_before: list[float],
        clipped_action: list[float],
        state_after: list[float],
    ) -> None:
        self.logs[f"{prefix}_arm_read_before"] = list(current_state_before)
        self.logs[f"{prefix}_reference_state"] = list(reference_state)
        self.logs[f"{prefix}_raw_target_joints"] = list(raw_target_joints)
        self.logs[f"{prefix}_clipped_action"] = list(clipped_action)
        self.logs[f"{prefix}_arm_read_after"] = list(state_after)
        self.logs[f"{prefix}_action_raw"] = list(raw_target_joints)
        self.logs[f"{prefix}_action_clipped"] = list(clipped_action)

        self.logs[f"{prefix}_raw_minus_reference"] = [
            r - ref for r, ref in zip(raw_target_joints, reference_state)
        ]
        self.logs[f"{prefix}_clipped_minus_reference"] = [
            c - ref for c, ref in zip(clipped_action, reference_state)
        ]
        self.logs[f"{prefix}_current_minus_reference"] = [
            cur - ref for cur, ref in zip(current_state_before, reference_state)
        ]
        self.logs[f"{prefix}_after_minus_clipped"] = [
            s - c for s, c in zip(state_after, clipped_action)
        ]

        should_print = (
            (prefix == "teleop" and self.debug_teleop_clip)
            or (prefix == "policy" and self.debug_policy_clip)
        )
        if not should_print:
            return

        print(f"[{prefix}] read_before      =", np.round(current_state_before, 4).tolist())
        print(f"[{prefix}] reference_state =", np.round(reference_state, 4).tolist())
        print(f"[{prefix}] raw_target       =", np.round(raw_target_joints, 4).tolist())
        print(f"[{prefix}] clipped_action   =", np.round(clipped_action, 4).tolist())
        print(f"[{prefix}] read_after       =", np.round(state_after, 4).tolist())
        print(
            f"[{prefix}] raw-ref          =",
            np.round(self.logs[f"{prefix}_raw_minus_reference"], 4).tolist(),
        )
        print(
            f"[{prefix}] clip-ref         =",
            np.round(self.logs[f"{prefix}_clipped_minus_reference"], 4).tolist(),
        )
        print(
            f"[{prefix}] cur-ref          =",
            np.round(self.logs[f"{prefix}_current_minus_reference"], 4).tolist(),
        )
        print(
            f"[{prefix}] after-clipped    =",
            np.round(self.logs[f"{prefix}_after_minus_clipped"], 4).tolist(),
        )
        print("=" * 60)

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise ConnectionError()

        if self.teleop is None and self.inference_time:
            self.teleop = SixAxisArmController_101()

        before_read_t = time.perf_counter()
        state_before_dict = self.arm.read()
        current_state_before = list(state_before_dict.values())
        action_dict_raw = self.teleop.get_action()
        raw_target_joints = list(action_dict_raw.values())
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        before_write_t = time.perf_counter()
        control_mode = self.teleop.get_control_mode()
        self.logs["teleop_control_mode"] = control_mode

        executed_action = None
        state_after = list(current_state_before)

        if control_mode == "ee":
            pose_target = self.teleop.get_pose_target()
            self.logs["teleop_pose_target"] = pose_target
            self.logs["teleop_gripper_target"] = getattr(self.teleop, "gripper", None)

            if pose_target is not None:
                self.arm.write_pose(pose_target, self.teleop.gripper)
                state_after = list(self.arm.read().values())
                self.logs["teleop_arm_read_before"] = list(current_state_before)
                self.logs["teleop_arm_read_after"] = list(state_after)
            else:
                executed_action = self._clip_teleop_targets(raw_target_joints, current_state_before)
                self.arm.write(executed_action)
                state_after = list(self.arm.read().values())
                reference_state = list(
                    self.logs.get("teleop_clip_reference_state", current_state_before)
                )
                self._store_action_trace(
                    "teleop",
                    raw_target_joints,
                    reference_state,
                    current_state_before,
                    executed_action,
                    state_after,
                )
        else:
            executed_action = self._clip_teleop_targets(raw_target_joints, current_state_before)
            self.arm.write(executed_action)
            state_after = list(self.arm.read().values())
            reference_state = list(
                self.logs.get("teleop_clip_reference_state", current_state_before)
            )
            self._store_action_trace(
                "teleop",
                raw_target_joints,
                reference_state,
                current_state_before,
                executed_action,
                state_after,
            )

        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        if not record_data:
            return

        state_tensor = torch.as_tensor(
            self._compact_state_for_policy(current_state_before), dtype=torch.float32
        )
        if executed_action is None:
            action_tensor = torch.as_tensor(
                self._compact_state_for_policy(raw_target_joints), dtype=torch.float32
            )
        else:
            action_tensor = torch.as_tensor(
                self._compact_state_for_policy(executed_action), dtype=torch.float32
            )

        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state_tensor
        action_dict["action"] = action_tensor
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Write the predicted actions from policy to the motors."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Piper is not connected. You need to run `robot.connect()`."
            )

        compact_action = action.tolist()
        current_state_before = list(self.arm.read().values())
        raw_target_joints = self._expand_policy_action(compact_action, current_state_before)

        if (
            self.policy_clip_reference_mode == "last_sent"
            and self._last_sent_policy_target is not None
        ):
            reference_state = list(self._last_sent_policy_target)
        else:
            reference_state = list(current_state_before)

        clipped_action = []
        for idx, (target, reference) in enumerate(
            zip(raw_target_joints, reference_state, strict=True)
        ):
            max_delta = (
                self.policy_gripper_delta_limit
                if idx == len(raw_target_joints) - 1
                else self.policy_joint_delta_limit
            )
            clipped_action.append(float(np.clip(target, reference - max_delta, reference + max_delta)))

        self.arm.write(clipped_action)
        state_after = list(self.arm.read().values())

        self.logs["policy_clip_reference_mode"] = self.policy_clip_reference_mode
        self._store_action_trace(
            "policy",
            raw_target_joints,
            reference_state,
            current_state_before,
            clipped_action,
            state_after,
        )

        self._last_sent_policy_target = list(clipped_action)
        return torch.as_tensor(self._compact_state_for_policy(clipped_action), dtype=action.dtype)

    def _clip_teleop_targets(self, target_joints: list[float], current_state: list[float]) -> list[float]:
        raw_input_joints = list(target_joints)
        gripper_index = len(raw_input_joints) - 1

        if self._teleop_filtered_target is None:
            self._teleop_filtered_target = list(current_state)

        if self._teleop_command_state is None:
            self._teleop_command_state = list(current_state)

        filtered_target_joints = []
        for idx, (raw, prev_filtered) in enumerate(
            zip(raw_input_joints, self._teleop_filtered_target, strict=True)
        ):
            alpha = self.teleop_target_alpha[idx] if idx < len(self.teleop_target_alpha) else 0.18
            filtered = (1.0 - alpha) * prev_filtered + alpha * raw
            filtered_target_joints.append(float(filtered))

        self._teleop_filtered_target = list(filtered_target_joints)

        next_command_state = list(self._teleop_command_state)
        clipped_action = []
        step_limits = []
        follow_limits = []
        err_current_list = []
        err_cmd_list = []
        command_prev_list = []
        command_new_list = []
        for idx, (target, current, cmd_prev) in enumerate(
            zip(filtered_target_joints, current_state, self._teleop_command_state, strict=True)
        ):
            err_current = float(target - current)
            abs_err_current = abs(err_current)
            err_current_list.append(err_current)
            command_prev_list.append(float(cmd_prev))

            enter_thr = self.teleop_settle_enter[idx]
            exit_thr = self.teleop_settle_exit[idx]

            if abs_err_current < enter_thr:
                self._teleop_settle_counter[idx] += 1
            else:
                self._teleop_settle_counter[idx] = 0

            if self._teleop_settle_counter[idx] >= self.teleop_settle_frames:
                self._teleop_settled[idx] = True

            if self._teleop_settled[idx]:
                if abs_err_current < exit_thr:
                    cmd_new = float(current)
                    next_command_state[idx] = cmd_new
                    err_cmd_list.append(float(target - cmd_prev))
                    step_limits.append(0.0)
                    follow_limits.append(0.0)
                    command_new_list.append(cmd_new)
                    clipped_action.append(cmd_new)
                    continue
                else:
                    self._teleop_settled[idx] = False
                    self._teleop_settle_counter[idx] = 0

            err_cmd = float(target - cmd_prev)
            abs_err_cmd = abs(err_cmd)

            if idx == gripper_index:
                if abs_err_cmd > 0.02:
                    step_limit = 0.0035
                elif abs_err_cmd > 0.008:
                    step_limit = 0.0020
                else:
                    step_limit = 0.0008
            elif idx == 0:
                if abs_err_cmd > 0.18:
                    step_limit = 0.07
                elif abs_err_cmd > 0.07:
                    step_limit = 0.04
                elif abs_err_cmd > 0.025:
                    step_limit = 0.02
                else:
                    step_limit = 0.006
            elif idx == 1:
                # Use a smooth proportional ramp for joint1 to avoid
                # visible step changes when err_cmd crosses bucket edges.
                step_limit = float(np.clip(0.115 * abs_err_cmd + 0.0008, 0.0008, 0.0320))
            elif idx in {2, 3}:
                if abs_err_cmd > 0.15:
                    step_limit = 0.055
                elif abs_err_cmd > 0.06:
                    step_limit = 0.03
                elif abs_err_cmd > 0.02:
                    step_limit = 0.015
                else:
                    step_limit = 0.005
            elif idx == 4:
                if abs_err_cmd > 0.15:
                    step_limit = 0.06
                elif abs_err_cmd > 0.06:
                    step_limit = 0.035
                elif abs_err_cmd > 0.02:
                    step_limit = 0.02
                else:
                    step_limit = 0.008
            else:
                if abs_err_cmd > 0.15:
                    step_limit = 0.04
                elif abs_err_cmd > 0.06:
                    step_limit = 0.02
                elif abs_err_cmd > 0.02:
                    step_limit = 0.010
                else:
                    step_limit = 0.003

            cmd_new = float(cmd_prev + np.clip(err_cmd, -step_limit, step_limit))

            follow_limit = (
                self.teleop_follow_error_limit[idx]
                if idx < len(self.teleop_follow_error_limit)
                else 0.06
            )
            cmd_new = float(np.clip(cmd_new, current - follow_limit, current + follow_limit))

            next_command_state[idx] = cmd_new
            err_cmd_list.append(err_cmd)
            step_limits.append(step_limit)
            follow_limits.append(follow_limit)
            command_new_list.append(cmd_new)
            clipped_action.append(cmd_new)

        self._teleop_command_state = list(next_command_state)

        self.logs["teleop_clip_reference_mode"] = "command_state_with_settle_zone"
        self.logs["teleop_current_state"] = list(current_state)
        self.logs["teleop_clip_reference_state"] = list(current_state)
        self.logs["teleop_action_raw"] = list(raw_input_joints)
        self.logs["teleop_action_input"] = list(raw_input_joints)
        self.logs["teleop_action_filtered"] = list(filtered_target_joints)
        self.logs["teleop_action_clipped"] = list(clipped_action)
        self.logs["teleop_action_settle_mask"] = list(self._teleop_settled)
        self.logs["teleop_settled_flags"] = list(self._teleop_settled)
        self.logs["teleop_settle_counter"] = list(self._teleop_settle_counter)
        self.logs["teleop_command_state"] = list(self._teleop_command_state)
        self.logs["teleop_command_prev"] = list(command_prev_list)
        self.logs["teleop_command_new"] = list(command_new_list)
        self.logs["teleop_err_current"] = list(err_current_list)
        self.logs["teleop_err_cmd"] = list(err_cmd_list)
        self.logs["teleop_step_limit"] = list(step_limits)
        self.logs["teleop_follow_limit"] = list(follow_limits)
        self.logs["teleop_settle_enter"] = list(self.teleop_settle_enter)
        self.logs["teleop_settle_exit"] = list(self.teleop_settle_exit)
        self.logs["teleop_input_minus_reference"] = [
            t - r for t, r in zip(raw_input_joints, current_state)
        ]
        self.logs["teleop_filtered_minus_reference"] = [
            t - r for t, r in zip(filtered_target_joints, current_state)
        ]
        self.logs["teleop_clipped_minus_reference"] = [
            c - r for c, r in zip(clipped_action, current_state)
        ]
        self.logs["teleop_input_minus_current"] = [
            t - c for t, c in zip(raw_input_joints, current_state)
        ]
        self.logs["teleop_filtered_minus_current"] = [
            t - c for t, c in zip(filtered_target_joints, current_state)
        ]
        self.logs["teleop_clipped_minus_current"] = [
            c - s for c, s in zip(clipped_action, current_state)
        ]

        self._last_sent_teleop_target = list(clipped_action)

        if self.debug_teleop_clip:
            self._teleop_debug_counter += 1
            if self._teleop_debug_counter % 10 == 0:
                print("[teleop-clip] mode =", self.logs["teleop_clip_reference_mode"])
                print("[teleop-clip] current   =", np.round(current_state, 4).tolist())
                print("[teleop-clip] input     =", np.round(raw_input_joints, 4).tolist())
                print("[teleop-clip] filtered  =", np.round(filtered_target_joints, 4).tolist())
                print("[teleop-clip] command   =", np.round(clipped_action, 4).tolist())
                print("[teleop-clip] settled   =", self._teleop_settled)
                for idx, (cur, raw, flt, cmd_prev, cmd_new, err_cur, err_cmd, step_limit, follow_limit, settled, counter) in enumerate(
                    zip(
                        current_state,
                        raw_input_joints,
                        filtered_target_joints,
                        command_prev_list,
                        command_new_list,
                        err_current_list,
                        err_cmd_list,
                        step_limits,
                        follow_limits,
                        self._teleop_settled,
                        self._teleop_settle_counter,
                        strict=True,
                    )
                ):
                    print(
                        "[teleop-joint] "
                        f"j{idx} cur={cur:+.4f} raw={raw:+.4f} flt={flt:+.4f} "
                        f"cmd_prev={cmd_prev:+.4f} cmd_new={cmd_new:+.4f} "
                        f"err_cur={err_cur:+.4f} err_cmd={err_cmd:+.4f} "
                        f"step={step_limit:.4f} follow={follow_limit:.4f} "
                        f"settled={int(settled)} cnt={counter}"
                    )
                print("-" * 60)

        return clipped_action

    def capture_observation(self) -> dict:
        """Capture current images and joint positions."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Piper is not connected. You need to run `robot.connect()`."
            )

        before_read_t = time.perf_counter()
        state = self.arm.read()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        full_state = list(state.values())
        state = torch.as_tensor(self._compact_state_for_policy(full_state), dtype=torch.float32)

        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        return obs_dict

    def consume_control_events(self) -> dict[str, bool]:
        if self.teleop is None:
            return {"exit_early": False, "rerecord_episode": False}
        return self.teleop.consume_control_events()

    def teleop_safety_stop(self):
        """Move to home position after recording one episode."""
        self.run_calibration()

    def __del__(self):
        if self.is_connected:
            self.disconnect()
            if not self.inference_time and self.teleop is not None:
                self.teleop.stop()
