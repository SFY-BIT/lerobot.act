import os
import pygame
import threading
import time
from typing import Dict

try:
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    from trac_ik import TracIK
    IK_IMPORT_ERROR = None
except ImportError as exc:
    np = None
    R = None
    TracIK = None
    IK_IMPORT_ERROR = exc


SONY_WIRELESS_CONTROLLER_NAME = "Sony Computer Entertainment Wireless Controller"
DEADZONE = 0.28
AXIS_SMOOTHING_ALPHA = 0.25
JOINT_STEP_SCALE = 0.006
DPAD_STEP_SCALE = 0.006
GRIPPER_STEP_SCALE = 0.001

SONY_BUTTON_MAP = {
    "a": 0,
    "b": 1,
    "x": 3,
    "y": 2,
    "lb": 4,
    "rb": 5,
    "lt": 6,
    "rt": 7,
    "back": 8,
    "start": 9,
    "home": 10,
    "l3": 11,
    "r3": 12,
}

SONY_AXIS_MAP = {
    "left_x": 0,
    "left_y": 1,
    "left_trigger": 2,
    "right_x": 3,
    "right_y": 4,
    "right_trigger": 5,
}

SONY_HAT_MAP = {
    "dpad": 0,
}

DEFAULT_BUTTON_MAP = SONY_BUTTON_MAP
DEFAULT_AXIS_MAP = SONY_AXIS_MAP
DEFAULT_HAT_MAP = SONY_HAT_MAP


class PiperTracIKKinematics:
    def __init__(
        self,
        urdf_path: str,
        base_link_name: str = "base_link",
        target_link_name: str = "link6",
        timeout: float = 0.005,
        epsilon: float = 0.00001,
        solver_type: str = "Speed",
    ):
        self.ik_solver = TracIK(
            base_link_name=base_link_name,
            tip_link_name=target_link_name,
            urdf_path=urdf_path,
            timeout=timeout,
            epsilon=epsilon,
            solver_type=solver_type,
        )
        lower_limits, upper_limits = self.ik_solver.joint_limits
        self.joint_limits = list(zip(lower_limits.tolist(), upper_limits.tolist()))

    def solve_fk(self, joint_angles: list[float]) -> "np.ndarray":
        joint_array = np.asarray(joint_angles, dtype=float)
        position, rotation_matrix = self.ik_solver.fk(joint_array)
        rotation = R.from_matrix(rotation_matrix)
        return np.concatenate((position, rotation.as_euler("xyz", degrees=True)))

    def solve_ik(
        self,
        target_position: "np.ndarray",
        target_euler_deg: "np.ndarray",
        initial_guess: list[float],
    ) -> list[float] | None:
        target_rotation = R.from_euler("xyz", target_euler_deg, degrees=True)
        solution = self.ik_solver.ik(
            target_position,
            target_rotation.as_matrix(),
            seed_jnt_values=np.asarray(initial_guess, dtype=float),
        )
        if solution is None:
            return None
        return solution.tolist()


class SixAxisArmController:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise Exception("未检测到手柄")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.device_name = self.joystick.get_name()
        self.button_map, self.axis_map, self.hat_map = self._select_maps(self.device_name)

        self.joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.gripper = 0.0
        self.speeds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.gripper_speed = 0.0
        self.control_mode = "joint"
        self.pose_translation_step = 0.001
        self.pose_rotation_step_deg = 0.5
        self.pose_target = None
        self._last_debug_snapshot = None
        self._axis_filters = {
            "left_x": 0.0,
            "left_y": 0.0,
            "right_x": 0.0,
            "right_y": 0.0,
        }
        self._last_y_pressed = False
        self._last_start_pressed = False
        self._last_lb_pressed = False
        self._last_rb_pressed = False
        self._pending_events = {
            "exit_early": False,
            "rerecord_episode": False,
        }

        self.joint_limits = [
            (-92000 / 57324.840764, 92000 / 57324.840764),
            (0 / 57324.840764, 170000 / 57324.840764),
            (-80000 / 57324.840764, 0 / 57324.840764),
            (-90000 / 57324.840764, 90000 / 57324.840764),
            (-77000 / 57324.840764, 60000 / 57324.840764),
            (-90000 / 57324.840764, 90000 / 57324.840764),
        ]
        self.kinematic = self._init_kinematics()
        if self.kinematic is not None:
            self._sync_pose_target_from_joints()

        self.running = True
        self.thread = threading.Thread(target=self.update_joints)
        self.thread.start()

    def _select_maps(self, device_name: str):
        if device_name == SONY_WIRELESS_CONTROLLER_NAME:
            return SONY_BUTTON_MAP, SONY_AXIS_MAP, SONY_HAT_MAP
        return DEFAULT_BUTTON_MAP, DEFAULT_AXIS_MAP, DEFAULT_HAT_MAP

    def _get_button(self, name: str) -> int:
        return self.joystick.get_button(self.button_map[name])

    def _get_axis(self, name: str, invert: bool = False) -> float:
        raw_value = self.joystick.get_axis(self.axis_map[name])
        raw_value = -raw_value if invert else raw_value
        if abs(raw_value) < DEADZONE:
            if name in self._axis_filters:
                self._axis_filters[name] = 0.0
            return 0.0

        if name in self._axis_filters:
            filtered = (1 - AXIS_SMOOTHING_ALPHA) * self._axis_filters[name] + AXIS_SMOOTHING_ALPHA * raw_value
            if abs(filtered) < 0.01:
                filtered = 0.0
            self._axis_filters[name] = filtered
            return filtered

        return raw_value

    def _get_hat(self, name: str) -> tuple[int, int]:
        return self.joystick.get_hat(self.hat_map[name])

    def _discover_urdf_path(self) -> str | None:
        candidates = [
            os.getenv("PIPER_URDF_PATH"),
            "/home/night/Gamepad_PiPER/piper/piper.urdf",
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return None

    def _init_kinematics(self) -> PiperTracIKKinematics | None:
        if np is None or R is None or TracIK is None:
            detail = f" ({IK_IMPORT_ERROR})" if IK_IMPORT_ERROR is not None else ""
            print(f"Piper EE mode unavailable: missing numpy/scipy/trac_ik dependencies{detail}.")
            return None

        urdf_path = self._discover_urdf_path()
        if urdf_path is None:
            print("Piper EE mode unavailable: URDF file not found. Set PIPER_URDF_PATH to enable it.")
            return None

        try:
            print(f"Initializing Piper EE mode with URDF: {urdf_path}")
            return PiperTracIKKinematics(urdf_path=urdf_path, base_link_name="base_link", target_link_name="link6")
        except Exception as exc:
            print(f"Piper EE mode unavailable: failed to initialize TracIK ({exc}).")
            return None

    def _sync_pose_target_from_joints(self) -> None:
        if self.kinematic is None:
            self.pose_target = None
            return
        try:
            self.pose_target = self.kinematic.solve_fk(self.joints)
        except Exception as exc:
            print(f"Failed to sync EE pose from joints: {exc}")
            self.pose_target = None

    def _toggle_control_mode(self) -> None:
        if self.control_mode == "joint":
            if self.kinematic is None:
                print("Cannot switch to EE mode: kinematics is unavailable.")
                return
            self._sync_pose_target_from_joints()
            if self.pose_target is None:
                print("Cannot switch to EE mode: failed to compute current end-effector pose.")
                return
            self.control_mode = "ee"
            print("Switched control mode to EE.")
            return

        self.control_mode = "joint"
        print("Switched control mode to joint.")

    def _debug_input_and_mode(
        self,
        start_pressed: bool,
        y_pressed: bool,
        lb_pressed: bool,
        rb_pressed: bool,
    ) -> None:
        snapshot = (
            start_pressed,
            y_pressed,
            lb_pressed,
            rb_pressed,
            self.control_mode,
            self.kinematic is not None,
        )
        if snapshot == self._last_debug_snapshot:
            return

        self._last_debug_snapshot = snapshot
        print(
            "[gamepad-debug] "
            f"start={int(start_pressed)} y={int(y_pressed)} "
            f"lb={int(lb_pressed)} rb={int(rb_pressed)} "
            f"mode={self.control_mode} "
            f"ee_ready={self.kinematic is not None}"
        )

    def _update_joint_control(
        self,
        left_x: float,
        left_y: float,
        right_x: float,
        right_y: float,
        up: bool,
        down: bool,
        left: bool,
        right: bool,
    ) -> None:
        target_speeds = [0.0] * 6
        target_speeds[0] = left_x * JOINT_STEP_SCALE
        target_speeds[1] = left_y * JOINT_STEP_SCALE
        target_speeds[2] = -right_y * JOINT_STEP_SCALE
        target_speeds[3] = right_x * JOINT_STEP_SCALE
        target_speeds[4] = -DPAD_STEP_SCALE if up else (DPAD_STEP_SCALE if down else 0.0)
        target_speeds[5] = DPAD_STEP_SCALE if right else (-DPAD_STEP_SCALE if left else 0.0)

        self.speeds = target_speeds

        for i in range(6):
            self.joints[i] += self.speeds[i]

        for i in range(6):
            min_val, max_val = self.joint_limits[i]
            self.joints[i] = max(min_val, min(max_val, self.joints[i]))

    def _update_ee_control(
        self,
        left_x: float,
        left_y: float,
        right_x: float,
        right_y: float,
        up: bool,
        down: bool,
        left: bool,
        right: bool,
    ) -> None:
        if self.kinematic is None:
            return

        if self.pose_target is None:
            self._sync_pose_target_from_joints()
            if self.pose_target is None:
                return

        hat_x = 1 if right else (-1 if left else 0)
        hat_y = 1 if up else (-1 if down else 0)
        d_local = np.array([-left_y, left_x, -right_y], dtype=float) * self.pose_translation_step
        r_local = np.array([hat_x, -hat_y, right_x], dtype=float) * self.pose_rotation_step_deg

        if not np.any(d_local) and not np.any(r_local):
            self.speeds = [0.0] * 6
            return

        current_position = self.pose_target[:3]
        current_rotation = R.from_euler("xyz", self.pose_target[3:], degrees=True)
        target_rotation = current_rotation * R.from_euler("xyz", r_local, degrees=True)
        target_position = current_position + current_rotation.apply(d_local)
        ik_solution = self.kinematic.solve_ik(
            target_position=target_position,
            target_euler_deg=target_rotation.as_euler("xyz", degrees=True),
            initial_guess=self.joints,
        )
        if ik_solution is None:
            return

        old_joints = list(self.joints)
        self.joints = ik_solution
        for i in range(6):
            min_val, max_val = self.joint_limits[i]
            self.joints[i] = max(min_val, min(max_val, self.joints[i]))
            self.speeds[i] = self.joints[i] - old_joints[i]

        self._sync_pose_target_from_joints()

    def consume_control_events(self) -> dict[str, bool]:
        events = dict(self._pending_events)
        self._pending_events = {key: False for key in self._pending_events}
        return events

    def update_joints(self):
        while self.running:
            try:
                pygame.event.pump()
            except Exception:
                self.stop()
                continue

            left_x = self._get_axis("left_x", invert=True)
            left_y = self._get_axis("left_y", invert=True)
            right_x = self._get_axis("right_x", invert=True)
            right_y = self._get_axis("right_y", invert=True)

            hat = self._get_hat("dpad")
            up = hat[1] == 1
            down = hat[1] == -1
            left = hat[0] == -1
            right = hat[0] == 1

            left_trigger = self._get_button("lt")
            right_trigger = self._get_button("rt")
            y_pressed = bool(self._get_button("y"))
            start_pressed = bool(self._get_button("start"))
            lb_pressed = bool(self._get_button("lb"))
            rb_pressed = bool(self._get_button("rb"))

            self._debug_input_and_mode(start_pressed, y_pressed, lb_pressed, rb_pressed)

            if rb_pressed and not self._last_rb_pressed:
                self._pending_events["exit_early"] = True

            if lb_pressed and not self._last_lb_pressed:
                self._pending_events["rerecord_episode"] = True
                self._pending_events["exit_early"] = True

            if start_pressed and not self._last_start_pressed:
                self._toggle_control_mode()

            if y_pressed and not self._last_y_pressed:
                self.go_home()
                self._last_y_pressed = True
                self._last_start_pressed = start_pressed
                self._last_lb_pressed = lb_pressed
                self._last_rb_pressed = rb_pressed
                time.sleep(0.02)
                continue

            self._last_y_pressed = y_pressed
            self._last_start_pressed = start_pressed
            self._last_lb_pressed = lb_pressed
            self._last_rb_pressed = rb_pressed

            if self.control_mode == "joint":
                self._update_joint_control(left_x, left_y, right_x, right_y, up, down, left, right)
            else:
                self._update_ee_control(left_x, left_y, right_x, right_y, up, down, left, right)

            self.gripper_speed = (
                GRIPPER_STEP_SCALE if right_trigger else (-GRIPPER_STEP_SCALE if left_trigger else 0.0)
            )

            self.gripper += self.gripper_speed

            self.gripper = max(0.0, min(0.08, self.gripper))
            time.sleep(0.005)

    def get_action(self) -> Dict:
        return {
            "joint0": self.joints[0],
            "joint1": self.joints[1],
            "joint2": self.joints[2],
            "joint3": self.joints[3],
            "joint4": self.joints[4],
            "joint5": self.joints[5],
            "gripper": self.gripper,
        }

    def get_control_mode(self) -> str:
        return self.control_mode

    def stop(self):
        self.running = False
        self.thread.join()
        pygame.quit()
        print("Gamepad exits")

    def go_home(self):
        self.joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.gripper = 0.0
        self.speeds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.gripper_speed = 0.0
        self._sync_pose_target_from_joints()

    def reset(self):
        self.go_home()


if __name__ == "__main__":
    arm_controller = SixAxisArmController()
    try:
        while True:
            print(arm_controller.get_action())
            print(arm_controller.consume_control_events())
            time.sleep(0.1)
    except KeyboardInterrupt:
        arm_controller.stop()
