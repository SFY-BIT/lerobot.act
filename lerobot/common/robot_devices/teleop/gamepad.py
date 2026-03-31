import json
import math
import os
import time
import pygame
import threading
from pathlib import Path
from typing import Dict

try:
    from pynput import keyboard as pynput_keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    pynput_keyboard = None
    PYNPUT_AVAILABLE = False

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

EE_SPEED_SCALE = 0.6
DEFAULT_SO101_LEADER_PORT = "/dev/ttyACM0"
DEFAULT_SO101_CALIBRATION = Path(
    "/home/night/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/R07252801.json"
)
JOINT0_OFFSET_DEG = 90.0
JOINT2_ZERO_DEG = 78.0
JOINT4_ZERO_DEG = 223.0
JOINT5_ZERO_DEG = 32.0
JOINT1_SMOOTHING_ALPHA = 0.35
PIPER_JOINT_LIMITS_RAD = [
    (-2.618, 2.618),
    (0.0, 3.14),
    (-2.967, 0.0),
    (-1.745, 1.745),
    (-1.22, 1.22),
    (-2.967, 2.967),
]
PIPER_JOINT_LIMITS_DEG = [(math.degrees(lower), math.degrees(upper)) for lower, upper in PIPER_JOINT_LIMITS_RAD]
DEBUG_SO101_MAPPING = os.getenv("LEROBOT_DEBUG_SO101_MAPPING", "0") == "1"
DEBUG_PRINT_INTERVAL_S = 0.2
SO101_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _detect_default_leader_port() -> str:
    return DEFAULT_SO101_LEADER_PORT


def _is_legacy_so101_calibration(calibration: dict) -> bool:
    required_keys = {"homing_offset", "drive_mode", "start_pos", "end_pos", "calib_mode", "motor_names"}
    return required_keys.issubset(calibration.keys())


def _is_full_so101_calibration(calibration: dict) -> bool:
    return all(name in calibration and isinstance(calibration[name], dict) for name in SO101_MOTOR_NAMES)


def _convert_full_so101_calibration(calibration: dict) -> dict:
    return {
        "homing_offset": [calibration[name]["homing_offset"] for name in SO101_MOTOR_NAMES],
        "drive_mode": [calibration[name]["drive_mode"] for name in SO101_MOTOR_NAMES],
        "start_pos": [calibration[name]["range_min"] for name in SO101_MOTOR_NAMES],
        "end_pos": [calibration[name]["range_max"] for name in SO101_MOTOR_NAMES],
        "calib_mode": ["DEGREE", "DEGREE", "DEGREE", "DEGREE", "DEGREE", "LINEAR"],
        "motor_names": SO101_MOTOR_NAMES,
    }


def _normalize_so101_calibration(calibration: dict) -> dict:
    if _is_legacy_so101_calibration(calibration):
        return calibration

    if _is_full_so101_calibration(calibration):
        return _convert_full_so101_calibration(calibration)

    raise ValueError("Unsupported SO101 calibration format.")


def _candidate_so101_calibration_paths() -> list[Path]:
    explicit = os.getenv("LEROBOT_SO101_CALIBRATION")
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())

    candidates.extend(
        [
            DEFAULT_SO101_CALIBRATION,
            Path(".cache/calibration/so101/main_leader.json"),
            Path(".cache/calibration/so100/main_leader.json"),
            Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "teleoperators" / "so101_leader" / "main.json",
            Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "teleoperators" / "so100_leader" / "main.json",
        ]
    )
    return candidates


def _load_so101_calibration(calibration: str | os.PathLike | dict | None) -> tuple[dict | None, Path | None]:
    if isinstance(calibration, dict):
        return _normalize_so101_calibration(calibration), None

    if calibration is not None:
        path = Path(calibration).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Leader calibration file not found: {path}")
        with open(path, encoding="utf-8") as f:
            return _normalize_so101_calibration(json.load(f)), path

    for candidate in _candidate_so101_calibration_paths():
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                return _normalize_so101_calibration(json.load(f)), candidate

    return None, None


def _wrap_deg(value: float) -> float:
    """Wrap angles into [-180, 180) to keep leader cross-zero jumps continuous."""
    return (value + 180.0) % 360.0 - 180.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _unwrap_near_reference(value: float, reference: float) -> float:
    """Return the angle equivalent to value that is closest to reference."""
    candidates = (value - 360.0, value, value + 360.0)
    return min(candidates, key=lambda candidate: abs(candidate - reference))


class _SO101LeaderController:
    """Leader-arm reader that mimics the gamepad controller interface used by PiperRobot."""

    def __init__(self, port: str | None = None, calibration: str | os.PathLike | dict | None = None, use_degrees: bool = False):
        self.port = port or _detect_default_leader_port()
        self.use_degrees = use_degrees
        self.calibration, self.calibration_path = _load_so101_calibration(calibration)
        self.bus = None
        self.control_mode = "joint"
        self.pose_target = None
        self.gripper = 0.0
        self._last_debug_print_t = 0.0
        self._last_debug_signature = None
        self._last_joint2_deg = None
        self._last_joint1_deg = None
        self._pending_events = {
            "exit_early": False,
            "rerecord_episode": False,
        }
        self._keyboard_listener = None

        if PYNPUT_AVAILABLE:
            self._keyboard_listener = pynput_keyboard.Listener(on_press=self._on_key_press)
            self._keyboard_listener.start()

    def _on_key_press(self, key):
        try:
            if hasattr(key, "char") and key.char == "q":
                self._pending_events["exit_early"] = True
                return False
        except Exception:
            return None
        return None

    def connect(self):
        if self.bus is not None and self.bus.is_connected:
            return

        from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
        from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

        config = FeetechMotorsBusConfig(
            port=self.port,
            motors={
                "shoulder_pan": (1, "sts3215"),
                "shoulder_lift": (2, "sts3215"),
                "elbow_flex": (3, "sts3215"),
                "wrist_flex": (4, "sts3215"),
                "wrist_roll": (5, "sts3215"),
                "gripper": (6, "sts3215"),
            },
        )
        self.bus = FeetechMotorsBus(config)
        self.bus.connect()

        if self.calibration is None:
            raise FileNotFoundError(
                "SO101 leader calibration file was not found. "
                "Set LEROBOT_SO101_CALIBRATION or place a calibration file in .cache/calibration/so100/main_leader.json."
            )

        self.bus.set_calibration(self.calibration)

    def _read_leader_state(self) -> dict[str, float]:
        if self.bus is None or not self.bus.is_connected:
            self.connect()

        values = self.bus.read("Present_Position")
        if hasattr(values, "tolist"):
            values = values.tolist()

        return dict(zip(self.bus.motor_names, values, strict=True))

    def _map_so101_to_piper(self, state: dict[str, float]) -> Dict:
        # Feetech leader readings are calibrated to degrees for rotational joints and [0, 100] for the gripper.
        joint1_deg_raw = state["shoulder_lift"]
        if self._last_joint1_deg is None:
            joint1_deg = joint1_deg_raw
        else:
            joint1_deg = (
                (1.0 - JOINT1_SMOOTHING_ALPHA) * self._last_joint1_deg
                + JOINT1_SMOOTHING_ALPHA * joint1_deg_raw
            )
        self._last_joint1_deg = joint1_deg

        elbow_deg_raw = state["elbow_flex"] - JOINT2_ZERO_DEG
        if self._last_joint2_deg is None:
            elbow_deg = _wrap_deg(elbow_deg_raw)
        else:
            elbow_deg = _unwrap_near_reference(elbow_deg_raw, self._last_joint2_deg)
        elbow_deg = _clamp(elbow_deg, *PIPER_JOINT_LIMITS_DEG[2])
        self._last_joint2_deg = elbow_deg
        joint_deg = [
            -(state["shoulder_pan"] - JOINT0_OFFSET_DEG),
            joint1_deg,
            elbow_deg,
            0.0,
            _wrap_deg(state["wrist_flex"] - JOINT4_ZERO_DEG),
            -_wrap_deg(state["wrist_roll"] - JOINT5_ZERO_DEG),
        ]
        joint_deg = [
            _clamp(value, lower_deg, upper_deg)
            for value, (lower_deg, upper_deg) in zip(joint_deg, PIPER_JOINT_LIMITS_DEG, strict=True)
        ]
        joint_rad = [math.radians(value) for value in joint_deg]
        gripper_m = max(0.0, min(0.08, state["gripper"] / 100.0 * 0.08))
        self.gripper = gripper_m

        return {
            "joint0": joint_rad[0],
            "joint1": joint_rad[1],
            "joint2": joint_rad[2],
            "joint3": joint_rad[3],
            "joint4": joint_rad[4],
            "joint5": joint_rad[5],
            "gripper": gripper_m,
        }

    def _debug_print_mapping(self, state: dict[str, float], action: dict[str, float]) -> None:
        if not DEBUG_SO101_MAPPING:
            return

        signature = tuple(round(state[name], 1) for name in SO101_MOTOR_NAMES)
        now = time.perf_counter()
        if signature == self._last_debug_signature and now - self._last_debug_print_t < DEBUG_PRINT_INTERVAL_S:
            return

        joint_deg = {f"joint{i}": round(math.degrees(action[f"joint{i}"]), 1) for i in range(6)}
        state_fmt = " ".join(f"{name}={state[name]:+.1f}" for name in SO101_MOTOR_NAMES)
        action_fmt = " ".join(f"{name}={value:+.1f}" for name, value in joint_deg.items())
        print(f"[so101-state] {state_fmt}")
        print(f"[piper-map] {action_fmt} gripper={action['gripper']:.4f}")

        self._last_debug_signature = signature
        self._last_debug_print_t = now

    def get_action(self) -> Dict:
        state = self._read_leader_state()
        action = self._map_so101_to_piper(state)
        self._debug_print_mapping(state, action)
        return action

    def get_control_mode(self) -> str:
        return self.control_mode

    def get_pose_target(self):
        return None

    def consume_control_events(self) -> dict[str, bool]:
        return dict(self._pending_events)

    def go_home(self):
        pass

    def reset(self):
        pass

    def stop(self):
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
        if self.bus is not None and self.bus.is_connected:
            self.bus.disconnect()


def SixAxisArmController_101(
    port: str | None = None,
    calibration: str | os.PathLike | dict | None = None,
    use_degrees: bool = False,
):
    return _SO101LeaderController(port=port, calibration=calibration, use_degrees=use_degrees)


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
        # End-effector mode moves more stably with a reduced step size.
        self.pose_translation_step = 0.001 * EE_SPEED_SCALE
        self.pose_rotation_step_deg = 0.5 * EE_SPEED_SCALE
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

        # Keep software limits aligned with the Piper URDF so EE IK clipping
        # matches the actual kinematic model used by the reference teleop code.
        self.joint_limits = [
            (-2.618, 2.618),
            (0.0, 3.14),
            (-2.967, 0.0),
            (-1.745, 1.745),
            (-1.22, 1.22),
            (-2.967, 2.967),
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
                print("[gamepad-event] RB pressed -> exit current phase early")
                self._pending_events["exit_early"] = True

            if lb_pressed and not self._last_lb_pressed:
                print("[gamepad-event] LB pressed -> rerecord current episode")
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

    def get_pose_target(self):
        return None if self.pose_target is None else self.pose_target.copy()

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
