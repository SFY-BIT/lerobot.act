"""
    Teleoperation Agilex Piper with a PS5 controller    
"""

import time
import torch
import numpy as np
from dataclasses import dataclass, field, replace

from lerobot.common.robot_devices.teleop.gamepad import SixAxisArmController
from lerobot.common.robot_devices.motors.utils import get_motor_names, make_motors_buses_from_configs
from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.robot_devices.robots.configs import PiperRobotConfig

class PiperRobot:
    def __init__(self, config: PiperRobotConfig | None = None, **kwargs):
        if config is None:
            config = PiperRobotConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)
        self.robot_type = self.config.type
        self.inference_time = self.config.inference_time # if it is inference time
        
        # build cameras
        self.cameras = make_cameras_from_configs(self.config.cameras)
        
        # build piper motors
        self.piper_motors = make_motors_buses_from_configs(self.config.follower_arm)
        self.arm = self.piper_motors['main']
        
        # build gamepad teleop
        if not self.inference_time:
            self.teleop = SixAxisArmController()
        else:
            self.teleop = None
        
        self.logs = {}
        self.is_connected = False
        self.policy_joint_delta_limit = 0.05
        self.policy_gripper_delta_limit = 0.01

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
        """Connect piper and cameras"""
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "Piper is already connected. Do not run `robot.connect()` twice."
            )
        
        # connect piper
        self.arm.connect(enable=True)
        print("piper conneted")

        # connect cameras
        for name in self.cameras:
            self.cameras[name].connect()
            self.is_connected = self.is_connected and self.cameras[name].is_connected
            print(f"camera {name} conneted")
        
        print("All connected")
        self.is_connected = True
        
        self.run_calibration()


    def disconnect(self) -> None:
        """move to home position, disenable piper and cameras"""
        # move piper to home position, disable
        if not self.inference_time:
            self.teleop.stop()

        # disconnect piper
        self.arm.safe_disconnect()
        print("piper disable after 5 seconds")
        time.sleep(5)
        self.arm.connect(enable=False)

        # disconnect cameras
        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self.is_connected = False


    def run_calibration(self):
        """move piper to the home position"""
        if not self.is_connected:
            raise ConnectionError()
        
        self.arm.apply_calibration()
        if not self.inference_time:
            self.teleop.reset()



    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise ConnectionError()
        
        if self.teleop is None and self.inference_time:
            self.teleop = SixAxisArmController()

        # read target pose state as 
        before_read_t = time.perf_counter()
        state = self.arm.read() # read current joint position from robot
        action = self.teleop.get_action() # target joint position from gamepad
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        # do action
        before_write_t = time.perf_counter()
        control_mode = self.teleop.get_control_mode()
        target_joints = list(action.values())
        if control_mode == "ee":
            pose_target = self.teleop.get_pose_target()
            if pose_target is not None:
                self.arm.write_pose(pose_target, self.teleop.gripper)
            else:
                self.arm.write(target_joints)
        else:
            self.arm.write(target_joints)
        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        if not record_data:
            return
        
        state = torch.as_tensor(list(state.values()), dtype=torch.float32)
        action = torch.as_tensor(list(action.values()), dtype=torch.float32)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionnaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict



    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Write the predicted actions from policy to the motors"""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Piper is not connected. You need to run `robot.connect()`."
            )

        target_joints = action.tolist()
        current_state = list(self.arm.read().values())

        clipped_action = []
        for idx, (target, current) in enumerate(zip(target_joints, current_state, strict=True)):
            max_delta = self.policy_gripper_delta_limit if idx == len(target_joints) - 1 else self.policy_joint_delta_limit
            clipped_action.append(float(np.clip(target, current - max_delta, current + max_delta)))

        self.logs["policy_action_raw"] = target_joints
        self.logs["policy_action_clipped"] = clipped_action

        # send to motors, torch to list
        target_joints = clipped_action
        self.arm.write(target_joints)

        return torch.as_tensor(target_joints, dtype=action.dtype)



    def capture_observation(self) -> dict:
        """capture current images and joint positions"""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "Piper is not connected. You need to run `robot.connect()`."
            )
        
        # read current joint positions
        before_read_t = time.perf_counter()
        state = self.arm.read()  # 6 joints + 1 gripper
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        state = torch.as_tensor(list(state.values()), dtype=torch.float32)

        # read images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionnaries and format to pytorch
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
        """ move to home position after record one episode """
        self.run_calibration()

    
    def __del__(self):
        if self.is_connected:
            self.disconnect()
            if not self.inference_time:
                self.teleop.stop()
