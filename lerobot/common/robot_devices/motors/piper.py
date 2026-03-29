import time
from typing import Dict
from piper_sdk import *
from lerobot.common.robot_devices.motors.configs import PiperMotorsBusConfig

class PiperMotorsBus:
    """
        对Piper SDK的二次封装
    """
    def __init__(self, 
                 config: PiperMotorsBusConfig):
        self.piper = C_PiperInterface_V2(config.can_name)
        self.piper.ConnectPort()
        self.motors = config.motors
        self.init_joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # [6 joints + 1 gripper] * 0.0
        self.safe_disable_position = [0.0, 0.0, 0.0, 0.0, 0.52, 0.0, 0.0]
        self.pose_factor = 1000 # 单位 0.001mm
        self.joint_factor = 57324.840764 # 1000*180/3.14， rad -> 度（单位0.001度）
        self.joint_command_deadband = 120
        self.gripper_command_deadband = 2000
        self._last_joint_command = None
        self._last_gripper_command = None
        self._last_move_mode = None

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]


    def connect(self, enable:bool) -> bool:
        '''
            使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
        '''
        enable_flag = False
        loop_flag = False
        # 设置超时时间（秒）
        timeout = 5
        # 记录进入循环前的时间
        start_time = time.time()
        while not (loop_flag):
            elapsed_time = time.time() - start_time
            print(f"--------------------")
            enable_list = []
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status)
            if(enable):
                enable_flag = all(enable_list)
                self.piper.EnableArm(7)
                self.piper.GripperCtrl(0,1000,0x01, 0)
            else:
                # move to safe disconnect position
                enable_flag = any(enable_list)
                self.piper.DisableArm(7)
                self.piper.GripperCtrl(0,1000,0x02, 0)
            print(f"使能状态: {enable_flag}")
            print(f"--------------------")
            if(enable_flag == enable):
                loop_flag = True
                enable_flag = True
            else: 
                loop_flag = False
                enable_flag = False
            # 检查是否超过超时时间
            if elapsed_time > timeout:
                print(f"超时....")
                enable_flag = False
                loop_flag = True
                break
            time.sleep(0.5)
        resp = enable_flag
        print(f"Returning response: {resp}")
        return resp
    
    def motor_names(self):
        return

    def set_calibration(self):
        return
    
    def revert_calibration(self):
        return

    def apply_calibration(self):
        """
            移动到初始位置
        """
        self.write(target_joint=self.init_joint_position)

    def write(self, target_joint:list):
        """
            Joint control
            - target joint: in radians
                joint_1 (float): 关节1角度 (-92000~92000) / 57324.840764
                joint_2 (float): 关节2角度 -1300 ~ 90000 / 57324.840764
                joint_3 (float): 关节3角度 2400 ~ -80000 / 57324.840764
                joint_4 (float): 关节4角度 -90000~90000 / 57324.840764
                joint_5 (float): 关节5角度 19000~-77000 / 57324.840764
                joint_6 (float): 关节6角度 -90000~90000 / 57324.840764
                gripper_range: 夹爪角度 0~0.08
        """
        joint_0 = round(target_joint[0]*self.joint_factor)
        joint_1 = round(target_joint[1]*self.joint_factor)
        joint_2 = round(target_joint[2]*self.joint_factor)
        joint_3 = round(target_joint[3]*self.joint_factor)
        joint_4 = round(target_joint[4]*self.joint_factor)
        joint_5 = round(target_joint[5]*self.joint_factor)
        gripper_range = round(target_joint[6]*1000*1000)

        joint_commands = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5]
        joints_changed = self._last_joint_command is None or any(
            abs(curr - prev) >= self.joint_command_deadband
            for curr, prev in zip(joint_commands, self._last_joint_command)
        )
        gripper_changed = (
            self._last_gripper_command is None
            or abs(gripper_range - self._last_gripper_command) >= self.gripper_command_deadband
        )

        if not joints_changed and not gripper_changed:
            return

        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00) # joint control
        self._last_move_mode = "joint"
        if joints_changed:
            self.piper.JointCtrl(*joint_commands)
            self._last_joint_command = joint_commands
        if gripper_changed:
            self.piper.GripperCtrl(gripper_range, 1000, 0x01, 0) # 单位 0.001°
            self._last_gripper_command = gripper_range

    def write_pose(self, target_pose: list | tuple, gripper: float):
        """
            End-effector pose control
            - target_pose: [x, y, z, rx, ry, rz]
              x/y/z in meters, rx/ry/rz in degrees
            - gripper: opening in meters
        """
        if len(target_pose) != 6:
            raise ValueError(f"Expected 6D end-effector pose, got {len(target_pose)} values.")

        xyz_cmd = [round(axis * 1_000_000) for axis in target_pose[:3]]
        rpy_cmd = [round(axis * 1000) for axis in target_pose[3:]]
        gripper_range = round(gripper * 1000 * 1000)

        if self._last_move_mode != "pose":
            # SDK requires switching to end-pose control mode before EndPoseCtrl.
            self.piper.ModeCtrl(0x01, 0x00, 100, 0x00)
            self._last_move_mode = "pose"

        self.piper.EndPoseCtrl(*(xyz_cmd + rpy_cmd))

        gripper_changed = (
            self._last_gripper_command is None
            or abs(gripper_range - self._last_gripper_command) >= self.gripper_command_deadband
        )
        if gripper_changed:
            self.piper.GripperCtrl(gripper_range, 1000, 0x01, 0)
            self._last_gripper_command = gripper_range
    

    def read(self) -> Dict:
        """
            返回统一单位的机器人状态:
            - joints: radians
            - gripper: meters
        """
        joint_msg = self.piper.GetArmJointMsgs()
        joint_state = joint_msg.joint_state

        gripper_msg = self.piper.GetArmGripperMsgs()
        gripper_state = gripper_msg.gripper_state
        
        return {
            "joint_1": joint_state.joint_1 / self.joint_factor,
            "joint_2": joint_state.joint_2 / self.joint_factor,
            "joint_3": joint_state.joint_3 / self.joint_factor,
            "joint_4": joint_state.joint_4 / self.joint_factor,
            "joint_5": joint_state.joint_5 / self.joint_factor,
            "joint_6": joint_state.joint_6 / self.joint_factor,
            "gripper": gripper_state.grippers_angle / 1_000_000,
        }
    
    def safe_disconnect(self):
        """ 
            Move to safe disconnect position
        """
        self.write(target_joint=self.safe_disable_position)
