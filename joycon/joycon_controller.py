# To Run on the host
'''
PYTHONPATH=src python -m lerobot.robots.xlerobot.xlerobot_host --robot.id=my_xlerobot
'''

# To Run the teleop:
'''
PYTHONPATH=src python -m examples.xlerobot.teleoperate_joycon
'''
 
# 基础速度控制说明：
# - 当按住任一底盘控制按钮（X 前进、B 后退、Y 左转、A 右转）时，速度会线性加速至最大速度
# - 释放按钮后，速度会线性减速至 0
# - 你可以通过修改以下参数来调整加速和减速斜率：
#   * BASE_ACCELERATION_RATE: 加速度斜率（速度/秒）
#   * BASE_DECELERATION_RATE: 减速度斜率（速度/秒）
#   * BASE_MAX_SPEED: 最大速度倍增器

import time
import numpy as np
import math

from lerobot.robots.xlerobot import XLerobotConfig, XLerobot
from lerobot.utils.robot_utils import busy_wait
# from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data
from lerobot.model.SO101Robot import SO101Kinematics
from joyconrobotics import JoyconRobotics

LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}
RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}

HEAD_MOTOR_MAP = {
    "head_motor_1": "head_motor_1",
    "head_motor_2": "head_motor_2",
}

class FixedAxesJoyconRobotics(JoyconRobotics):
    def __init__(self, device, **kwargs):
        super().__init__(device, **kwargs)
        
        # Set different center values for left and right Joy-Cons
        if self.joycon.is_right():
            self.joycon_stick_v_0 = 1900 
            self.joycon_stick_h_0 = 2100
        else:  # left Joy-Con
            self.joycon_stick_v_0 = 2300 
            self.joycon_stick_h_0 = 2000
        self.gripper_state = 0.0
        # Gripper control related variables
        self.gripper_speed = 0.4  # Gripper open/close speed (degrees/frame)
        self.gripper_direction = 1  # 1 means open, -1 means close
        self.gripper_min = 0  # Minimum angle (fully closed)
        self.gripper_max = 90  # Maximum angle (fully open)
        self.last_gripper_button_state = 0  # Record previous frame button state for detecting press events
    
    def common_update(self):
        # 添加属性检查，避免未初始化错误
        if not hasattr(self, 'joycon_stick_v_0'):
            self.joycon_stick_v_0 = 1900 if self.joycon.is_right() else 2300
        if not hasattr(self, 'joycon_stick_h_0'):
            self.joycon_stick_h_0 = 2100 if self.joycon.is_right() else 2000
        
        # Modified update logic: joystick only controls fixed axes
        speed_scale1 = 0.0001 # Speed scaling factor
        speed_scale2 =0.0005
        # Get current orientation data to print pitch
        orientation_rad = self.get_orientation()
        roll, pitch, yaw = orientation_rad

        
        # Vertical joystick: controls X and Z axes (forward/backward)
        joycon_stick_v = self.joycon.get_stick_right_vertical() if self.joycon.is_right() else self.joycon.get_stick_left_vertical()
        joycon_stick_v_threshold = 300
        joycon_stick_v_range = 1000
        if joycon_stick_v > joycon_stick_v_threshold + self.joycon_stick_v_0:
            self.position[0] += speed_scale1 * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[0] * self.direction_reverse[0] * math.cos(pitch)
            self.position[2] += speed_scale1 * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[1] * self.direction_reverse[1] * math.sin(pitch)
        elif joycon_stick_v < self.joycon_stick_v_0 - joycon_stick_v_threshold:
            self.position[0] += speed_scale1 * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[0] * self.direction_reverse[0] * math.cos(pitch)
            self.position[2] += speed_scale1 * (joycon_stick_v - self.joycon_stick_v_0) / joycon_stick_v_range *self.dof_speed[1] * self.direction_reverse[1] * math.sin(pitch)
        
        # Horizontal joystick: only controls Y axis (left/right)
        joycon_stick_h = self.joycon.get_stick_right_horizontal() if self.joycon.is_right() else self.joycon.get_stick_left_horizontal()
        joycon_stick_h_threshold = 300
        joycon_stick_h_range = 1000
        if joycon_stick_h > joycon_stick_h_threshold + self.joycon_stick_h_0:
            self.position[1] += speed_scale2 * (joycon_stick_h - self.joycon_stick_h_0) / joycon_stick_h_range * self.dof_speed[1] * self.direction_reverse[1]
        elif joycon_stick_h < self.joycon_stick_h_0 - joycon_stick_h_threshold:
            self.position[1] += speed_scale2 * (joycon_stick_h - self.joycon_stick_h_0) / joycon_stick_h_range * self.dof_speed[1] * self.direction_reverse[1]
        
        # Z-axis button control
        joycon_button_up = self.joycon.get_button_r() if self.joycon.is_right() else self.joycon.get_button_l()
        if joycon_button_up == 1:
            self.position[2] += speed_scale2 * self.dof_speed[2] * self.direction_reverse[2]
        
        joycon_button_down = self.joycon.get_button_r_stick() if self.joycon.is_right() else self.joycon.get_button_l_stick()
        if joycon_button_down == 1:
            self.position[2] -= speed_scale2 * self.dof_speed[2] * self.direction_reverse[2]
        
        # Home button reset logic (simplified version)
        joycon_button_home = self.joycon.get_button_home() if self.joycon.is_right() else self.joycon.get_button_capture()
        if joycon_button_home == 1:
           
            self.position = self.offset_position_m.copy()
        
        # Gripper control logic (hold for linear increase/decrease mode)
        for event_type, status in self.button.events():
            if (self.joycon.is_right() and event_type == 'plus' and status == 1) or (self.joycon.is_left() and event_type == 'minus' and status == 1):
                self.reset_button = 1
                self.reset_joycon()
            elif self.joycon.is_right() and event_type == 'a':
                self.next_episode_button = status
            elif self.joycon.is_right() and event_type == 'y':
                self.restart_episode_button = status
            else: 
                self.reset_button = 0
        
        # Gripper button state detection and direction control
        gripper_button_pressed = False
        if self.joycon.is_right():
            # Right Joy-Con uses ZR button
            if not self.change_down_to_gripper:
                gripper_button_pressed = self.joycon.get_button_zr() == 1
            else:
                gripper_button_pressed = self.joycon.get_button_stick_r_btn() == 1
        else:
            # Left Joy-Con uses ZL button
            if not self.change_down_to_gripper:
                gripper_button_pressed = self.joycon.get_button_zl() == 1
            else:
                gripper_button_pressed = self.joycon.get_button_stick_l_btn() == 1
        
        # Detect button press events (from 0 to 1) to change direction
        if gripper_button_pressed and self.last_gripper_button_state == 0:
            # Button just pressed, change direction
            self.gripper_direction *= -1
            print(f"[GRIPPER] Direction changed to: {'Open' if self.gripper_direction == 1 else 'Close'}")
        
        # Update button state record
        self.last_gripper_button_state = gripper_button_pressed
        
        # Linear control of gripper open/close when holding gripper button
        if gripper_button_pressed:
            # Check if exceeding limits
            new_gripper_state = self.gripper_state + self.gripper_direction * self.gripper_speed
            
            # If exceeding limits, stop moving
            if new_gripper_state >= self.gripper_min and new_gripper_state <= self.gripper_max:
                self.gripper_state = new_gripper_state
            # If exceeding limits, stay at current position, don't change direction

        

        # Button control state
        if self.joycon.is_right():
            if self.next_episode_button == 1:
                self.button_control = 1
            elif self.restart_episode_button == 1:
                self.button_control = -1
            elif self.reset_button == 1:
                self.button_control = 8
            else:
                self.button_control = 0
        
        return self.position, self.gripper_state, self.button_control
    
class SimpleTeleopArm:
    def __init__(self, joint_map, initial_obs, kinematics, prefix="right", kp=1.0):
        self.joint_map = joint_map
        self.prefix = prefix
        self.kp = kp
        self.kinematics = kinematics
       
        # Initial joint positions
        self.joint_positions = {
            "shoulder_pan": initial_obs[f"{prefix}_arm_shoulder_pan.pos"],
            "shoulder_lift": initial_obs[f"{prefix}_arm_shoulder_lift.pos"],
            "elbow_flex": initial_obs[f"{prefix}_arm_elbow_flex.pos"],
            "wrist_flex": initial_obs[f"{prefix}_arm_wrist_flex.pos"],
            "wrist_roll": initial_obs[f"{prefix}_arm_wrist_roll.pos"],
            "gripper": initial_obs[f"{prefix}_arm_gripper.pos"],
        }
        
        # Set initial x/y to fixed values
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        
        # Set step size
        self.degree_step = 2.0
        self.xy_step = 0.5
        
        # P control target positions, set to zero position
        self.target_positions = {
           'shoulder_pan': 0.0,
            'shoulder_lift': -100.0,
            'elbow_flex': 100.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }
        self.zero_pos = {
            'shoulder_pan': 0.0,
            'shoulder_lift': -100.0,
            'elbow_flex': 100.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }

    def move_to_zero_position(self, robot):
        print(f"[{self.prefix}] Moving to Zero Position: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()
        
        # Reset kinematics variables to initial state
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        
        # Explicitly set wrist_flex
        self.target_positions["wrist_flex"] = 0.0
        
        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_joycon_input(self, joycon_pose, gripper_state):
        """Handle Joy-Con input, update arm control - based on 6_so100_joycon_ee_control.py"""
        x, y, z, roll_, pitch_, yaw = joycon_pose
        
        # 计算俯仰控制 - 与 6_so100_joycon_ee_control.py 保持一致
        pitch = -pitch_ * 60 + 10
        
        # 设置坐标 - 与 6_so100_joycon_ee_control.py 保持一致
        current_x = 0.0+ x
        current_y = 0.0 + z
         # 将 x 和 z 作为增量更新 current_x/current_y（小步长）
        # 当 y < -0.08 时，禁止 z 方向的任何增量（每次 z 增量为 0）
        self.current_x += x * self.xy_step

        if y < -0.08:
            # 禁止 z 增量
            z_inc = 0.0
        else:
            z_inc = z * self.xy_step
        self.current_y += z_inc

        # 计算横滚 - 与 6_so100_joycon_ee_control.py 保持一致
        roll = roll_ * 45
        
        print(f"[{self.prefix}] pitch: {pitch}")
        
        # 将 y 值用于控制 shoulder_pan 关节 - 与 6_so100_joycon_ee_control.py 保持一致
        y_scale = 150.0  # 缩放因子，可根据需要调整
        self.target_positions["shoulder_pan"] = y * y_scale
        
        # 使用逆运动学计算关节角 - 与 6_so100_joycon_ee_control.py 保持一致
        try:
            joint2_target, joint3_target = self.kinematics.inverse_kinematics(current_x, current_y)
            self.target_positions["shoulder_lift"] = joint2_target
            self.target_positions["elbow_flex"] = joint3_target
        except Exception as e:
            print(f"[{self.prefix}] IK failed: {e}")
        
        # 设置 wrist_flex - 与 6_so100_joycon_ee_control.py 保持一致
        self.target_positions["wrist_flex"] = -self.target_positions["shoulder_lift"] - self.target_positions["elbow_flex"] + pitch
        
        # 设置 wrist_roll - 与 6_so100_joycon_ee_control.py 保持一致
        self.target_positions["wrist_roll"] = roll
        self.target_positions["gripper"] = gripper_state
        # 抓手控制现在在主循环中直接设置，此处无需处理
        pass

    # def p_control_action(self, robot):
    #     obs = robot.get_observation()
    #     current = {j: obs[f"{self.prefix}_arm_{j}.pos"] for j in self.joint_map}
    #     action = {}
    #     for j in self.target_positions:
    #         error = self.target_positions[j] - current[j]
    #         control = self.kp * error
    #         action[f"{self.joint_map[j]}.pos"] = current[j] + control
    #     return action
    def p_control_action(self, robot):
        obs = robot.get_observation()
        current = {j: obs[f"{self.prefix}_arm_{j}.pos"] for j in self.joint_map}
        
        # 初始化平滑属性，如果不存在
        if not hasattr(self, 'smoothing_tau'):
            self.smoothing_tau = 0.6 # 平滑时间常数（秒，可调整）
        if not hasattr(self, 'last_smooth_time'):
            self.last_smooth_time = time.time()
        if not hasattr(self, 'smoothed_targets'):
            self.smoothed_targets = self.target_positions.copy()
        
        # 使用指数平滑更新平滑目标
        now = time.time()
        dt = now - self.last_smooth_time
        self.last_smooth_time = now
        
        action = {}
        for j in self.target_positions:
            if j not in self.smoothed_targets:
                self.smoothed_targets[j] = current.get(j, self.target_positions[j])
            
            # 指数平滑：smoothed = smoothed + alpha * (target - smoothed)
            alpha = 1 - math.exp(-dt / self.smoothing_tau) if dt > 0 else 1.0
            self.smoothed_targets[j] += alpha * (self.target_positions[j] - self.smoothed_targets[j])
            
            # 使用平滑目标进行 P 控制
            error = self.smoothed_targets[j] - current[j]
            control = self.kp * error
            action[f"{self.joint_map[j]}.pos"] = current[j] + control
        
        return action

    # def p_control_action(self, robot):
    #     obs = robot.get_observation()
    #     current = {j: obs[f"{self.prefix}_arm_{j}.pos"] for j in self.joint_map}
        
    #     # 对前三个关节应用平滑控制
    #     smoothed_positions = self.smooth_controller.update(self.target_positions, current)
        
    #     action = {}
    #     for j in self.target_positions:
    #         if j in ["shoulder_pan", "shoulder_lift", "elbow_flex"]:
    #             # 对前三个关节使用平滑后的位置
    #             error = smoothed_positions[j] - current[j]
    #         else:
    #             # 对其他关节（wrist_flex, wrist_roll, gripper）使用直接控制
    #             error = self.target_positions[j] - current[j]
            
    #         control = self.kp * error
    #         action[f"{self.joint_map[j]}.pos"] = current[j] + control
    #     return action
    
    # def p_control_action(self, robot):
    #     """
    #     平滑 + PID 跟踪：
    #     - 对 target 先做指数平滑并做速率限制（与之前实现一致）
    #     - 使用每关节的 PID 控制器跟踪平滑后的目标（带积分防风和微分项）
    #     - 最终输出位置命令：current + pid_output（并对单帧改变量做限幅）
    #     """
    #     obs = robot.get_observation()
    #     now = time.time()

    #     # 平滑参数
    #     tau = getattr(self, "smoothing_time", 0.06)
    #     max_rate = getattr(self, "max_rate", 20.0)    # deg/sec, 用于单帧限幅
    #     deadband = getattr(self, "deadband", 0.02)

    #     # PID 参数（可在实例上调整）
    #     kp = getattr(self, "pid_kp", self.kp)
    #     ki = getattr(self, "pid_ki", 0.0)
    #     kd = getattr(self, "pid_kd", 0.0)
    #     max_integral = getattr(self, "pid_max_integral", 1000.0)

    #     if not hasattr(self, "_smooth_state"):
    #         self._smooth_state = {}

    #     action = {}
    #     for j in self.target_positions:
    #         obs_key = f"{self.prefix}_arm_{j}.pos"
    #         current = float(obs.get(obs_key, 0.0))
    #         target = float(self.target_positions[j])

    #         state = self._smooth_state.get(j)
    #         if state is None:
    #         # 初始化平滑与 PID 状态
    #           state = {
    #             "output": current,
    #             "last_time": now,
    #             "integral": 0.0,
    #             "last_error": 0.0,
    #         }
    #         self._smooth_state[j] = state

    #         dt = max(1e-6, now - state["last_time"])

    #         # 指数平滑（低通滤波）
    #         alpha = 1.0 - math.exp(-dt / max(tau, 1e-6))
    #         filtered = state["output"] + alpha * (target - state["output"])

    #         # 先做速率限制（基于位置变化的最大速率）
    #         max_delta = max_rate * dt
    #         delta = filtered - state["output"]
    #         if abs(delta) > max_delta:
    #           filtered = state["output"] + math.copysign(max_delta, delta)

    #         # 小误差直接置为目标，消除微抖
    #         if abs(filtered - target) < deadband:
    #           filtered = target

    #         # PID 控制部分
    #         error = filtered - current

    #         # 积分累积并防风
    #         state["integral"] += error * dt
    #         # anti-windup
    #         if state["integral"] > max_integral:
    #            state["integral"] = max_integral
    #         elif state["integral"] < -max_integral:
    #            state["integral"] = -max_integral

    #         # 微分（差分）
    #         derivative = (error - state["last_error"]) / dt

    #         # PID 输出（单位与关节角度一致）
    #         pid_out = kp * error + ki * state["integral"] + kd * derivative

    #         # 将 PID 输出在单帧内限幅，避免大幅抖动步幅
    #         if abs(pid_out) > max_delta:
    #           pid_out = math.copysign(max_delta, pid_out)

    #         # 小误差不动
    #         if abs(error) < deadband:
    #            pid_out = 0.0

    #         # 更新状态
    #         state["output"] = filtered
    #         state["last_time"] = now
    #         state["last_error"] = error

    #         # 输出为当前观测 + 控制量（仍为位置命令）
    #         action[f"{self.joint_map[j]}.pos"] = current + pid_out

    #     return action
    # def p_control_action(self, robot):
    #     """
    #     使用二次多项式（quadratic easing）对臂电机目标位置做平滑插值：
    #     desired(t) = start + (target - start) * (tau^2)，tau = (t - start_time) / T
    #     当 target 发生显著变化时重置起始点和时间。
    #     """
    #     obs = robot.get_observation()
    #     now = time.time()

    #     # 平滑时长（秒），可按需调整
    #     if not hasattr(self, "smoothing_time"):
    #         self.smoothing_time = 0.2  # 默认 0.2 秒平滑时间

    #     # 存储每个关节的平滑状态
    #     if not hasattr(self, "_arm_smoothing"):
    #         self._arm_smoothing = {}

    #     # 读取当前关节位姿
    #     current = {}
    #     for j in self.joint_map:
    #         obs_key = f"{self.prefix}_arm_{j}.pos"
    #         current[j] = obs.get(obs_key, 0.0)

    #     action = {}
    #     for j in self.target_positions:
    #         obs_key = f"{self.prefix}_arm_{j}.pos"
    #         curr_obs = current.get(j, 0.0)
    #         target = float(self.target_positions[j])

    #         state = self._arm_smoothing.get(j)
    #         # 如果没有状态或者目标发生显著变化，则重置平滑轨迹
    #         if state is None or abs(target - state["target"]) > 1e-3:
    #             self._arm_smoothing[j] = {
    #                 "start": curr_obs,
    #                 "target": target,
    #                 "start_time": now,
    #                 "duration": max(1e-3, self.smoothing_time),
    #             }
    #             state = self._arm_smoothing[j]

    #         t = now - state["start_time"]
    #         T = state["duration"]

    #         # 计算平滑后的期望位置（quadratic easing-in）
    #         if t >= T:
    #             desired = state["target"]
    #         else:
    #             tau = max(0.0, min(1.0, t / T))
    #             desired = state["start"] + (state["target"] - state["start"]) * (tau * tau)

    #         # P 控制器跟踪平滑后的位置
    #         error = desired - curr_obs
    #         control = self.kp * error
    #         action[f"{self.joint_map[j]}.pos"] = curr_obs + control

    #     return action

class SimpleHeadControl:
    def __init__(self, initial_obs, kp=1):
        self.kp = kp
        self.degree_step = 2  # Move 2 degrees each time
        # Initialize head motor positions
        self.target_positions = {
            "head_motor_1": initial_obs.get("head_motor_1.pos", 0.0),
            "head_motor_2": initial_obs.get("head_motor_2.pos", 0.0),
        }
        self.zero_pos = {"head_motor_1": 0.0, "head_motor_2": 0.0}

    def move_to_zero_position(self, robot):
        print("[HEAD] Moving to Zero Position: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()
        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_joycon_input(self, joycon):
        """Handle left Joy-Con directional pad input to control head motors"""
        # Get left Joy-Con directional pad state
        button_up = joycon.joycon.get_button_up()      # Up: head_motor_1+
        button_down = joycon.joycon.get_button_down()  # Down: head_motor_1-
        button_left = joycon.joycon.get_button_left()  # Left: head_motor_2+
        button_right = joycon.joycon.get_button_right() # Right: head_motor_2-
        
        if button_up == 1:
            self.target_positions["head_motor_2"] += self.degree_step
            print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")
        if button_down == 1:
            self.target_positions["head_motor_2"] -= self.degree_step
            print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")
        if button_left == 1:
            self.target_positions["head_motor_1"] += self.degree_step
            print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")
        if button_right == 1:
            self.target_positions["head_motor_1"] -= self.degree_step
            print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")

    def p_control_action(self, robot):
        obs = robot.get_observation()
        action = {}
        for motor in self.target_positions:
            current = obs.get(f"{HEAD_MOTOR_MAP[motor]}.pos", 0.0)
            error = self.target_positions[motor] - current
            control = self.kp * error
            action[f"{HEAD_MOTOR_MAP[motor]}.pos"] = current + control
        return action
 

def get_joycon_base_action(joycon, robot):
    """
    Get base control commands from Joy-Con
    X: forward, B: backward, Y: left turn, A: right turn
    """
    # Get button states
    button_x = joycon.joycon.get_button_x()  # forward
    button_b = joycon.joycon.get_button_b()  # backward
    button_y = joycon.joycon.get_button_y()  # left turn
    button_a = joycon.joycon.get_button_a()  # right turn
    
    # Build key set (simulate keyboard input)
    pressed_keys = set()
    
    if button_x == 1:
        pressed_keys.add('k')  # forward
        print("[BASE] Forward")
    if button_b == 1:
        pressed_keys.add('i')  # backward
        print("[BASE] Backward")
    if button_y == 1:
        pressed_keys.add('u')  # left turn
        print("[BASE] Left turn")
    if button_a == 1:
        pressed_keys.add('o')  # right turn
        print("[BASE] Right turn")
    
    # Convert to numpy array and get base action
    keyboard_keys = np.array(list(pressed_keys))
    base_action = robot._from_keyboard_to_base_action(keyboard_keys) or {}
    
    return base_action

# 基础速度控制参数 - 可调整的加减速斜率
BASE_ACCELERATION_RATE = 2.0  # 加速度斜率（速度/秒）
BASE_DECELERATION_RATE = 2.5  # 减速度斜率（速度/秒）
BASE_MAX_SPEED = 1.5         # 最大速度倍增器

MIN_VELOCITY_THRESHOLD = 0.02 # 减速期间发送给电机的最小速度阈值

# # 机械臂平滑控制参数 - 可调斜率
# ARM_ACCELERATION_RATE = 2.0   # 加速度斜率（度/秒）
# ARM_DECELERATION_RATE = 5.0   # 减速度斜率（度/秒）
# ARM_MAX_SPEED = 2.0           # 最大速度倍增器
# ARM_MIN_VELOCITY_THRESHOLD = 0.1 # 减速期间发送给电机的最小速度阈值
# class SmoothArmController:
#     """对前三个关节实现加速/减速的平滑机械臂控制器"""
    
#     def __init__(self):
#         self.current_speeds = {
#             "shoulder_pan": 0.0001,
#             "shoulder_lift": 0.0001,
#             "elbow_flex": 0.0001
#         }
#         self.last_time = time.time()
#         self.last_directions = {
#             "shoulder_pan": 0.0,
#             "shoulder_lift": 0.0,
#             "elbow_flex": 0.0
#         }
#         self.is_moving = {
#             "shoulder_pan": False,
#             "shoulder_lift": False,
#             "elbow_flex": False
#         }
    
#     def update(self, target_positions, current_positions):
#         """更新平滑控制并返回平滑后的目标位置"""
#         current_time = time.time()
#         dt = current_time - self.last_time if current_time - self.last_time > 1e-6 else 1e-6
#         self.last_time = current_time
        
#         smoothed_positions = target_positions.copy()
        
#         for joint in ["shoulder_pan", "shoulder_lift", "elbow_flex"]:
#             target = target_positions.get(joint, 0.0)
#             current = current_positions.get(joint, 0.0)
            
#             # 计算需要移动的方向和幅度
#             error = target - current
#             abs_error = abs(error)
            
#             if abs_error > 0.01:  # 仅当误差显著时移动
#                 # 确定方向
#                 direction = 1.0 if error > 0 else -1.0
                
#                 # 检查是否开始移动
#                 if not self.is_moving[joint]:
#                     self.is_moving[joint] = True
#                     print(f"[ARM] Starting {joint} movement")
                
#                 # 存储当前方向以便减速时使用
#                 self.last_directions[joint] = direction
#                 # 如果初始速度为0，可以给个小的启动速度避免死区
#                 if self.current_speeds[joint] < 1e-6:
#                     self.current_speeds[joint] = min(ARM_MAX_SPEED, 0.01)
                
#                 # 加速
#                 self.current_speeds[joint] += ARM_ACCELERATION_RATE * dt
#                 self.current_speeds[joint] = min(self.current_speeds[joint], ARM_MAX_SPEED)
                
#                 # 计算移动步长
#                 movement_step = self.current_speeds[joint] * dt * direction
#                 # 防止越过目标：步长不超过误差
#                 if abs(movement_step) > abs_error:
#                     movement_step = error
#                 # 应用移动
#                 smoothed_positions[joint] = current + movement_step
                
#             else:
#                 # 误差不显著 - 减速
#                 if self.is_moving[joint]:
#                     self.is_moving[joint] = False
#                     print(f"[ARM] Starting {joint} deceleration")
                
#                 # 减速期间使用最后方向
#                 if self.current_speeds[joint] > 0.01 and self.last_directions[joint] != 0:
#                     direction = self.last_directions[joint]
#                     movement_step = self.current_speeds[joint] * dt * direction
                    
#                     # 在减速期间确保最小速度
#                     if abs(movement_step) < ARM_MIN_VELOCITY_THRESHOLD:
#                         movement_step = ARM_MIN_VELOCITY_THRESHOLD if direction > 0 else -ARM_MIN_VELOCITY_THRESHOLD
#                     # 同样避免越过目标
#                     if abs(movement_step) > abs_error:
#                         movement_step = error
#                     smoothed_positions[joint] = current + movement_step
#                 else:
#                     smoothed_positions[joint] = current
                
#                 # 减速
#                 self.current_speeds[joint] -= ARM_DECELERATION_RATE * dt
#                 self.current_speeds[joint] = max(self.current_speeds[joint], 0.0)
#             print(f"[{smoothed_positions[joint]}] Joint: {joint} ")
#             print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
#         return smoothed_positions

def get_joycon_speed_control(joycon):
    """
    Get speed control from Joy-Con - linear acceleration and deceleration
    Linearly accelerate to maximum speed when holding any base control button, linearly decelerate to 0 when released
    """
    global current_base_speed, last_update_time, is_accelerating
    
    # Initialize global variables
    if 'current_base_speed' not in globals():
        current_base_speed = 0.0
        last_update_time = time.time()
        is_accelerating = False
    
    current_time = time.time()
    dt = current_time - last_update_time
    last_update_time = current_time
    
    # Check if any base control buttons are pressed
    button_x = joycon.joycon.get_button_x()  # forward
    button_b = joycon.joycon.get_button_b()  # backward
    button_y = joycon.joycon.get_button_y()  # left turn
    button_a = joycon.joycon.get_button_a()  # right turn
    
    any_base_button_pressed = any([button_x, button_b, button_y, button_a])
    
    if any_base_button_pressed:
        # Button pressed - accelerate
        if not is_accelerating:
            is_accelerating = True
            print("[BASE] Starting acceleration")
        
        # Linear acceleration
        current_base_speed += BASE_ACCELERATION_RATE * dt
        current_base_speed = min(current_base_speed, BASE_MAX_SPEED)
        
    else:
        # No button pressed - decelerate
        if is_accelerating:
            is_accelerating = False
            print("[BASE] Starting deceleration")
        
        # Linear deceleration
        current_base_speed -= BASE_DECELERATION_RATE * dt
        current_base_speed = max(current_base_speed, 0.0)
    
    # Print current speed (optional, for debugging)
    if abs(current_base_speed) > 0.01:  # Only print when speed is not 0
        print(f"[BASE] Current speed: {current_base_speed:.2f}")
    
    return current_base_speed


# def main():
#     FPS = 30
    
#     # Try to use saved calibration file to avoid recalibrating each time
#     # You can modify robot_id here to match your robot configuration
#     robot_config = XLerobotConfig(id="my_xlerobot")  # Can be modified to your robot ID
#     robot = XLerobot(robot_config)
    
#     try:
#         robot.connect()
#         print(f"[MAIN] Successfully connected to robot")
#         if robot.is_calibrated:
#             print(f"[MAIN] Robot is calibrated and ready to use!")
#         else:
#             print(f"[MAIN] Robot requires calibration")
#     except Exception as e:
#         print(f"[MAIN] Failed to connect to robot: {e}")
#         print(f"[MAIN] Robot config: {robot_config}")
#         print(f"[MAIN] Robot: {robot}")
#         return

#     # _init_rerun(session_name="xlerobot_teleop_joycon")

#     # Initialize right Joy-Con controller - based on 6_so100_joycon_ee_control.py
#     print("[MAIN] Initializing right Joy-Con controller...")
#     joycon_right = FixedAxesJoyconRobotics(
#         "right",
#         dof_speed=[2, 2, 2, 1, 1, 1]
#     )
#     print(f"[MAIN] Right Joy-Con controller connected")
#     print("[MAIN] Initializing left Joy-Con controller...")
#     joycon_left = FixedAxesJoyconRobotics(
#         "left",
#         dof_speed=[2, 2, 2, 1, 1, 1]
#     )
#     print(f"[MAIN] Left Joy-Con controller connected")

#     # Init the arm and head instances
#     obs = robot.get_observation()
#     kin_left = SO101Kinematics()
#     kin_right = SO101Kinematics()
#     left_arm = SimpleTeleopArm(LEFT_JOINT_MAP, obs, kin_left, prefix="left")
#     right_arm = SimpleTeleopArm(RIGHT_JOINT_MAP, obs, kin_right, prefix="right")
#     head_control = SimpleHeadControl(obs)

#     # Move both arms and head to zero position at start
#     left_arm.move_to_zero_position(robot)
#     right_arm.move_to_zero_position(robot)
#     head_control.move_to_zero_position(robot)

#     try:
#         while True:
#             pose_right, gripper_right, control_button_right = joycon_right.get_control()
#             print(f"pose_right: {pose_right}, gripper_right: {gripper_right}, control_button_right: {control_button_right}")
#             pose_left, gripper_left, control_button_left = joycon_left.get_control()
#             print(f"pose_left: {pose_left}, gripper_left: {gripper_left}, control_button_left: {control_button_left}")

#             if control_button_right == 8:  # reset button
#                 print("[MAIN] Reset to zero position!")
#                 right_arm.move_to_zero_position(robot)
#                 left_arm.move_to_zero_position(robot)
#                 head_control.move_to_zero_position(robot)
#                 continue

#             # Handle gripper control - directly use Joy-Con gripper state
#             right_arm.target_positions["gripper"] = gripper_right
#             left_arm.target_positions["gripper"] = gripper_left
            
#             right_arm.handle_joycon_input(pose_right, gripper_right)
#             right_action = right_arm.p_control_action(robot)
#             left_arm.handle_joycon_input(pose_left, gripper_left)
#             left_action = left_arm.p_control_action(robot)
#             head_control.handle_joycon_input(joycon_left) # Pass joycon_left to head_control
#             head_action = head_control.p_control_action(robot)

#             base_action = get_joycon_base_action(joycon_right, robot)
#             speed_multiplier = get_joycon_speed_control(joycon_right)
            
#             if base_action:
#                 for key in base_action:
#                     if 'vel' in key or 'velocity' in key:  
#                         base_action[key] *= speed_multiplier 

#             # Merge all actions
#             action = {**left_action, **right_action, **head_action, **{}, **base_action}
#             robot.send_action(action)

#             obs = robot.get_observation()
#             # log_rerun_data(obs, action)
#     finally:
#         joycon_right.disconnect()
#         joycon_left.disconnect()
#         robot.disconnect()
#         print("Teleoperation ended.")

# if __name__ == "__main__":
#     main()