 #!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys

import torch
import numpy as np

import time
from queue import Queue
from typing import Any

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .configuration_joycon import JoyconTeleopConfig
from .joycon_controller import (
    FixedAxesJoyconRobotics,
    SimpleTeleopArm,
    SimpleHeadControl,
    get_joycon_base_action,
    get_joycon_speed_control,
    LEFT_JOINT_MAP,
    RIGHT_JOINT_MAP,
)

logger = logging.getLogger(__name__)


class JoyconTeleop(Teleoperator):
    """
    Teleop class to use joycon inputs for control. This class wraps the example
    implementation found in examples/xlerobot_teleop_joycon.py and exposes it as a
    Teleoperator that can be instantiated by the factory from a config.

    Notes:
    - connect() accepts an optional `robot` keyword. When provided the teleop will
      use the robot for computing P-control actions and base conversions. If no
      robot is provided, get_action() will raise DeviceNotConnectedError.
    """

    config_class = JoyconTeleopConfig
    name = "joycon"
    is_connected = True

    def __init__(self, config: JoyconTeleopConfig):
        super().__init__(config)
        self.config = config

        self.controllers_right: FixedAxesJoyconRobotics | None = None
        self.controllers_left: FixedAxesJoyconRobotics | None = None
        self.left_arm: SimpleTeleopArm | None = None
        self.right_arm: SimpleTeleopArm | None = None
        self.head_control: SimpleHeadControl | None = None
        self._connected = False
        self.robot = None

    @property
    def action_features(self) -> dict:
            # Return a generic action feature matching the xlerobot action space,
            # including left/right arm (6 joints each), head (2) and base velocities (x,y,theta).
            return {
                "dtype": "float32",
                "shape": (17,),
                "names": {
                    # left arm (6)
                    "left_shoulder_pan": 0,
                    "left_shoulder_lift": 1,
                    "left_elbow": 2,
                    "left_wrist1": 3,
                    "left_wrist2": 4,
                    "left_wrist3": 5,
                    # right arm (6)
                    "right_shoulder_pan": 6,
                    "right_shoulder_lift": 7,
                    "right_elbow": 8,
                    "right_wrist1": 9,
                    "right_wrist2": 10,
                    "right_wrist3": 11,
                    # head (2)
                    "head_pan": 12,
                    "head_tilt": 13,
                    # base velocities (3)
                    "x.vel": 14,
                    "y.vel": 15,
                    "theta.vel": 16,
                },
            }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, robot=None, calibrate: bool = True  ) -> None:  # allow optional robot
        # Store robot reference when provided (useful to compute p-control and base mapping)
        self.robot = robot
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

         

        # Instantiate hardware/controllers
        # These constructors will raise if Joy-Con hardware is not available.
        self.controllers_right = FixedAxesJoyconRobotics("right", dof_speed=[2, 2, 2, 1, 1, 1])
        self.controllers_left = FixedAxesJoyconRobotics("left", dof_speed=[2, 2, 2, 1, 1, 1])

        # If robot provided, use its observation to initialize arms/head
        if self.robot is not None:
            obs = self.robot.get_observation()
            kin_left = SimpleTeleopArm  # placeholder for typing
            kin_right = SimpleTeleopArm
            # instantiate SO101 kinematics used by Example (we import inside joycon_controller file)
            from lerobot.model.SO101Robot import SO101Kinematics

            kin_l = SO101Kinematics()
            kin_r = SO101Kinematics()
            self.left_arm = SimpleTeleopArm(LEFT_JOINT_MAP, obs, kin_l, prefix="left")
            self.right_arm = SimpleTeleopArm(RIGHT_JOINT_MAP, obs, kin_r, prefix="right")
            self.head_control = SimpleHeadControl(obs)


        self._connected = True
        logger.info(f"{self} connected")

    @property
    def is_calibrated(self) -> bool:
        # Joy-Con doesn't require motor calibration
        return True

    def calibrate(self) -> None:
        # No-op for Joy-Con wrapper
        return None

    def configure(self) -> None:
        # No-op
        return None

    def get_action(self) -> dict[str, Any]:
        """Read joycons and compute a robot action dictionary.

        Requires that connect(robot=...) was called so self.robot is set. If not set,
        an error is raised because arm P-control needs robot observations to compute goals.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected. Call connect() first.")

        if self.robot is None:
            raise DeviceNotConnectedError(f"{self} needs a robot instance passed to connect(robot=...) to compute actions.")

        # Read controllers
        pose_right, gripper_right, control_button_right = self.controllers_right.get_control()
        pose_left, gripper_left, control_button_left = self.controllers_left.get_control()

        # Reset behavior: if right control button signals reset (8 in example), move to zero
        if control_button_right == 8:
            if self.right_arm is not None and self.left_arm is not None and self.head_control is not None:
                self.right_arm.move_to_zero_position(self.robot)
                self.left_arm.move_to_zero_position(self.robot)
                self.head_control.move_to_zero_position(self.robot)

        # Update targets using joycon poses
        if self.right_arm is not None:
            self.right_arm.handle_joycon_input(pose_right, gripper_right)
        if self.left_arm is not None:
            self.left_arm.handle_joycon_input(pose_left, gripper_left)
        if self.head_control is not None:
            # head uses left joycon directional pad
            self.head_control.handle_joycon_input(self.controllers_left)

        # Compute P-control actions by querying current robot observation
        # Note: p_control_action expects robot to read current joint positions
        right_action = self.right_arm.p_control_action(self.robot) if self.right_arm is not None else {}
        left_action = self.left_arm.p_control_action(self.robot) if self.left_arm is not None else {}
        head_action = self.head_control.p_control_action(self.robot) if self.head_control is not None else {}

        # Compute base action directly (x.vel, y.vel, theta.vel) from right joycon buttons
        # This mirrors get_joycon_base_action but produces direct velocity keys used by XLerobot
        base_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        # Simple mapping: X forward, B backward, Y left, A right
        if self.controllers_right.joycon.get_button_x() == 1:
            base_action["x.vel"] += get_joycon_speed_control(self.controllers_right)
        if self.controllers_right.joycon.get_button_b() == 1:
            base_action["x.vel"] -= get_joycon_speed_control(self.controllers_right)
        if self.controllers_right.joycon.get_button_y() == 1:
            base_action["y.vel"] += get_joycon_speed_control(self.controllers_right)
        if self.controllers_right.joycon.get_button_a() == 1:
            base_action["y.vel"] -= get_joycon_speed_control(self.controllers_right)

        # Merge all actions
        action = {**left_action, **right_action, **head_action, **base_action}
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # No force feedback for Joy-Con in this wrapper
        return None

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            if self.controllers_left is not None:
                self.controllers_left.disconnect()
            if self.controllers_right is not None:
                self.controllers_right.disconnect()
        except Exception:
            logger.debug("Error while disconnecting controllers", exc_info=True)

        self._connected = False
        self.robot = None
