#!/usr/bin/env python

import argparse
import logging
import time
from typing import Any

import torch
import zmq

from lerobot.common.robot_devices.robots.utils import make_robot
from lerobot.common.robot_devices.utils import busy_wait


DROP_JOINT_INDEX = 3
DEFAULT_CAMERA_MAPPING = {
    "observation.images.one": "observation.images.camera0",
    "observation.images.two": "observation.images.camera1",
}


def parse_camera_mappings(values: list[str] | None) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_CAMERA_MAPPING)

    mapping: dict[str, str] = {}
    for value in values:
        local_key, remote_key = value.split("=", maxsplit=1)
        if not local_key.startswith("observation.images."):
            local_key = f"observation.images.{local_key}"
        if not remote_key.startswith("observation.images."):
            remote_key = f"observation.images.{remote_key}"
        mapping[local_key] = remote_key
    return mapping


def compress_state(full_state: list[float]) -> list[float]:
    return [value for idx, value in enumerate(full_state) if idx != DROP_JOINT_INDEX]


def expand_action(compact_action: list[float], fixed_joint4: float) -> list[float]:
    if len(compact_action) != 6:
        raise ValueError(f"Expected 6-D action from pi0 server, got {len(compact_action)} values.")

    full_action = []
    compact_idx = 0
    for full_idx in range(7):
        if full_idx == DROP_JOINT_INDEX:
            full_action.append(float(fixed_joint4))
        else:
            full_action.append(float(compact_action[compact_idx]))
            compact_idx += 1
    return full_action


def build_request(
    observation: dict[str, Any],
    task: str,
    camera_mapping: dict[str, str],
) -> tuple[dict[str, Any], float]:
    full_state = observation["observation.state"].tolist()
    fixed_joint4 = float(full_state[DROP_JOINT_INDEX])

    images: dict[str, Any] = {}
    for local_key, remote_key in camera_mapping.items():
        if local_key not in observation:
            continue
        images[remote_key] = observation[local_key].cpu().numpy()

    if not images:
        raise ValueError(
            f"No mapped images found in observation. Observation keys: {list(observation.keys())}, "
            f"camera_mapping: {camera_mapping}"
        )

    payload = {
        "type": "infer",
        "task": task,
        "state": compress_state(full_state),
        "images": images,
    }
    return payload, fixed_joint4


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Piper locally and delegate pi0 inference to a remote GPU.")
    parser.add_argument("--server", default="tcp://127.0.0.1:5555", help="ZeroMQ server endpoint.")
    parser.add_argument("--task", required=True, help="Instruction string to send to pi0.")
    parser.add_argument("--fps", type=int, default=10, help="Control loop rate.")
    parser.add_argument("--duration-s", type=float, default=60.0, help="How long to run control for.")
    parser.add_argument("--robot-type", default="piper", help="Robot type to instantiate locally.")
    parser.add_argument(
        "--camera-mapping",
        action="append",
        help=(
            "Map local image keys to remote policy keys, e.g. "
            "--camera-mapping one=camera0 --camera-mapping two=camera1"
        ),
    )
    parser.add_argument(
        "--fixed-joint4",
        type=float,
        default=None,
        help="Optional constant to hold joint_4 at instead of using the initial observed value.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    camera_mapping = parse_camera_mappings(args.camera_mapping)
    robot = make_robot(args.robot_type, inference_time=True)
    robot.connect()

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(args.server)

    try:
        socket.send_pyobj({"type": "reset"})
        reset_reply = socket.recv_pyobj()
        if reset_reply.get("status") != "ok":
            raise RuntimeError(f"Failed to reset remote policy: {reset_reply}")

        start_t = time.perf_counter()
        step_idx = 0
        fixed_joint4 = args.fixed_joint4

        while time.perf_counter() - start_t < args.duration_s:
            loop_start_t = time.perf_counter()

            observation = robot.capture_observation()
            request, inferred_joint4 = build_request(observation, args.task, camera_mapping)
            if fixed_joint4 is None:
                fixed_joint4 = inferred_joint4

            socket.send_pyobj(request)
            reply = socket.recv_pyobj()
            if reply.get("status") != "ok":
                raise RuntimeError(f"Remote inference failed: {reply}")

            full_action = expand_action(reply["action"], fixed_joint4)
            robot.send_action(torch.tensor(full_action, dtype=torch.float32))

            dt_s = time.perf_counter() - loop_start_t
            logging.info("step=%s loop_dt_ms=%.2f", step_idx, dt_s * 1000)
            if args.fps is not None:
                busy_wait(1 / args.fps - dt_s)
            step_idx += 1
    finally:
        try:
            robot.disconnect()
        finally:
            socket.close(0)
            context.term()


if __name__ == "__main__":
    main()
