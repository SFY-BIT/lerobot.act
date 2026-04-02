import argparse
import time
from pathlib import Path

try:
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf
except ImportError:
    viser = None
    yourdfpy = None
    ViserUrdf = None

from lerobot.common.robot_devices.teleop.gamepad import SixAxisArmController_101


DEFAULT_URDF_PATH = Path("/home/night/Gamepad_PiPER/piper/piper.urdf")
DEFAULT_MESH_DIR = Path("/home/night/Gamepad_PiPER/piper/meshes")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SO101 leader actions on a virtual Piper arm.")
    parser.add_argument("--port", type=str, default=None, help="Leader serial port. Defaults to gamepad.py config.")
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to SO101 calibration file. Defaults to gamepad.py config.",
    )
    parser.add_argument("--hz", type=float, default=30.0, help="Visualization update frequency.")
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_URDF_PATH, help="Path to Piper URDF.")
    parser.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR, help="Path to Piper mesh directory.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable browser visualization and print joint values only.",
    )
    return parser.parse_args()


def build_visualizer(urdf_path: Path, mesh_dir: Path):
    if viser is None or yourdfpy is None or ViserUrdf is None:
        raise RuntimeError("Missing optional dependencies: install `viser` and `yourdfpy`.")
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2.0, height=2.0)
    urdf = yourdfpy.URDF.load(str(urdf_path), mesh_dir=str(mesh_dir))
    vis = ViserUrdf(server, urdf, root_node_name="/base_link")
    return server, vis


def action_to_vis_cfg(action: dict[str, float]) -> list[float]:
    gripper = max(0.0, min(0.08, action["gripper"]))
    finger = gripper / 2.0
    return [
        action["joint0"],
        action["joint1"],
        action["joint2"],
        action["joint3"],
        action["joint4"],
        action["joint5"],
        finger,
        -finger,
    ]


def format_action(action: dict[str, float]) -> str:
    keys = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "gripper"]
    return " ".join(f"{key}={action[key]:+.3f}" for key in keys)


def main():
    args = parse_args()
    controller = SixAxisArmController_101(port=args.port, calibration=args.calibration)

    vis = None
    if not args.headless:
        try:
            _, vis = build_visualizer(args.urdf_path, args.mesh_dir)
            print("Virtual Piper viewer running. Open the viser URL shown above in your browser.")
        except Exception as exc:
            print(f"Falling back to headless mode: {exc}")

    controller.connect()
    period = 1.0 / max(args.hz, 1e-3)
    last_print = 0.0

    try:
        while True:
            action = controller.get_action()
            events = controller.consume_control_events()
            if events.get("exit_early", False):
                print("Virtual teleop stopped by 'q'.")
                break
            if vis is not None:
                vis.update_cfg(action_to_vis_cfg(action))
            now = time.perf_counter()
            if vis is None and now - last_print >= 0.2:
                print(format_action(action))
                last_print = now
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nVirtual teleop stopped.")
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
