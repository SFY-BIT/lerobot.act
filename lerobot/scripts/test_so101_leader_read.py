import argparse
import time

from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.teleop.gamepad import (
    DEFAULT_SO101_LEADER_PORT,
    SO101_MOTOR_NAMES,
    _load_so101_calibration,
)


def build_bus(port: str) -> FeetechMotorsBus:
    config = FeetechMotorsBusConfig(
        port=port,
        motors={
            "shoulder_pan": (1, "sts3215"),
            "shoulder_lift": (2, "sts3215"),
            "elbow_flex": (3, "sts3215"),
            "wrist_flex": (4, "sts3215"),
            "wrist_roll": (5, "sts3215"),
            "gripper": (6, "sts3215"),
        },
    )
    return FeetechMotorsBus(config)


def main():
    parser = argparse.ArgumentParser(description="Continuously read SO101 leader positions and report communication stability.")
    parser.add_argument("--port", default=DEFAULT_SO101_LEADER_PORT, help="Leader serial port.")
    parser.add_argument("--calibration", default=None, help="Calibration json path. Defaults to gamepad lookup.")
    parser.add_argument("--iterations", type=int, default=500, help="Number of read attempts.")
    parser.add_argument("--interval", type=float, default=0.02, help="Sleep time between reads in seconds.")
    parser.add_argument(
        "--motor",
        choices=SO101_MOTOR_NAMES,
        default=None,
        help="Read only one motor instead of the full sync-read group.",
    )
    args = parser.parse_args()

    calibration, calibration_path = _load_so101_calibration(args.calibration)
    if calibration is None:
        raise FileNotFoundError("Could not find SO101 calibration file.")

    bus = build_bus(args.port)
    print(f"[leader-test] port={args.port}")
    print(f"[leader-test] calibration={calibration_path}")
    print(f"[leader-test] iterations={args.iterations} interval={args.interval:.3f}s")
    print(f"[leader-test] motor={args.motor or 'ALL'}")

    ok_count = 0
    err_count = 0
    first_error = None
    start_t = time.perf_counter()

    try:
        bus.connect()
        bus.set_calibration(calibration)
        print("[leader-test] connected")

        for i in range(1, args.iterations + 1):
            t0 = time.perf_counter()
            try:
                if args.motor is None:
                    values = bus.read("Present_Position")
                    values = values.tolist() if hasattr(values, "tolist") else list(values)
                    state_items = list(zip(SO101_MOTOR_NAMES, values, strict=True))
                else:
                    value = bus.read("Present_Position", args.motor)
                    if hasattr(value, "item"):
                        value = value.item()
                    state_items = [(args.motor, float(value))]

                ok_count += 1
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if i == 1 or i % 20 == 0:
                    state_fmt = " ".join(f"{name}={value:+.1f}" for name, value in state_items)
                    print(f"[leader-test] ok #{i} dt={dt_ms:.2f}ms {state_fmt}")
            except Exception as exc:
                err_count += 1
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if first_error is None:
                    first_error = repr(exc)
                print(f"[leader-test] err #{i} dt={dt_ms:.2f}ms {type(exc).__name__}: {exc}")

            time.sleep(args.interval)
    finally:
        elapsed = time.perf_counter() - start_t
        try:
            if bus.is_connected:
                bus.disconnect()
        except Exception:
            pass

        print("[leader-test] summary")
        print(f"[leader-test] elapsed_s={elapsed:.2f}")
        print(f"[leader-test] ok={ok_count}")
        print(f"[leader-test] err={err_count}")
        print(f"[leader-test] success_rate={(ok_count / max(1, ok_count + err_count)) * 100:.1f}%")
        if first_error is not None:
            print(f"[leader-test] first_error={first_error}")


if __name__ == "__main__":
    main()
