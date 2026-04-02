#!/usr/bin/env python

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import zmq

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.utils import cast_stats_to_numpy, load_json, load_jsonlines
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy


def _is_pretrained_policy_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file() and (path / "model.safetensors").is_file()


def resolve_policy_path(policy_path: str) -> str:
    requested_path = Path(policy_path).expanduser()
    if requested_path.exists():
        return str(requested_path)

    # Keep relative inputs untouched so Hugging Face repo ids like "org/model" still work.
    if not requested_path.is_absolute():
        return policy_path

    candidate_paths: list[Path] = []
    basename = requested_path.name

    if len(requested_path.parts) >= 4 and requested_path.parts[1] == "home":
        relative_from_home = Path(*requested_path.parts[3:])
        candidate_paths.append(Path.home() / relative_from_home)

    search_roots = [
        Path.cwd(),
        Path.cwd().parent,
        Path.home(),
        Path("/mnt/hdd") / Path.home().name,
    ]
    for root in search_roots:
        candidate_paths.append(root / basename)
        candidate_paths.append(root / "models" / basename)

    seen: set[Path] = set()
    for candidate in candidate_paths:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_pretrained_policy_dir(candidate):
            logging.warning(
                "Local policy path %s does not exist. Falling back to detected local model directory %s.",
                requested_path,
                candidate,
            )
            return str(candidate)

    tried = ", ".join(str(path) for path in seen)
    raise FileNotFoundError(
        f"Local pretrained path does not exist: {requested_path.resolve()}. "
        f"Tried fallback locations: {tried}"
    )


def _load_stats_json(stats_file: Path) -> dict[str, dict]:
    return cast_stats_to_numpy(load_json(stats_file))


def _load_episodes_stats_jsonl(episodes_stats_file: Path) -> dict[str, dict]:
    episodes_stats = load_jsonlines(episodes_stats_file)
    aggregated = aggregate_stats([cast_stats_to_numpy(item["stats"]) for item in episodes_stats])
    return aggregated


def load_dataset_stats(stats_path: str | None) -> dict[str, dict] | None:
    if not stats_path:
        return None

    requested_path = Path(stats_path).expanduser()
    if not requested_path.exists():
        raise FileNotFoundError(f"Stats path does not exist: {requested_path.resolve()}")

    if requested_path.is_dir():
        candidates = [
            requested_path / "meta" / "episodes_stats.jsonl",
            requested_path / "meta" / "stats.json",
            requested_path / "episodes_stats.jsonl",
            requested_path / "stats.json",
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            if candidate.name == "episodes_stats.jsonl":
                stats = _load_episodes_stats_jsonl(candidate)
            else:
                stats = _load_stats_json(candidate)
            logging.info("Loaded dataset stats from %s", candidate)
            return stats

        raise FileNotFoundError(
            "Could not find dataset stats under the provided directory. "
            f"Tried: {', '.join(str(path) for path in candidates)}"
        )

    if requested_path.name == "episodes_stats.jsonl":
        stats = _load_episodes_stats_jsonl(requested_path)
        logging.info("Loaded aggregated dataset stats from %s", requested_path)
        return stats

    if requested_path.name == "stats.json":
        stats = _load_stats_json(requested_path)
        logging.info("Loaded dataset stats from %s", requested_path)
        return stats

    raise ValueError(
        "Unsupported stats path. Provide a dataset root directory, `meta/stats.json`, "
        "or `meta/episodes_stats.jsonl`."
    )


def build_batch(payload: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "observation.state": torch.tensor(payload["state"], dtype=torch.float32, device=device).unsqueeze(0),
        "task": [payload["task"]],
    }

    for key, image in payload["images"].items():
        tensor = torch.from_numpy(image).to(device=device, dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1).contiguous() / 255.0
        batch[key] = tensor.unsqueeze(0)

    return batch


def _find_missing_normalization_entries(policy: PI0Policy) -> list[str]:
    missing_entries: list[str] = []

    for module_name in ("normalize_inputs", "normalize_targets", "unnormalize_outputs"):
        module = getattr(policy, module_name, None)
        if module is None:
            continue

        for name, parameter in module.named_parameters(recurse=True):
            if torch.isinf(parameter).any():
                missing_entries.append(f"{module_name}.{name}")

    return missing_entries


def validate_policy_normalization(policy: PI0Policy, policy_path: str) -> None:
    missing_entries = _find_missing_normalization_entries(policy)
    if not missing_entries:
        return

    joined = "\n".join(f"- {name}" for name in missing_entries)
    raise ValueError(
        "The loaded policy is missing normalization statistics required for LeRobot inference.\n"
        f"Policy path: {policy_path}\n"
        "The following normalization entries are still uninitialized (infinity):\n"
        f"{joined}\n"
        "This usually means the exported model.safetensors does not include normalization buffers."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a remote pi0 inference server.")
    parser.add_argument("--policy-path", required=True, help="Local path to the pi0 pretrained_model directory.")
    parser.add_argument(
        "--stats-path",
        default=None,
        help="Optional dataset stats source: dataset root, meta/stats.json, or meta/episodes_stats.jsonl.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device to run inference on.")
    parser.add_argument("--bind", default="tcp://127.0.0.1:5555", help="ZeroMQ bind address.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device(args.device)
    policy_path = resolve_policy_path(args.policy_path)
    dataset_stats = load_dataset_stats(args.stats_path)
    logging.info("Loading pi0 policy from %s on %s", policy_path, device)
    if dataset_stats is not None:
        logging.info("Injecting external dataset stats into PI0Policy during load.")
    policy = PI0Policy.from_pretrained(policy_path, map_location=str(device), dataset_stats=dataset_stats)
    policy.to(device)
    policy.eval()
    policy.reset()
    validate_policy_normalization(policy, policy_path)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(args.bind)
    logging.info("Listening on %s", args.bind)

    try:
        while True:
            message = socket.recv_pyobj()
            msg_type = message.get("type")

            if msg_type == "reset":
                policy.reset()
                socket.send_pyobj({"status": "ok"})
                continue

            if msg_type != "infer":
                socket.send_pyobj({"status": "error", "error": f"Unsupported message type: {msg_type}"})
                continue

            try:
                batch = build_batch(message, device)
                with torch.inference_mode():
                    action = policy.select_action(batch)
                socket.send_pyobj({"status": "ok", "action": action.squeeze(0).detach().cpu().tolist()})
            except Exception as exc:  # noqa: BLE001
                logging.exception("Inference failed")
                socket.send_pyobj({"status": "error", "error": repr(exc)})
    finally:
        socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
