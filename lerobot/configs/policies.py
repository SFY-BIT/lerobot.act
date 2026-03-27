import abc
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, TypeVar

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError

from lerobot.common.optim.optimizers import OptimizerConfig
from lerobot.common.optim.schedulers import LRSchedulerConfig
from lerobot.common.utils.hub import HubMixin
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

# Generic variable that is either PreTrainedConfig or a subclass thereof
T = TypeVar("T", bound="PreTrainedConfig")


def _looks_like_local_path(path_str: str) -> bool:
    return (
        os.path.sep in path_str
        or (os.path.altsep is not None and os.path.altsep in path_str)
        or path_str.startswith(".")
        or path_str.startswith("~")
    )


def _load_config_dict(config_file: str) -> dict | None:
    try:
        with open(config_file) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


@dataclass
class PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):
    """
    Base configuration class for policy models.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        input_shapes: A dictionary defining the shapes of the input data for the policy.
        output_shapes: A dictionary defining the shapes of the output data for the policy.
        input_normalization_modes: A dictionary with key representing the modality and the value specifies the
            normalization mode to apply.
        output_normalization_modes: Similar dictionary as `input_normalization_modes`, but to unnormalize to
            the original scale.
    """

    n_obs_steps: int = 1
    normalization_mapping: dict[str, NormalizationMode] = field(default_factory=dict)

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    def __post_init__(self):
        self.pretrained_path = None

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractproperty
    def observation_delta_indices(self) -> list | None:
        raise NotImplementedError

    @abc.abstractproperty
    def action_delta_indices(self) -> list | None:
        raise NotImplementedError

    @abc.abstractproperty
    def reward_delta_indices(self) -> list | None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError

    @abc.abstractmethod
    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
        raise NotImplementedError

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        for _, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION:
                return ft
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: Type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs,
    ) -> T:
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        local_path = Path(model_id).expanduser()
        if local_path.is_dir():
            if CONFIG_NAME in os.listdir(local_path):
                config_file = os.path.join(local_path, CONFIG_NAME)
            else:
                print(f"{CONFIG_NAME} not found in {local_path.resolve()}")
        elif local_path.is_file():
            config_file = str(local_path)
        elif _looks_like_local_path(model_id):
            raise FileNotFoundError(f"Local pretrained path does not exist: {local_path.resolve()}")
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        # HACK: this is very ugly, ideally we'd like to be able to do that natively with draccus
        # something like --policy.path (in addition to --policy.type)
        cli_overrides = policy_kwargs.pop("cli_overrides", [])
        config_data = _load_config_dict(config_file) if config_file else None
        if isinstance(config_data, dict) and "type" not in config_data:
            # Ensure policy subclasses are registered before we try legacy fallback parsing.
            import lerobot.common.policies  # noqa: F401

            parse_errors = []
            for choice_name in cls.get_known_choices():
                subcls = cls.get_choice_class(choice_name)
                try:
                    return draccus.parse(subcls, config_file, args=cli_overrides)
                except Exception as exc:  # noqa: BLE001
                    parse_errors.append(f"{choice_name}: {exc}")

            raise ValueError(
                f"Could not infer policy type from config without a 'type' field: {config_file}. "
                f"Tried choices: {', '.join(parse_errors)}"
            )

        return draccus.parse(cls, config_file, args=cli_overrides)
