"""Interface for a Policy interacting in CRISP."""

import json
import logging
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import LeRobotDatasetMetadata, get_policy_class
from typing_extensions import override

from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv
from crisp_gym.policy.policy import Action, Observation, Policy, register_policy
from crisp_gym.util.lerobot_features import concatenate_state_features, numpy_obs_to_torch
from crisp_gym.util.setup_logger import setup_logging

try:
    from lerobot.policies.factory import make_pre_post_processors
    USE_LEROBOT_PROCESSORS = True
    logging.info("Found lerobot pre/post processor support.")
except ImportError:
    USE_LEROBOT_PROCESSORS = False
    logging.warning("No lerobot pre/post processor support found.")


logger = logging.getLogger(__name__)


def _ensure_task_ids_in_batch(
    batch: dict[str, Any], task_id: int | None, device: torch.device
) -> dict[str, Any]:
    """Ensure observation.task_ids exists in a policy input batch.

    Some policy preprocessor pipelines may drop unknown keys. For policies using
    task one-hot conditioning, we enforce this key right before inference.
    """
    if task_id is None:
        return batch
    if "observation.task_ids" in batch:
        return batch

    batch = dict(batch)
    batch["observation.task_ids"] = torch.tensor([[int(task_id)]], device=device, dtype=torch.long)
    return batch


def _get_policy_type_from_train_config_json(pretrained_path: str) -> str:
    """Read policy type directly from train_config.json.

    This fallback is used when TrainPipelineConfig.from_pretrained() fails due to
    schema mismatches between checkpoint and installed LeRobot version.
    """
    config_path = Path(pretrained_path) / "train_config.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)

    policy_cfg = cfg.get("policy", {})
    policy_type = policy_cfg.get("type")
    if not policy_type:
        raise ValueError(
            f"Could not determine policy type from {config_path}. Expected key policy.type."
        )
    return policy_type


@register_policy("lerobot_policy")
class LerobotPolicy(Policy):
    """A policy implementation that wraps a LeRobot policy for use in CRISP environments.

    This class runs LeRobot policy inference in a separate process and communicates with the
    environment to generate actions based on observations. It is intended for direct use in
    CRISP-based manipulation environments.
    """

    def __init__(
        self,
        pretrained_path: str,
        env: ManipulatorBaseEnv,
        overrides: dict | None = None,
        task_id: int | None = None,
        subfolder: str | None = None,
    ):
        """Initialize the policy.

        Args:
            pretrained_path (str): Path to the pretrained policy model.
            env (ManipulatorBaseEnv): The environment in which the policy will be applied.
            overrides (dict | None): Optional overrides for the policy configuration.
            subfolder (str | None): Optional subfolder inside a HF Hub repo to load from.
                When set and pretrained_path is not a local directory, the subfolder is
                resolved to a local path via huggingface_hub.snapshot_download.
        """
        if subfolder is not None and not Path(pretrained_path).exists():
            from huggingface_hub import snapshot_download

            logger.info(
                f"Resolving HF subfolder '{subfolder}' of '{pretrained_path}' to local path."
            )
            local_root = snapshot_download(
                repo_id=pretrained_path,
                allow_patterns=[f"{subfolder}/*"],
            )
            pretrained_path = str(Path(local_root) / subfolder)
            logger.info(f"Using resolved pretrained_path: {pretrained_path}")

        self.parent_conn, self.child_conn = Pipe()
        self.env = env
        self.overrides = overrides if overrides is not None else {}
        self.task_id = task_id

        # Capture picklable snapshots for the worker (spawn start method requires pickling).
        warmup_obs_sample = env.observation_space.sample()
        env_metadata = env.get_metadata()

        self.inf_proc = Process(
            target=inference_worker,
            kwargs={
                "conn": self.child_conn,
                "pretrained_path": pretrained_path,
                "warmup_obs_sample": warmup_obs_sample,
                "env_metadata": env_metadata,
                "overrides": self.overrides,
                "task_id": self.task_id,
            },
            daemon=True,
        )
        self.inf_proc.start()

    @override
    def make_data_fn(self) -> Callable[[], Tuple[Observation, Action]]:  # noqa: ANN002, ANN003
        """Generate observation and action by communicating with the inference worker."""

        def _fn() -> tuple:
            """Function to apply the policy in the environment.

            This function observes the current state of the environment, sends the observation
            to the inference worker, receives the action, and steps the environment.

            Returns:
                tuple: A tuple containing the observation from the environment and the action taken.
            """
            logger.debug("Requesting action from policy...")
            obs_raw: Observation = self.env.get_obs()

            obs_raw["observation.state"] = concatenate_state_features(obs_raw)
            if self.task_id is not None:
                obs_raw["observation.task_ids"] = np.array([int(self.task_id)], dtype=np.int64)

            self.parent_conn.send(obs_raw)
            action: Action = self.parent_conn.recv().squeeze(0).to("cpu").numpy()
            logger.debug(f"Action: {action}")

            try:
                self.env.step(action, block=False)
            except Exception as e:
                logger.exception(f"Error during environment step: {e}")

            return obs_raw, action

        return _fn

    @override
    def reset(self):
        """Reset the policy state."""
        self.parent_conn.send("reset")

    @override
    def shutdown(self):
        """Shutdown the policy and release resources."""
        self.parent_conn.send(None)
        self.inf_proc.join()


def inference_worker(
    conn: Connection,
    pretrained_path: str,
    warmup_obs_sample: dict,
    env_metadata: dict,
    overrides: dict | None = None,
    task_id: int | None = None,
):  # noqa: ANN001
    """Policy inference process: loads policy on GPU, receives observations via conn, returns actions, and exits on None.

    Args:
        conn (Connection): The connection to the parent process for sending and receiving data.
        pretrained_path (str): Path to the pretrained policy model.
        warmup_obs_sample (dict): Picklable sample from env.observation_space used for warmup.
        env_metadata (dict): Picklable snapshot of env.get_metadata() for dataset metadata checks.
        overrides (dict | None): Optional overrides for the policy configuration.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        from lerobot.utils.import_utils import register_third_party_plugins

        register_third_party_plugins()
    except ImportError:
        logger.warning(
            "[Inference] Could not import third-party plugins for LeRobot. Continuing without them."
        )
    logger.info("[Inference] Starting inference worker...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Inference] Using device: {device}")

        logger.info(f"[Inference] Loading training config from {pretrained_path}...")

        train_config: TrainPipelineConfig | None = None
        policy_type: str | None = None
        try:
            train_config = TrainPipelineConfig.from_pretrained(pretrained_path)

            _check_dataset_metadata(train_config, env_metadata, logger)

            logger.info("[Inference] Loaded training config.")
            logger.debug(f"[Inference] Train config: {train_config}")

            if train_config.policy is None:
                raise ValueError(
                    f"Policy configuration is missing in the pretrained path: {pretrained_path}. "
                    "Please ensure the policy is correctly configured."
                )
            policy_type = train_config.policy.type
        except Exception as e:
            logger.warning(
                "[Inference] Failed to parse full TrainPipelineConfig (%s). "
                "Falling back to reading policy.type from train_config.json.",
                e,
            )
            policy_type = _get_policy_type_from_train_config_json(pretrained_path)
            logger.warning(
                "[Inference] Skipping dataset metadata compatibility check due to config schema mismatch."
            )

        logger.info("[Inference] Loading policy...")
        policy_cls = get_policy_class(policy_type)
        policy = policy_cls.from_pretrained(pretrained_path)

        for override_key, override_value in (overrides or {}).items():
            logger.warning(
                f"[Inference] Overriding policy config: {override_key} = {getattr(policy.config, override_key)} -> {override_value}"
            )
            setattr(policy.config, override_key, override_value)

        logger.info(
            f"[Inference] Loaded {policy.name} policy with {pretrained_path} on device {device}."
        )
        policy.reset()
        policy.to(device).eval()

        if USE_LEROBOT_PROCESSORS:
            preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=pretrained_path)

        warmup_obs_raw = dict(warmup_obs_sample)
        warmup_obs_raw["observation.state"] = concatenate_state_features(warmup_obs_raw)
        if task_id is not None:
            warmup_obs_raw["observation.task_ids"] = np.array([int(task_id)], dtype=np.int64)
        warmup_obs = numpy_obs_to_torch(warmup_obs_raw)
        if USE_LEROBOT_PROCESSORS:
            warmup_obs = preprocessor(warmup_obs)
        warmup_obs = _ensure_task_ids_in_batch(warmup_obs, task_id, device)

        logger.info("[Inference] Warming up policy...")
        elapsed_list = []
        with torch.inference_mode():
            import time

            for _ in range(100):
                start = time.time()
                _ = policy.select_action(warmup_obs)
                end = time.time()
                elapsed = end - start
                elapsed_list.append(elapsed)

            torch.cuda.synchronize()

        avg_elapsed = sum(elapsed_list) / len(elapsed_list)
        std_elapsed = np.std(elapsed_list)
        max_elapsed = max(elapsed_list)
        min_elapsed = min(elapsed_list)
        logger.info(
            f"[Inference] Warm-up timing over 100 runs: "
            f"avg={avg_elapsed * 1000:.2f}ms, std={std_elapsed * 1000:.2f}ms, max={max_elapsed * 1000:.2f}ms, min={min_elapsed * 1000:.2f}ms"
        )

        logger.info("[Inference] Warm-up complete")

        while True:
            obs_raw = conn.recv()
            if obs_raw is None:
                break
            if obs_raw == "reset":
                logger.info("[Inference] Resetting policy")
                policy.reset()
                if USE_LEROBOT_PROCESSORS:
                    preprocessor.reset()
                    postprocessor.reset()
                continue

            with torch.inference_mode():
                if task_id is not None and "observation.task_ids" not in obs_raw:
                    obs_raw["observation.task_ids"] = np.array([int(task_id)], dtype=np.int64)
                obs = numpy_obs_to_torch(obs_raw)
                logger.info("[Inference] Starting inference...")
                t0 = time.perf_counter()
                if USE_LEROBOT_PROCESSORS:
                    obs = preprocessor(obs)
                obs = _ensure_task_ids_in_batch(obs, task_id, device)
                action = policy.select_action(obs)
                if USE_LEROBOT_PROCESSORS:
                    action = postprocessor(action)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                logger.info(f"[Inference] Done in {elapsed_ms:.2f}ms")

            logger.debug(f"[Inference] Computed action: {action}")
            conn.send(action)
    except Exception as e:
        logger.exception(f"[Inference] Exception in inference worker: {e}")

    conn.close()
    logger.info("[Inference] Worker shutting down")


def _check_dataset_metadata(
    train_config: TrainPipelineConfig,
    env_metadata: dict,
    logger: logging.Logger,
    keys_to_skip: list[str] | None = None,
):
    """Check if the dataset metadata matches the environment configuration.

    Args:
        train_config (TrainPipelineConfig): The training pipeline configuration.
        env_metadata (dict): Snapshot of env.get_metadata() captured in the parent process.
        logger (logging.Logger): Logger for logging information.
        keys_to_skip (list[str] | None): List of metadata keys to skip during comparison.
    """
    if keys_to_skip is None:
        keys_to_skip = []

    def _warn_if_not_equal(key: str, env_val: Any, policy_val: Any):
        if env_val != policy_val:
            logger.warning(
                f"[Inference] Mismatch in metadata for key '{key}': "
                f"env has '{env_val}', policy has '{policy_val}'."
            )

    def _warn_if_missing(key: str):
        logger.warning(f"[Inference] Key '{key}' not found in environment metadata.")

    try:
        metadata = LeRobotDatasetMetadata(repo_id=train_config.dataset.repo_id)
        logger.debug(f"[Inference] Loaded dataset metadata: {metadata}")

        path_to_metadata = Path(metadata.root / "meta" / "crisp_meta.json")
        if path_to_metadata.exists():
            logger.info(
                "[Inference] Found crisp_meta.json in dataset, comparing environment and policy configs..."
            )
            with open(path_to_metadata, "r") as f:
                dataset_metadata = json.load(f)
            for key, value in dataset_metadata.items():
                if key in keys_to_skip:
                    continue
                if isinstance(value, dict):
                    if key not in env_metadata:
                        _warn_if_missing(key)
                        continue
                    for subkey, subvalue in value.items():
                        if subkey not in env_metadata[key]:
                            _warn_if_missing(f"{key}.{subkey}")
                            continue
                        _warn_if_not_equal(
                            f"{key}.{subkey}",
                            env_metadata[key].get(subkey),
                            subvalue,
                        )
                else:
                    if key not in env_metadata:
                        _warn_if_missing(key)
                    _warn_if_not_equal(key, env_metadata.get(key), value)

    except Exception as e:
        logger.warning(f"[Inference] Could not load dataset metadata: {e}")
        logger.info("[Inference] Skipping metadata comparison.")
