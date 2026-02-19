"""Record camera-only datasets in LeRobot format without requiring robot connectivity.

Sample command to start recording:
    pixi run -e humble-lerobot crisp-record-cameras-only \
        --repo-id <your_account>/<repo_name> \
        --camera-config default_camera_recording

Episode controls (keyboard manager):
    r: start/stop recording
    s: save episode
    d: delete episode
    q: quit

Sucess labeling:
With --no-label-success: each saved episode gets success_score=1.0 automatically (implicit logic).
With --label-success: after each saved episode, you are prompted for a score in [0.0, 1.0].

To add/remove cameras, edit a YAML under `config/camera_recording/`.
"""

import argparse
import datetime
import json
import logging
import termios
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rclpy
import yaml
from crisp_py.camera import Camera
from crisp_py.camera.camera_config import CameraConfig

from crisp_gym.config.path import find_config, list_configs_in_folder
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.record.recording_manager_config import RecordingManagerConfig
from crisp_gym.util import prompt
from crisp_gym.util.setup_logger import setup_logging

try:
    import cv2
except ImportError:
    cv2 = None


logger = logging.getLogger(__name__)


DEFAULT_PLACEHOLDER_ACTION_DIM = 7
DEFAULT_PLACEHOLDER_STATE_DIM = 13


def _list_camera_recording_configs() -> list[str]:
    """List available camera-only recording config names."""
    configs = list_configs_in_folder("camera_recording")
    return sorted([config.stem for config in configs if config.suffix == ".yaml"])


def _resolve_camera_config_path(config_name_or_path: str | None) -> Path:
    """Resolve a camera recording YAML config from path or config name."""
    if config_name_or_path is not None:
        as_path = Path(config_name_or_path)
        if as_path.exists():
            return as_path

        resolved = find_config(f"camera_recording/{config_name_or_path}.yaml")
        if resolved is not None:
            return resolved

        raise FileNotFoundError(
            f"Could not find camera recording config '{config_name_or_path}'. "
            "Provide a valid path or one of the available config names."
        )

    available = _list_camera_recording_configs()
    if not available:
        raise FileNotFoundError(
            "No camera recording config found. Please create one in 'config/camera_recording/'."
        )

    chosen = prompt.prompt(
        "Please select the camera recording configuration name.",
        options=available,
        default=available[0],
    )
    resolved = find_config(f"camera_recording/{chosen}.yaml")
    if resolved is None:
        raise FileNotFoundError(f"Could not resolve selected config '{chosen}'.")
    return resolved


def _coerce_camera_source(source: Any) -> int | str:
    """Convert numeric camera sources to int and keep paths/URLs as strings."""
    if isinstance(source, int):
        return source

    if isinstance(source, str):
        stripped = source.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped

    raise ValueError(f"Unsupported camera source type: {type(source)}")


def _ask_enable_success_labeling(cli_value: bool | None) -> bool:
    """Resolve whether to enable manual per-episode success score prompts."""
    if cli_value is not None:
        return cli_value

    response = prompt.prompt(
        "Do you want to manually label saved episodes with a success score?",
        options=["yes", "no"],
        default="yes",
    )
    return response == "yes"


def _prompt_success_score() -> float:
    """Ask for a success score and keep asking until we get a valid value.

    This is intentionally strict: only floats in [0.0, 1.0] are accepted.
    """
    _drain_stdin()

    while True:
        response = prompt.prompt(
            "Enter success score for this saved episode (0.0 to 1.0)",
            default="1.0",
        )

        candidate = _sanitize_score_candidate(response)
        try:
            score = float(candidate)
        except ValueError:
            logger.warning("Invalid number. Please enter a float between 0.0 and 1.0.")
            continue

        if 0.0 <= score <= 1.0:
            return score

        logger.warning("Out of range. Success score must be between 0.0 and 1.0.")


def _sanitize_score_candidate(response: str) -> str:
    """Clean score text in case control keys leak into stdin.

    The recording controls use keys like `r`, `s`, `d`, `q`. Depending on terminal
    timing, those keypresses can leak into stdin and become part of the next prompt input.
    We keep intended numeric input (e.g. "1", "0.0") and strip those leaked controls.
    """
    stripped = response.strip()
    lowered = stripped.lower()

    cleaned = "".join(ch for ch in lowered if ch not in {"r", "s", "d", "q"})
    if cleaned:
        return cleaned
    return lowered


def _drain_stdin() -> None:
    """Flush pending keyboard input before score prompt.

    This avoids the common case where a previous hotkey press is consumed as score input.
    """
    try:
        termios.tcflush(0, termios.TCIFLUSH)
    except Exception:
        # Some environments do not expose a flushable stdin (e.g. non-interactive runs).
        # In that case we simply continue; input validation still guards correctness.
        return


def _append_success_label(
    labels_file: Path,
    saved_order_index: int,
    episode_index_estimate: int,
    episode_count_after_save: int,
    task: str,
    success_score: float,
) -> None:
    """Append one episode-level success label to parquet metadata.

    We keep this as a sidecar file under `meta/` so it stays easy to query
    and does not interfere with LeRobot's core metadata lifecycle.
    """
    labels_file.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "saved_order_index": int(saved_order_index),
        "episode_index_estimate": int(episode_index_estimate),
        "episode_count_after_save": int(episode_count_after_save),
        "task": str(task),
        "success_score": float(success_score),
        "is_success": bool(success_score >= 0.5),
        "timestamp_utc": datetime.datetime.now(tz=datetime.UTC).isoformat(),
    }

    if labels_file.exists():
        labels_df = pd.read_parquet(labels_file)
        labels_df = pd.concat([labels_df, pd.DataFrame([row])], ignore_index=True)
    else:
        labels_df = pd.DataFrame([row])

    labels_df.to_parquet(labels_file, index=False)


def _count_existing_labels(labels_file: Path) -> int:
    """Return how many success labels already exist in parquet metadata."""
    if labels_file.exists():
        return int(len(pd.read_parquet(labels_file)))

    return 0


class CameraRig:
    """Multi-camera capture rig supporting ROS-topic and OpenCV backends."""

    def __init__(
        self,
        config: dict[str, Any],
        fps: int,
        backend: str = "auto",
        namespace: str = "",
    ) -> None:
        """Initialize camera rig and load cameras for the selected backend."""
        self.config = config
        self.fps = fps
        self.namespace = namespace
        self.backend = self._resolve_backend(backend)

        self.captures: list[Any] = []
        self.cameras: list[Camera] = []
        self.camera_specs: list[dict[str, Any]] = []

        if self.backend == "ros_topics":
            self._init_ros_cameras()
        elif self.backend == "opencv":
            self._init_opencv_cameras()
        else:
            raise ValueError(f"Unsupported backend '{self.backend}'.")

    def _resolve_backend(self, backend: str) -> str:
        """Resolve backend choice from explicit arg or config structure."""
        if backend != "auto":
            return backend

        if "camera_configs" in self.config:
            return "ros_topics"
        if "cameras" in self.config:
            return "opencv"

        raise ValueError(
            "Could not infer camera backend from config. "
            "Use 'camera_configs' (ROS topics) or 'cameras' (OpenCV)."
        )

    def _init_ros_cameras(self) -> None:
        """Initialize cameras from ROS topics using crisp_py Camera objects."""
        camera_configs = self.config.get("camera_configs", [])
        if not camera_configs:
            raise ValueError("ROS backend expects a non-empty 'camera_configs' list.")

        if not rclpy.ok():
            rclpy.init()

        for camera_cfg in camera_configs:
            if not isinstance(camera_cfg, dict):
                raise ValueError("Each item in 'camera_configs' must be a dictionary.")

            cfg = CameraConfig(**camera_cfg)
            camera = Camera(namespace=self.namespace, config=cfg)
            self.cameras.append(camera)

        for camera in self.cameras:
            camera.wait_until_ready(timeout=5)
            frame = camera.current_image

            if frame is None:
                raise RuntimeError(
                    f"Camera '{camera.config.camera_name}' is ready but has no current image."
                )

            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(
                    f"Camera '{camera.config.camera_name}' returned shape {frame.shape}. "
                    "Expected HxWx3 image."
                )

            self.camera_specs.append(
                {
                    "name": camera.config.camera_name,
                    "source": camera.config.camera_color_image_topic,
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "backend": "ros_topics",
                }
            )

            logger.info(
                "Camera '%s' ready from ROS topic '%s' with resolution %dx%d.",
                camera.config.camera_name,
                camera.config.camera_color_image_topic,
                frame.shape[1],
                frame.shape[0],
            )

    def _init_opencv_cameras(self) -> None:
        """Initialize cameras from OpenCV sources."""
        if cv2 is None:
            raise ImportError(
                "OpenCV is required for OpenCV backend. "
                "Install it in your environment, e.g. `pixi add opencv-python` or `pip install opencv-python`."
            )

        cameras = self.config.get("cameras", [])
        if not cameras:
            raise ValueError("OpenCV backend expects a non-empty 'cameras' list.")

        for camera in cameras:
            name = camera.get("name")
            source = _coerce_camera_source(camera.get("source"))
            width = int(camera.get("width", 640))
            height = int(camera.get("height", 480))
            convert_bgr_to_rgb = bool(camera.get("convert_bgr_to_rgb", True))

            if not name:
                raise ValueError("Each camera entry must define 'name'.")

            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera '{name}' from source '{source}'.")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, float(self.fps))

            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                raise RuntimeError(f"Camera '{name}' did not provide an initial frame.")

            frame_height, frame_width, channels = frame.shape
            if channels != 3:
                cap.release()
                raise ValueError(
                    f"Camera '{name}' returned frame shape {frame.shape}. Expected 3 channels."
                )

            self.captures.append(cap)
            self.camera_specs.append(
                {
                    "name": name,
                    "source": source,
                    "width": frame_width,
                    "height": frame_height,
                    "convert_bgr_to_rgb": convert_bgr_to_rgb,
                    "backend": "opencv",
                }
            )

            logger.info(
                f"Camera '{name}' ready from source '{source}' with resolution {frame_width}x{frame_height}."
            )

    def get_features(self) -> dict[str, dict[str, Any]]:
        """Build LeRobot features for configured cameras and human-action placeholders."""
        features: dict[str, dict[str, Any]] = {
            "observation.state.human_action": {
                "dtype": "float32",
                "shape": (DEFAULT_PLACEHOLDER_STATE_DIM,),
                "names": [
                    f"human_action_{i}" for i in range(DEFAULT_PLACEHOLDER_STATE_DIM)
                ],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (DEFAULT_PLACEHOLDER_STATE_DIM,),
                "names": [f"state_{i}" for i in range(DEFAULT_PLACEHOLDER_STATE_DIM)],
            },
            "action": {
                "dtype": "float32",
                "shape": (DEFAULT_PLACEHOLDER_ACTION_DIM,),
                "names": [f"action_{i}" for i in range(DEFAULT_PLACEHOLDER_ACTION_DIM)],
            },
        }

        for camera in self.camera_specs:
            features[f"observation.images.{camera['name']}"] = {
                "dtype": "video",
                "shape": (camera["height"], camera["width"], 3),
                "names": ["height", "width", "channels"],
                "video_info": {
                    "video.fps": self.fps,
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }

        return features

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata for this camera-only dataset recording setup."""
        return {
            "type": "camera_only",
            "backend": self.backend,
            "fps": self.fps,
            "namespace": self.namespace,
            "cameras": self.camera_specs,
        }

    def read_observation(self) -> dict[str, Any]:
        """Read one multi-camera observation."""
        obs: dict[str, Any] = {
            "observation.state.human_action": np.zeros(
                (DEFAULT_PLACEHOLDER_STATE_DIM,), dtype=np.float32
            ),
        }

        if self.backend == "ros_topics":
            for camera, ros_camera in zip(self.camera_specs, self.cameras, strict=True):
                frame = ros_camera.current_image
                if frame is None:
                    raise RuntimeError(
                        f"Failed to read current image from ROS camera '{camera['name']}'."
                    )
                obs[f"observation.images.{camera['name']}"] = frame
        else:
            if cv2 is None:
                raise ImportError(
                    "OpenCV is required for OpenCV backend. "
                    "Install it in your environment, e.g. `pixi add opencv-python` or `pip install opencv-python`."
                )
            for camera, cap in zip(self.camera_specs, self.captures, strict=True):
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(
                        f"Failed to read frame from camera '{camera['name']}' (source={camera['source']})."
                    )

                if camera["convert_bgr_to_rgb"]:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                obs[f"observation.images.{camera['name']}"] = frame

        return obs

    def close(self) -> None:
        """Release all camera captures."""
        for cap in self.captures:
            cap.release()


def main() -> None:
    """Record camera-only data in LeRobot format with interactive episode controls."""
    parser = argparse.ArgumentParser(
        description=(
            "Record camera-only LeRobot datasets without robot state/action streaming. "
            "Supports ROS-topic camera configs (same structure as env configs) and OpenCV sources."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  pixi run -e humble-lerobot crisp-record-cameras-only "
            "--repo-id user/camera_demo --camera-config left_stack_v6_like --camera-backend ros_topics\n"
            "  pixi run -e humble-lerobot crisp-record-cameras-only "
            "--repo-id user/camera_demo --camera-config /tmp/my_cameras.yaml --camera-backend opencv --fps 20\n\n"
            "Keyboard controls: r=start/stop, s=save, d=delete, q=quit"
        ),
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Repository ID for the dataset, e.g. username/my_dataset",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["Human demonstrates the task in front of the camera."],
        help="List of task descriptions to sample from per episode.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="camera_only",
        help="Robot type string stored in LeRobot metadata.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Recording FPS.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to record.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume recording of an already existing dataset.",
    )
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to push the dataset to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--recording-manager-type",
        type=str,
        default="keyboard",
        help="Type of recording manager to use. Supported: keyboard, ros.",
    )
    parser.add_argument(
        "--camera-config",
        type=str,
        default=None,
        help="Camera config name (from config/camera_recording) or full YAML path.",
    )
    parser.add_argument(
        "--camera-backend",
        type=str,
        default="auto",
        choices=["auto", "ros_topics", "opencv"],
        help="Camera backend. 'auto' infers backend from config keys.",
    )
    parser.add_argument(
        "--camera-namespace",
        type=str,
        default="left",
        help="ROS namespace prefix for camera topics when using the ROS backend (default: left).",
    )
    parser.add_argument(
        "--label-success",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable manual per-saved-episode success score prompts. "
            "If disabled, saved episodes are labeled implicitly with success_score=1.0. "
            "If omitted, the script asks once at startup."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logger level.",
    )

    args = parser.parse_args()
    setup_logging(level=args.log_level)

    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg:<30}: {value}")

    if args.repo_id is None:
        args.repo_id = prompt.prompt(
            "Please enter the repository ID for the dataset (e.g., 'username/dataset_name'):"
        )

    repo_id = str(args.repo_id)
    robot_type = str(args.robot_type)
    num_episodes = int(args.num_episodes)
    fps = int(args.fps)
    resume = bool(args.resume)
    push_to_hub = bool(args.push_to_hub)
    manual_success_labeling = _ask_enable_success_labeling(args.label_success)

    camera_config_path = _resolve_camera_config_path(args.camera_config)
    logger.info(f"Using camera config: {camera_config_path}")

    with open(camera_config_path, "r") as file:
        camera_config = yaml.safe_load(file) or {}

    if (args.recording_manager_type == "ros" or args.camera_backend == "ros_topics") and not rclpy.ok():
        rclpy.init()

    camera_rig = None
    try:
        camera_rig = CameraRig(
            config=camera_config,
            fps=fps,
            backend=args.camera_backend,
            namespace=args.camera_namespace,
        )
        features = camera_rig.get_features()
        logger.debug(f"Using the features: {features}")

        recording_config = RecordingManagerConfig(
            features=features,
            repo_id=repo_id,
            robot_type=robot_type,
            num_episodes=num_episodes,
            fps=fps,
            resume=resume,
            push_to_hub=push_to_hub,
        )

        recording_manager = make_recording_manager(
            recording_manager_type=args.recording_manager_type,
            config=recording_config,
        )
        recording_manager.wait_until_ready()
        logger.info("Recording manager is ready.")

        meta_dir = recording_manager.dataset_directory / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / "crisp_meta.json", "w") as file:
            json.dump(camera_rig.get_metadata(), file, indent=4)

        labels_file = meta_dir / "episode_success.parquet"
        saved_order_index = _count_existing_labels(labels_file)
        if manual_success_labeling:
            logger.info(
                "Manual success labeling enabled. Scores will be written to %s",
                labels_file,
            )
        else:
            logger.info(
                "Manual success labeling disabled. Saved episodes will be labeled implicitly with success_score=1.0 in %s",
                labels_file,
            )

        tasks = list(args.tasks)

        # Keep action shape tied to dataset features so placeholder actions always match schema.
        action_shape = tuple(features.get("action", {}).get("shape", (1,)))

        def data_fn() -> tuple[dict[str, Any], np.ndarray]:
            """Read one observation and return a schema-aligned placeholder action."""
            obs = camera_rig.read_observation()
            action = np.zeros(action_shape, dtype=np.float32)
            return obs, action

        with recording_manager:
            while not recording_manager.done():
                logger.info(
                    f"→ Episode {recording_manager.episode_count + 1} / {recording_manager.num_episodes}"
                )
                task = tasks[np.random.randint(0, len(tasks))] if tasks else "No task specified."
                logger.info(f"▷ Task: {task}")

                previous_episode_count = recording_manager.episode_count

                recording_manager.record_episode(
                    data_fn=data_fn,
                    task=task,
                )

                # We only write a label if the episode actually got saved.
                # This keeps labels aligned with real dataset episodes.
                was_saved = recording_manager.episode_count > previous_episode_count
                if was_saved:
                    # Manual mode asks once per saved episode; auto mode writes 1.0 directly.
                    success_score = (
                        _prompt_success_score() if manual_success_labeling else 1.0
                    )
                    episode_count_after_save = recording_manager.episode_count
                    episode_index_estimate = episode_count_after_save - 1
                    _append_success_label(
                        labels_file=labels_file,
                        saved_order_index=saved_order_index,
                        episode_index_estimate=episode_index_estimate,
                        episode_count_after_save=episode_count_after_save,
                        task=task,
                        success_score=success_score,
                    )
                    logger.info(
                        "Saved success score %.3f for saved_order_index=%d (episode_index_estimate=%d).",
                        success_score,
                        saved_order_index,
                        episode_index_estimate,
                    )
                    saved_order_index += 1

        logger.info("Finished recording.")

    except Exception as e:
        logger.exception(f"An error occurred during recording: {e}.")

    finally:
        if camera_rig is not None:
            logger.info("Releasing cameras.")
            camera_rig.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()