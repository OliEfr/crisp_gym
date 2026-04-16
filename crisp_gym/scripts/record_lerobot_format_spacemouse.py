"""Script showcasing how to record data in Lerobot Format with a SpaceMouse."""

import argparse
import json
import logging

import numpy as np
import rclpy

import crisp_gym  # noqa: F401
from crisp_gym.config.home import HomeConfig
from crisp_gym.envs.manipulator_env import make_env
from crisp_gym.envs.manipulator_env_config import list_env_configs
from crisp_gym.record.record_functions import make_teleop_spacemouse_fn
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.teleop.teleop_spacemouse import SpaceMouseTeleop
from crisp_gym.util import prompt
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.setup_logger import setup_logging


def _parse_signs(flag_value: str, flag_name: str) -> tuple[int, int, int]:
    """Parse a 3-char string of '+' / '-' into a sign tuple."""
    if len(flag_value) != 3 or any(c not in "+-" for c in flag_value):
        raise ValueError(f"{flag_name} must be a 3-char string of '+' or '-', got {flag_value!r}")
    return tuple(1 if c == "+" else -1 for c in flag_value)  # type: ignore[return-value]


def main():
    """Record data in Lerobot Format using a SpaceMouse teleoperation setup."""
    parser = argparse.ArgumentParser(description="Record data in Lerobot Format with a SpaceMouse")
    parser.add_argument("--repo-id", type=str, default="test")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["pick the lego block."],
    )
    parser.add_argument("--robot-type", type=str, default="franka")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "--push-to-hub", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--recording-manager-type", type=str, default="keyboard")
    parser.add_argument("--follower-config", type=str, default=None)
    parser.add_argument("--follower-namespace", type=str, default=None)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("--home-config-noise", type=float, default=0.0)

    parser.add_argument(
        "--spacemouse-trans-scale",
        type=float,
        default=0.2,
        help="Meters per second at full SpaceMouse deflection (|axis|=1.0).",
    )
    parser.add_argument(
        "--spacemouse-rot-scale",
        type=float,
        default=0.5,
        help="Radians per second at full SpaceMouse deflection (|axis|=1.0).",
    )
    parser.add_argument(
        "--spacemouse-gripper-rate",
        type=float,
        default=2.0,
        help="Gripper units per second while a button is held. 2.0 = full sweep in 0.5s.",
    )
    parser.add_argument("--spacemouse-deadzone", type=float, default=0.05)
    parser.add_argument(
        "--spacemouse-trans-signs",
        type=str,
        default="+-+",
        help="Sign flips for SpaceMouse x/y/z (3-char string of '+' or '-').",
    )
    parser.add_argument(
        "--spacemouse-rot-signs",
        type=str,
        default="+++",
        help="Sign flips for SpaceMouse roll/pitch/yaw (3-char string of '+' or '-').",
    )

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    setup_logging(level=args.log_level)

    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg:<30}: {value}")

    if args.follower_namespace is None:
        args.follower_namespace = prompt.prompt(
            "Please enter the follower robot namespace (e.g., 'left', 'right', ...)",
            default="right",
        )
        logger.info(f"Using follower namespace: {args.follower_namespace}")

    if args.follower_config is None:
        follower_configs = list_env_configs()
        args.follower_config = prompt.prompt(
            "Please enter the follower robot configuration name.",
            options=follower_configs,
            default=follower_configs[0],
        )
        logger.info(f"Using follower configuration: {args.follower_config}")

    trans_signs = _parse_signs(args.spacemouse_trans_signs, "--spacemouse-trans-signs")
    rot_signs = _parse_signs(args.spacemouse_rot_signs, "--spacemouse-rot-signs")

    leader: SpaceMouseTeleop | None = None
    try:
        env = make_env(
            env_type=args.follower_config,
            control_type="cartesian",
            namespace=args.follower_namespace,
        )

        leader = SpaceMouseTeleop(
            trans_scale=args.spacemouse_trans_scale,
            rot_scale=args.spacemouse_rot_scale,
            gripper_rate=args.spacemouse_gripper_rate,
            deadzone=args.spacemouse_deadzone,
            trans_signs=trans_signs,
            rot_signs=rot_signs,
        )
        leader.wait_until_ready()
        logger.info("SpaceMouse leader ready.")

        keys_to_ignore = []
        features = get_features(env=env, ignore_keys=keys_to_ignore)
        logger.debug(f"Using the features: {features}")

        recording_manager = make_recording_manager(
            recording_manager_type=args.recording_manager_type,
            features=features,
            repo_id=args.repo_id,
            robot_type=args.robot_type,
            num_episodes=args.num_episodes,
            fps=args.fps,
            resume=args.resume,
            push_to_hub=args.push_to_hub,
        )
        recording_manager.wait_until_ready()
        logger.info("Recording manager is ready.")

        env_metadata = env.get_metadata()

        with open(recording_manager.dataset_directory / "meta" / "crisp_meta.json", "w") as f:
            json.dump(env_metadata, f, indent=4)

        logger.info(
            f"Environment metadata saved to {recording_manager.dataset_directory / 'meta' / 'crisp_meta.json'}"
        )

        logger.info("Homing follower before starting with recording.")

        env.wait_until_ready()
        env.home(home_config=HomeConfig.CLOSE_TO_TABLE.randomize(noise=args.home_config_noise))
        env.reset()

        tasks = list(args.tasks)

        def on_start():
            """Hook function to be called when starting a new episode."""
            env.robot.reset_targets()
            env.reset()
            leader.reset_pose()

        def on_end():
            """Hook function to be called when stopping the recording."""
            env.robot.reset_targets()
            random_home = HomeConfig.CLOSE_TO_TABLE.randomize(noise=args.home_config_noise)
            env.robot.home(blocking=False, home_config=random_home)
            env.gripper.open()

        with recording_manager:
            while not recording_manager.done():
                logger.info(
                    f"→ Episode {recording_manager.episode_count + 1} / {recording_manager.num_episodes}"
                )

                teleop_fn = make_teleop_spacemouse_fn(env, leader)

                task = tasks[np.random.randint(0, len(tasks))] if tasks else "No task specified."
                logger.info(f"▷ Task: {task}")

                recording_manager.record_episode(
                    data_fn=teleop_fn,
                    task=task,
                    on_start=on_start,
                    on_end=on_end,
                )

        logger.info("Homing follower.")
        env.home()

        logger.info("Closing the environment.")
        env.close()

        logger.info("Finished recording.")

    except TimeoutError as e:
        logger.exception(f"Timeout error occurred during recording: {e}.")
        logger.error(
            "Please check if the robot container is running and the namespace is correct."
            "\nYou can check the topics using `ros2 topic list` command."
        )

    except Exception as e:
        logger.exception(f"An error occurred during recording: {e}.")

    finally:
        if leader is not None:
            leader.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
