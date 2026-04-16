"""Class defining the teleoperation for a 3Dconnexion SpaceMouse device."""

import threading
import time

import numpy as np
import pyspacemouse
from crisp_py.utils.geometry import Pose
from scipy.spatial.transform import Rotation


class SpaceMouseTeleop:
    """Class to handle teleoperation using a 3Dconnexion SpaceMouse.

    The SpaceMouse produces per-tick 6-DoF deltas. This class integrates those
    deltas into a synthetic absolute :class:`Pose` so it can be consumed by the
    same recording pipeline as :class:`TeleopStreamedPose`.
    """

    def __init__(
        self,
        trans_scale: float = 0.2,
        rot_scale: float = 1.0,
        gripper_rate: float = 2.0,
        deadzone: float = 0.05,
        trans_signs: tuple[int, int, int] = (1, 1, 1),
        rot_signs: tuple[int, int, int] = (1, 1, 1),
        initial_gripper: float = 1.0,
        poll_hz: float = 250.0,
    ):
        """Initialize the SpaceMouseTeleop class.

        Args:
            trans_scale: Meters per second at full puck deflection (|axis| = 1.0).
            rot_scale: Radians per second at full puck deflection (|axis| = 1.0).
            gripper_rate: Gripper units per second while a button is held.
                Full 0->1 sweep takes ``1 / gripper_rate`` seconds.
            deadzone: Per-axis deadzone in SpaceMouse units. Values below are zeroed.
            trans_signs: Per-axis sign multipliers (+1 or -1) for x/y/z.
            rot_signs: Per-axis sign multipliers (+1 or -1) for roll/pitch/yaw.
            initial_gripper: Initial gripper target in [0, 1]. Defaults to 1.0 (open)
                to match the env's post-home state.
            poll_hz: Polling frequency of the background thread.
        """
        self.trans_scale = trans_scale
        self.rot_scale = rot_scale
        self.gripper_rate = gripper_rate
        self.deadzone = deadzone
        self.trans_signs = np.array(trans_signs, dtype=np.float64)
        self.rot_signs = np.array(rot_signs, dtype=np.float64)
        self._initial_gripper = float(np.clip(initial_gripper, 0.0, 1.0))
        self._poll_period = 1.0 / poll_hz

        self._lock = threading.Lock()
        self._position = np.zeros(3, dtype=np.float64)
        self._orientation = Rotation.identity()
        self._gripper_value = self._initial_gripper
        self._ready = False
        self._stop = threading.Event()

        self._cached_xyz = np.zeros(3, dtype=np.float64)
        self._cached_rpy = np.zeros(3, dtype=np.float64)
        self._cached_button0 = False
        self._cached_button1 = False

        self._device = pyspacemouse.open()
        if self._device is None:
            raise RuntimeError("Failed to open SpaceMouse device. Is it connected?")
        self._ready = True

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _apply_deadzone(self, v: float) -> float:
        return 0.0 if abs(v) < self.deadzone else v

    def _poll_loop(self):
        last_time = time.time()
        MAX_DRAIN = 64
        while not self._stop.is_set():
            # Drain all queued HID reports, keep the freshest non-stale state.
            # pyspacemouse signals an empty queue with state.t < 0.
            for _ in range(MAX_DRAIN):
                state = self._device.read()
                if state.t < 0:
                    break
                self._cached_xyz = np.array(
                    [
                        self._apply_deadzone(state.y),
                        self._apply_deadzone(state.x),
                        self._apply_deadzone(state.z),
                    ],
                    dtype=np.float64,
                )
                self._cached_rpy = np.array(
                    [
                        self._apply_deadzone(state.roll),
                        self._apply_deadzone(state.pitch),
                        self._apply_deadzone(state.yaw),
                    ],
                    dtype=np.float64,
                )
                self._cached_button0 = (
                    bool(state.buttons[0]) if len(state.buttons) > 0 else False
                )
                self._cached_button1 = (
                    bool(state.buttons[1]) if len(state.buttons) > 1 else False
                )

            now = time.time()
            dt = now - last_time
            last_time = now

            dxyz = self._cached_xyz * self.trans_signs * self.trans_scale * dt
            drpy = self._cached_rpy * self.rot_signs * self.rot_scale * dt

            delta_rot = Rotation.from_euler("xyz", drpy)

            # Two-button continuous gripper: btn0 held -> close, btn1 held -> open.
            gripper_delta = 0.0
            if self._cached_button0:
                gripper_delta -= self.gripper_rate * dt
            if self._cached_button1:
                gripper_delta += self.gripper_rate * dt

            with self._lock:
                self._position = self._position + dxyz
                self._orientation = delta_rot * self._orientation
                if gripper_delta != 0.0:
                    self._gripper_value = float(
                        np.clip(self._gripper_value + gripper_delta, 0.0, 1.0)
                    )

            time.sleep(self._poll_period)

    @property
    def last_pose(self) -> Pose:
        """Get the current synthetic pose integrated from SpaceMouse deltas."""
        with self._lock:
            return Pose(position=self._position.copy(), orientation=self._orientation)

    @property
    def last_gripper(self) -> float:
        """Get the current gripper target value."""
        with self._lock:
            return self._gripper_value

    def is_ready(self) -> bool:
        """Return True once the SpaceMouse device is opened."""
        return self._ready

    def wait_until_ready(self, timeout: float = 5.0):
        """Wait until the SpaceMouse device is opened and polling."""
        start_time = time.time()
        while not self.is_ready():
            if time.time() - start_time > timeout:
                raise TimeoutError("Timed out waiting for SpaceMouse to be ready.")
            time.sleep(0.05)

    def reset_pose(self):
        """Zero the synthetic pose. Gripper state persists across episodes.

        Call at the start of each episode so the first delta is zero, matching
        the first-step-skip behavior of the streamed-pose recorder. The gripper
        target is deliberately not reset — it carries over from the previous
        episode like the physical leader gripper in a leader-follower setup.
        """
        with self._lock:
            self._position = np.zeros(3, dtype=np.float64)
            self._orientation = Rotation.identity()

    def close(self):
        """Stop the polling thread and release the SpaceMouse device."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            self._device.close()
        except Exception:
            pass
