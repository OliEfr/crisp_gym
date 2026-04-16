import pyspacemouse

# Context manager (recommended) - automatically closes device
with pyspacemouse.open() as device:
    while True:
        state = device.read()
        print(state.x, state.y, state.z)


pixi run -e kilted-lerobot crisp-record-spacemouse \
  --repo-id OliverHausdoerfer/test_1 \
  --follower-config left_robot_env_oliver --follower-namespace left \
  --fps 15 \
  --tasks "siemens" --no-push-to-hub --num-episodes 1