# crisp_gym & Lerobot Command Reference

Collection of useful commands for deployment, training, recording, dataset inspection, and ROS camera operations in the crisp_gym and Lerobot environments. Each section is labeled wiht headlines and so on for quick reference.

---

## Deployment Commands

### Deploy Policy (crisp_gym-trained model)

Deploy a trained policy from crisp_gym using the `crisp-deploy-policy` command.

```sh
pixi run -e humble-lerobot crisp-deploy-policy  --path /home/maxchr/repos/crisp_gym/outputs/train/2026-02-13/15-48-01_smolvla_lego_stack_with_lang_30k/15-48-01_smolvla_lego_stack_with_lang_30k/checkpoints/last/pretrained_model \
  --policy-config lerobot_policy \
  --env-namespace left \
  --repo-id max-chr/stack_v6_lang_depl_02 \
  --robot-type franka
```

### Deploy Policy (Lerobot-trained model)

```sh
pixi run -e humble-lerobot crisp-deploy-policy  --path /home/maxchr/repos/lerobot/outputs/train/2026-02-10/15-42-24_smolvla_stack_v6/checkpoints/last/pretrained_model \
  --policy-config lerobot_policy \
  --env-namespace left \
  --repo-id max-chr/stack_v6_depl_01 \
  --robot-type franka
```

### Deploy Policy (No-Language Model)

```sh
pixi run -e humble-lerobot crisp-deploy-policy  --path  /home/maxchr/repos/crisp_gym/outputs/train/2026-02-17/10-08-02_smolvla_lego_stack_with_lang_30k/checkpoints/last/pretrained_model \
  --policy-config lerobot_policy \
  --env-namespace left \
  --repo-id max-chr/stack_v6_nolang_depl_03 \
  --robot-type franka
```

---

## Recording Commands

### Record Human-Only Episode (Cameras Only)

Record a human demonstration using only camera data.

```sh
pixi run -e humble-lerobot crisp-record-cameras-only \
  --repo-id max-chr/stack_v6_camera_only_04 \
  --camera-config left_stack_v6_like \
  --num-episodes 2 --fps 15 --push-to-hub
```

#### For New Camera Setup

```sh
pixi run -e humble-lerobot crisp-record-cameras-only \
  --repo-id max-chr/human_rec_base_camera_only_01  \
  --camera-config left_human_rec_base \
  --num-episodes 2 --fps 15 
```

#### Push Dataset to HuggingFace Hub (if forgotten)

```sh
hf upload max-chr/simple_tasks_human_rec_v0 /home/maxchr/data/hf/lerobot/max-chr/simple_tasks_human_rec_v0 --repo-type dataset
```


#### Push Dataset to LSY HuggingFace Hub

```sh
hf upload LSY-lab/simple_tasks_human_rec_v0 /home/maxchr/data/hf/lerobot/max-chr/simple_tasks_human_rec_v0 --repo-type dataset
```


#### Final Human Recording Setup for Cmaera-only (with explicit task)

```sh
pixi run -e humble-lerobot crisp-record-cameras-only \
  --camera-config human_rec_v1 \
  --fps 15 \
  --repo-id max-chr/simple_tasks_human_rec_v0 \
  --tasks "Pick the syringe from the plate and place it in the cup" --resume  --num-episodes 100
```

```sh
pixi run -e humble-lerobot crisp-record-cameras-only \
  --camera-config human_rec_v1 \
  --fps 15 \
  --repo-id max-chr/simple_tasks_human_rec_v0 \
  --tasks "Pick the syringe from the table and place it on the plate" --resume --num-episodes 100
```
```sh
pixi run -e humble-lerobot crisp-record-cameras-only \
  --camera-config human_rec_v1 \
 --fps 15 \
  --repo-id max-chr/simple_tasks_human_rec_v0 \
  --tasks "Pick up the cup and place it in the basket" --resume --num-episodes 48 
```

Task prompts for generalization:
1. "Pick the syringe from the plate and place it in the cup"
2. "Pick the syringe from the table and place it on the plate"
3. "Pick up the cup and place it in the basket"

Randomizations:
45-50 episodes per Task
Fine grained randomization blocks 
always in every episode: placement of cup, plate, basket within 5-10cm radius, operator dress code (i.e. sleeve, type of sleeve) (i.e. diagonally, swap positions of late and cup, multiple syringes in cup,..)
- camera positions (slight angle and hight adjustments of 2 or more cameras)every 10 episodes

Big randomization blocks:
- lighting (light on, off, flashlight, blinds up)
- background for table and wall (poster, blanket, blank wall, person standing there)

### Final  Recording for TELEOP uon human camera setup (with explicit task)

#### Start recording (keyboard mode):
```sh
pixi run -e humble-lerobot crisp-record-leader-follower \
  --repo-id max-chr/simple_tasks_teleop_v0 \
  --leader-config right_leader \
  --leader-namespace right \
  --follower-config left_human_rec_v1 \
  --follower-namespace left \
  --recording-manager-type keyboard \
  --fps 15 \
  --tasks "Pick up the cup and place it in the basket" \
  --no-push-to-hub --resume \
  --num-episodes 4
```
#### Start recording (franka button mode):

```sh
pixi run -e humble-lerobot crisp-record-leader-follower \
  --repo-id max-chr/simple_tasks_teleop_v0 \
  --leader-config right_leader \
  --leader-namespace right \
  --follower-config left_human_rec_v1 \
  --follower-namespace left \
  --recording-manager-type ros \
  --fps 15 \
  --tasks "Pick up the cup and place it in the basket" \
  --no-push-to-hub --resume \
  --num-episodes 4
```


---

## Dataset Inspection & Visualization

> **Important:** Run these dataset-viz commands from the `lerobot` repository.

### Visualize Recorded Episodes

Inspect or visualize recorded episodes using various methods:

```sh
pixi run -e lerobot python -m lerobot.scripts.visualize_dataset --repo-id max-chr/stack_v6_camera_only_01  --episode-index 0
pixi run -e humble-lerobot lerobot.scripts.visualize_dataset --repo-id max-chr/stack_v6_camera_only_01  --episode-index 0
pixi run -e humble-lerobot visualize_dataset --repo-id max-chr/stack_v6_camera_only_01  --episode-index 0
```

#### Approach to visualize dataset with rerun: Rerun .rrd File for Robust Visualization

#### On Local Dataset

```sh
python -m lerobot.scripts.lerobot_dataset_viz --repo-id stack_v6_camera_only_04 --root /home/maxchr/data/hf/lerobot/max-chr/stack_v6_camera_only_04 --episode-index 3 --display-compressed-images 0 --mode local --save 1 --output-dir /tmp/lerobot_viz_camera_only_04
rerun /tmp/lerobot_viz_camera_only_04/stack_v6_camera_only_04_episode_3.rrd
```

#### With Adapted Camera

```sh
python -m lerobot.scripts.lerobot_dataset_viz --repo-id human_rec_base_camera_only_03 --root /home/maxchr/data/hf/lerobot/max-chr/human_rec_base_camera_only_03 --episode-index 0 --display-compressed-images 0 --mode local --save 1 --output-dir /tmp/lerobot_viz_camera_only_hum_03
rerun /tmp/lerobot_viz_camera_only_hum_03/human_rec_base_camera_only_03_episode_0.rrd 
```

#### Final Human Recording Setup Dataset Viz

```sh
python -m lerobot.scripts.lerobot_dataset_viz --repo-id human_rec_v1_2 --root /home/maxchr/data/hf/lerobot/max-chr/human_rec_v1_2 --episode-index 1 --display-compressed-images 0 --mode local --save 1 --output-dir /tmp/lerobot_viz_human_rec_v1_2
rerun /tmp/lerobot_viz_human_rec_v1_2/human_rec_v1_2_episode_0.rrd
rerun /tmp/lerobot_viz_human_rec_v1_1/human_rec_v1_1_episode_0.rrd
```
##### for our datatset simple_tasks_human_rec_v0

```sh
python -m lerobot.scripts.lerobot_dataset_viz --repo-id simple_tasks_human_rec_v0 --root /home/maxchr/data/hf/lerobot/max-chr/simple_tasks_human_rec_v0 --display-compressed-images 0 --mode local --save 1 --output-dir /tmp/lerobot_viz_simple_tasks_human_rec_v0 --episode-index 0 
rerun /tmp/lerobot_viz_simple_tasks_human_rec_v0/simple_tasks_human_rec_v0_episode_0.rrd

```

##### for our datatset simple_tasks_teleop_v0

```sh
python -m lerobot.scripts.lerobot_dataset_viz --repo-id simple_tasks_teleop_v0 --root /home/maxchr/data/hf/lerobot/max-chr/simple_tasks_teleop_v0 --episode-index 0 --display-compressed-images 0 --mode local --save 1 --output-dir /tmp/lerobot_viz_simple_tasks_teleop_v0
rerun /tmp/lerobot_viz_simple_tasks_teleop_v0/simple_tasks_teleop_v0_episode_0.rrd

```


#### Distant Streaming Version

Producer (Terminal A):
```sh
python -m lerobot.scripts.lerobot_dataset_viz --repo-id stack_v6 --root /home/maxchr/datasets/stack_v6 --episode-index 0 --display-compressed-images 0 --mode distant --web-port 9088
```
Viewer (Terminal B):
```sh
open http://127.0.0.1:9088 in a browser
# OR
rerun --connect rerun+http://127.0.0.1:9088/proxy
```

> **Note:** In this lerobot version, `--ws-port` is declared but not used by `lerobot_dataset_viz.py`; `--mode distant` serves the web viewer via `--web-port`.

---

### Visualize Cameras in ROS

```sh
ros2 run rqt_image_view rqt_image_view
```

- **Wrist View:** Select `/left/left_wrist_camera/color/image_rect_raw`
- **Third-Person View:** Select `/right/right_down_third_person_camera/color/image_raw`

Human dataset camera perspective:

```sh
rqt --perspective-file human_darqt --perspective-file human_dataset_cameras.perspectivetaset_cameras_2.perspective
```

## ROS Camera Commands

### List and Inspect Camera Topics

```sh
ros2 topic list | grep -E 'image|camera_info'
ros2 topic type /your/camera/color/image_raw
ros2 topic hz /your/camera/color/image_raw
ros2 topic echo /your/camera/color/image_raw --once
```


---

## Training Commands

### Standard Training Command

```sh
lerobot-train \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.load_vlm_weights=true \
  --policy.train_expert_only=true \
  --policy.num_vlm_layers=16 \
  --policy.n_action_steps=50 \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/stack_v6 \
  --dataset.root=/home/maxchr/datasets/stack_v6 \
  --dataset.video_backend=pyav \
  --dataset.use_pretrained_stats=true \
  --dataset.image_transforms.train_percentage=0.9 \
  --steps=15000 \
  --batch_size=32 \
  --eval_freq=2500 \
  --optimizer.type=adamw \
  --optimizer.betas=[0.9,0.95] \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps=500 \
  --scheduler.num_decay_steps=14500 \
  --scheduler.peak_lr=1e-4 \
  --scheduler.decay_lr=2.5e-6 \
  --save_freq=1500 \
  --log_freq=100 \
  --wandb.enable=true \
  --wandb.project=franka-stacking-v6 \
  --job_name=smolvla_stack_v6_finetuned \
  --use_policy_training_preset=false
```

### Add Instruction to Dataset File

Add an instruction column to a parquet dataset file:

```sh
pixi run -e humble-lerobot python - <<'PY'
import pandas as pd
p = "/home/maxchr/datasets/stack_v6/meta/tasks.parquet"
df = pd.read_parquet(p)
df["instruction"] = "Pick up the small lego block and stack it on top of the other bigger lego block."
df.to_parquet(p, index=False)
print(df)
PY
```

### Training Command with Language Input

```sh
lerobot-train \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.load_vlm_weights=true \
  --policy.train_expert_only=true \
  --policy.num_vlm_layers=16 \
  --policy.n_action_steps=50 \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/stack_v6 \
  --dataset.root=/home/maxchr/datasets/stack_v6 \
  --dataset.video_backend=pyav \
  --dataset.use_pretrained_stats=true \
  --policy.input_features.instruction.type=LANGUAGE \
  --policy.input_features.instruction.shape=[48] \
  --policy.normalization_mapping.LANGUAGE=IDENTITY \
  --steps=15000 \
  --batch_size=32 \
  --eval_freq=2500 \
  --optimizer.type=adamw \
  --optimizer.betas=[0.9,0.95] \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps=500 \
  --scheduler.num_decay_steps=14500 \
  --scheduler.peak_lr=1e-4 \
  --scheduler.decay_lr=2.5e-6 \
  --save_freq=1500 \
  --log_freq=100 \
  --wandb.enable=true \
  --wandb.project=franka-stacking-v6 \
  --job_name=smolvla_stack_v6_finetuned_lang \
  --use_policy_training_preset=false
```

---

## Miscellaneous / Code Fixes

### Patch for Action and State Logging in lerobot_dataset_viz in lerobot repo, currently not needed anymore 

> **Suggestion:** Move this to a code documentation or troubleshooting section if not needed as a command.

```python
if ACTION in batch:
    action = batch[ACTION][i]
    if isinstance(action, torch.Tensor) and action.ndim == 0:
        rr.log(f"{ACTION}/0", rr.Scalars(action.item()))
    else:
        for dim_idx, val in enumerate(action):
            rr.log(f"{ACTION}/{dim_idx}", rr.Scalars(val.item()))

# display each dimension of observed state space (e.g. agent position in joint space)
if OBS_STATE in batch:
    state = batch[OBS_STATE][i]
    if isinstance(state, torch.Tensor) and state.ndim == 0:
        rr.log("state/0", rr.Scalars(state.item()))
    else:
        for dim_idx, val in enumerate(state):
            rr.log(f"state/{dim_idx}", rr.Scalars(val.item()))
```
