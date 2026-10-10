# Plant-watering arm

Built on top of SmolVLA.

## Day 1: data collection → real-to-sim transfer → LeRobot dataset

### Task and scene

A Panda arm imitates watering a plant: it grabs a mug from one of four start positions and pours it into one of three plant pots (left, middle, right). The simulation is a custom MuJoCo scene. Water is modeled as 30 small spheres inside the mug.

**Language commands** (one per plant, used as the language instruction):

- Water the left plant
- Water the middle plant
- Water the right plant

**Success**:

- **Watered:** > 15 spheres end up in the commanded pot.
- **Put back:** the mug ends within 3 cm and 15° of where the demonstration put it down.

### Data acquisition

- **Camera:** iPhone 16 Pro, HD, 60 FPS. Camera calibrated with a ChArUco board.
- **Mug tracking:** 40 mm ArUco marker on the mug.
- **Demonstrations:** 4 mug start positions × 3 target pots × 4 repetitions = 48 recorded pours.

```bash
./run calibrate        # lens intrinsics from the board video -> dataset/calibration/intrinsics.json
```

### Real-to-sim transfer

- **Mug pose:** in every frame, the mug's position and orientation are estimated using the ArUco marker and camera intrinsics. A ChArUco board on the floor gives the camera's pose in the fixed floor frame.

```bash
./run track            # mug pose per frame, floor frame -> <clip>/track.npz, track.json
./run replay           # sim mug follows each tracked pour, no robot -> <clip>/replay.json, replay.mp4
```

### Retargeting
Execute robot movement to perform the recorded pouring.

1. Find a grasp from which the robot can pick up the mug and complete the whole pour (IK follows within joint limits). Search space is different hand tilts of 35°, 40° and 50° (pointing down at the mug).
2. **Grasp pose:** the hand points at the mug from the robot's shoulder on the mug's side, fingers closing left/right, 3 cm above the mug's bottom.
3. **Start pose:** the hand 20 cm above the table, in the same arm posture as the chosen grasp.
4. **Reach** (scripted): up to a safe height, across to 12 cm behind the mug while turning into the grasp orientation, down, slide in horizontally, close the gripper.
5. **Follow** (from the recording): the grasp is fixed relative to the mug, so each hand target is the recorded mug pose times that offset. Played at 0.6× real speed, control rate of 20 Hz. We also aim 0.1 s ahead because the controller lags.
6. **Release** (scripted): open, slide back out, lift, return to the start pose.

```bash
./run robot            # the Panda grabs the mug, follows the pour, puts it back -> <clip>/robot.json, robot_states.npz, robot.mp4
./run watch            # play the saved robot runs in the viewer
```

### LeRobot export
Turn successful robot runs into training set for SmolVLA. 
- Same format of `HuggingFaceVLA/smolvla_libero`. No physics. Same inputs for training as evalution. 
- Rate: 20fps


| Feature | Shape | Content |
|---|---|---|
| `observation.images.image` | 256×256×3 video | scene camera (`agentview`), rotated 180° like LIBERO |
| `observation.images.image2` | 256×256×3 video | wrist camera, rotated the same way |
| `observation.state` | 8 float32 | gripper position (3), gripper rotation as axis-angle (3), finger positions (2) |
| `action` | 7 float32 | change in gripper position and rotation, scaled to [-1, 1] (6), gripper (-1 open, +1 close) |
| `task` | text | "Water the left / middle / right plant" |

Saved [on Hugging Face](https://huggingface.co/datasets/edgarcancinoe/water_pour) keeping the standard LeRobot layout (`meta/`, `data/` parquet, `videos/` MP4 per camera). Visualize [here](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fedgarcancinoe%2Fwater_pour%2Fepisode_0).

```bash
./run export           # successful robot runs -> LeRobot dataset (dataset/data/lerobot/)
```