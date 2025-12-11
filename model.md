## Guide to adding additional states and action spaces to the codebase

### Step 1 - Registering a new embodiment

For each new embodiment navigate to `external/rldb/rldb/utils.py`](./external/rldb/rldb/utils.py). (Note that since rldb is a submodule, the navigation link here won't work :<)

In the `Embodiment` Enum:

```python
class EMBODIMENT(Enum):
    EVE_RIGHT_ARM = 0
    EVE_LEFT_ARM = 1
    EVE_BIMANUAL = 2
    ARIA_RIGHT_ARM = 3
    ARIA_LEFT_ARM = 4
    ARIA_BIMANUAL = 5
    MY_NEW_EMBODIMENT = 6
```

Note, `6` is arbitrary, you can use whatever int id you would like.

---

### Step 2 - Modify train yaml

Navigate to [`./egomimic/hydra_configs/train.yaml`](./egomimic/hydra_configs/train.yaml). Replace the data schematic with the relevant keys for your embodiment. The current `data_schematic` would look something like:

```yaml
data_schematic: # Dynamically fill in these shapes from the dataset
  _target_: egomimic.rldb.utils.DataSchematic
  schematic_dict:
    eve_right_arm:
      front_img_1: #batch key
        key_type: camera_keys # key type
        lerobot_key: observations.images.front_img_1 # dataset key
      right_wrist_img:
        key_type: camera_keys
        lerobot_key: observations.images.right_wrist_img
      joint_positions:
        key_type: proprio_keys
        lerobot_key: observations.state.joint_positions
      actions_joints:
        key_type: action_keys
        lerobot_key: actions_joints
      embodiment:
        key_type: metadata_keys
        lerobot_key: metadata.embodiment
    aria_right_arm:
      front_img_1:
        key_type: camera_keys
        lerobot_key: observations.images.front_img_1
      ee_pose:
        key_type: proprio_keys
        lerobot_key: observations.state.ee_pose
      actions_cartesian:
        key_type: action_keys
        lerobot_key: actions_cartesian
      embodiment:
        key_type: metadata_keys
        lerobot_key: metadata.embodiment
  viz_img_key:
    eve_right_arm:
      front_img_1
    aria_right_arm:
      front_img_1
```

Modify this to your new embodiment. Note that the name of the embodiment should match with the one you added to the enum:

```yaml
data_schematic:
  _target_: egomimic.rldb.utils.DataSchematic
  schematic_dict:
    my_new_embodiment:
      front_img_1: #batch key - This corresponds to what you want the key to be called in the batch
        key_type: camera_keys # key type - This corresponds to the type of obs modality
        lerobot_key: observations.images.front_img_1 # dataset key - This corresponds to what the key is stored as in the dataset
      right_wrist_img:
        key_type: camera_keys
        lerobot_key: observations.images.right_wrist_img
      joint_positions:
        key_type: proprio_keys
        lerobot_key: observations.state.joint_positions
      actions_joints:
        key_type: action_keys
        lerobot_key: actions_joints
      embodiment:
        key_type: metadata_keys
        lerobot_key: metadata.embodiment
    aria_right_arm:
      front_img_1:
        key_type: camera_keys
        lerobot_key: observations.images.front_img_1
      ee_pose:
        key_type: proprio_keys
        lerobot_key: observations.state.ee_pose
      actions_cartesian:
        key_type: action_keys
        lerobot_key: actions_cartesian
      embodiment:
        key_type: metadata_keys
        lerobot_key: metadata.embodiment
  viz_img_key:
    my_new_embodiment:
      front_img_1
    aria_right_arm:
      front_img_1
```

Currently the `data_schematic` contains `joint_positions` as the proprioceptive observation, and `front_img_1` and `right_wrist_img` as the image observation. To remove these if your embodiment doesn't have it, just delete them. To add additional obs, under `my_new_embodiment`, add:

```yaml
      my_new_obs: # name you want your new observation to be in the batch
        key_type: key_type #change key_type to either image_keys or proprio_keys depending on type
        lerobot_key: my.new_key.stored.in.dataset # what the new observation is stored inside the dataset
```

Adding new action spaces functions in a similar manner, but the `key_type` is `action_keys`.

---

### Step 3 - Modify your model yaml

Navigate to [`./egomimic/hydra_configs/model/hpt_cotrain.yaml`](./egomimic/hydra_configs/model/hpt_cotrain.yaml) or alternate HPT yaml file. Note, the current `hpt_cotrain.yaml` contains a training setup for our eve arm.

Then, modify the domains:

```yaml
domains: ["my_new_embodiment", "aria_right_arm"]
```

Note that the name of the embodiment should match with the one you added to the enum.

Again, currently the yaml is set up with the input observation and output action modality of the Eve arm. To change this, let us assume we are adding `my_new_obs_1` which is an image modality that requires a ResNet encoder and `my_new_obs_2` which is a proprioceptive modality with dimension `10`.

Add this to the `stem_specs` under `my_new_embodiment`:

```yaml
      my_new_obs_1:
        _target_: egomimic.models.hpt_nets.MLPPolicyStem
        input_dim: 512 # ResNet output feature dim
        output_dim: 512
        widths: [512]
        specs:
          random_horizon_masking: false
          cross_attn:
            crossattn_latent: 16
            crossattn_heads: 8
            crossattn_dim_head: 64
            crossattn_modality_dropout: 0.1
            modality_embed_dim: 512
      state_my_new_obs_2: # the prefix state tells the model it is an additional low-dim state modality
        _target_: egomimic.models.hpt_nets.MLPPolicyStem
        input_dim: 10 # input dimension
        output_dim: 512 # HPT embed_dim
        widths: [512]
        specs:
          random_horizon_masking: false
          cross_attn:
            crossattn_latent: 16
            crossattn_heads: 8
            crossattn_dim_head: 64
            crossattn_modality_dropout: 0.1
            modality_embed_dim: 512
```

For the `my_new_obs_1`, we also need to add an additional image encoder. Add the following to `encoder_specs`:

```yaml
    my_new_obs_1:
      _target_: egomimic.models.hpt_nets.ResNet
      output_dim: 512
      num_of_copy: 1
```

---

### Step 4 (Optional) - A guide to the flexible action head system

There are 3 kinds of decoder head types for the HPT action head. There is the standard head (labelled as the embodiment name in `head_specs`); the shared head (labelled as `shared` in `head_specs`); auxiliary head (labelled as `{embodiment_name}_{action_key}` in `head_specs`).

The standard head corresponds to the default action output of the embodiment. The shared head corresponds to an action space shared between any 2 embodiments - In this case, with the human embodiment. The auxiliary head(s) corresponds to any additional action spaces required such as dexterous hand etc. You may only have 1 default head and 1 shared head but as many auxiliary heads per embodiment as needed.

Here is how `head_specs` and other parts of the yaml look if I add:
a default `7` dim action head for `my_new_embodiment`, a shared `3` dim action head, and an auxiliary `10` dim action head for `my_new_embodiment`. I assume that the default action key is `actions_joints`, auxiliary action key is `actions_auxiliary` and the shared action key is `actions_shared`:

```yaml
 head_specs:
    my_new_embodiment : 
      _target_: egomimic.models.hpt_nets.MLPPolicyHead
      input_dim: 512
      output_dim: 7
      widths: [256, 512]
      tanh_end: false
      dropout: true    
    aria_right_arm : 
      _target_: egomimic.models.hpt_nets.MLPPolicyHead
      input_dim: 512
      output_dim: 3
      widths: [256, 512]
      tanh_end: false
      dropout: true    
    shared :
      _target_: egomimic.models.hpt_nets.MLPPolicyHead
      input_dim: 512
      output_dim: 3
      widths: [256, 512]
      tanh_end: false
      dropout: true
    my_new_embodiment_actions_auxiliary :
      _target_: egomimic.models.hpt_nets.MLPPolicyHead
      input_dim: 512
      output_dim: 10
      widths: [256, 512]
      tanh_end: false
      dropout: true
```

You would specify the auxiliary action keys and shared action keys in the following manner under `robomimic_model`:

```yaml
  shared_ac_key : "actions_shared"
  auxiliary_ac_keys:
    my_new_embodiment: ["actions_auxiliary"]
```

Note that these action keys have to match their `batch_key` inside the `data_schematic` in `train.yaml`.
