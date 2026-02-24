# Hydra Workflow in EgoVerse (Minimal Guide)

This codebase uses **Hydra** for config composition and **no ConfigStore** (no Python-side registration). Classes are wired purely via YAML `_target_` and `hydra.utils.instantiate()`.

---

## 1. Minimal Example: One Class + One Config

### Step 1: Implement your class

Your class must be importable and its `__init__` must accept the same keys you put in the config (as keyword arguments).

```python
# egomimic/utils/my_module.py
class MyService:
    def __init__(self, foo: str, bar: int = 10):
        self.foo = foo
        self.bar = bar
```

### Step 2: Add a YAML config

Config lives under `egomimic/hydra_configs/`. Use `_target_` = fully qualified class path; other keys become constructor kwargs.

```yaml
# egomimic/hydra_configs/my_group/my_config.yaml
_target_: egomimic.utils.my_module.MyService
foo: hello
bar: 20
```

### Step 3: Compose it from the main config

**Option A – Defaults (recommended)**  
In `train.yaml`, add a default so your config is loaded under a key:

```yaml
# egomimic/hydra_configs/train.yaml
defaults:
  - model: hpt_bc_flow_eva
  - paths: default
  - my_group: my_config   # <-- your config merged under key "my_group"
  - _self_
# ...
```

Then the composed `cfg` has `cfg.my_group` with `_target_`, `foo`, `bar`.

**Option B – Override from CLI**

```bash
python egomimic/trainHydra.py my_group=my_config
```

### Step 4: Instantiate in code

In `trainHydra.py` (or any function that receives `cfg`):

```python
from omegaconf import DictConfig

def run(cfg: DictConfig):
    obj = hydra.utils.instantiate(cfg.my_group)
    # obj is MyService(foo="hello", bar=20)
```

That’s the full loop: **class → YAML with `_target_` → defaults or CLI → `hydra.utils.instantiate(cfg.xxx)`**.

---

## 2. How “registration” works here (no ConfigStore)

- **There is no explicit “register class” step.**  
- The “contract” is: **`_target_` = dot path to a callable (class or function)** that Hydra can import; the rest of the config is passed as kwargs.  
- So “registering” = **putting the right `_target_` in the right YAML** (and making sure that module is importable).

---

## 3. Config layout and defaults

- **Config root**: `egomimic/hydra_configs/` (see `config_path` in `@hydra.main`).  
- **Main file**: `config_name="train.yaml"`.  
- **Defaults** in `train.yaml`:

  ```yaml
  defaults:
    - model: hpt_bc_flow_eva   # → model/hpt_bc_flow_eva.yaml → cfg.model
    - data: test_RBY1          # → data/test_RBY1.yaml        → cfg.data
    - trainer: ddp            # → trainer/ddp.yaml           → cfg.trainer
    - my_group: my_config     # → my_group/my_config.yaml    → cfg.my_group
    - _self_
  ```

  Each line means: “load `group_name/option.yaml` and merge under key `group_name`.”  
  `_self_` means “merge the rest of `train.yaml` after the defaults.”

- **Optional group**: `- debug: null` means group `debug` exists but no file is selected by default; you can still do e.g. `trainer=debug` if you have `trainer/debug.yaml`.

---

## 4. Nested instantiation (`_target_` inside YAML)

You can nest configs; each nested block with `_target_` is instantiated when you call `instantiate` on a parent that passes that block down (e.g. as a nested config).

Example from the codebase (conceptually):

```yaml
# Inside model/hpt_bc_flow_eva.yaml
robomimic_model:
  _target_: egomimic.algo.hpt.HPT
  kinematics_solver:
    _target_: egomimic.robot.eva.eva_kinematics.EvaMinkKinematicsSolver
    model_path: /path/to/model.xml
```

When the top-level `cfg.model` is instantiated, Hydra recursively instantiates `kinematics_solver` and passes it as the `kinematics_solver` kwarg to `HPT`.

---

## 5. Passing extra kwargs at runtime

Sometimes the config doesn’t have everything (e.g. objects created earlier). You can pass extra kwargs into `instantiate`:

```python
# From trainHydra.py
datamodule = hydra.utils.instantiate(
    cfg.data, train_datasets=train_datasets, valid_datasets=valid_datasets
)
model = hydra.utils.instantiate(
    cfg.model, robomimic_model={"data_schematic": data_schematic}
)
```

Those kwargs override or supplement the keys from the config.

---

## 6. Partial instantiation

For objects that are created but not called yet (e.g. optimizer to be passed to the model later), use `_partial_: true`:

```yaml
optimizer:
  _target_: torch.optim.AdamW
  _partial_: true
  lr: 1e-4
  weight_decay: 0.0001
```

Then `hydra.utils.instantiate(cfg.optimizer)` returns a **partial** callable (e.g. `functools.partial(AdamW, lr=..., weight_decay=...)`), not an optimizer instance.

---

## 7. Resolvers and interpolation

- **Interpolation**: `output_dir: ${hydra:runtime.output_dir}` reuses Hydra’s runtime output dir.  
- **Custom resolver**: In `trainHydra.py`, `OmegaConf.register_new_resolver("eval", eval)` allows `${eval: '1+1'}` → 2.  
- **Cross-config refs**: e.g. `_${data.dataset.data_schematic}` references another part of the composed config (used when that value is set elsewhere).

---

## 8. Workflow checklist when adding your own component

1. **Implement** the class (or function) in an importable module; `__init__` args = config keys you’ll put in YAML.  
2. **Add YAML** under `hydra_configs/<group>/<option>.yaml` with:
   - `_target_: your.package.module.YourClass`
   - Other keys = constructor kwargs.  
3. **Wire it in** either:
   - add `- <group>: <option>` to `defaults` in `train.yaml`, or  
   - override from CLI: `python egomimic/trainHydra.py <group>=<option>`.  
4. **Instantiate** where you need it: `obj = hydra.utils.instantiate(cfg.<group>)`, optionally passing extra kwargs.  
5. **Nested objects**: add nested blocks with their own `_target_` in the same YAML; Hydra will instantiate them when building the parent.

No separate “registration” step is required beyond the YAML `_target_` and an importable class.
