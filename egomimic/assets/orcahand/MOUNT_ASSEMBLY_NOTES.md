# FR3 + Gavin 90° mount + ORCA right hand — assembly notes

Combined URDF: `fr3_mount_orcahand.urdf` (FR3 arm → mount → ORCA right hand, extended pose).
Left-hand counterpart: `fr3_mount_orcahand_left.urdf` — see the LEFT section at the bottom;
it is generated, not hand-aligned.

## Derived fixed joints (the hard-won numbers)

Both transforms were aligned by hand in Blender and read out relative to the
correct link frames. Keep these so the assembly never has to be re-derived.

### `mount_joint`  (parent `fr3_link8` → child `mount`)
```xml
<origin xyz="0.031815 0.080326 0.002538" rpy="2.375311 -0.000087 -1.561465"/>
```

### `mount_to_hand` (parent `mount` → child `right_tower`)
```xml
<origin xyz="0.120598 0.098351 0.000260" rpy="0.814307 3.141593 0.000000"/>
```

## Key facts / conventions
- **Mount STL** `Gavin90DegMount_edited_P1S.stl` is in **millimetres** → referenced
  with `scale="0.001 0.001 0.001"` and an identity visual `<origin>`.
- The **`mount` link frame is the STL's own (0,0,0) corner** with the mount's
  orientation. Both joint origins above are expressed to that corner frame, so the
  visual mesh sits at identity in the link. (If you re-derive in Blender, the corner
  frame = `mount.matrix_world @ Translate(mesh_local_min)` with scale stripped.)
- Mount mating faces (both carry a Ø50 / 4×M6 ISO-9409-1-50 pattern, PCD 50 mm):
  - **FACE_B** (±45° clocked holes) → **Franka flange** (`fr3_link8`).
  - **FACE_A** (0/90 clocked holes) → **ORCA hand** (`right_tower`).
- `fr3_link8` (flange frame) = `fr3_link7` + fixed `(0,0,0.107)`, no rotation.

## How the combined URDF was built
- FR3 visual meshes were the Collada `.dae` originals; Phobos/trimesh chokes on their
  missing UVs, so they were converted (via trimesh, no axis remap) to STL.
- The ORCA `world` and `world2right_tower_fixed_jointbody` stub links and their two
  joints (`world2right_tower_fixed`, `world2right_tower_fixed_offset`) were dropped;
  the hand was re-rooted at `right_tower` directly under `mount`.

## Packaging (self-contained)
- All referenced meshes are copied under `meshes/franka/` (16) and `meshes/orca/` (68);
  the mount STL sits at the assets root. The URDF uses **relative** mesh paths
  resolved against its own directory, so this folder is portable — move/copy the whole
  `assets/` dir and it still loads.

## Source URDFs (unmodified)
Both live in external ROS description packages rather than in this repo, so they are
given relative to their package root. `build_left_mount_assembly.py` reads the ORCA
root from `ORCAHAND_DESCRIPTION_DIR`, defaulting to `~/orcahand_description`.

- `franka_description` → `urdfs/fr3.urdf`  (default `~/franka_description`)
- `orcahand_description` → `v1/models/urdf/orcahand_right_extended.urdf`

# LEFT assembly: FR3 + mirrored mount + ORCA left hand

`fr3_mount_orcahand_left.urdf` is **generated** by `build_left_mount_assembly.py`
(run with the isaacsim conda env; needs trimesh) and verified by
`validate_left_mount_assembly.py` (run with `/usr/bin/python3`; needs pinocchio).
Never hand-edit the derived numbers — regenerate instead.

## Derivation (no Blender involved)
The vendor left hand is the right hand mirrored across the tower's local x=0
plane (`S_T = diag(-1,1,1)`, zero translation). The mirrored mount mesh
`meshes/mount/Gavin90DegMount_mirrored_P1S.stl` is the right STL reflected
across its local x and re-cornered so AABB min stays at the link origin
(`S_M = Trans(0.155000061, 0, 0) · diag(-1,1,1)`). With the assembly mirrored
across the flange xz-plane (`S_F = diag(1,-1,1)`):

```
mount_joint_left   = S_F · mount_joint_right   · S_M⁻¹
mount_to_hand_left = S_M · mount_to_hand_right · S_T⁻¹   (independent of S_F)
```

### `mount_joint`  (parent `fr3_link8` → child `mount`)
```xml
<origin xyz="0.033261 0.074667 0.002551" rpy="2.375311 0.000087 -1.580128"/>
```

### `mount_to_hand` (parent `mount` → child `left_tower`)
```xml
<origin xyz="0.034402 0.098351 0.000260" rpy="-2.327286 0.000000 3.141593"/>
```

## Physical clocking caveat
The ISO-9409-1-50 patterns on both mating faces are mirror-symmetric, so a
mirrored print mates with the flange and the left tower. But the sim numbers
assume the mirror plane is the flange **xz**-plane; if the physical mirrored
mount is clocked differently on the real flange (90° steps are possible),
premultiply the corresponding RotZ onto `S_F` in the builder and regenerate.

## Joint mirror sign map (validated)
`q[left_wrist] = -q[right_wrist]`; **all 16 finger joints map with +1**
(vendor axis conventions make finger angles transfer unchanged). Limits map
accordingly (the wrist's swap+negate, fingers' identical).

## Validation results (2026-08-24)
- pinocchio FK, 20 random in-limit configs, all hand links + mount relative to
  the flange: max position error 4.9e-7 m, max rotation error 6.1e-7 (pure
  6-decimal URDF rounding).
- Per-joint composite inertias mirror exactly (mass, COM, tensor via D·I·D).
- pytorch_kinematics builds all the chains the app/retargeter uses
  (base→fr3_link8, base→left_tower, left_tower→left_palm, 5 fingertip chains)
  and agrees on the flange→tower mirror to 4.8e-7.
