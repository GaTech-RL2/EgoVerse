# Sharpa kinematic models

The hand URDFs are unchanged copies from `sharpa-urdf-usd-xml`, Wave V3.0.4.
Their upstream license and notice are included in this directory. The platform
URDF describes NTH POC1.0 and was supplied separately from the hand package.
SHA-256 values in the registry identify each URDF's exact bytes. Mesh packages
are not included or required for forward kinematics.

Hand keypoints are expressed relative to the declared `ee_pose_link` (palm),
which differs from the flange root. Use `dexmate_bimanual` as the episode's
`embodiment` and `dexmate_nth_poc1` as its morphology platform identifier.
