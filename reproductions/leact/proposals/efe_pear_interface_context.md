# PEAR action and observation context

Both PEAR++ students receive the same per-environment language guide alongside
the native JSON specification, current sensor values and named RGB images. The
policy still produces reasoning followed by the constrained JSON action.

The guide describes action dimensions, exact component/arm/joint order, bounds,
units and scaling, absolute versus delta semantics, native coordinate frames,
gripper signs and action duration. Direction statements come from public native
metadata. World, robot-base, end-effector and camera frames stay distinct.

For example, the recorded MetaWorld declaration says world +x is the robot's
right, +y is forward away from its base, and +z is up. Thus negative translation
deltas move left, toward the base and down respectively. Its native translation
scale is 1 cm per unit per control step, and its gripper uses -1 open / +1 close.
These conventions must not be transferred to another robot: RoboCasa's arm
commands use the robot-base frame, and DexJoCo uses its asymmetric 46-component
arm/finger target ordering.

The observation guide lists the delivered cameras in image order, actual RGB
image sizes, declared camera motion/mount/calibration fields, image row/column
directions and proprioception names. It explains that `proprio.now[i]` matches
`proprio.layout[i]` and supplies a field/type template for the current public
observation. It does not insert example state values or task answers. Advertised
depth capability is distinguished from the RGB-only images actually delivered.

The shared native PEAR tool supplies the guide on every observation and logs
`interface_context_version` and `interface_context_sha256`. The formatter is
deterministic and independent of student identity. Native action parsing,
control execution, reward and held-out membership are unchanged.

The initial prompt budget is 16,384 tokens, configurable with `MAX_PROMPT`, to
accommodate the guide and multimodal metadata. The response budget and PEAR
score-centering recipe are unchanged. Long-context memory/numerical
qualification remains necessary before primary RL.

Coverage: wired into the shared EFE native PEAR observation path and numerical
probe conversion. Broader native GR00T YAM and reasoning adapters must route
their verified public robot/sensor metadata through the same contract as they
are implemented. Omitted frame, joint-axis or sensor conventions stay
unspecified; the formatter does not invent them. Existing frozen workflows and
their historical qualification receipts are not rewritten by this change.
