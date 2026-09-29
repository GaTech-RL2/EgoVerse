# Learn useful Astra corrections, then remove Astra

This is a new, explicitly authorized follow-up to the interrupted representation
study. The inference account cap does not prevent offline distillation of its
recorded evidence. No fresh provider requests are made in this experiment.

The frozen [protocol](configs/learned_correction_recipe_v1.json) compares native
pi05, a recorded task/clock schedule, a learned observation-conditioned text
selector, a learned flow-output correction, and that correction with the learned
selector's gate. The head arms keep original native text and visual conditioning;
the gated arm does not apply TEI/TLI a second time. All five arms run on every
test reset, including resets where native succeeds.

Training uses all ten successful actual Astra TEI/TLI trajectories from the
complete seed29 phase study and the two verified pilot text rescues: twelve
trajectories, 1,248 executed actions, 255 replan windows, eight OOD compositions.
Seven successful original phase baselines supply another 158 native anchor
windows. Four fresh native standard LIBERO10 tasks (IDs0–3, prescribed reset0)
provide additional anchors only when they succeed; failed collection and its
cost remain in the ledger. These are few independent demonstrations despite the
larger number of windows. Arrays and decisions are bound to archived evidence.

The selector sees pooled native final-prefix visual and instruction features and
raw robot state, with no clock, task ID, future progress or privileged outcome.
It learns a native/intervene gate, a categorical operator/source-pair choice and
a class-conditioned alpha. Equivalent swapped source pairs are canonicalized.
Its gate imitates teacher selection; it is not a calibrated failure detector.
Gate probability must exceed0.5; ties defer. Decisions occur every25actions and
apply to each fresh five-action replan. Training statistics and program classes
come only from the training corpus. No rollout memory is retrieved by learned
arms during evaluation; the fixed standard-demo text banks remain part of TLI.

The flow correction is a zero-initialized1,024→7 residual projection plus bias
(7,175 learned parameters), added only to the first five physical action rows.
All pretrained tensors remain frozen. Eight fixed native-distribution noise/time
draws per training window supply detached action-expert features. Only actually
executed controller values are labels, converted through the exact native input
profile. A new original-condition native chunk imputes the remaining physical
rows; clean padding is zero. Masking the loss does not remove attention's
dependence on that imputed suffix. This is a constrained, masked flow objective,
not full-chunk demonstration training. Its residual is0.5*tanh(raw/0.5); native
teacher-velocity anchors penalize changes to successful earlier behavior.

Evaluation uses seed61 and reset IDs1,2 on all twenty OOD tasks (40 paired
episodes), plus prescribed resets1,2 on the four standard LIBERO10 tasks
(8 paired episodes). These48episodes times five arms mean240physical rollouts.
OOD budgets are300actions; the original standard LIBERO10 budget is520. Native
Gaussian noise is freshly drawn at each replan from the same episode/step stream
for all arms. Astra, learning and trajectory retrieval are disabled. No reset or
method is removed because of its outcome; no parameters are selected on these
evaluation results. This is transfer on known compositions and new resets, with
a limited ID retention panel, not a new-task generalization claim.

Use one OSMO L40S training worker, then five L40S evaluation workers after
training/checkpoint gates pass. Record parameter and corpus hashes, optimizer
steps/losses, every physical rollout, reset/noise identities, native parity,
training and execution time, solver calls, gate frequency and original videos.
New provider/token usage is zero by construction; historical teacher acquisition
costs are reported separately, never erased. Report per-task paired wins/losses
and complete denominators, rather than only successful videos. Keep failed
training or interrupted evaluation work and withhold incomplete-cohort rates.

Local visual editing remains a separate proposed intervention experiment. This
recipe first tests whether already useful text corrections can become autonomous
behavior and whether a learned gate preserves ID behavior.
