# Dexterous / contact-rich world models, WAMs and policies; learning from play, failures and human hand data

Method note: arxiv.org could not be reached from this session (DNS blocked). No PDFs or abstract pages were fetched. Everything below comes from web-search result summaries of arXiv, OpenReview, project pages and blogs. Numbers marked "reported" are the authors' own claims as the search surfaced them, not checked against the papers' tables. Several 2026 papers (arXiv IDs 26xx) were found only by title or abstract snippet.

## Q1. Which papers learn world models or dynamics for multi-fingered hands, what do they show, and where do they fail?

### Takeaway
There are two lineages. The older one, PDDM (2019), learns dynamics models for MPC on a real 24-DoF hand and names dropping objects and contact making and breaking as the core difficulties. The newer one, from late 2025 to 2026, trains action-conditioned video or latent world models on egocentric human hand data (DexWM, Dexterous World Models, DexAC-WM). These are scored mainly with video and keypoint metrics (FVD, FID, PCK). Robot-transfer evidence is thin, and DexWM's is limited to grasp, reach and place. A tactile world-model cluster appeared in 2026, but this session found only titles and abstracts for it.

### Cited Findings
**Classical learned-dynamics baseline**
- PDDM (Nagabandi, Konolige, Levine, Kumar), "Deep Dynamics Models for Learning Dexterous Manipulation", CoRL 2019 (PMLR v100), arXiv 1909.11652. It does online MPC with an ensemble of deep dynamics models for uncertainty, plus a sampling optimizer that models covariance between action samples. Reported: contact-rich dexterous skills learned on a real 24-DoF hand. — [PMLR](https://proceedings.mlr.press/v100/nagabandi20a.html); [arXiv](https://arxiv.org/abs/1909.11652)
- The PDDM authors frame the difficulty this way. In-hand manipulation and finger gaiting need finely balanced contact forces, repeatedly breaking and re-establishing contacts, and control of unactuated objects. With free-floating objects, most maneuvers drop the object, which creates "sharp discontinuities in the dynamics". Precision is hard to reach when planning with approximate models. — [PMLR PDF](https://proceedings.mlr.press/v100/nagabandi20a/nagabandi20a.pdf)
- A 2025 survey lists PDDM among model-based methods and calls sim-to-real (perception, dynamics and actuation mismatch) the remaining bottleneck. — [Survey arXiv 2510.10903](https://arxiv.org/pdf/2510.10903)

**World models from human hand video (2025–2026)**
- DexWM, "World Models for Learning Dexterous Hand-Object Interactions from Human Videos", arXiv 2512.13644 (Dec 2025, revised Mar 2026). It predicts future latent states conditioned on past states and dexterous actions, where the actions are finger keypoints taken from egocentric video. Training uses more than 900 h of human data plus non-dexterous robot data. Reported finding: predicting visual features alone is not enough, so they add an auxiliary hand-consistency loss to enforce accurate hand configuration. Reported transfer is to unseen robot tasks, but only grasping, reaching and placing. — [arXiv abs](https://arxiv.org/abs/2512.13644); [HTML](https://arxiv.org/html/2512.13644)
- "Dexterous World Models", Byungjun Kim, Taeksoo Kim, Junyoung Lee, Hanbyul Joo, arXiv 2512.17907 (cs.CV, Dec 2025). It is a video diffusion model that takes a static 3D scene plus hand motion and predicts the resulting scene change (grasp, open, move) while keeping camera and scene consistent. Training data is synthetic egocentric interactions plus fixed-camera real videos. The authors call it an early step toward interactive digital twins. The search showed no robot-control evaluation. — [arXiv](https://arxiv.org/abs/2512.17907); [HF](https://huggingface.co/papers/2512.17907)
- DexAC-WM, "Not All Actions Are Equal: Rethinking Conditioning for Dexterous World Model", Zizhao Yuan et al. (10 authors), arXiv 2606.27325 (Jun 2026). It argues that high-DoF dexterous actions are heterogeneous, with values spanning orders of magnitude between arm-scale and finger motions. It keeps per-dimension semantics through action tokenization and aligns actions with visual dynamics through local refinement and global modulation, plus an object-scene semantic branch. It trains on EgoDex and EgoVerse-A. Reported: +6.18 PCK@10 on EgoDex and +3.0 on EgoVerse, with lower FVD; best FID/FVD 106.6/284.40 on EgoDex and 139.60/830.03 on EgoVerse. It is evaluated only on video and keypoint fidelity. — [arXiv](https://arxiv.org/abs/2606.27325); [HTML](https://arxiv.org/html/2606.27325v1)
- Older related work: "Structured World Models from Human Videos" (arXiv 2308.10901), and Wh0, "Generative World Models as Scalable Sources of Egocentric Human Hand Manipulation Data" (arXiv 2606.22136), which uses a world model as a data generator. Only titles were seen. — [2308.10901](https://arxiv.org/pdf/2308.10901); [Wh0](https://arxiv.org/pdf/2606.22136)
- Egocentric WAMs (titles only): EgoWAM, "World Action Models Beyond Pixels with In-the-Wild Egocentric Human Data" (arXiv 2607.08436), and EgoGenesis, "Egocentric World-Action Modeling with Online Anchored Projective Memory and Action-3D RoPE" (arXiv 2607.28243). — [EgoWAM](https://arxiv.org/pdf/2607.08436); [EgoGenesis](https://arxiv.org/pdf/2607.28243)

**Tactile and visuo-tactile world models (mostly 2026)**
- DTWM, "Dexterous Tactile World Model" (Ziyao Zeng, Xiatao Sun, Hao Wang, et al.), arXiv 2609.34286. It is a video world model for egocentric manipulation conditioned on video plus tactile signals from a glove on each hand. Reported finding: conditioning on tactile location and magnitude improves prediction more than a binary contact state does. — [arXiv PDF](https://arxiv.org/pdf/2609.34286)
- TouchWorld (arXiv 2607.07287) is described as a predictive-and-reactive tactile foundation model for dexterous manipulation. Its tactile world model provides future visuo-tactile subgoals for contact-aware execution. — [arXiv](https://arxiv.org/pdf/2607.07287)
- Dream-Tac (arXiv 2606.08737) is a unified tactile world-action model on a pretrained video-generation backbone, evaluated on six real contact-rich tasks. — [arXiv HTML](https://arxiv.org/html/2606.08737v1)
- Titles only, no results seen: ContactWorld (what representations matter for vision-tactile latent WMs, 2606.13877); FeelWorld (hierarchical contact prediction and planning, 2607.24267); TacWAM (2607.28391); ViTacWorld (scaling visuo-tactile WMs, 2607.22530); TacPAC (tactile prediction and real-time action correction in WAMs, 2609.05266); DeCAL (contact-aware latent co-imagination for dexterous VLAs, 2609.09119); Dream-TacVLA (2512.23864), cited by the others. — [ContactWorld](https://arxiv.org/pdf/2606.13877); [FeelWorld](https://arxiv.org/pdf/2607.24267); [TacWAM](https://arxiv.org/pdf/2607.28391); [ViTacWorld](https://arxiv.org/pdf/2607.22530); [TacPAC](https://arxiv.org/pdf/2609.05266); [DeCAL](https://arxiv.org/pdf/2609.09119)
- "When Does Touch Matter? Charting the Vision-Interaction Gap in Cluttered Dexterous Grasping" (arXiv 2609.24068): title only. — [arXiv](https://arxiv.org/pdf/2609.24068)

### Inferences
- The human-video dexterous WMs (DexWM, DexAC-WM, Dexterous World Models) are evaluated mostly on perceptual and keypoint fidelity. Closed-loop dexterous control with these models (in-hand reorientation, finger gaiting) is not shown in what was found. DexWM's robot transfer covers grasp, reach and place, which are coarse tasks.
- DexWM needed a hand-consistency loss and DexAC-WM needed per-dimension action tokenization. Both suggest that generic action conditioning (one MLP-embedded action vector) under-uses finger-level actions. Fine finger actions get swamped by arm motion.
- The discontinuities PDDM described in 2019 (drops, contact switching) have no measured counterpart in the 2025–2026 video-WM papers, as far as the snippets show. Whether video WMs predict drops and slip faithfully is untested in the evidence found.

### Gaps
- No full-text numbers for DexWM's robot results, baselines or failure analysis (arXiv blocked).
- ManipGen and MANO-based hand-object dynamics papers were not searched within budget. Neither was hand-object interaction prediction (for example HOI forecasting).
- No quantitative results for most tactile-WM papers beyond DTWM's qualitative finding.

## Q2. Learning from play: Play-LMP, MimicPlay, C-BeT, dexterous play. Do any evaluate on objectives unseen during play?

### Takeaway
Play-learning results are mostly on parallel-jaw arms. MimicPlay (Franka) tests compositions of trained subgoals that never appeared together, and reports one-shot generalization to new temporal compositions. That is compositional novelty, not truly new objectives. The clearest dexterous result is Play2Perfect (RSS 2026 dexterity workshop). Task-agnostic in-hand play pretraining makes sparse-reward RL for precise assembly work, but only if the play forces finger-level in-hand manipulation. Loose play objectives do not transfer.

### Cited Findings
- MimicPlay, "MimicPlay: Long-Horizon Imitation Learning by Watching Human Play" (Chen Wang et al.), arXiv 2302.12422 (CoRL 2023). A goal-conditioned latent planner trained on human play predicts future 3D hand trajectories from goal images, and a low-level robot controller tracks them. It runs on a Franka arm (not a dexterous hand) across 6 environments and 14 tasks. Reported: more than 50% better than prior methods. The "Study Desk" setting tests compositional generalization: Easy chains two trained tasks, Medium and Hard add new subgoal transitions. They claim one-shot generalization to unseen temporal compositions. — [arXiv](https://arxiv.org/pdf/2302.12422); [project](https://mimic-play.github.io/)
- C-BeT (goal-conditioned Behavior Transformer on teleoperated robot play) is a MimicPlay baseline. MimicPlay notes that C-BeT needs 4.5 h of play data for long-horizon tasks. — [MimicPlay arXiv](https://ar5iv.labs.arxiv.org/html/2302.12422)
- Hyper-GoalNet (arXiv 2512.00085) compares against C-BeT and MimicPlay variants in Robosuite. The snippet shows no dexterous-hand results. — [arXiv](https://arxiv.org/pdf/2512.00085)
- Play2Perfect, "What Matters in Dexterous Play Pretraining for Precise Assembly?", arXiv 2606.26428, RSS 2026 Dexterous Manipulation workshop. It is a two-stage RL pipeline: task-agnostic play with diverse objects in free space, then sparse-reward RL fine-tuning for contact-rich assembly with zero-shot sim-to-real. It ablates object diversity, training objective, trajectory diversity and goal precision. Findings as summarized:
  - Pretraining transfers best when it forces in-hand finger manipulation rather than fixed-grasp transport.
  - Loose play objectives do not reliably transfer to precise assembly.
  - From-scratch RL stays near zero success.
  - One summary reports about 60% zero-shot real success.
  - Hardware: KUKA iiwa 14 with Sharpa hand, and Flexiv Rizon 4s with Wuji Hand 2. Tasks: peg-in-hole, beam assembly, furniture-bench.
  - Sources: [arXiv HTML](https://arxiv.org/html/2606.26428v1); [workshop PDF](https://dex-manipulation.github.io/rss2026/pdfs/19_Play2Perfect_What_Matters_i.pdf); [project](https://play2perfect.github.io/); [Sharpa blog](https://www.sharpa.com/blogs/research/play2perfect-what-matters-in-dexterous-play-pretraining-for-precise-assembly)
- EgoDex's inverse-dynamics benchmark conditions on a goal image of the end state. Reported: visual goal-conditioning significantly improves trajectory prediction. This is goal-conditioned learning from large human hand data, with 194 tasks. — [EgoDex arXiv HTML](https://arxiv.org/html/2505.11709v1)

### Inferences
- In this evidence, "unseen objectives" in play learning has meant new compositions of seen subgoals (MimicPlay) or fine-tuning to a new task with RL (Play2Perfect). It has not meant zero-shot achievement of truly new goals with a dexterous hand.
- Play2Perfect suggests that for hands, the content of the play matters more than its volume. Play that never exercises finger-level contact gives a weak prior for contact-rich tasks. This is relevant to anyone using EgoDex/EgoVerse-style task-agnostic human hand data as world-model pretraining.
- No paper found trains a world model specifically on dexterous play data and then plans for unseen objectives.

### Gaps
- Play-LMP (Lynch et al., "Learning Latent Plans from Play", 2019) was not retrieved this session, so its details and evaluation are unverified here.
- No dexterous-hand play-based world-model paper was found. This may be an absence in the literature or a search limitation.

## Q3. Failure / negative data: effects on policies and world models

### Takeaway
Failure data is used in three ways. It feeds value or advantage signals for VLA policy improvement (π*0.6 RECAP reports more than 2x throughput and about half the failure rate on the hardest tasks). It serves as explicit negative guidance for diffusion and flow VLAs (AFIL/DAG-VLA). And it is increasingly recognized as necessary for world models. Several 2025–2026 papers report that world models trained on mostly-success data (for example OXE) "dream of success": they predict successful grasps even when the arm is far from the object. None of the failure-data work found is dexterous-hand-specific.

### Cited Findings
- π*0.6 / RECAP (Physical Intelligence), "π*0.6: a VLA That Learns From Experience", arXiv 2511.14759 (Nov 2025). A value function flags failures and estimates time to completion. Each action's advantage is binarized into an indicator that the policy is conditioned on, and the conditioning is dropped 30% of the time in training. Reported: more than double the task throughput and roughly half the failure rate on the hardest tasks. The paper admits remaining failures, for example box placing. — [arXiv HTML](https://arxiv.org/html/2511.14759v1); [PI PDF](https://www.pi.website/download/pistar06.pdf)
- Commentary on RECAP, not measured:
  - Advantage labels are only as good as the value estimates behind them. — [blog](https://fusheng-ji.github.io/blog/posts/from-pi0-to-pi07-open-pi-series/)
  - Binarizing discards magnitude, and there is no ablation against continuous conditioning. — [terryum.ai](https://www.terryum.ai/en/posts/2511-pi0-6-recap)
  - RECAP depends on human interventions to find better actions. — [Sarrocco blog](https://federicosarrocco.com/blog/pi-star-06-recap)
- AFIL / DAG-VLA, "Failing Forward: Adaptive Failure-Informed Learning for Vision-Language-Action Models", arXiv 2605.08434. It uses failure trajectories as negative guidance for diffusion and flow VLAs, with a shared VLM backbone learning from both success and failure. It notes that RL approaches (π0.6) and diagnostic systems (RoboFAC) need heavy failure-mode design and in-domain data. — [arXiv HTML](https://arxiv.org/html/2605.08434v1)
- VLAC, "A Vision-Language-Action-Critic Model", arXiv 2509.15937. It outputs reward and action tokens and learns progress estimates from more than 4,000 h of annotated manipulation data. It constructs negative samples to detect regression or stagnation, and is used in asynchronous real-world RL that learns from successes and failures. — [arXiv](https://arxiv.org/pdf/2509.15937)
- World models and failures:
  - "World Models Dream of Success: Diagnosing and Repairing Failure Insensitivity in Robot World Models", arXiv 2610.09134. It diagnoses world models that fail to tell successful actions from failing ones, with repairs in simulation and on a physical robot. Fine-tuned variants up-sample policy failures. It compares policy failures with counterfactual failures on LIBERO. — [arXiv HTML](https://arxiv.org/html/2610.09134)
  - WMPO, arXiv 2511.09515 (ICLR 2026). OXE trajectories are mostly successes, so failures are under-represented. Without fine-tuning on the policy's own rollouts, imagined failures stay "brittle and unfaithful". — [arXiv HTML](https://arxiv.org/html/2511.09515); [ICLR](https://proceedings.iclr.cc/paper_files/paper/2026/file/6605c7d1c1ae1521b328d036df5be1ae-Paper-Conference.pdf)
  - RLVR-World, arXiv 2505.13934. The model tends to predict successful grasps even when the arm is still away from the object. — [arXiv](https://arxiv.org/pdf/2505.13934)
  - PiL-World, arXiv 2606.05773. It fine-tunes the world model on successes plus failed teleoperated trajectories, while the π0 policy being evaluated is trained on successes only. — [arXiv](https://arxiv.org/pdf/2606.05773)
  - dWorldEval, arXiv 2604.22152. It trains on expert successes plus failure rollouts from undertrained checkpoints. Rollout drift causes false negatives that break correlation with real-world performance. — [arXiv](https://arxiv.org/pdf/2604.22152)
  - GigaWorld-1, arXiv 2607.02642. It trains on real, policy-rollout and simulation data, and reports simulating failures such as spilling. — [arXiv HTML](https://arxiv.org/html/2607.02642)
- Failure detection and recovery, titles only, not verified: SAFECAST (2608.04246), FailBench, "How Reliable are VLMs at Judging Robot Task Success?" (2609.03611), RePO-VLA (2605.09410), DreamAvoid (2605.11750), B2FF (2606.09258), "Hide-and-Seek in Trajectories" (2605.30834), FIERCE progress-failure feedback (2609.18651). — [SAFECAST](https://arxiv.org/pdf/2608.04246); [FailBench](https://arxiv.org/pdf/2609.03611); [RePO-VLA](https://arxiv.org/pdf/2605.09410); [DreamAvoid](https://arxiv.org/pdf/2605.11750); [FIERCE](https://arxiv.org/pdf/2609.18651)

### Inferences
- For world models, failure data looks necessary rather than optional. A WM trained on success-only demos (which is what most human hand datasets are, EgoDex included) is likely to predict success-biased futures. That is especially damaging for dexterous tasks, where drops and slips are the dominant failure modes PDDM named.
- For policies, failure data helps mainly through a critic or value or advantage signal, or through negative guidance, not through plain BC.

### Gaps
- No dexterous-hand-specific study of failure data (for example drops or slips in in-hand manipulation) for world models or policies was found.
- No head-to-head numbers on success-only versus mixed data were found across papers. The quantitative effect sizes in AFIL and "Dream of Success" were not retrieved.

## Q4. Human hand video as world-model / WAM vs policy training data

### Takeaway
Human hand data clearly helps policies when co-trained with some robot data: EgoMimic up to +200% reported, HAT about +100% relative OOD gain, Being-H0 data-efficiency gains. Fully zero-robot-data transfer (Phantom) works for coarse and pinch tasks but degrades on precise contact-rich tasks. As world-model data, human hand video (EgoDex, EgoVerse) is now used to train dexterous WMs, but evaluation stays at video and keypoint fidelity. The open issues are retargeting precision, hand-pose estimation under occlusion, the lack of force and tactile signals, and the embodiment gap.

### Cited Findings
- EgoDex (Hoque, Huang, Yoon, Sivapurapu, Zhang; Apple), arXiv 2505.11709, ICLR 2026 poster. Dataset: 829 h at 30 FPS, 90M frames, 338k demos, 194 tabletop tasks, Apple Vision Pro hand tracking. Benchmarks: dexterous trajectory prediction and inverse dynamics with a goal image. Reported: performance scales with data, and visual goal-conditioning significantly improves prediction. Per a secondary summary, about 200M-parameter models were sufficient. Limitations per a secondary summary: mostly tabletop, and pose tracking can be inaccurate under heavy occlusion or fast motion. — [arXiv HTML](https://arxiv.org/html/2505.11709v1); [ICLR](https://iclr.cc/virtual/2026/poster/10010617)
- EgoMimic (Kareer et al., 2024), arXiv 2410.24221. Co-trains on egocentric human video with 3D hand tracking plus teleoperated robot data. Reported: up to 200% relative improvement (project page says 34–228%, so the two need reconciling). It generalizes to objects and scenes seen only in human data. One extra hour of hand data beats one extra hour of robot data. The robot uses a gripper, not a dexterous hand. — [arXiv HTML](https://arxiv.org/html/2410.24221v1); [project](https://egomimic.github.io)
- EgoVLA, arXiv 2507.12440. Pretrains a VLA on egocentric human video with a unified human/robot hand action space, then fine-tunes on few robot demos. Stated limits: it needs hand and wrist pose annotations, and it cannot deploy without fine-tuning on a moderate amount of robot data. — [arXiv](https://arxiv.org/pdf/2507.12440)
- Being-H0, arXiv 2507.15597 (ICML 2026). A dexterous VLA pretrained on large-scale human video hand motion. Real tests used 20 trials per task. Reported to beat GR00T N1.5 and InternVL3, and to match or exceed InternVL3 without hand pretraining using a fraction of the data. Per-task numbers could not be reliably extracted. — [arXiv](https://arxiv.org/abs/2507.15597); [project](https://beingbeyond.github.io/Being-H0/)
- HAT, "Humanoid Policy ~ Human Policy", arXiv 2503.13441. A unified human/humanoid state-action space with differentiable retargeting, co-trained with robot data. Reported: nearly 100% relative OOD improvement over robot-only training, and about 50% higher overall success in one analysis. It names camera-sensor and end-effector-appearance domain gaps. — [arXiv HTML](https://arxiv.org/html/2503.13441)
- Phantom, "Training Robots Without Robots Using Only Human Videos", arXiv 2503.00779. Reported: 92% on the book task and 72% on rope, and up to 84% on a different robot in new settings. Stated limits: hand pose estimation is less precise than teleop, it depends on demo quality and pose precision, and it focuses on pinch grasps. — [arXiv HTML](https://arxiv.org/html/2503.00779v2)
- H2RBench (arXiv 2609.24778) independently finds Phantom competitive on coarse tasks but weaker on precise contact-rich tasks (stacking, insertion). It attributes this to retargeted actions that are not geometrically precise relative to the objects. — [arXiv](https://arxiv.org/pdf/2609.24778)
- DexMV (Qin et al., ECCV 2022, UCSD) pairs a simulated multi-finger hand with a vision pipeline (3D hand and object pose plus retargeting) that turns human video into demos. Reported: it solves tasks RL alone cannot. — [ECCV](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5261_ECCV_2022_paper.php); [arXiv](https://arxiv.org/pdf/2108.05877)
- Other titles, not verified: UniDex (robot foundation suite for dexterous control from egocentric human video, 2603.22264); UniDexTok (unified dexterous hand tokenizer, 2606.10683); human-centric transferable tactile pretraining (2607.01067); Ego-Pi (VLA fine-tuning on egocentric human and robot data, 2606.08107); Human-as-Humanoid (2606.32009); DexImit (bimanual dexterous from monocular human video, 2602.10105). — [UniDex](https://arxiv.org/pdf/2603.22264); [UniDexTok](https://arxiv.org/pdf/2606.10683); [tactile pretrain](https://arxiv.org/pdf/2607.01067); [Ego-Pi](https://huggingface.co/papers/2606.08107.md); [DexImit](https://arxiv.org/pdf/2602.10105)
- World-model use of human hand data: DexWM (900+ h human data, hand-consistency loss), DexAC-WM (EgoDex + EgoVerse-A), DTWM (glove tactile + egocentric video). See Q1. — [DexWM](https://arxiv.org/abs/2512.13644); [DexAC-WM](https://arxiv.org/abs/2606.27325); [DTWM](https://arxiv.org/pdf/2609.34286)

### Inferences
- For policies, the consistent result is that human data plus a little robot data beats robot data alone, and that zero-robot-data transfer breaks down on precision and contact.
- For world models, human data supplies visual dynamics at scale but has no action-to-force grounding. The WM learns "what the hand-object scene looks like next given hand keypoints", not robot-hand contact mechanics. Whether that transfers to planning with a robot hand that has different kinematics and friction is the open question. DexWM's coarse-task transfer is the only evidence found.
- Human demos are almost all successes, so human-data WMs inherit the success bias described in Q3.

### Gaps
- ZeroMimic and DexVLA were not covered by results within budget.
- No paper found directly compares human video as WM training data against the same data as policy training data, on the same dexterous tasks.

## Q5. Stated open problems for dexterous world models

### Takeaway
The sources name these problems: contact discontinuities and object drops (PDDM), heterogeneous high-DoF action scaling (DexAC-WM), hand-configuration fidelity (DexWM), occlusion and fast-motion pose errors in human data (EgoDex), retargeting imprecision on contact-rich tasks (Phantom, H2RBench), failure insensitivity (Dream of Success, WMPO), and tactile sensing (DTWM: graded tactile beats binary contact). Workshop calls (RSS 2026) pose foundation-policy generality and safe compliance of high-DoF hands as open questions.

### Cited Findings
- Contact switching and drops create sharp dynamics discontinuities, and precision is hard with approximate models. — [PDDM](https://proceedings.mlr.press/v100/nagabandi20a/nagabandi20a.pdf)
- High-DoF action heterogeneity: arm versus finger values differ by orders of magnitude and need per-dimension conditioning. — [DexAC-WM](https://arxiv.org/html/2606.27325v1)
- Visual-feature prediction alone does not keep hand configuration accurate. — [DexWM](https://arxiv.org/abs/2512.13644)
- Partial observability and occlusion: hand tracking is inaccurate under heavy occlusion or fast motion (secondary summary of EgoDex limitations). — [liner review](https://liner.com/en/review/egodex-learning-dexterous-manipulation-from-largescale-egocentric-video)
- Tactile: graded tactile location and magnitude conditioning beats binary contact for WM prediction. — [DTWM](https://arxiv.org/pdf/2609.34286)
- Stochastic failures and success bias in WMs. — [Dream of Success](https://arxiv.org/html/2610.09134); [WMPO](https://arxiv.org/html/2511.09515)
- Retargeting precision limits contact-rich transfer from human video. — [H2RBench](https://arxiv.org/pdf/2609.24778)
- The RSS 2026 Dexterous Manipulation workshop (13 July 2026) asks whether one foundation policy can handle everyday dexterous tasks or each needs its own, and how to make high-DoF hands safely compliant during contact. — [workshop](https://dex-manipulation.github.io/rss2026/)
- Related venues: the IROS 2026 tactile-dexterity workshop and the ICRA 2025 dexterity workshop (contents not read). — [IROS26](https://sites.google.com/view/iros26-tactile-dexterity/home); [ICRA25](https://sites.google.com/view/dexterity-workshop-icra2025)

### Inferences
- Planning in high-DoF action spaces with learned dexterous WMs is barely evaluated. None of the 2025–2026 dexterous video WMs found reports MPC or CEM planning success on in-hand tasks, while PDDM (2019) did so with a low-dimensional state model. This is a notable regression in evaluation rigor, and an open opportunity.
- Slip and drop stochasticity and occlusion by the hand are named as problems, but no paper found measures WM accuracy specifically on these events.

### Gaps
- No workshop document was found that lists "contact discontinuity / slip / occlusion for world models" as explicit open problems. These come from individual papers.
- Full workshop topic lists for RSS 2024/2025 and CoRL dexterity workshops were not retrieved.
