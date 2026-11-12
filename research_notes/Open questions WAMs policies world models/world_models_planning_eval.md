# Action-conditioned world models for robot manipulation: planning and policy evaluation (2023–2026)

Research note scope and method: these notes came from 16 web searches run in November 2026. arxiv.org and alphaxiv.org could not be reached from this session, which failed with DNS errors, so no full paper text was fetched. Many numbers below come from search-result extracts of the primary arXiv pages. Some come from secondary summaries (emergentmind, themoonlight, liner); where that is the case, the line says so. A writer should check key numbers against the PDFs before quoting them as final.

## Q1. How well do world-model rollouts predict real policy success, and where do they fail?

### Takeaway
Reported correlations between success in the world model and real success are high: Pearson about 0.78–0.99, with low rank violation. They cover small sets of policies, mostly in-distribution scenes, and usually come from the authors themselves. Absolute success is often biased, either optimistic (models "dream of success") or pessimistic (Veo). Correlation drops sharply for novel objects. Models struggle with off-expert or counterfactual actions, contact outcomes and long horizons.

### Cited Findings
**Headline correlation numbers (measured, by the authors)**
- **Evaluating Gemini Robotics Policies in a Veo World Simulator** (Gemini Robotics Team, Google DeepMind; arXiv:2512.10675, Dec 2025, rev. Jan 2026). Veo is fine-tuned so it can be conditioned on the current image plus a sequence of future robot poses. The validation used 1,600+ real trials across 8 Gemini Robotics checkpoints and 5 bimanual tasks. — [arXiv abs](https://arxiv.org/abs/2512.10675v2)
  - Nominal scenes: Pearson 0.88, MMRV 0.03. OOD scenes (made with Gemini 2.5 Flash Image edits): Pearson 0.86, MMRV 0.06. — [arXiv HTML v1](https://arxiv.org/html/2512.10675v1), via search extract
  - Pearson by OOD axis: background 0.91, small distractors 0.86, large distractors 0.77, novel objects 0.56. The authors put the low novel-object figure down to all policies doing poorly there. Predicted absolute success rates are consistently *lower* than real ones. — [arXiv HTML v1](https://arxiv.org/html/2512.10675v1), via search extract
  - The model is also used for "predictive red teaming" of semantic safety in edited scenes. The authors write that "we are still in the early days of video modeling for robotics" (Sec. 7). — [arXiv abs](https://arxiv.org/abs/2512.10675v2)
  - Critique (secondary): the paper gives no confidence intervals, bootstrapping or significance tests, and does not justify per-condition sample sizes. — [Emergent Mind summary](https://www.emergentmind.com/papers/2512.10675)
- **WorldGym: World Model as an Environment for Policy Evaluation**, also titled "Evaluating Robot Policies in a World Model" (Quevedo et al.; ICLR 2026). An action-conditioned video model is rolled out Monte Carlo style with a VLM as the reward model. It reports Pearson r = 0.78 between world-model and real success rates across 17 tasks, and a 3.3% average gap in mean success rate, "near the standard error". — [project page](https://world-model-eval.github.io/abstract); [ICLR 2026 listing](https://mlanthology.org/iclr/2026/quevedo2026iclr-worldgym/)
- **WorldEval: World Model as Real-World Robot Policies Evaluator** (Yaxuan Li, Yichen Zhu, Junjie Wen, Chaomin Shen, Yi Xu; Midea / ECNU; arXiv:2505.19017, May 2025). "Policy2Vec" feeds a policy's latent action embeddings into a pretrained video model; a summary says this is WAN 2.1. Success is judged by a VLM (Gemini-2.0, per summary). The authors claim a "strong correlation" with real-world performance and that it beats real-to-sim baselines. I did not retrieve the exact coefficient. Per a review, unsafe actions produce visibly corrupted or collapsed videos. — [arXiv](https://arxiv.org/abs/2505.19017); [review](https://www.themoonlight.io/review/worldeval-world-model-as-real-world-robot-policies-evaluator)
- **Ctrl-World: A Controllable Generative World Model for Robot Manipulation** (Guo et al.; ICLR 2026; arXiv:2510.10125). A multi-view model initialized from a pretrained video diffusion model and trained on DROID (95k trajectories, 564 scenes). It supports policy-in-the-loop rollouts, and the authors claim it "accurately rank[s]" policies without real rollouts (their Fig. 7). Supervised fine-tuning on synthetic successful imagined trajectories raised policy success by 44.7%. I did not retrieve the exact correlation coefficient. — [arXiv](https://arxiv.org/html/2510.10125v3); [ICLR listing](https://mlanthology.org/iclr/2026/guo2026iclr-ctrlworld/)
- **RoboWorld: Fast and Reliable Neural Simulators for Generalist Robot Policy Evaluation** (arXiv:2607.01060, 2026). Across 8 policies it reports Pearson r = 0.989 and Spearman ρ = 0.970 against the real-world RoboArena leaderboard; full replication costs about 100 H100 GPU-hours. — [arXiv PDF](https://arxiv.org/pdf/2607.01060)
- **RoboScape: Physics-informed Embodied World Model** (arXiv:2506.23135, 2025). Using Diffusion Policy, it reports Pearson 0.953 with the *ground-truth simulator*, not real hardware, and says IRASim, iVideoGPT and Genie correlate much more weakly. — [arXiv](https://arxiv.org/pdf/2506.23135)
- **dWorldEval: Scalable Robotic Policy Evaluation via Discrete Diffusion World Model** (arXiv:2604.22152, Apr 2026). Uses a discrete diffusion world model with sparse keyframe memory. The authors claim that WorldGym, WorldEval and Ctrl-World show weaker correlation and higher rank violation, which they attribute to "insufficient action controllability". This is one group's comparison. — [arXiv](https://arxiv.org/abs/2604.22152); [HTML](https://arxiv.org/html/2604.22152v1)
- **Cosmos-Surg-dVRK** (arXiv:2510.16240) applies a Cosmos world foundation model to automated online evaluation of surgical-robot policies. I did not retrieve its numbers. — [arXiv](https://arxiv.org/pdf/2510.16240)
- 1X frames world-model evaluation of humanoid policies as "Evaluating Bits, not Atoms" (1X World Model Team technical report). I did not retrieve its correlation numbers. — [1X PDF](https://www.1x.tech/1x-world-model.pdf); [1X blog](https://www.1x.tech/discover/world-model-self-learning)

**Where they fail**
- **Off-expert or counterfactual actions.** "Do Robotic World Models Really Follow Actions?" (arXiv:2608.24885, 2026; WorldEcho benchmark) finds current models handle expert actions reasonably well. On off-expert action queries they either ignore the commanded action or produce visually invalid rollouts. — [arXiv](https://arxiv.org/html/2608.24885v1)
- **Optimism bias.** MiraBench (arXiv:2605.29360) reports that models often predict success even under failure-inducing actions, and that visual fidelity is a poor proxy for action fidelity. — [arXiv](https://arxiv.org/html/2605.29360)
- **Action alignment without the right outcome.** WorldOracle (arXiv:2610.18826) benchmarks against real executions and finds that strong action alignment does not guarantee the correct physical outcome, which bears on contact results. — [arXiv](https://arxiv.org/html/2610.18826)
- **Right ranking, wrong scale.** A reproduction study (arXiv:2611.02699) backs a robotic world model's ranking claims but finds its absolute scale wrong. — [arXiv](https://arxiv.org/html/2611.02699)
- **Hallucinated objects.** RoboTrustBench (arXiv:2606.01600) finds instruction-following video models can "complete" tasks by hallucinating absent objects or changing object attributes under counterfactual instructions, and can generate unsafe behaviour under adversarial ones. — [arXiv](https://arxiv.org/pdf/2606.01600)
- **Long horizon and memory.** Dreamer 4's memory is limited to a 9.6 s context, and its inventory predictions can be imprecise. — [Dreamer 4 project](https://danijar.com/project/dreamer4/); [summary](https://www.emergentmind.com/papers/2509.24527). "Persistent Robot World Models: Stabilizing Multi-Step Rollouts via RL" (arXiv:2603.25685) targets rollout drift directly. — [arXiv](https://arxiv.org/pdf/2603.25685)
- **Unstable ranking metrics.** The PRA and MMRV ranking metrics are unstable for small or poorly separated policy sets (arXiv:2606.15032). — [arXiv](https://arxiv.org/abs/2606.15032). Separately, the MMRV-origin paper (SIMPLER, arXiv:2405.05941) argues that Pearson ignores the range of values it is computed over, and that a linear relation is not needed if improvements are preserved. — [arXiv](https://arxiv.org/pdf/2405.05941)

### Inferences
- Reported correlations are not comparable across papers. They differ in policy count (8 to roughly 17), in reference (real hardware vs. simulator for RoboScape), in success judge (VLM vs. human) and in metric. Most come from the model's own authors.
- The pattern across the Veo paper (novel objects 0.56), WorldEcho, MiraBench and RoboTrustBench is consistent. Correlation holds when policies differ in ways visible in-distribution, and breaks when the decisive event is an unusual action, a new object or a subtle contact outcome. Those are exactly the cases where cheap evaluation would be most valuable.
- Bias in absolute success runs in both directions (Veo pessimistic; 1XWM and LIBERO-based models optimistic). So only relative rankings are currently trustworthy, and only within a task.

### Gaps
- I could not retrieve exact correlation coefficients for WorldEval or Ctrl-World, nor the full Sec. 7 limitations of the Veo paper, because arxiv.org was unreachable.
- I found no independent, third-party replication of the Veo, WorldGym or Ctrl-World correlations on a common policy set. The closest is the "Right Ranking, Wrong Scale" reproduction.
- I found no correlation numbers stratified by contact-rich vs. free-space tasks, or by object size.

## Q2. Does the world model need failure, off-policy or exploratory data to rank actions correctly?

### Takeaway
Yes, the evidence increasingly points that way. Models trained mostly on successful demonstrations show optimism or "failure insensitivity". Adding failures, especially *counterfactual* failures (replays of perturbed actions whose outcome is verified), measurably improves success/failure discrimination. Direct evidence that this improves *policy ranking* is still thin.

### Cited Findings
- **World Models Dream of Success: Diagnosing and Repairing Failure Insensitivity in Robot World Models** (arXiv:2610.09134, Oct 2026). Fine-tuning on counterfactual failures gives a success–failure value gap of 0.124, against 0.014 for the same number of freshly collected on-policy failures. Failure counts, successful replay data and training budget were held equal. The method, CureWM, fine-tunes released world models on execution-verified counterfactual replays without changing architecture or objective. On 484 held-out LIBERO failure counterfactuals, "optimism" fell from 80% (fine-tuned on official data) to 30–43% across four CureWM models. Stated limitation: it needs executable counterfactual actions and verifiable outcomes. — [arXiv abs](https://arxiv.org/abs/2610.09134); [HTML](https://arxiv.org/html/2610.09134)
- 1X: "without failure examples, 1XWM generations show optimistic bias towards success." — [1X World Model report](https://www.1x.tech/1x-world-model.pdf)
- **DreamTrue: Action-Faithful Robot World Model with Counterfactual Post-Training** (arXiv:2610.12468) applies counterfactual post-training for action faithfulness. I did not retrieve its numbers. — [arXiv](https://arxiv.org/html/2610.12468)
- WorldEcho (arXiv:2608.24885) shows a gap between expert and off-expert actions. An "expanded-coverage" Cosmos-Predict2.5 variant had lower raw trajectory error (NDTW), which suggests broader action coverage in training helps action following. — [arXiv](https://arxiv.org/html/2608.24885v1)
- Ctrl-World trains on DROID, a large teleoperated dataset with varied scenes. Its policy-improvement loop fine-tunes on *successful* imagined rollouts (+44.7%). — [arXiv](https://arxiv.org/html/2510.10125v3)
- The evaluation position paper (arXiv:2606.15032) treats an "interventional" level of testing (called "counterfactual" in v1) as the minimal bar for calling something a decision world model. — [arXiv](https://arxiv.org/html/2606.15032)

### Inferences
- The "Dream of Success" result suggests *which* failures you train on matters, not only that you include some. Counterfactual perturbations of successful trajectories are about 9x more informative than on-policy failures at equal count. That favours targeted, intervention-style data collection over passive collection of failed rollouts.
- Demonstration-heavy datasets (teleop, egocentric human video) probably leave models optimistic about failure-inducing actions unless they are complemented with perturbation or play data.

### Gaps
- I found no controlled study that measures policy-*ranking* accuracy (MMRV or Spearman) as a function of the fraction of failures in training data.
- I found no quantitative counterfactual-action accuracy results for the large models (Veo, Cosmos, Genie 3).

## Q3. Stochastic vs. deterministic prediction; JEPA latent vs. generative pixel models; mode averaging

### Takeaway
The main argument is theoretical. Deterministic pixel regression averages over modes and blurs; diffusion or generative models represent multimodal futures; JEPA avoids pixels by predicting deterministically in latent space. Empirically, deterministic latent planners (DINO-WM, V-JEPA 2-AC) dominate goal-image *planning*, and generative video models dominate *policy evaluation*. I found no controlled head-to-head study on multimodal robot tasks.

### Cited Findings
- **V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning** (Meta FAIR; arXiv:2506.09985, 2025). V-JEPA 2-AC is post-trained as action-conditioned and deployed zero-shot on Franka arms in two new labs, planning with CEM toward image goals. Reported pick-and-place success: cups 80% vs. 15% for Octo, boxes 65% vs. 10%, with 10 trials per lab. Not compute-matched: V-JEPA 2-AC used 10x more candidate samples. Stated limitations: sensitivity to camera position, long-horizon error accumulation and search-space size, and reliance on image goals rather than language. — [arXiv PDF](https://arxiv.org/pdf/2506.09985); [summary](https://www.themoonlight.io/en/review/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning). Planning cost: about 16 s per step for a single-view predictor, per a follow-up citing it. — [DUET-DINO arXiv:2609.10506](https://arxiv.org/pdf/2609.10506)
- A follow-up (DUET-DINO) reports that an earlier V-JEPA 2-AC* checkpoint beats later trained V-JEPA 2 baselines, which suggests that "longer training on more diverse data degraded spatial planning". — [arXiv](https://arxiv.org/pdf/2609.10506)
- **Reconstruction or Semantics? What Makes a Latent Space Useful for Robotic World Models** (arXiv:2605.06388) frames diffusion's joint denoising over a whole horizon as a way to capture multimodal futures and reduce compounding error. It calls DINO-WM and V-JEPA 2-AC "complementary ... rather than direct competitors" and leaves the JEPA comparison out of scope. — [arXiv](https://arxiv.org/pdf/2605.06388)
- Newer JEPA variants add stochasticity or decision alignment: Flow-JEPA (flow matching for "robust latent dynamics", arXiv:2608.29029), D-JEPA (decision-aligned latent world model, arXiv:2609.24749) and FF-JEPA (long-horizon latent planners, arXiv:2606.09311). — [Flow-JEPA](https://arxiv.org/pdf/2608.29029); [D-JEPA](https://arxiv.org/pdf/2609.24749); [FF-JEPA](https://arxiv.org/pdf/2606.09311)
- Mode-averaging argument (LeCun's framing: the optimal pixel-loss prediction of a stochastic future is the average of all futures). This comes from secondary commentary. — [Ken Huang substack](https://kenhuangus.substack.com/p/world-models-architectures-and-the); [Aman primer](https://aman.ai/primers/world-modeling/)
- **Dreamer 4: Training Agents Inside of Scalable World Models** (Danijar Hafner, Wilson Yan; arXiv:2509.24527, 2025). It is the first agent to get Minecraft diamonds from offline data only, a task needing more than 20k actions. Diamonds were found in 0.7% of 1,000 60-minute episodes, against 0% for offline-finetuned VPT. A secondary review notes the paper has no systematic study of model exploitation or reward hacking. — [project](https://danijar.com/project/dreamer4/); [arXiv PDF](https://arxiv.org/pdf/2509.24527); [summary](https://www.emergentmind.com/papers/2509.24527)

### Inferences
- The latent-deterministic vs. pixel-generative split follows the use case. Planning needs fast, many-sample, differentiable or cheap evaluation of costs. Policy evaluation needs rollouts a VLM or human can judge visually, and needs outcome diversity. Mode averaging matters most for evaluation, where a blurry "average" outcome can hide whether a grasp failed, and for risk-sensitive planning, which needs the tail of outcomes.
- The DUET-DINO observation (more diverse training hurt spatial planning) hints that representation quality for planning is not monotone in pretraining scale.

### Gaps
- I found no controlled benchmark comparing a stochastic and a deterministic latent world model on manipulation tasks with genuinely multimodal outcomes.
- I found no evidence on whether sampling multiple futures from a generative world model improves planning or ranking compared with a single sample.

## Q4. Planning with world models: test-time objectives, new objective classes, and planners for high-dimensional actions

### Takeaway
Nearly all manipulation world-model planning is tested with **goal images**, sometimes with subgoals; language goals are rare. I found **no** work testing genuinely new objective classes zero-shot on a *learned latent or video* world model, such as constraints, risk or avoidance. That work exists only with analytic or explicit dynamics models. The usual planners are CEM and MPPI, with hierarchical or sampling-based kinodynamic variants appearing in 2026.

### Cited Findings
- **Hierarchical Planning with Latent World Models (HWM)** (arXiv:2604.03208, 2026). Zero-shot hierarchical MPC with MPPI at two temporal scales over learned latent world models. Real Franka pick-and-place from a single goal image: 70% success, against 0% for single-level V-JEPA 2-AC (V-JEPA 2-AC scored 0/0/30% across three tasks). The authors claim it is the first world-model planner to show zero-shot, non-greedy real-robot manipulation from pixels with one goal image. On a 75-step Push-T variant, success rose from 17% (plain DINO-WM) to 61%. — [arXiv HTML](https://arxiv.org/html/2604.03208v2); [ResearchGate](https://www.researchgate.net/publication/403530027_Hierarchical_Planning_with_Latent_World_Models)
- Gao, Yang and Kingston (preprint) apply sampling-based kinodynamic planning on an off-the-shelf DINO-WM. In simulation it matches or exceeds CEM and MPPI with up to 13.8x lower planning time; on a real block-pushing robot it improves success by 20%. — via [search extract of HWM-related results](https://arxiv.org/html/2604.03208v2) (the primary link was not retrieved; see Gaps)
- V-JEPA 2-AC lists reliance on image goals as a limitation. — [arXiv PDF](https://arxiv.org/pdf/2506.09985)
- Constrained MPPI (Shield-MPPI with control barrier function quadratic programs, BC-MPPI with a probabilistic constraint layer arXiv:2510.00272, COSMIK-MPPI arXiv:2604.10358) handles safety and avoidance, but with explicit dynamics models. Soft-penalty MPPI can still violate constraints when no good feasible sample is drawn. — [BC-MPPI](https://arxiv.org/pdf/2510.00272); [COSMIK-MPPI](https://arxiv.org/html/2604.10358)
- The 2024 DINO-WM figures, transcribed by a later paper from Maes et al., are 100 / 79 / 74 / 86 across four environments. — [via FIRM-WM arXiv:2609.22816](https://arxiv.org/pdf/2609.22816)
- IRASim plans by simulating candidate trajectories and picking the one with the best predicted value. — [survey arXiv:2605.00080](https://arxiv.org/html/2605.00080v1)

### Inferences
- A flagship test of "planning on objectives not seen in training" is still missing: for example, "reach the goal while keeping the cup upright" or "avoid region X", added as a cost on a frozen latent world model. Latent spaces like DINO's are not built so that arbitrary constraints are easy to express as costs. That is a likely reason the field defaults to goal-image distance.
- High-dimensional action spaces (bimanual, dexterous) are handled by *policies* evaluated in world models (Veo, Ctrl-World) rather than by planners. Sampling planners have been shown mostly on low-dimensional end-effector actions.

### Gaps
- I found no zero-shot constraint, risk or avoidance objective on a learned visual world model for manipulation.
- I found no planner results for action spaces above about 7–14 DoF using latent world models.
- I did not verify the Gao/Yang/Kingston paper title or ID.

## Q5. Model exploitation, uncertainty estimation, and metrics beyond pixel MSE and FVD

### Takeaway
Simulator exploitation is widely named as the central risk of optimizing policies inside learned world models. Ensembles are costly for video models and the theory behind them does not fit well, so video uncertainty quantification is immature: latent probes and hallucination predictors are 2025–26 proposals. The field is converging on decision-centric metrics: action following, success/failure discrimination, and policy ranking (Pearson, Spearman, MMRV, PRA).

### Cited Findings
- A 2026 manipulation world-model survey (arXiv:2606.00113) says that once a policy is optimized in an imagined world, "it may discover artifacts that improve imagined reward without improving real manipulation". It recommends evaluating under "adversarial or optimization-heavy use". — [arXiv](https://arxiv.org/html/2606.00113)
- Classical defenses carried over from offline MBRL: MOPO penalizes imagined reward by ensemble uncertainty; RAMBO-RL trains an adversarial dynamics model. — [survey arXiv:2606.00133](https://arxiv.org/pdf/2606.00133)
- Video world models break standard Bayesian uncertainty assumptions (frames are not i.i.d.), and ensembles are computationally prohibitive. "World Models That Know When They Don't Know" (arXiv:2512.05927) proposes a calibrated probe, C^3, on video latents that localizes hallucinated regions. — [arXiv](https://arxiv.org/html/2512.05927)
- "Hallucination in World Models is Predictable and Preventable" (arXiv:2606.27326) predicts hallucination from the world model's own encoder, dynamics and decoder signals, with no auxiliary networks. — [arXiv](https://arxiv.org/pdf/2606.27326)
- Metrics:
  - The position paper (arXiv:2606.15032) says action-, outcome- and policy-level evidence should outrank artifact quality. A model with good FVD or VLM-judged plausibility should not rank highly if it fails action controllability, reward fidelity or policy ranking. It argues against collapsing metrics into one score. — [arXiv](https://arxiv.org/html/2606.15032)
  - WorldEcho uses integrity-gated error, pose-aware NDTW and a visual-integrity pass rate. Raw NDTW correlated with human action ratings. — [arXiv](https://arxiv.org/html/2608.24885v1)
  - Surveys propose task success, policy-ranking fidelity and executability diagnostics, scored together with action sensitivity and control utility. — [survey arXiv:2609.16697](https://arxiv.org/html/2609.16697v1)
  - Benchmarks include WorldArena (arXiv:2602.08971) and WorldArena 2.0 (arXiv:2605.17912). In WorldArena 2.0, Cosmos-Predict-2.5 (action) and RoboScape did best as RL environments, but no world model beat training on the ground-truth simulator. — [WorldArena 2.0](https://arxiv.org/pdf/2605.17912)
- The 1X World Model Challenge (ICCV 2025 phase) still scored on pixel and token metrics. The winner, Team Revontuli, reached 23.0 dB PSNR on the sampling track (adapted Wan-2.2 TI2V-5B) and Top-500 cross-entropy 6.6386 on the compression track. — [arXiv:2510.07092](https://arxiv.org/html/2510.07092); [1xgpt repo](https://github.com/1x-technologies/1xgpt)

### Inferences
- Policy-level metrics are now standard in evaluation papers. Their statistical fragility (small policy sets, no confidence intervals) is the next issue to fix; per-condition bootstrap confidence intervals and larger policy pools are the obvious remedy.
- I found no measured study of planner exploitation in a manipulation video world model, in the sense of a "planner finds actions that score well in the model and fail in reality". It is asserted, not quantified.

### Gaps
- I found no quantitative optimizer's-curse study (predicted vs. realized return as a function of planning budget) for visual manipulation world models.

## Q6. Explicitly stated open problems and limitations

### Takeaway
The limitations authors state themselves converge on a short list. That list is set out below with citations.

### Cited Findings
- **Action faithfulness off the data manifold.** Models ignore off-expert actions or render them invalidly. — [arXiv:2608.24885](https://arxiv.org/html/2608.24885v1)
- **Optimism or failure insensitivity.** — [arXiv:2610.09134](https://arxiv.org/abs/2610.09134); [MiraBench](https://arxiv.org/html/2605.29360); [1X report](https://www.1x.tech/1x-world-model.pdf)
- **Physical outcome errors despite good action alignment.** — [WorldOracle](https://arxiv.org/html/2610.18826)
- **Novel objects and OOD.** Correlation in the Veo study drops to 0.56. — [arXiv:2512.10675](https://arxiv.org/html/2512.10675v1)
- **Camera sensitivity, long horizon and image-only goals.** These are V-JEPA 2-AC's stated limitations. — [arXiv:2506.09985](https://arxiv.org/pdf/2506.09985)
- **Short memory.** Dreamer 4 has a 9.6 s context and imprecise inventory prediction. — [Dreamer 4](https://danijar.com/project/dreamer4/)
- **Exploitation and unstudied reward hacking.** — [arXiv:2606.00113](https://arxiv.org/html/2606.00113); [Dreamer 4 summary](https://www.emergentmind.com/papers/2509.24527)
- **Uncertainty quantification for video is young.** — [arXiv:2512.05927](https://arxiv.org/html/2512.05927)
- **Unstable ranking metrics on small policy sets.** — [arXiv:2606.15032](https://arxiv.org/abs/2606.15032)
- **Learned simulators do not yet beat true simulators for RL.** — [WorldArena 2.0](https://arxiv.org/pdf/2605.17912)
- **Large-model status.** Cosmos 3 (NVIDIA, arXiv:2606.02800, Jun 2026) is an "omnimodal" world model. Its Cosmos3-Nano-Policy-DROID ranked first on the RoboArena real-world leaderboard as of 30 May 2026, a policy result and not an evaluation-correlation result. — [Cosmos 3 report](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf); [arXiv](https://arxiv.org/pdf/2606.02800)
- **Later surveys** that list open problems: "Robotic Video World Models: A Survey of Applications, Research Challenges, Future Directions" (arXiv:2601.07823); "World Model for Robot Learning: A Comprehensive Survey" (arXiv:2605.00080); "World-Action Models for Robot Learning and Control: A Survey" (arXiv:2609.16074); "World Models for Embodied Intelligence: From Plausible to Controllable to Actionable" (arXiv:2609.16697). — [2601.07823](https://arxiv.org/html/2601.07823v2); [2605.00080](https://arxiv.org/html/2605.00080v1); [2609.16074](https://arxiv.org/pdf/2609.16074); [2609.16697](https://arxiv.org/html/2609.16697v1)
- **Other 2026 successors seen in passing** (not analysed): World-Gymnast (RL in a world model, arXiv:2602.02454), SC3-Eval (arXiv:2606.18610), GigaWorld-1 (a roadmap for policy-evaluation world models, arXiv:2607.02642), XPolicyLab (arXiv:2608.09892), DreamX-Phi 1.0 (arXiv:2608.13489), WALL-SS (arXiv:2608.26239). — [World-Gymnast](https://arxiv.org/html/2602.02454); [SC3-Eval](https://arxiv.org/pdf/2606.18610); [GigaWorld-1](https://arxiv.org/pdf/2607.02642); [WALL-SS](https://arxiv.org/pdf/2608.26239)

### Inferences
- The unresolved core is causal and interventional fidelity: getting the *consequence of an action the data never showed* right, including failure. Ranking among in-distribution policies is largely working; counterfactual and safety-relevant prediction is not.

### Gaps
- **Not researched in this session (no sources retrieved):** PETS/PDDM, TD-MPC2, DreamerV3, IRIS, DIAMOND, Genie 1 and Genie 3, and IRASim's own numbers. Background identifiers that were *not verified here*: PETS arXiv:1805.12114; PDDM arXiv:1909.11652; DreamerV3 arXiv:2301.04104; TD-MPC2 arXiv:2310.16828; IRIS arXiv:2209.00588; DIAMOND arXiv:2405.12399; Genie arXiv:2402.15391; DINO-WM arXiv:2411.04983; IRASim arXiv:2406.14540. Genie 3 (DeepMind, Aug 2025) is a blog/announcement; I found no robot-manipulation policy-evaluation numbers for it.
- **Not found:** World-model workshop calls (ICLR/ICML/CoRL 2025) listing open problems.
- **Not retrieved:** Full text of the Veo paper's Sec. 7 and of the 1X correlation results.
