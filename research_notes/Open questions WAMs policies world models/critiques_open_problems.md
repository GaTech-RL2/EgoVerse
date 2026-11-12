# Critiques, position papers, surveys and benchmarks: what the field says is unresolved about world models vs policies vs joint world-action models (2024–2026)

Method note: arxiv.org and github.io were blocked for direct fetch in this session (DNS not allowed), so every finding below comes from search-engine summaries of abstracts/excerpts, alphaXiv/HF/aggregator pages, or company pages. Numbers marked "(secondary)" come from a citing paper or aggregator, not the original paper. Check them against the PDFs before quoting. arXiv IDs are kept so the writer can check them.

## 1. Surveys and position papers: what open problems do they list?

### Takeaway
2025–2026 surveys agree on a stable list of open problems: action grounding/alignment, physical and multi-view consistency, long-horizon memory and drift, closed-loop neural simulation, real-time inference cost, embodiment transfer, missing unified datasets, and above all evaluation. The newest position papers say the field optimises visual fidelity, and that fidelity is not evidence of decision utility.

### Cited Findings
- "A Comprehensive Survey on World Models for Embodied AI" (arXiv 2510.16732, Oct 2025, rev. Nov 2025) proposes a taxonomy along functionality (decision-coupled vs general-purpose), temporal modelling, and spatial representation. Its open problems include no unified large-scale dataset across navigation/manipulation/driving, plus weak long-horizon dynamics and inefficient spatial representations. — [arXiv 2510.16732](https://arxiv.org/html/2510.16732v1)
- "A Survey: Learning Embodied Intelligence from Physical Simulators and World Models" (arXiv 2507.00917, Jul 2025) treats simulators and world models as complementary and lists open challenges. — [arXiv 2507.00917](https://arxiv.org/abs/2507.00917)
- "World-Action Models for Robot Learning and Control: A Survey" (arXiv 2609.16074, Sep 2026; project page rcl-robotics.github.io/Awesome-World-Action-Models) lists these open challenges: action alignment/grounding, world-action factorization, spatial and multi-view consistency, long-horizon memory, neural simulation for closed-loop policy learning, efficient/real-time inference, embodiment transfer, and safe deployment. It contrasts "dual-system" designs (separate world expert and action expert joined at an interface) with "shared-backbone" designs. It names the tension between multimodal future prediction and accurate action decoding. It states that fidelity alone does not show action grounding or control utility, so whether imagined futures improve closed-loop control stays open. Reward-free and offline RL at scale is another open challenge. — [arXiv 2609.16074](https://arxiv.org/html/2609.16074v1)
- "World Model for Robot Learning: A Comprehensive Survey" (arXiv 2605.00080, 2026) notes that world-model-based policies cost far more compute than VLAs in training and inference, and that diffusion video prediction has high latency from iterative denoising. Some works (VLA-JEPA, LAWM, HiLAM) use predictive objectives only to shape latent representations and skip online video rollout at deployment. — [arXiv 2605.00080](https://arxiv.org/html/2605.00080v1)
- "How Should World Models Be Evaluated for Embodied Decision-Making? A Decision-Making-Centric Position" by Yang Yu, Shiyuan Zhang, Yifei Sheng, Haoxiang Ren, Haoxin Lin (arXiv 2606.15032; v1 13 Jun 2026, v2 28 Jun 2026). Its position: for decision-making world models, visual realism matters less than reliable interventional reasoning, policy evaluation, planning and policy optimisation under distribution shift and long-horizon rollout. It proposes an L0–L7 evaluation ladder running from visual plausibility to policy-optimisation utility. L4 is the first genuinely interventional test, and L5–L7 give the most direct evidence of decision usefulness. It warns against collapsing levels into one score. Its survey findings:
  - Many VLA/RL papers report policy success but diagnose the world model only with reconstruction metrics.
  - Fixed-policy ranking agreement is reported in only about a dozen works.
  - WorldArena and RoboWM-Bench suggest perceptual quality does not reliably predict functional performance.
  — [arXiv 2606.15032](https://arxiv.org/abs/2606.15032); [cspaper mirror](https://cspaper.org/openprint/20260726.0001v1)
- "From Generative Engines to Actionable Simulators: The Imperative of Physical Grounding in World Models" (arXiv 2601.15533, Jan 2026) argues that visual realism is an unreliable proxy for world understanding. — [arXiv 2601.15533](https://arxiv.org/pdf/2601.15533)
- "From Generation to Simulation: How Far Are World Models from Being True Simulators?" (Tong Wang et al., arXiv 2608.23070, Aug 2026) maps 200 works (2018–Jun 2026) onto eight simulator capabilities. World models reach "functional substitution" in interaction and controllability for specific scenarios, but fall short of traditional simulators on formal guarantees of physical laws and structured state feedback. By route:
  - Autoregressive: broadest interaction coverage.
  - Diffusion: best visual fidelity.
  - Latent dynamics: supports control but generates visuals poorly.
  - JEPA: efficient (~15x faster) but its output is hard to read.
  It calls for rigorous definitions of capability, gap and evaluation so that "world model as simulator" becomes an agenda rather than a slogan. — [arXiv 2608.23070](https://arxiv.org/pdf/2608.23070)
- Other 2026 roadmap/definition papers seen but not read: "World Models for Embodied Intelligence: From Plausible to Controllable to Actionable" (arXiv 2609.16697), "A Definition and Roadmap for World Models" (arXiv 2607.06401), "From World Models to World Action Models: A Concise Tutorial for Robotics" (arXiv 2607.00836), "From World Action Models to Embodied Brains: A Roadmap for Open-World Physical Intelligence" (arXiv 2607.11689), "World Models for Robotic Manipulation: A Survey" (arXiv 2606.00113), "World Action Models: A Survey" (arXiv 2606.20781), "OpenWorldLib: A Unified Codebase and Definition of Advanced World Models" (arXiv 2604.04707). — [2609.16697](https://arxiv.org/pdf/2609.16697), [2607.06401](https://arxiv.org/html/2607.06401), [2607.00836](https://arxiv.org/pdf/2607.00836), [2607.11689](https://arxiv.org/pdf/2607.11689), [2606.00113](https://arxiv.org/html/2606.00113), [2606.20781](https://arxiv.org/html/2606.20781), [2604.04707](https://arxiv.org/pdf/2604.04707)

### Inferences
- The surveys moved from "world models for embodied AI" (2025) to "world-action models" (mid/late 2026). The unit of study is now the joint model, and the open question is whether the world-prediction part causally helps the action part.
- The most often repeated open problem is evaluation: decision-relevant, interventional, closed-loop evaluation rather than video fidelity. A paper that measures decision utility (policy ranking, planning gain, counterfactual correctness) speaks directly to the stated gap.

### Gaps
- I could not read the full open-problems sections of 2510.16732, 2609.16697 or 2606.00113 (arXiv blocked). Their exact lists of stochasticity/counterfactual problems are unverified.
- I did not confirm what the L0–L7 levels of 2606.15032 are, beyond L0 (visual plausibility), L4 (first interventional) and L7 (policy optimisation).

## 2. Debates: world models vs bigger policies/VLAs; pixel vs latent vs state; video pretraining; generative vs deterministic; model-based vs model-free

### Takeaway
None of these debates is settled. The best controlled evidence, from 2026, is mixed:
- WAMs (video-pretrained world-action models) are more robust to visual perturbations than standard VLAs, but a VLA trained on diverse data (pi0.5) matches or beats them.
- Generated frames seem to help actions mainly through temporal structure rather than pixel detail.
- Auxiliary world-modelling losses do not always help.
- Latent (JEPA) planning is faster but has been compared only to a few pixel baselines.

### Cited Findings
- **WAM vs VLA (controlled robustness study).** "Do World Action Models Generalize Better than VLAs? A Robustness Study" (arXiv 2603.22078, Mar 2026; v3 exists).
  - Authorship is inconsistent across sources. One search summary listed Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine; another summary attributed it to Huawei and collaborators. Verify on the abs page.
  - Setup: LIBERO-Plus and RoboTwin 2.0-Plus under visual and language perturbations.
  - WAMs show a robustness edge, especially against visual distractors. The authors attribute this partly to video pretraining, since VLA backbones are trained mostly on static image-text and lack priors over visual dynamics.
  - Counterexample: pi0.5 reaches comparable or better overall robustness (85.7% on LIBERO-Plus) with diverse robot and web data. The authors say data diversity can compensate for no explicit world modelling.
  - MOTUS had the highest initial-state robustness, 85.0% on RoboTwin 2.0-Plus (secondary).
  - The paper frames the debate as whether the value lies in world modelling or whether policies already model dynamics implicitly.
  — [arXiv 2603.22078](https://arxiv.org/abs/2603.22078); [HF page](https://huggingface.co/papers/2603.22078); [alphaXiv](https://www.alphaxiv.org/abs/2603.22078)
- **Design ablations of WAMs.** "What Matters in Designing World Action Models: An Empirical Study" (arXiv 2609.24048, Sep 2026). Controlled experiments on RoboCasa-GR1, LIBERO and LIBERO-Plus, with real-robot checks on DROID. Findings:
  - Generated future frames influence actions mainly through temporal structure rather than exact visual detail, most strongly under distribution shift.
  - Fixed temporal relations pre-encoded in latents become brittle under shift.
  - Auxiliary world-modelling losses do not always help; it depends on the training schedule.
  - Causal structure, latent representation and objective must be designed together.
  — [arXiv 2609.24048](https://arxiv.org/html/2609.24048)
- "What Matters for World Modeling in Vision-Language-Action Models?" (arXiv 2610.19825, Oct 2026) finds that how well a world representation works depends on the paradigm. 2D semantic representations give more consistent gains, and world modelling beats action-only baselines in several settings. — [arXiv 2610.19825](https://arxiv.org/html/2610.19825)
- **Video pretraining for control, evidence for.** NVIDIA "Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning" (arXiv 2601.16163, Jan 2026).
  - Results: 98.5% average on LIBERO (Spatial 98.1, Object 100, Goal 98.2, Long 97.6) and 93.6% average on 4 ALOHA bimanual tasks.
  - Ablations: training from scratch without the pretrained video model drops average success by 3.9 points; removing the auxiliary policy/world-model objectives costs 1.5 points.
  - Claims to beat diffusion policies, other video-model policies, and VLAs fine-tuned on the same demos.
  - Caveats (secondary, from other papers): 82.2% on LIBERO-Plus and 52.1% on the RoboRecover recovery benchmark; ~2.1B params with ~1413 ms latency.
  — [NVIDIA project page](https://research.nvidia.com/labs/cosmos-lab/cosmos-policy/); [alphaXiv](https://www.alphaxiv.org/abs/2601.16163); caveats via [LaWAM 2606.15768](https://arxiv.org/pdf/2606.15768) and [RoboRecover 2609.28952](https://arxiv.org/pdf/2609.28952)
- **Pixel vs latent (LeCun/JEPA).** LeCun's position (2022 white paper "A Path Towards Autonomous Machine Intelligence") is that pixel reconstruction wastes capacity on unpredictable detail and gives blurry predictions, so the model should predict in latent space. Meta's evidence, V-JEPA 2 (arXiv 2506.09985, Jun 2025):
  - V-JEPA 2-AC was post-trained on under 62 h of unlabelled robot video (DROID).
  - It was deployed zero-shot on Franka arms in two labs for image-goal pick-and-place, with no data from those environments and no task-specific reward.
  - Secondary sources report ~16 s per action for latent planning vs ~4 min for a Cosmos pixel-generation planner. Unverified, and that is one baseline only.
  — [V-JEPA 2 alphaXiv](https://www.alphaxiv.org/abs/2506.09985); critique of its long-horizon and causal-reasoning limits: [The Decoder](https://the-decoder.com/metas-latest-model-highlights-the-challenge-ai-faces-in-long-term-planning-and-causal-reasoning/)
- Follow-ups that question pure latent prediction: "Is Forward Prediction Enough? Physical State Grounding for JEPA World Models" (arXiv 2608.06799), "D-JEPA: A Decision-Aligned Latent World Model" (arXiv 2609.24749), "LeWorldModel: Stable End-to-End JEPA from Pixels" (arXiv 2603.19312). Their titles alone show the open issues: latent predictions may lack physical-state grounding and decision alignment, and end-to-end JEPA training is unstable (collapse). — [2608.06799](https://arxiv.org/pdf/2608.06799), [2609.24749](https://arxiv.org/pdf/2609.24749), [2603.19312](https://arxiv.org/pdf/2603.19312)
- **Planning with generated futures (does sampling plus selection help?).**
  - "Beyond Visual Quality: A Study of Test-Time Planning with World Action Models" (arXiv 2609.24745, Sep 2026, Oxford and collaborators) estimates an oracle upper bound by picking the sampled candidate with the best realised outcome. Selectors based on visual quality, physical consistency or task progression recover little of that headroom; learned value gives a small gain. Conclusion: producing consequential action choices is different from recognising them in generated futures, so evaluate WAM predictions by decision usefulness. — [arXiv 2609.24745](https://arxiv.org/abs/2609.24745)
  - "Is the Future Compatible? Diagnosing Dynamic Consistency in World Action Models" (Ruan et al., arXiv 2605.07514, May 2026) proposes action-state consistency (do predicted actions match predicted state transitions?) as a missing reliability axis. It separates successful from failed rollouts and tracks learned value estimates. A value-free consensus selection uses it. Failure mode: predictions that collapse to a static background score well on consistency while the task fails. — [arXiv 2605.07514](https://arxiv.org/html/2605.07514)
  - The two papers partly conflict: little recoverable gain from future-based selectors vs useful consistency-based selection. Related: World-Coherent Decoding (arXiv 2609.02159) ranks candidates by video surprisal and action-path effort. — [2609.02159](https://arxiv.org/pdf/2609.02159)
- Compact-policy distillation as a middle path: "Think Like a World Model, Act Like a VLA: Distilling World-Model Representations into Compact Robot Policies" (arXiv 2609.24682). The title shows the stance that world-model value may lie in representations, not rollouts. — [arXiv 2609.24682](https://arxiv.org/pdf/2609.24682)

### Inferences
- **Settled-ish:**
  - Video and world-model pretraining gives some benefit to visual robustness and data efficiency (Cosmos Policy ablation, WAM robustness study).
  - Visual fidelity is a poor proxy for control utility (several independent papers).
  - Pixel rollout at test time is expensive.
- **Unsettled:**
  - Whether world modelling helps beyond what data diversity buys (pi0.5 counterexample).
  - Whether explicit future generation at inference is needed, or only as a training signal (temporal-structure finding, latent-only WAMs).
  - Whether test-time planning over sampled futures beats the policy prior (Beyond Visual Quality vs Is the Future Compatible).
  - Pixel vs latent for planning, which has had no matched-compute head-to-head.
- **Generative/multimodal vs deterministic prediction** is raised only as a design tension ("multimodal prediction vs accurate action decoding" in 2609.16074). I found no controlled study isolating stochastic vs deterministic predictors for control.

### Gaps
- No Sergey Levine blog post or recorded debate surfaced. His CoRL 2024 generalist-policies talk is known only through a third-party Medium note ([note](https://medium.com/@jianming.wang07/note-on-generalist-robot-policies-from-sergey-levine-in-corl-2024-46f9b7a90471)), which I did not read.
- No Physical Intelligence-authored post on world models or video prediction was found. PI's public line remains VLAs (pi0.5, pi0.7), and pi0.5 is used in the literature as the "data diversity instead of world model" counterexample ([Black Scarab deep dive](https://www.blackscarab.ai/insights/physical-intelligence-generalist-robot-policy-guide), secondary).
- No Chelsea Finn statement was found in this pass.
- No study compared pure policy vs world-model-plus-planning vs joint WAM on identical data and compute across all three arms. The closest are 2603.22078 (WAM vs VLA, but different pretraining data), 2609.24048 (ablations within WAMs), and 2609.24745 (WAM with and without test-time selection).

## 3. Benchmarks/evaluations for world models and their criticisms: what is measured vs what matters for control

### Takeaway
Benchmarks have moved from perceptual/physics plausibility (WorldModelBench, EWMBench, WorldScore, VBench-style metrics) to functional utility (WorldArena 1/2, WMBench/GigaWorld-1, WorldFailBench, RoboWM-Bench). The functional benchmarks also show that perceptual scores correlate weakly with control usefulness. Policy-evaluation correlation (does the world model rank policies like reality?) has become the de facto functional test. Industrial labs (1X, Google DeepMind) report high correlations on their own policies, but with no shared protocol.

### Cited Findings
- **WorldArena** (Shang et al., Tsinghua FIB lab, arXiv 2602.08971, Feb 2026).
  - Perception side: 16 metrics across 6 sub-dimensions, plus human evaluation.
  - Functional side: world model as synthetic data engine, as policy evaluator (correlation with real/sim outcomes), and as action planner.
  - Combines these into one index, EWMScore.
  - Motivation: prior evaluation focused on perceptual fidelity and overlooked functional utility for downstream decisions.
  - EWMScore correlates only weakly with downstream action-planning success, r = 0.36 (secondary, reported by Motubrain arXiv 2604.27792).
  — [arXiv 2602.08971](https://arxiv.org/abs/2602.08971); [GitHub](https://github.com/tsinghua-fib-lab/WorldArena); r=0.36 via [Motubrain](https://arxiv.org/pdf/2604.27792)
- **WorldArena 2.0** (arXiv 2605.17912, May 2026) extends to more modalities, online RL and sim-to-real testbeds. — [arXiv 2605.17912](https://arxiv.org/html/2605.17912v1)
- **Policy-evaluation correlation on WorldArena** (secondary, from the BWM paper arXiv 2607.29302): Pearson correlation with RoboTwin outcomes is Ctrl-World 0.986, BWM 0.978, WoW 0.959, Cosmos-Predict 2.5 0.483. A general video foundation model correlates poorly as a policy evaluator unless it is adapted. — [BWM arXiv 2607.29302](https://arxiv.org/pdf/2607.29302)
- **GigaWorld-1 / WMBench**, "GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation" (arXiv 2607.02642, Jul 2026): a benchmark for world models as robot policy evaluators that adopts a subset of WorldArena's metrics. — [arXiv 2607.02642](https://arxiv.org/pdf/2607.02642)
- **WorldFailBench**, "Do World Models Understand Policy Failures?" (arXiv 2610.33565, Oct 2026): tests whether world models reproduce policy failures, which policy evaluation needs and which success-biased training data under-represents. — [arXiv 2610.33565](https://arxiv.org/html/2610.33565)
- **EWMBench**: robot-manipulation video quality (visual-scene consistency, motion correctness, semantic alignment/diversity). Not centred on policy-evaluation correlation. **WorldModelBench**: judges video generators on instruction following, commonsense and physics. Position work argues that such benchmarks do not establish policy-evaluation or optimisation utility. — summarised via [2606.15032](https://arxiv.org/html/2606.15032v1)
- **Physical-fidelity diagnostics:** WorldSimProbe, "Diagnosing Simulator Faithfulness in Action-Conditioned World Models for Embodied Manipulation" (arXiv 2608.09298). — [summary](https://tldr.takara.ai/p/2608.09298)
- **1X**, "1X World Model: Evaluating Bits, not Atoms" (technical report; undated in results, likely 2025). It learns a simulator from raw sensor data to evaluate 1X policies across many scenarios and claims high correlation between world-model and real-world evaluations. Example claim: with a true 15% success-rate gap between two policies, a world model with 70% accuracy picks the better policy 90% of the time. The argument is that pairwise ranking, not absolute success prediction, is the useful metric. — [1X report PDF](https://www.1x.tech/1x-world-model.pdf); [1X page](https://www.1x.tech/discover/1x-world-model)
- **Google DeepMind**, "Evaluating Gemini Robotics Policies in a Veo World Simulator" (arXiv 2512.10675, Dec 2025). A fine-tuned, action-conditioned Veo generates the next seconds of video from the policy's actions. Predictions were validated against 1600+ real-world trials across 8 generalist policy checkpoints and 5 tasks, covering nominal, out-of-distribution and safety red-teaming evaluation. — [arXiv 2512.10675](https://arxiv.org/abs/2512.10675); [HF](https://huggingface.co/papers/2512.10675)
- **Other policy-evaluator world models:**
  - "WorldEval: World Model as Real-World Robot Policies Evaluator" (arXiv 2505.19017, May 2025) — [arXiv 2505.19017](https://arxiv.org/html/2505.19017)
  - "Evaluating Robot Policies in a World Model" (arXiv 2506.00613, Jun 2025) — [arXiv 2506.00613](https://arxiv.org/html/2506.00613v1)
  - PiL-World (arXiv 2606.05773), a chunk-wise world model for VLA policy-in-the-loop evaluation — [2606.05773](https://arxiv.org/pdf/2606.05773)
  - "Interactive World Simulator for Robot Policy Training and Evaluation" (arXiv 2603.08546) — [2603.08546](https://arxiv.org/pdf/2603.08546)
  - "Persistent Robot World Models: Stabilizing Multi-Step Rollouts via RL" (arXiv 2603.25685), which targets long-horizon drift — [2603.25685](https://arxiv.org/pdf/2603.25685)
- **Earlier/older benchmark lineage** (named in the task, not re-verified here): WorldScore, PhysBench, VBench. No fresh primary source was pulled for them in this pass.

### Inferences
- **What is measured:**
  - Per-frame and video fidelity (FVD, perceptual metrics).
  - Instruction following and physical-commonsense judgements.
  - More recently, policy-ranking correlation on a few policies and tasks.
- **What matters for control but is rarely measured** (per 2606.15032, 2609.24745 and 2605.07514):
  - Interventional/counterfactual accuracy (same state, different action, correct divergence).
  - Faithful reproduction of failures.
  - Action-state consistency.
  - Ranking agreement across many policies and under distribution shift.
  - Planning gain relative to an oracle.
  - Long-horizon closed-loop drift.
- Policy-evaluation correlations are reported on small policy sets (8 checkpoints for Veo; a handful on RoboTwin), so high Pearson r values are statistically fragile. No benchmark reports confidence intervals over many policies; this is a gap I observed, not a claim from a source.

### Gaps
- I could not verify WorldArena's own correlation numbers or the 77% vs 66% pi0.5 figure that one search summary mentioned.
- No criticisms of WorldScore or PhysBench specifically were retrieved.

## 4. Workshop calls for papers 2025–2026: listed topics/questions

### Takeaway
2026 has a dense workshop circuit on world models for robotics: ICLR 2026 World Models, the Embodied AI Workshop 2026 themed "World Models for Embodied AI", ECCV 2026 Safe World Models, and NeurIPS 2026 Robot Learning with World Models. Their calls list world models for action, for evaluation, and for safety/trustworthiness. Full topic lists could not be fetched.

### Cited Findings
- **ICLR 2026, 2nd Workshop on World Models** (Rio de Janeiro, 27 Apr 2026). Scope: model-based RL, causality, sequential modelling, environment simulation, diffusion models, video generation, 3D reconstruction, spatial intelligence, robotics, embodied AI. — [workshop site](https://sites.google.com/view/iclr-2026-workshop-world-model/home)
- **Embodied AI Workshop 2026 (EAI, CVPR-affiliated series).** Overarching theme "World Models for Embodied AI"; call topics include "World Models for Action and Evaluation". The page's date fields looked inconsistent in search. — [embodied-ai.org](https://embodied-ai.org/)
- **ECCV 2026, "Safe World Models for Trustworthy Embodied AI"** (Malmö, 9 Sep 2026; papers due 15 Jul 2026). Focuses on safety and trustworthiness of world models. — [workshop site](https://trustworthy-world-models.github.io/ECCV2026/)
- **NeurIPS 2026, "Robot Learning with World Models"** (Sydney). Focus is the intersection of world models and robotics; full papers up to 8 pages; deadline 29 Aug / 2 Sep 2026 AoE. — [robowm-ws.github.io](https://robowm-ws.github.io/)

### Inferences
- "World models for evaluation" and "world models for action" now appear explicitly as CfP topics. Safety and trustworthiness (ECCV) is a newer axis.

### Gaps
- The full NeurIPS 2026 RoboWM and EAI 2026 topic lists, panel questions and organiser names could not be fetched (github.io and some sites blocked).
- No ICML 2026, CoRL 2025/2026 or RSS 2025/2026 world-model workshop CfP was retrieved in this pass.
- No OpenReview reviewer discussions were retrieved.

## 5. Practitioner and lab views on the key unanswered question

### Takeaway
- Industry labs with world-model programmes (1X, Google DeepMind, NVIDIA) present policy evaluation and data generation as the near-term use of world models, and back this with correlation evidence on their own systems.
- Meta/LeCun argue for latent predictive models for planning.
- PI's public work keeps scaling VLAs with diverse data, which the literature treats as the main counterexample to "you need explicit world models".

### Cited Findings
- 1X: a world model learned from raw sensor data is used to evaluate policies at scale ("Evaluating Bits, not Atoms"). — [1X](https://www.1x.tech/1x-world-model.pdf)
- Google DeepMind: an action-conditioned Veo is used for nominal, OOD and red-team policy evaluation, validated on 1600+ real trials. — [arXiv 2512.10675](https://arxiv.org/abs/2512.10675)
- NVIDIA: video foundation models (Cosmos) are fine-tuned directly into policies and planners. — [Cosmos Policy](https://research.nvidia.com/labs/cosmos-lab/cosmos-policy/)
- Meta/LeCun: JEPA latent prediction plus planning instead of pixel generation; zero-shot robot planning with V-JEPA 2-AC. — [V-JEPA 2](https://www.alphaxiv.org/abs/2506.09985)
- Robustness-study framing: "whether the primary value lies in world modeling or whether policies may already implicitly model world dynamics". — [arXiv 2603.22078](https://arxiv.org/abs/2603.22078)
- Industry commentary contrasts PI's hierarchical VLM-over-pi0 (Hi Robot) with 1X's "imagine future video, then inverse model to actions" approach. This is a secondary blog. — [Robotensor blog](https://www.robotensor.ai/blog/robotics-in-2026-from-imitation-to-self-improving-physical-intelligence)

### Inferences
- The question practitioners share is when a learned world model is trustworthy enough to replace real-world evaluation or to improve a policy beyond what more diverse data gives. This is a decision-utility question, not a fidelity question.

### Gaps
- No direct quotes from talks by Levine, Finn, LeCun (2025–2026) or PI staff were retrieved. Treat the stances above as drawn from their papers and products, not from statements.

## 6. Controlled studies comparing policy vs world-model-plus-planning vs joint models on the same data

### Takeaway
Few true same-data, three-way comparisons exist. The available controlled evidence comes from ablations within one model family or from cross-family comparisons with unequal pretraining. This is itself a gap the field repeatedly flags.

### Cited Findings
- Cosmos Policy: same robot demos, with vs without video pretraining (−3.9 points without) and with vs without auxiliary world-model/value objectives (−1.5 points); compared against VLAs fine-tuned on the same demos. — [NVIDIA](https://research.nvidia.com/labs/cosmos-lab/cosmos-policy/)
- WAM vs VLA robustness (LIBERO-Plus, RoboTwin 2.0-Plus): WAMs more visually robust; pi0.5 matches with diverse data. Pretraining data differs across arms. — [arXiv 2603.22078](https://arxiv.org/abs/2603.22078)
- WAM design ablations (RoboCasa-GR1, LIBERO, LIBERO-Plus, DROID real): temporal structure matters more than visual detail; auxiliary world-modelling losses help only conditionally. — [arXiv 2609.24048](https://arxiv.org/html/2609.24048)
- Test-time planning with WAMs: oracle selection shows large headroom, but practical selectors recover little. — [arXiv 2609.24745](https://arxiv.org/abs/2609.24745)
- World modelling in VLAs: gains over action-only baselines depend on the paradigm; 2D semantic targets are most consistent. — [arXiv 2610.19825](https://arxiv.org/html/2610.19825)
- OpenWAM (arXiv 2609.07398) is an open, modular platform for systematic WAM pretraining. It could serve as a substrate for matched comparisons; not read. — [arXiv 2609.07398](https://arxiv.org/pdf/2609.07398)

### Inferences
- **Questions a paper could test to fill the gaps:**
  - Matched-data, matched-compute comparison of (a) a policy only, (b) a policy plus a separate world model with planning/selection, (c) a joint WAM, evaluated in-distribution, under visual/language/viewpoint shift, and on recovery.
  - Isolating whether world-model value comes from representation learning (training-time) or from test-time rollouts.
  - Whether generative/stochastic prediction beats deterministic or latent prediction for action quality.
  - Measuring counterfactual/interventional accuracy and policy-ranking agreement, with confidence intervals across many policies, alongside fidelity.
  - Data-diversity controls (the pi0.5 counterexample) to separate "world modelling" from "more varied data".

### Gaps
- No study found that runs all three arms on identical data and compute.
- No study found that cleanly isolates stochastic vs deterministic predictors for manipulation control.
- Evidence on egocentric human video specifically as world-model pretraining for control was not covered in this pass.
