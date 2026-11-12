# World Action Models (WAMs) vs. plain policies: claims, ablations, open questions

Method note (read first): in this session arxiv.org, research.nvidia.com and *.github.io could not be fetched (DNS blocked by the session's network allowlist). Everything below comes from search-engine snippets of primary papers and project pages, plus third-party summaries (alphaxiv, emergentmind, liner, themoonlight, papernotes) where noted. Items marked **[unverified]** are known from the paper abstract or prior knowledge but were not re-checked against full text in this session. Exact table numbers should be checked against the PDFs before quoting.

Glossary used here: a **WAM** jointly models future observations (pixels, video latents or compact embeddings) AND actions in one network. The term "World Action Model" is used explicitly by DreamZero (NVIDIA, 2602.15922) and by a 2026 wave of follow-ups (Fast-WAM, GE-Act 2.0, OA-WAM, Motubrain, BadWAM, WISE, and others). There are at least three 2026 WAM surveys (2606.20781, 2609.16074, 2607.00836).

## Q1. Does jointly predicting future video/latents improve action prediction over an equally sized policy, and under what conditions? Are the ablations controlled?

### Takeaway
Many papers report gains. The best-controlled evidence is within-paper ablations that keep the architecture and drop the video objective (GR-1, Seer, VPP, Cosmos Policy, UWM). These usually show gains, which are large on CALVIN-style benchmarks and small on saturated ones (Cosmos Policy reports about +1.5% from future-state/value prediction). Cross-paper "WAM beats VLA" headlines (DreamZero, GigaWorld-Policy) confound backbone, pretraining corpus and data scale. The cleanest regime where WAMs win is generalization: unseen tasks or environments, visual perturbations and diverse non-repetitive data. Camera-viewpoint and initial-state perturbations remain weak points. I found no compute-matched, data-matched negative result.

### Cited Findings
**Within-paper ablations (same architecture, objective removed)**
- GR-1 (Wu et al., ByteDance, ICLR 2024, arXiv 2312.13139) is a GPT-style model that predicts future frames and actions. It was pretrained on Ego4D video, about 0.8M clips. On CALVIN, 5-task-in-a-row success rises from 38.3% to 73.1% and average length from 3.06 to 4.21 versus the prior state of the art. Ablation: a variant trained from scratch without video pretraining but with video prediction still beats the variant with neither on CALVIN. On the real robot, though, it is worse, which the authors attribute to weaker video prediction without pretraining. The full model beats both variants everywhere. — [GR-1 ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/2c37c5bcef24b9541550261dcd63261b-Paper-Conference.pdf); [project page](https://gr1-manipulation.github.io/)
- GR-2 (ByteDance, arXiv 2410.06158) pretrains on about 38M text-video pairs, versus 0.8M for GR-1. The video generation acts as a planner, and actions try to "replay" the predicted video. — [GR-2](https://arxiv.org/pdf/2410.06158); corpus-size comparison from a survey: [2507.01925](https://arxiv.org/pdf/2507.01925)
- Seer, "Predictive Inverse Dynamics Models are Scalable Learners for Robotic Manipulation" (Tian et al., ICLR 2025, arXiv 2412.15109), predicts foresight (future image) plus inverse dynamics end to end and sets a CALVIN ABC-D state of the art with average length 4.28. Its ablation says that combining the conditional visual-foresight objective with inverse-dynamics prediction gives the best results in both pretraining and fine-tuning. The paper also reports "w/o PT" and "scratch" variants, but I did not retrieve their numbers. — [Seer ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/file/e5b5c402bb7bd5e60bede6961d6fe39e-Paper-Conference.pdf); [project](https://nimolty.github.io/Seer/)
- VPP, "Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations" (Hu et al., ICML 2025, arXiv 2412.14803), reports an 18.6% relative improvement on CALVIN ABC-D over the prior state of the art. One ablation row, "VPP w/o Calvin video", gives average length 3.31 at about 140ms. The text suggests the video-model training data matters. — [VPP arXiv](https://arxiv.org/abs/2412.14803); [ICML poster](https://icml.cc/virtual/2025/poster/44705)
- Cosmos Policy (Kim et al., NVIDIA/Stanford, ICLR 2026, arXiv 2601.16163) fine-tunes Cosmos-Predict2-2B with no architecture change. Actions, future state images and values are encoded as latent frames. As a direct policy it gets 98.5% on LIBERO and 67.1% on RoboCasa. A summary reports that predicting future states and values alongside actions improves the policy by +1.5%, which is small on a near-saturated benchmark. — [arXiv](https://arxiv.org/html/2601.16163); [papernotes summary](https://en.papernotes.org/ICLR2026/robotics/cosmos_policy_fine-tuning_video_models_for_visuomotor_control_and_planning/); [liner review](https://liner.com/review/cosmos-policy-finetuning-video-models-for-visuomotor-control-and-planning)
- UWM, "Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets" (Zhu, Yu, Feng, Burchfiel, Shah, Gupta, RSS 2025, arXiv 2504.02792), is one transformer with separate diffusion timesteps for video and action. It reports beating baselines (policy-only pretraining) in-distribution and OOD. Co-training on action-free video improves it further, for example Stack-Bowls from 86% to 92% in that example. DROID pretraining took about 24h on 4×A100 for 100K steps. A design-choice ablation exists (Table VII/VIII), but I did not retrieve its contents. — [UWM arXiv](https://arxiv.org/html/2504.02792v3); [project](https://weirdlabuw.github.io/uwm/); [code](https://github.com/WEIRDLabUW/unified-world-model)
- PAD, "Prediction with Action: Visual Policy Learning via Joint Denoising Process" (Guo et al., NeurIPS 2024, arXiv 2411.18179), jointly diffuses future images and actions in a DiT and can co-train on action-free video. **[unverified]** It reports about 26% relative improvement on Meta-World over prior methods, but its result numbers did not appear in search. — [PAD](https://arxiv.org/abs/2411.18179)
- FLARE, "Robot Learning with Implicit World Modeling" (NVIDIA, CoRL 2025, PMLR v305, arXiv 2505.15659), aligns a few learnable "future tokens" in a DiT policy with compact embeddings of future observations, with no pixel generation. It reports up to 26% over prior baselines on two simulated imitation benchmarks. — [PMLR](https://proceedings.mlr.press/v305/zheng25a.html); [OpenReview](https://openreview.net/forum?id=c8wl1O12GH)

**Cross-model comparisons (confounded)**
- DreamZero, "World Action Models are Zero-shot Policies" (Ye, Ge, …, Fan; NVIDIA GEAR, arXiv 2602.15922), is a 14B WAM on an image-to-video diffusion backbone. On 10 entirely unseen tasks it gets 39.5% average task progress, versus under 1% for from-scratch VLAs and 16.3% for pretrained VLAs. It claims more than 2× generalization to new tasks and environments over state-of-the-art VLAs. A later NVIDIA blog reports 1750 Elo on the April 2026 RoboArena leaderboard versus 1622 for π0.5. Backbone and pretraining differ from the baselines, so this is not a controlled test of the joint-prediction objective. — [DreamZero paper](https://arxiv.org/html/2602.15922v1); [NVIDIA blog](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/)
- DreamZero data ablation, with model and data budget fixed: on AgiBot pick-and-place-easy, diverse data gives 50% ± 6.3% task progress versus 33% ± 4.2% for repetitive data. The authors argue joint video+action prediction is what lets the model learn from diverse, non-repetitive data. This is one task family, and there is no "same data, no video" arm in the snippet. — [DreamZero](https://arxiv.org/pdf/2602.15922)
- GigaWorld-Policy reports about +14% average real-world success over π0.5. In a data-efficiency study, conditioning on WAM video-prior representations improves sample efficiency. — [GigaWorld-Policy](https://arxiv.org/html/2603.17240v2)
- DSWAM explicitly matches robot, pretraining data, post-training data and success criteria against a VLA baseline. It notes that WAM-vs-VLA comparisons are otherwise "dominated by differences in data scale, embodiment, or evaluation protocol". — [DSWAM](https://arxiv.org/pdf/2607.04927)
- "Do World Action Models Generalize Better than VLAs? A Robustness Study" (Huawei, arXiv 2603.22078) introduces LIBERO-Plus and RoboTwin 2.0-Plus, each with 7 perturbation dimensions. WAMs are generally more robust, especially to visual distractors: Cosmos-Policy gets 82.2% on LIBERO-Plus and LingBot-VA 74.2% on RoboTwin 2.0-Plus. π0.5 can match WAM robustness on some tasks but needs far more diverse robot data and objectives. Camera-viewpoint and robot-initial-state perturbations remain hard for WAMs. Hybrid "partial video" models land in between. — [alphaxiv](https://www.alphaxiv.org/abs/2603.22078); [HF papers](https://huggingface.co/papers/2603.22078); [bopi summary](https://bopi-bot.github.io/bopi-research/papers/2603.22078/)
- mimic-video (arXiv 2512.15692) argues control reduces to visual prediction: policy performance scales with video-model quality, shown in an oracle-style analysis. — [mimic-video](https://arxiv.org/pdf/2512.15692)

### Inferences
- Joint prediction appears most valuable when (a) a large video-pretrained backbone is available, (b) evaluation stresses unseen tasks, scenes or visual distractors, and (c) data is diverse rather than repetitive. On saturated in-distribution benchmarks such as LIBERO, the marginal gain is small (about 1.5 points in Cosmos Policy).
- The GR-1 sim-vs-real discrepancy suggests that video prediction without pretraining helps less, or even hurts, in the real world. The "video objective" benefit may come largely from the pretrained prior rather than the joint loss itself.
- No retrieved paper runs a compute-matched comparison that counts video-generation FLOPs, where a policy gets the same parameters and training compute but no video loss. This is a key gap.

### Gaps
- Exact numbers for the UWM design ablation, the Seer w/o pretraining ablation, PAD results and the GR-2 ablations were not retrieved (arXiv blocked).
- I found no published compute-matched, data-matched negative result where removing the video objective does not hurt. Absence may reflect publication bias rather than evidence.

## Q2. Is the benefit from the auxiliary prediction loss (representation learning) or from using the predicted future at inference (planning/guidance)? Which papers separate these?

### Takeaway
Increasingly, the evidence says most of the benefit comes from training-time video supervision, i.e. representation learning. Fast-WAM (arXiv 2603.16666) is the clearest isolation: keeping video co-training but skipping test-time generation stays competitive and runs about 4× faster. Explicit inference-time use of predicted futures adds measurable gains mainly when paired with a value or verifier for search: Cosmos Policy best-of-N gives +12.5 points on real ALOHA, and WISE does test-time scaling. Other separating papers include UVA, VPP, FLARE and DreamZero.

### Cited Findings
- Fast-WAM, "Do World Action Models Need Test-time Future Imagination?" (Yuan et al., arXiv 2603.16666), keeps future-video co-training but masks the future branch at test time. It is about 4× faster (190 ms) with competitive accuracy. Survey 2606.20781 ("World Action Models: A Survey — Dream Less, Act More") frames it as isolating the value of test-time generation once co-training is kept. — [WAM survey 2606.20781](https://arxiv.org/html/2606.20781); [awesome-WAM list](https://github.com/world-action-models/awesome-world-action-models/blob/main/README.md)
- BadWAM (arXiv 2607.15207) independently states that video modeling during training gives substantial benefits even when test-time future generation is removed. RepWAM (2606.13674) notes Fast-WAM is competitive with imagine-then-execute variants. — [BadWAM](https://arxiv.org/pdf/2607.15207); [RepWAM](https://arxiv.org/pdf/2606.13674)
- UVA, "Unified Video Action Model" (Li, Gao, Sadigh, Song, RSS 2025, arXiv 2503.00200), uses a joint video-action latent with decoupled lightweight diffusion heads, so video decoding can be skipped at inference. Action-only inference is about 95ms with 16 steps, versus 100–625ms extra if video is generated. Its speed is comparable to Diffusion Policy with better multi-task performance. — [UVA arXiv](https://arxiv.org/html/2503.00200v1); [UVA PDF](https://unified-video-action-model.github.io/static/UVA_paper.pdf)
- VPP uses predictive representations from a video diffusion model (a single forward/denoising pass) rather than full video generation, which is a representation-use design. — [VPP](https://arxiv.org/pdf/2412.14803)
- FLARE predicts only compact future embeddings, never pixels, and still gets gains. Gains are therefore achievable from latent future-prediction as an auxiliary signal. — [FLARE](https://proceedings.mlr.press/v305/zheng25a.html)
- DreamZero found that at 14B scale, dropping video generation gave minimal speedup, because diffusion steps and DiT blocks dominate latency. It also notes a train/test mismatch: training denoises video and action at the same noise level, while few-step inference produces clean actions with video still noisy. — [DreamZero](https://arxiv.org/html/2602.15922v1)
- Cosmos Policy is the explicit inference-time use case. It samples N action proposals, predicts each one's future state and value (expected reward-to-go) with a planning model refined on rollout data, and executes the argmax. This adds about 12.5 points on real bimanual ALOHA. The released code supports best-of-N only for ALOHA. — [liner](https://liner.com/review/cosmos-policy-finetuning-video-models-for-visuomotor-control-and-planning); [deepwiki of repo](https://deepwiki.com/nvlabs/cosmos-policy)
- WISE, "Seeing Through the Illusion: Test-Time Scaling for World Action Models" (arXiv 2611.03585), keeps the WAM frozen. It scores sampled rollouts on action validity, inverse-dynamics consistency with the imagined motion, and task progress, then executes the best one. — [WISE](https://arxiv.org/abs/2611.03585)
- "What Makes Video World Model Latents Action-Relevant: Prediction over Reconstruction" (arXiv 2606.07687) argues the prediction objective, not reconstruction, makes latents action-relevant. — [2606.07687](https://arxiv.org/html/2606.07687)
- GR-1 and GR-2 condition actions on generated frames inside one autoregressive model. GR-2 frames generation as a planner whose actions "replay" the video. — [GR-2](https://arxiv.org/pdf/2410.06158)
- Older "video-then-IDM" pipelines put the predicted future squarely in the inference path. UniPi (Du et al., NeurIPS 2023, arXiv 2302.00111) does text-conditioned video generation followed by an inverse-dynamics model, at about 24s per inference by one blog's count. Gen2Act (Bharadhwaj et al., arXiv 2409.16283) generates a human video zero-shot and runs a video-conditioned policy. Dreamitate (Liang et al., CoRL 2024, arXiv 2406.16862) fine-tunes video diffusion on human tool-use demonstrations and tracks the tool in generated video. **[UniPi, Gen2Act and Dreamitate details unverified this session]** — [UniPi](https://arxiv.org/abs/2302.00111); [Gen2Act](https://arxiv.org/abs/2409.16283); [Dreamitate](https://arxiv.org/pdf/2406.16862); [UVA blog on UniPi latency](https://medium.com/@zljdanceholic/a-unified-video-action-model-how-uva-folds-video-generation-and-action-prediction-into-one-a698bbd7cf3d)

### Inferences
- The field is converging on "dream during training, act without dreaming" (the 2606.20781 survey's "Video-Generation-Free" category) for latency reasons. That makes WAMs, operationally, policies with a video-prediction auxiliary loss and a video-pretrained init.
- Inference-time use of the future pays off mainly when there is a scorer (value head, verifier or IDM-consistency check). Generated video alone, without selection, is not shown to beat skipping it.
- A clean 2×2 is still needed: {video loss on/off} × {test-time imagination on/off}, at matched compute and data, on OOD tasks. Fast-WAM covers part of this. Its full tables need checking.

### Gaps
- Fast-WAM's ablation numbers, including whether skipping imagination hurts on OOD or long-horizon tasks, were not retrieved.
- I did not find whether Cosmos Policy's +12.5 planning gain was compared against an equal-compute alternative such as more policy samples with a learned Q-function and no video.

## Q3. Action-free data (human video, play) through a WAM, and inference-time cost/latency

### Takeaway
Action-free video is a central selling point. UWM and PAD handle it natively: fix the action branch at max noise, or mask the action loss. Video Policy, FLARE, DreamZero and DreamGen report gains from human or action-free video, mostly on generalization to new objects and tasks. Latency is the dominant practical cost. Full video generation runs at roughly 0.1–6 s per step, which is why a 14B model needed a 38× engineering speedup to reach about 7Hz. Lightweight or skip-video designs reach about 95–190ms.

### Cited Findings
- UWM handles action-free video by fixing the action diffusion timestep at maximum noise. Co-training on action-free video improves in- and out-of-distribution results. — [UWM](https://arxiv.org/html/2504.02792v3); [project](https://weirdlabuw.github.io/uwm/)
- Video Policy, "Video Generators are Robot Policies" (Liang, Tokmakov, Liu, Sudhakar, Shah, Ambrus, Vondrick; Columbia/TRI; arXiv 2508.00795, Aug 2025), trains video prediction first and then actions. Task success is closely tied to generated-video quality, and action-free video data helps generalization to new tasks. It claims strong generalization to unseen objects, backgrounds and tasks from minimal demonstrations. — [arXiv abs](https://arxiv.org/abs/2508.00795); [code](https://github.com/cvlab-columbia/videopolicy)
- FLARE uses action-free human video and reports a large generalization boost to a novel object with unseen geometry from as few as 1 robot demonstration. — [alphaxiv FLARE](https://www.alphaxiv.org/abs/2505.15659)
- DreamZero adapts to a new robot (YAM) with 30 minutes of play data (55 trajectories) while keeping zero-shot generalization. — [DreamZero GitHub](https://github.com/dreamzero0/dreamzero); [emergentmind](https://www.emergentmind.com/papers/2602.15922)
- WAM-TTT, "Steering World-Action Models by Watching Human Play at Test Time" (arXiv 2607.06988), uses human play at test time. Zero-WAM (2608.26103) does in-context world-action modeling from human videos for open-ended task generalization. Search returned titles only; content is unverified. — [WAM-TTT](https://arxiv.org/pdf/2607.06988); [Zero-WAM](https://arxiv.org/pdf/2608.26103)
- DreamGen (Jang et al., NVIDIA, arXiv 2505.12705) takes a different route, not joint modeling. It fine-tunes video world models to synthesize "neural trajectories" and labels them with pseudo-actions from an inverse-dynamics model or latent actions. **[unverified]** It reports a humanoid learning 22 new behaviors from single-task teleop data. — [DreamGen](https://arxiv.org/abs/2505.12705)
- Latency numbers:
  - UVA action head: 15ms (16 steps) or 93ms (100 steps). Total UVA: about 95ms or 173ms. Video generation would add 100ms or 625ms. Measured on RTX 3080 (real) and L40 (sim). — [UVA](https://unified-video-action-model.github.io/static/UVA_paper.pdf)
  - Cosmos Policy: action + future state + value in parallel in 0.61s on 1 H100 with 5 denoising steps. — [liner](https://liner.com/review/cosmos-policy-finetuning-video-models-for-visuomotor-control-and-planning)
  - DreamZero: 5.7s down to 150ms (38× on GB200), via about 9× (H100) or 16× (GB200) systems optimizations plus DreamZero-Flash, which biases video timesteps to high noise and caches velocities to go from 16 to 4 effective DiT steps. That gives about 7Hz closed loop. The authors say diffusion steps remain the primary bottleneck. A separate paper (Vidarc) lists π0.5 at 30Hz for context. — [DreamZero](https://arxiv.org/html/2602.15922v1); [Vidarc](https://arxiv.org/pdf/2512.17661)
  - Fast-WAM: 190ms, about 4× faster than generating. — [awesome-WAM](https://github.com/world-action-models/awesome-world-action-models/blob/main/README.md)
  - VPP: about 140ms (ablation row). — [VPP](https://arxiv.org/pdf/2412.14803)
- Survey 2609.16074 lists "efficient inference" or "real-time inference" as an open challenge. — [survey 2609.16074](https://arxiv.org/abs/2609.16074)

### Inferences
- Gains from human video are reported mostly as qualitative or small-n generalization wins (a novel object, new tasks). I found no scaling curve of success versus hours of action-free human video at fixed robot data.
- Latency forces either action-only decoding or heavy distillation and caching. Either way, the claimed benefit of "planning in imagination" goes unused at deployment in most current systems.

### Gaps
- No retrieved study quantifies human-video vs. robot-video value per hour inside a WAM, or how the embodiment gap affects it. This is relevant for egocentric human data such as EgoVerse.
- Training-compute cost relative to a VLA of matched parameter count is rarely reported. UWM's 24h on 4×A100 for DROID is one data point.

## Q4. Test-time flexibility: goal changes, guidance with new costs, forward/inverse dynamics from one model. Evaluated on genuinely new objectives?

### Takeaway
UWM and UVA (and Motubrain) show by construction that one model can act as policy, forward dynamics, inverse dynamics and video predictor, by choosing which modality's noise level to clamp. Cosmos Policy and WISE use the model's future predictions to rerank actions under a value or verifier. These evaluations mostly check that each mode works on the training task distribution. I found no paper that evaluates guidance toward a genuinely new cost function never seen in training, and the WISE/World Action Verifier line is the closest.

### Cited Findings
- UWM controls per-modality diffusion timesteps, which lets one model act as policy, forward dynamics, inverse dynamics or video generator. **[mode list from paper abstract, unverified this session]** — [UWM](https://weirdlabuw.github.io/uwm/)
- UVA uses masked training, so one model serves as policy, video generator, and forward/inverse dynamics. **[unverified this session]** — [UVA](https://arxiv.org/abs/2503.00200)
- Motubrain (arXiv 2604.27792) uses a UniDiffuser formulation for policy, world modeling, inverse dynamics and more in one model. — [Motubrain](https://arxiv.org/abs/2604.27792)
- Cosmos Policy does best-of-N planning with predicted future state and value, where the value is learned from rollout data on the same tasks. It is not a new objective. — [Cosmos Policy](https://arxiv.org/pdf/2601.16163)
- WISE keeps the WAM frozen and scores rollouts by action validity, IDM consistency with imagined motion, and task progress. "Test-Time Scaling for World Action Models via Zero-Shot Geometric Evaluation" (2607.17454) does zero-shot geometric scoring. — [WISE](https://arxiv.org/abs/2611.03585); [2607.17454](https://arxiv.org/pdf/2607.17454)
- World Action Verifier (arXiv 2604.01985) uses cycle consistency between a sparse inverse model and a forward world model to locate forward-model errors. — [WAV](https://arxiv.org/html/2604.01985)
- AR-WAM (2609.23578) swaps language for visual prompts (target bounding box + operation token), a form of goal re-specification. — [AR-WAM](https://arxiv.org/abs/2609.23578)

### Inferences
- "Flexibility" claims are mostly about the inference modes the model can serve, not about optimizing new objectives. Guidance-style control, e.g. classifier guidance with a new cost such as avoiding a region or minimizing contact force, appears untested in the retrieved papers.

### Gaps
- No retrieved evaluation of WAM guidance toward held-out cost functions or constraints.
- No retrieved quantitative comparison of the UWM/UVA inverse- or forward-dynamics modes against dedicated IDM or FDM models.

## Q5. Reported failure modes

### Takeaway
Named failure modes include:
- video/action inconsistency (GE-Act 2.0's "validity gap", BadWAM's "dream right but act wrong")
- train/test noise-level mismatch (DreamZero)
- weakness to camera-viewpoint and initial-state shifts (robustness study)
- dependence on generated-video quality (Video Policy, mimic-video)
- latency (all)
- no memory or long-horizon evaluation (DreamZero)
- compounding error and objective mismatch (world-model surveys)

Contact-rich or dexterous evidence is thin in what I retrieved.

### Cited Findings
- GE-Act 2.0, "Pretraining and Scaling a World-Action Model for Robotic Manipulation" (arXiv 2609.05588, a successor to AgiBot's Genie Envisioner), names a "validity gap": joint optimization introduces conflicting video and action supervision. It proposes selective optimization. — [GE-Act 2.0](https://arxiv.org/pdf/2609.05588)
- BadWAM, "When World-Action Models Dream Right but Act Wrong" (arXiv 2607.15207), documents a mismatch between correct imagined futures and wrong actions (title-level; details not retrieved). — [BadWAM](https://arxiv.org/pdf/2607.15207)
- DreamZero's train/test mismatch: training denoises both modalities at the same noise level, while few-step inference outputs clean actions with noisy video. — [DreamZero](https://arxiv.org/html/2602.15922v1)
- The robustness study finds camera-viewpoint and robot-initial-state perturbations remain challenging for WAMs. — [2603.22078](https://www.alphaxiv.org/abs/2603.22078)
- Video Policy reports task success tightly coupled to generated-video quality, so failed generations mean failed actions. — [Video Policy](https://arxiv.org/abs/2508.00795)
- Robust-WAM (2608.05903), CSWAM (2609.18462), "Steering Robustness into WAMs via Mechanistic Interpretability" (2607.14943), "The Right Future for Action" (2609.23369) and "Beyond Task Success: Behavioral and Representational Diagnostics for WAM and VLA" (2606.01095) all target WAM OOD or robustness failures. This suggests robustness remains contested. Search returned titles only. — [Robust-WAM](https://arxiv.org/pdf/2608.05903); [CSWAM](https://arxiv.org/pdf/2609.18462); [Steering](https://arxiv.org/pdf/2607.14943); [Right Future](https://arxiv.org/pdf/2609.23369); [Beyond Task Success](https://arxiv.org/pdf/2606.01095)
- DSWAM targets "fine-grained" manipulation, implying prior WAMs were weak there (title-level). — [DSWAM](https://arxiv.org/pdf/2607.04927)
- The robot world-model survey (2605.00080) lists compounding errors and objective mismatch as open challenges. Its project page says benchmarks should measure functional utility for robot learning, not future-frame realism. — [survey 2605.00080](https://arxiv.org/pdf/2605.00080); [project](https://ntumars.github.io/wm-robot-survey/)
- Earlier video-first pipelines such as UniPi suffered error propagation from video to IDM and slow inference, which UVA was designed to address. — [UVA](https://arxiv.org/html/2503.00200v1)

### Inferences
- The recurring core issue is that the visual-prediction objective and the action objective are not guaranteed to agree. Pixel-plausible futures can be action-irrelevant or physically wrong at contacts, which motivates the action-relevant-latent line (2606.07687, 2609.23369, RepWAM).

### Gaps
- No retrieved quantitative study of hallucinated contacts, or of physical inconsistency rates between generated video and executed action.
- Dexterous-hand or contact-rich benchmarks for WAMs versus VLAs were not found in this search.

## Q6. Explicitly stated open questions / limitations, and experiments the community wants

### Takeaway
The surveys converge on six open problems: action alignment and grounding, world-action factorization, spatial and multi-view consistency, long-horizon memory, neural simulation for closed-loop learning, and efficient inference. Embodiment transfer and safe deployment are also listed. Paper-level stated limitations are memory (DreamZero) and latency (DreamZero, UVA's motivation).

### Cited Findings
- "World-Action Models for Robot Learning and Control: A Survey" (arXiv 2609.16074) gives a taxonomy by representation, transition modeling, action interface and architecture. Its six open challenges are action alignment, world-action factorization, spatial/multi-view consistency, long-horizon memory, neural simulation for closed-loop policy learning, and efficient inference. Its evaluation section adds action grounding, real-time inference, embodiment transfer and safe deployment. It cautions that generation fidelity does not show action grounding or control utility. — [survey 2609.16074](https://arxiv.org/abs/2609.16074)
- "World Action Models: A Survey — Dream Less, Act More" (arXiv 2606.20781) defines a model as a WAM only if its predicted future directly helps produce, score, verify or train the action. Its trend is toward generation-free inference. — [2606.20781](https://arxiv.org/html/2606.20781); [taxonomy site](https://world-action-models.github.io/); [X summary](https://x.com/rohanpaul_ai/status/2081627897735840117)
- "From World Models to World Action Models: A Concise Tutorial for Robotics" (2607.00836) notes that a world model alone is not a policy. — [tutorial](https://arxiv.org/pdf/2607.00836)
- DreamZero, stated limitation: it does not evaluate or post-train on tasks that require memory. Diffusion steps remain the latency bottleneck. — [DreamZero](https://arxiv.org/html/2602.15922v1)
- "Think Like a World Model, Act Like a VLA: Distilling World-Model Representations into Compact Robot Policies" (2609.24682) is an emerging direction: distill WAM representations into fast policies. Title-level only. — [2609.24682](https://arxiv.org/pdf/2609.24682)

### Inferences
Experiments the community appears to want, inferred from the gaps above:
1. Compute- and data-matched 2×2 ablations of {video loss} × {test-time imagination} on OOD tasks.
2. Scaling curves of action-free human video hours at fixed robot data.
3. Guidance toward held-out objectives.
4. Contact-rich or dexterous benchmarks.
5. Memory and long-horizon tasks.
6. Measurement of video-action consistency at contacts.
7. Viewpoint-robustness fixes.

### Gaps
- OpenReview reviewer discussions (e.g. for FLARE, at [OpenReview c8wl1O12GH](https://openreview.net/forum?id=c8wl1O12GH) and [rFiLBM4YCh](https://openreview.net/forum?id=rFiLBM4YCh)) could not be read. Reviewer-raised concerns therefore remain uncollected.
- villa-X (Microsoft, arXiv 2507.23682; latent action modeling in VLAs) and the original Genie Envisioner (AgiBot, arXiv 2508.05635; GE-Base video model + GE-Act action decoder + GE-Sim) were not retrieved in detail **[unverified]**: [villa-X](https://arxiv.org/abs/2507.23682); [Genie Envisioner](https://arxiv.org/abs/2508.05635).
- Recent WAM titles found but not read: OA-WAM ([2605.06481](https://arxiv.org/abs/2605.06481); object slots + 16-step flow-matching action chunk), Streaming-WAM ([2609.28927](https://arxiv.org/pdf/2609.28927)), PatchWAM ([2609.25961](https://arxiv.org/pdf/2609.25961)), World Dreamer v5-lite ([2610.37518](https://arxiv.org/abs/2610.37518); 78.8% mean over 4 real multi-robot tasks), SelfWAM ([2608.00725](https://arxiv.org/pdf/2608.00725)), UniNav ([2608.03244](https://arxiv.org/pdf/2608.03244)), GigaWorld-Policy-0.5 ([2607.13960](https://arxiv.org/pdf/2607.13960)), MT-WAM ([2609.21474](https://arxiv.org/pdf/2609.21474)), Veo-Act ([2604.04502](https://arxiv.org/pdf/2604.04502)), VLAW ([2602.12063](https://arxiv.org/pdf/2602.12063)), 4D WAM ([2604.26694](https://arxiv.org/pdf/2604.26694)).
