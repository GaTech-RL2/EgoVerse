@Ryan Punamiya **I’d start with Bench2Dex for dexterous hands, and EmbodiedSWE’s assembly tasks for a bigger jump in planning complexity.** If your Astra-guided setup solves LIBERO-10 in two rollouts, the useful difficulty is having several consequential decisions and failures to diagnose across attempts.

These are the strongest candidates I found:

| Benchmark / simulator | Tasks I’d try | Why it fits your experiment | Main limitation |
|---|---|---|---|
| **Bench2Dex / Isaac Lab** | **Jigsaw Puzzle Assembly** with IIWA7 + Sharpa hands; **Fridge Wine Interhand Pour** with UR5 + RH5DG2 hands | Jigsaw adds spatial assembly. Pouring requires opening the fridge, retrieving a bottle, transferring between hands, pouring, returning it, and closing up. Astra could guide hand assignment, ordering, and recovery. | Stock goals are quite explicit; difficulty may primarily come from control. **Code, demonstrations, and policy checkpoints are released.** [Tasks and evaluation](https://bench2dex.github.io/doc/) |
| **EmbodiedSWE / Isaac Sim + Newton** | **IKEA table assembly**; also investigate its PC motherboard and robot assembly scenes | My strongest candidate for complicated task execution: assembling parts creates dependencies, alignment problems, and intermediate states that can require revising the plan. The suite advertises task horizons up to approximately 30 minutes. | Built around coding agents with simulator access. Using Astra to guide a separate policy requires adapting that interface. IKEA assembly has registered G1 and GR1-T2 configurations with fingered hands; other scenes use different embodiments. [Project](https://embodiedswe.github.io/), [assembly configurations](https://github.com/EmbodiedSWE/EmbodiedSWE/blob/main/robobench/suites/assembly/configs/envs.py) |
| **DexVerse / Isaac Lab** | **OvenBakeSalmon** with two floating Shadow hands; **CleanTable** with a Shadow hand | Oven task combines seasoning prerequisites, loading, closing, and operating the oven. Cleanup combines object assignment, receptacle access, and restoring the final state. Useful foundations for harder semantic variants. | Earlier release: Shadow assets available through gated download; demonstrations still listed as forthcoming. [Code and release status](https://github.com/ycyao216/DexVerse), [task definitions](https://arxiv.org/html/2607.08751v1) |

**My initial pilot would be three tasks: Bench2Dex jigsaw, Bench2Dex fridge pouring, and EmbodiedSWE IKEA assembly.** That covers spatial reasoning, coordination across stages, and assembly dependencies. These are plausible steps up; published results don’t establish how many rollouts *your* Astra-guided system would need.

For **semantic difficulty that you can deliberately scale**, I’d also consider a **custom puzzle cabinet in Isaac Lab with a Shadow hand**:

> Retrieve the requested component, work out which tool and latch sequence opens its compartment, keep another compartment closed, then restore and lock the cabinet.

You can vary tool compatibility, latch dependencies, distractors, and goal constraints. Keep the mechanism fixed across attempts so experience helps. This would require building a task, but it directly tests whether Astra can discover and revise a strategy.

For the pilot, compare **guidance only, learning only, both, and neither**, at 1, 2, 4, 8, and 16 rollouts. Record failed stages as well as success: you want evidence that guidance corrects decisions, alongside improvements in hand control.