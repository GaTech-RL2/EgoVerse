Created **16 additional LIBERO task definitions**, in two separate families: eight object–destination compositions and eight source-spatial-relation × destination compositions. These are separate from the paper’s ten Goal OOD and ten Spatial OOD tasks.

**Simulator validation passed: 16 tasks × 2 seeds = 32 checks. Policy evaluation is pending; there is no success-rate claim.**

[Visual task catalog](index.html) · [Task manifest](manifest.json) · [Simulator receipt](validation/receipt.json) · [Existing measured dashboard](../../learned_correction_recipe/dashboard/index.html)

| Family | Task instruction | Definition |
|---|---|---|
| Goal composition | put the milk in the bowl | [BDDL](bddl/astra_goal_composition/put_the_milk_in_the_bowl.bddl) |
| Goal composition | put the butter on the stove | [BDDL](bddl/astra_goal_composition/put_the_butter_on_the_stove.bddl) |
| Goal composition | put the chocolate pudding in the bowl | [BDDL](bddl/astra_goal_composition/put_the_chocolate_pudding_in_the_bowl.bddl) |
| Goal composition | put the alphabet soup on the plate | [BDDL](bddl/astra_goal_composition/put_the_alphabet_soup_on_the_plate.bddl) |
| Goal composition | put the tomato sauce on the plate | [BDDL](bddl/astra_goal_composition/put_the_tomato_sauce_on_the_plate.bddl) |
| Goal composition | put the ketchup on the stove | [BDDL](bddl/astra_goal_composition/put_the_ketchup_on_the_stove.bddl) |
| Goal composition | put the orange juice on top of the cabinet | [BDDL](bddl/astra_goal_composition/put_the_orange_juice_on_top_of_the_cabinet.bddl) |
| Goal composition | put the bbq sauce on top of the cabinet | [BDDL](bddl/astra_goal_composition/put_the_bbq_sauce_on_top_of_the_cabinet.bddl) |
| Spatial composition | pick up the black bowl next to the cookie box and place it on the stove | [BDDL](bddl/astra_spatial_composition/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_stove.bddl) |
| Spatial composition | pick up the black bowl next to the cookie box and place it on top of the cabinet | [BDDL](bddl/astra_spatial_composition/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_top_of_the_cabinet.bddl) |
| Spatial composition | pick up the black bowl next to the ramekin and place it on the stove | [BDDL](bddl/astra_spatial_composition/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_stove.bddl) |
| Spatial composition | pick up the black bowl next to the ramekin and place it on top of the cabinet | [BDDL](bddl/astra_spatial_composition/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_top_of_the_cabinet.bddl) |
| Spatial composition | pick up the black bowl between the plate and the ramekin and place it on the stove | [BDDL](bddl/astra_spatial_composition/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_stove.bddl) |
| Spatial composition | pick up the black bowl between the plate and the ramekin and place it on top of the cabinet | [BDDL](bddl/astra_spatial_composition/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_top_of_the_cabinet.bddl) |
| Spatial composition | pick up the black bowl on the ramekin and place it on the stove | [BDDL](bddl/astra_spatial_composition/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_stove.bddl) |
| Spatial composition | pick up the black bowl on the ramekin and place it on top of the cabinet | [BDDL](bddl/astra_spatial_composition/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_top_of_the_cabinet.bddl) |

Novelty was checked against 150 supplied task files: 130 original LIBERO definitions and the paper’s 20 OOD definitions. The comparison uses declared object types, not instance-name aliases; stove base/cook-region aliases are normalized. For Goal additions, the object type × destination goal is absent from all supplied goal atoms. For Spatial additions, the source relation × destination combination is absent; its individual object/destination pair can be familiar. All correction-teacher tasks used in the learned recipe are contained in the compared paper set.

This establishes novelty relative to those task definitions, not certified novelty relative to the checkpoint’s entire training data. The full training inventory is unknown. Existing object meshes and task templates are reused; these are compositional tasks, not new-object-category tasks or a visual-corruption suite.

Validation runs the real modified LIBERO parser and simulator (MuJoCo 3.2.3, robosuite 1.4.1), using fixed seeds 71 and 73 on CPU. Each reset is checked before and after ten zero-action stabilization steps; both camera observations must be nonblank. A constructed positive witness moves the target object over the destination and checks the actual success predicate. Restoring the saved initial simulator state must be exact and return success to false. These witnesses are teleports, not controller trajectories; they do not establish robotic reachability or policy solvability. The 32 saved states are validation fixtures, not yet a frozen policy-evaluation split. Preview images show seed 71 after stabilization, with the same display orientation as the policy harness.

All tasks inherit LIBERO’s existing goal predicates. For example, the bowl-placement tasks use its `On` contact/support predicate, as in the original cream-cheese-in-bowl and paper wine-in-bowl tasks; this is not a new volumetric-containment test. The original predicates and published task files were not modified.

A future comparison should freeze a new paired reset manifest, run every method on every case with matched noise and a declared action cap, and report these two families separately from the paper benchmark. Keep this task set out of correction training if it is to test transfer. The current recorded-schedule control has no exact-instruction teacher for these tasks and would default to native; any new teacher acquisition or retrieval policy must be declared as a separate method/development split.

The immutable definition-time input is retained in [definition_manifest.json](definition_manifest.json); the simulator receipt binds its exact hash. The public [manifest](manifest.json) adds the completed validation status and receipt pointer. [publication_manifest.json](publication_manifest.json) binds all delivered definitions, previews, states, source snapshots and licenses.

Recreate definitions with `python -m astra_reversal.ood_extensions --original-root ORIGINAL_LIBERO --ood-root PAPER_REPOSITORY --output NEW_DIRECTORY`. Validate with `python -m astra_reversal.validate_ood_extensions --tasks NEW_DIRECTORY --libero-root PAPER_REPOSITORY/third_party/modified_libero --output NEW_VALIDATION_DIRECTORY`, in an environment with the pinned simulator dependencies. Source the project environment before Python tooling, as required by AGENTS.md.

Attribution: Goal templates derive from [QuanyiLi/pi0-text-latent](https://github.com/QuanyiLi/pi0-text-latent), revision `587a6cbf64f16c7b87fa5805dc0ed934192239a4`, including its modified LIBERO. Spatial templates derive from [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), revision `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`. Changes replace the named source object for Goal tasks, change destinations and objects of interest, and provide matching task instructions. Original layouts and distractors are retained. [LIBERO MIT license](LIBERO_LICENSE.txt) and [paper repository Apache 2.0 license](PAPER_REPOSITORY_LICENSE.txt) are included.
