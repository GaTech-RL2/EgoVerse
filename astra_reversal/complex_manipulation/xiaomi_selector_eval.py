"""Matched native retries and observation-driven Astra method selection."""

import argparse
import copy
import gzip
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from astra_reversal.records import digest
from .worker import write_json
from .xiaomi_eval import Recorder, RecordedEnvironment
from . import xiaomi_teacher as teacher


def video_feedback(video, steps):
    import imageio.v2 as imageio
    with imageio.get_reader(video, input_params=["-threads", "1"]) as reader:
        count = reader.count_frames()
        indices = [round((count-1)*fraction) for fraction in (.25,.5,.75,1)]
        return [{"origin":"prior_rollout", "step":round(steps*index/max(count-1,1)), "state":None,
                 "images":{"left_right_wrist_panorama":teacher.wire.png_wire(reader.get_data(index))}}
                for index in indices]


class Guide:
    def __init__(self, client, output, history, history_context, episode_id):
        self.client, self.output, self.history = client, output, history
        self.history_context, self.episode_id = history_context, episode_id
        self.calls, self.next_review, self.active, self.previous = 0, 0, None, None
        self.previous_images, self.decisions = None, []

    def prepare(self, states, images, instruction, step, last_audit):
        if step < self.next_review:
            return self.active, self.previous_images
        if self.calls == 16:
            self.active = None
            return None, None
        snapshots = copy.deepcopy(self.history)
        if self.previous:
            snapshots.append({**self.previous,"origin":"previous_live"})
        current = {"origin":"current", "step":step, "state":np.asarray(states[-1]).tolist(),
                   "images":{name:teacher.wire.png_wire(images[key][-1]) for name,key in CAMERAS.items()}}
        snapshots.append(current)
        request = teacher.build_request(episode_id=self.episode_id, request_index=self.calls,
            step=step, snapshots=snapshots, context={"original_instruction":instruction,
            "remaining_calls_including_this":16-self.calls, "prior_rollout":self.history_context,
            "previous_decisions":self.decisions[-3:], "last_intervention_audit":last_audit,
            "available_methods":list(teacher.METHODS), "weights_frozen":True})
        folder = self.output / "guidance"
        folder.mkdir(exist_ok=True)
        with gzip.open(folder / f"request_{self.calls:02d}.json.gz", "wt") as stream:
            json.dump(request,stream)
        # A provider error is an incomplete episode, never a native fallback.
        self.calls += 1
        proposal = self.client.propose(request)
        write_json(folder / f"proposal_{self.calls-1:02d}.json",proposal)
        old_images = self.previous_images
        if proposal["method"] in ("image","vei","vli"):
            import imageio.v2 as imageio
            from .xiaomi_interventions import perturb_images, blend_images
            source = perturb_images(images,proposal,old_images)
            for label, views in (("observed",images),("source",source)):
                imageio.imwrite(folder/f"vision_{self.calls-1:02d}_{label}.png",
                                np.concatenate([views[k][-1] for k in CAMERAS.values()],axis=1))
            if proposal["method"] == "image":
                applied = blend_images(images,source,proposal["alpha"])
                imageio.imwrite(folder/f"vision_{self.calls-1:02d}_applied.png",
                                np.concatenate([applied[k][-1] for k in CAMERAS.values()],axis=1))
        self.previous_images = {key:np.array(images[key][-1],copy=True) for key in CAMERAS.values()}
        self.previous = current
        self.active = proposal
        self.next_review = step+proposal["next_review_controls"]
        self.decisions.append({k:proposal[k] for k in ("method","subgoal","alpha","observed_evidence",
                                                     "completion_signal","next_review_controls")})
        # The source remains the prior review frame until this edit expires.
        self.active_previous_images = old_images
        return self.active, old_images


class PolicyClient:
    def __init__(self, engine, recorder, case_index, attempt, guide=None):
        self.engine, self.record, self.guide = engine, recorder, guide
        self.case_index, self.attempt = case_index, attempt

    def infer(self, states, images, instruction):
        r = self.record
        settings = previous = None
        if self.guide:
            settings, previous = self.guide.prepare(states,images,instruction,r.steps,
                                                    self.engine.last_audit if r.queries else {})
            if settings is not None:
                previous = self.guide.active_previous_images
        seed = 20261009 + self.case_index*10000 + self.attempt*1000 + r.queries
        start = time.perf_counter()
        actions = self.engine.infer(states,images,instruction,settings=settings,previous=previous,seed=seed)
        seconds = time.perf_counter()-start
        if actions.ndim != 2 or actions.shape[1]!=12 or len(actions)<16 or not np.isfinite(actions).all():
            raise ValueError("Native decoded action contract violated")
        r.queries += 1
        r.policy_seconds += seconds
        audit = {"query":r.queries,"control_step":r.steps,"seconds":seconds,
                 "actions_sha256":digest(actions),**self.engine.last_audit}
        with (r.directory / "policy_queries.jsonl").open("a") as stream:
            stream.write(json.dumps(audit)+"\n")
        write_json(r.directory / "progress.json", {"queries":r.queries,"control_steps":r.steps,
            "teacher_calls":self.guide.calls if self.guide else 0,"method":audit["method"],
            "elapsed_seconds":time.perf_counter()-r.started})
        return actions


def integration_state(raw):
    import mujoco
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    model, data = raw.sim.model._model, raw.sim.data._data
    value = np.zeros(mujoco.mj_stateSize(model,spec),np.float64)
    mujoco.mj_getState(model,data,value,spec)
    return value


def restore_anchor(env, anchor):
    import mujoco
    raw = env.unwrapped.env
    raw.reset_from_xml_string(anchor["xml"])
    mujoco.mj_setState(raw.sim.model._model,raw.sim.data._data,anchor["integration_state"],
                       mujoco.mjtState.mjSTATE_INTEGRATION)
    raw.sim.forward()
    raw.rng.bit_generator.state = copy.deepcopy(anchor["environment_rng"])
    np.random.set_state(anchor["numpy_rng"])
    random.setstate(anchor["python_rng"])
    return env.unwrapped.get_observation(raw._get_observations(force_update=True))


def fingerprint(env, obs):
    raw = env.unwrapped.env
    return {"model_sha256":hashlib.sha256(raw.sim.model.get_xml().encode()).hexdigest(),
            "state_sha256":digest(raw.sim.get_state().flatten()),
            "integration_state_sha256":digest(integration_state(raw)),
            "environment_rng_sha256":digest(raw.rng.bit_generator.state),
            "observation_sha256":digest(obs),"instruction":obs["annotation.human.task_description"]}


class ReplayEnvironment(RecordedEnvironment):
    def __init__(self, env, recorder, anchor, expected, resets):
        super().__init__(env,recorder)
        self.anchor, self.expected, self.resets = anchor, expected, resets

    def reset(self, *, seed):
        obs = restore_anchor(self.env,self.anchor)
        self.resets["policy_restore_resets"] += 1
        actual = fingerprint(self.env,obs)
        if actual != self.expected:
            write_json(self.recorder.root / "reset_mismatch.json",{"actual":actual,"expected":self.expected})
            raise ValueError("Prospective paired reset failed exact fingerprint checks")
        self.recorder.terminated = False
        self.recorder.reset(self.env,obs,seed)
        write_json(self.recorder.directory / "paired_reset.json",{"verified":True,**actual})
        return obs, {"success":False}

    def close(self):
        pass  # The owned environment closes after both methods and attempts.


def preflight(engine, entry, initial, output):
    import torch
    from .xiaomi_interventions import perturb_images, blend_images
    states = np.repeat(entry.observation_to_state(initial)[None],4,axis=0)
    images = {k:np.repeat(initial[k][None],4,axis=0) for k in entry.CAMERA_KEYS}
    instruction = str(initial["annotation.human.task_description"])
    data = engine.inputs(states,images,instruction)
    torch.manual_seed(991)
    reference = engine.model(**data).actions
    rows = []
    for method in teacher.METHODS:
        write_json(output / "preflight_progress.json",{"testing":method,"completed_checks":rows})
        settings = {**copy.deepcopy(teacher.DEFAULTS),"method":method}
        neutral = engine.normalized(data,settings=settings,source=data,instruction=instruction,seed=991)
        if not torch.equal(reference,neutral):
            raise ValueError("Neutral setting changed native actions: "+method)
        if method == "native":
            rows.append({"method":method,"neutral_bitwise_equal":True})
            continue
        settings["alpha"] = .5
        if method in ("phase_prompt","tei","tli"):
            settings["subgoal"] = "Pick up the corn and put it in the left container."
        elif method in ("image","vei","vli"):
            settings["roi"] = [.2,.2,.8,.8]
        else:
            settings.update(alpha=1,translation_bias=[0,0,.08],noise_sigma=.05)
        active = engine.infer(states,images,instruction,settings=settings,seed=991)
        decoded = engine.processor.decode_action(reference,robot_type="robocasa365")[0,:,:12].float().cpu().numpy()
        change = float(np.max(np.abs(active-decoded)))
        if not np.isfinite(active).all() or change <= 1e-6:
            raise ValueError("Active intervention had no finite measured effect: "+method)
        rows.append({"method":method,"neutral_bitwise_equal":True,"active_action_linf":change,
                     "audit":engine.last_audit.copy()})
        restored = engine.normalized(data,seed=991)
        if not torch.equal(reference,restored):
            raise ValueError("Native model was not restored after intervention: "+method)
    # Quantify numerical reversal drift without calling it an identity operation.
    pure = {**copy.deepcopy(teacher.DEFAULTS),"method":"frs","alpha":1}
    reversed_actions = engine.normalized(data,settings=pure,instruction=instruction,seed=991)
    value = {"passed":True,"policy_actions":0,"methods":rows,
             "unperturbed_full_reversal_normalized_linf":float((reference-reversed_actions).float().abs().max()),
             "unperturbed_full_reversal_active_normalized_linf":float(
                 (reference-reversed_actions)[data["action_mask"].bool()].float().abs().max()),
             "protected_inputs": ["state","action_mask"],"native_restoration_bitwise_equal":True}
    write_json(output / "intervention_preflight.json",value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True,exist_ok="continuation" in protocol)
    from .xiaomi_selector_resume import load_anchor, load_parent, pending_trials
    rows, resets = load_parent(args.output,protocol)
    pending = set(pending_trials(protocol['cases'],rows))
    sys.path.insert(0,"/opt/astra-xiaomi/sources/xiaomi/eval_robocasa365")
    import entry
    import torch
    torch.set_num_threads(2)
    import gymnasium as gym
    import robocasa  # noqa:F401
    from transformers import AutoModel, AutoProcessor
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action
    from astra_reversal.codex_relay import CodexRelayClient, ensure_server
    from .xiaomi_interventions import NativeEngine, CAMERAS as camera_keys
    global CAMERAS
    CAMERAS = camera_keys
    ensure_server()
    # Decode historical evidence before allocating the CUDA model.
    history = {c["id"]:video_feedback(Path("/opt/astra-results/history")/c["id"]/"rollout.mp4",c["old_steps"])
               for c in protocol["cases"]}
    processor = AutoProcessor.from_pretrained("/opt/astra-xiaomi/weights",trust_remote_code=True,use_fast=False)
    model = AutoModel.from_pretrained("/opt/astra-xiaomi/weights",trust_remote_code=True,
                                     attn_implementation="flash_attention_2",dtype=torch.bfloat16).cuda().to(torch.bfloat16)
    builder = entry.EvalClient.__new__(entry.EvalClient)
    builder.crop_ratio = .95
    engine = NativeEngine(model,processor,builder._build_messages)
    with np.load(Path("/opt/astra-results/history")/protocol["cases"][0]["id"]/"initial_observation.npz",allow_pickle=False) as archive:
        initial = {k:archive[k] for k in archive.files if k!="simulator_state"}
    preflight(engine,entry,initial,args.output)

    def save_summary():
        write_json(args.output / "summary.json",{"cases":len(protocol["cases"]),"maximum_episodes":24,
            "completed_episodes":len(rows),"episodes":rows,"resets":resets,
            "control_steps":sum(r["steps"] for r in rows),"reset_free_action_chunks":sum(r["policy_queries"] for r in rows),
            "teacher_calls":sum(r["teacher_calls"] for r in rows)})

    for case_index, case in enumerate(protocol["cases"]):
        if not any(key[0] == case['id'] for key in pending):
            continue
        root = args.output / case["id"]
        reuse_anchor = root.exists()
        root.mkdir(exist_ok=reuse_anchor)
        if get_task_horizon(case["task"]) != case["horizon"]:
            raise ValueError("Native task horizon differs from the protocol")
        counts = {"case":case["id"],"constructor_started":1,"constructor_setup_resets":None,
                  "anchor_selection_resets":0,"diagnostic_restore_resets":0,"policy_restore_resets":0,
                  "origin_workflow":os.environ['ASTRA_RUN_ID'],"reused_parent_anchor":reuse_anchor}
        resets.append(counts)
        save_summary()
        np.random.seed(case["constructor_seed"])
        env = gym.make("robocasa/"+case["task"],split="pretrain",seed=case["constructor_seed"],disable_env_checker=True)
        counts["constructor_setup_resets"] = 1
        try:
            obs,_ = env.reset(seed=case["seed"])
            counts["anchor_selection_resets"] += 1
            raw = env.unwrapped.env
            if raw._check_success():
                raise ValueError("Initially successful anchor is not a valid trial")
            if reuse_anchor:
                anchor, expected = load_anchor(root)
                for _ in range(2):
                    obs = restore_anchor(env,anchor)
                    counts['diagnostic_restore_resets'] += 1
                    if fingerprint(env,obs) != expected:
                        raise ValueError('Continuation did not reproduce the saved initial fingerprint')
                if raw._check_success():
                    raise ValueError('Saved anchor is already successful')
            else:
                anchor = {"xml":raw.sim.model.get_xml(),"integration_state":integration_state(raw),
                          "environment_rng":copy.deepcopy(raw.rng.bit_generator.state),
                          "numpy_rng":np.random.get_state(),"python_rng":random.getstate()}
                restore_anchor(env,anchor)
                counts["diagnostic_restore_resets"] += 1
                # Save the simulator's canonical serialization after its first replay.
                anchor["xml"] = raw.sim.model.get_xml()
                obs = restore_anchor(env,anchor)
                counts["diagnostic_restore_resets"] += 1
                expected = fingerprint(env,obs)
                with gzip.open(root/"anchor_model.xml.gz","wt") as stream:stream.write(anchor["xml"])
                np.savez_compressed(root/"anchor_state.npz",integration_state=anchor["integration_state"])
                write_json(root/"anchor_rng.json",{"environment_rng":anchor["environment_rng"],
                    "numpy_rng":[anchor["numpy_rng"][0],anchor["numpy_rng"][1].tolist(),*anchor["numpy_rng"][2:]],
                    "python_rng":anchor["python_rng"]})
                write_json(root/"anchor.json",expected)
            completed = {arm:any(r['case']==case['id'] and r['arm']==arm and r['success'] for r in rows)
                         for arm in ('native','astra')}
            astra_history = history[case["id"]]
            history_context = {"source":"historical qualification failure","success":False,
                               "steps":case["old_steps"],"exact_model_equivalence_to_current_anchor":False}
            for attempt in (1,2):
                for arm in ("native","astra"):
                    if completed[arm] or (case['id'],arm,attempt) not in pending:continue
                    folder = root/f"{arm}{attempt}"
                    folder.mkdir()
                    record = Recorder(folder,arm,case["task"],case["horizon"])
                    guide = None
                    if arm == "astra":
                        client = CodexRelayClient(model="gpt-6-astra",family="xiaomi_selector",
                            reasoning_effort="medium",response_log=str(folder/"provider.jsonl"),
                            timeout=protocol.get('continuation',{}).get('relay_timeout_seconds',300))
                        guide = Guide(client,folder,astra_history,history_context,f"{case['id']}-astra{attempt}")
                    wrapped = ReplayEnvironment(env,record,anchor,expected,counts)
                    class GymProxy:
                        def make(self,*args,**kwargs):return wrapped
                    native_args = entry.parse_args([])
                    native_args.split, native_args.seed, native_args.num_trials = "pretrain",case["seed"],1
                    native_args.save_videos, native_args.video_stride, native_args.video_fps = True,2,10
                    try:
                        result = entry.evaluate_task(case["task"],0,native_args,
                            PolicyClient(engine,record,case_index,attempt,guide),GymProxy(),get_task_horizon,
                            convert_action,folder,episode_indices=[0],show_progress=False,write_task_stats=False)
                    except Exception as exc:
                        write_json(folder/"incomplete.json",{"type":type(exc).__name__,"message":str(exc),
                            "steps":getattr(record,"steps",0),"queries":getattr(record,"queries",0),
                            "teacher_calls":guide.calls if guide else 0,"counts_as_completed_failure":False})
                        save_summary()
                        raise
                    row = record.finish(result["episodes"][0])
                    row.update(case=case["id"],arm=arm,attempt=attempt,teacher_calls=guide.calls if guide else 0,
                               origin_workflow=os.environ['ASTRA_RUN_ID'],
                               origin_source_revision=os.environ['ASTRA_SOURCE_REVISION'])
                    write_json(folder/"episode.json",row)
                    rows.append(row)
                    completed[arm] = row["success"]
                    save_summary()
                    if arm == "astra" and not row["success"] and attempt == 1:
                        video = folder/case["task"]/f"episode_000_seed_{case['seed']}_failure.mp4"
                        astra_history = video_feedback(video,row["steps"])
                        history_context = {"source":"own guided attempt1 failure","success":False,"steps":row["steps"],
                                           "exact_model_equivalence_to_current_anchor":True,"decisions":guide.decisions}
        finally:
            env.close()
            save_summary()
    write_json(args.output / "completed.json",{"complete":True,"cases":len(protocol["cases"]),"episodes":len(rows)})


if __name__ == "__main__":
    main()
