"""Measure each student's speed on identical frozen public observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data import file_sha256, fingerprint, write_json
from .iterate import parse_sample
from .run_student import phase, workers
from .speed import stats
from .student import Backend, SETTINGS, action_reward, decision_seed, split_output
from .student_data import forward_prompt, load_corpus


def selected_states(corpus_path, root):
    states, corpus = load_corpus(corpus_path)
    audit = root/'training/bc/sft/selection.jsonl'
    eligible = {r['state_id'] for r in map(json.loads, audit.read_text().splitlines()) if r['admitted']}
    selected = [s for s in states if s['state_id'] in eligible]
    if len(selected) != len(eligible):
        raise ValueError('Frozen timing states differ from matched warmstart admission')
    return selected, corpus


def worker(args):
    states, corpus = selected_states(args.corpus, args.root)
    stage = json.loads(args.stage.read_text())
    thinking = stage['method'] != 'bc'
    subset = states[args.shard::args.shards]
    path = args.output/f'shard-{args.shard:02d}.jsonl'
    args.output.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    if [s['state_id'] for s in existing] != [s['state_id'] for s in subset[:len(existing)]]:
        raise ValueError('Diagnostic resume state identities differ')
    if len(existing)==len(subset): return
    backend = Backend(stage['checkpoint'])
    _, warmup = backend.generate('Reply with one short word.', seed=0, thinking=thinking)
    write_json(args.output/f'startup-{args.shard}.json',dict(load_seconds=backend.load_seconds,warmup_seconds=warmup))
    for state in subset[len(existing):]:
        prompt = forward_prompt(state, thinking=thinking)
        generated, elapsed = backend.generate(prompt, seed=decision_seed(0,state['state_id'],0,'fixed'),thinking=thinking)
        raw = generated[0]
        try:
            if raw['finish_reason']!='stop':raise ValueError('Truncated generation')
            reasoning, action = split_output(raw['text'],thinking=thinking,cubes=state['cubes'])
            valid,error=True,None
        except (ValueError,TypeError,KeyError) as exception:
            reasoning,action,valid,error='',None,False,str(exception)
        row=dict(state_id=state['state_id'],task=state['task'],stage=stage['id'],
            prompt_sha256=fingerprint(dict(prompt=prompt,thinking=thinking,settings=SETTINGS)),
            latency_seconds=elapsed,generated=raw,valid=valid,error=error,action=action,
            action_match_reward=action_reward(action,state['expert_action']),
            checkpoint_sha256=stage['checkpoint_sha256'])
        with path.open('a') as stream:stream.write(json.dumps(row,sort_keys=True)+'\n')


def paired(reference, student):
    left={r['state_id']:r for r in reference['rows']};right={r['state_id']:r for r in student['rows']}
    if left.keys()!=right.keys() or any(left[k]['prompt_sha256']!=right[k]['prompt_sha256'] for k in left):
        raise ValueError('Fixed-observation speed ratio requires identical states and prompts')
    before=sum(r['latency_seconds'] for r in left.values());after=sum(r['latency_seconds'] for r in right.values())
    return dict(reference=reference['stage'],student=student['stage'],states=len(left),speedup=before/after,
        reference_action_match_reward=reference['action_match_reward'],student_action_match_reward=student['action_match_reward'],
        reference_invalid=reference['invalid'],student_invalid=student['invalid'],
        interpretation='Same public prompts, H100 deployment and sampling settings; full generation latency, including invalid answers')


def driver(args):
    if not (args.root/'train-complete.json').exists():
        raise ValueError('Run this diagnostic after the first training study has published all checkpoints')
    states,_=selected_states(args.corpus,args.root)
    stages=[json.loads(p.read_text()) for p in sorted((args.root/'ready').glob('*.json'))]
    base=next(s for s in stages if s['id']=='base')
    stages.append({**base,'id':'base-no-thinking','method':'bc','label':'Base Qwen3-8B · action-only diagnostic'})
    results=[]
    priority = ['base', 'base-no-thinking', 'joint-e3', 'leact-r1-e3', 'exit-r1-e3', 'bc-e3']
    for stage in sorted(stages,key=lambda s:(priority.index(s['id']) if s['id'] in priority else len(priority),s['id'])):
        output=args.root/'fixed-observations'/stage['id'];spec=output/'stage.json'
        if spec.exists() and json.loads(spec.read_text())!=stage:raise ValueError('Fixed diagnostic stage changed')
        write_json(spec,stage)
        phase(args.root,'fixed','fixed_observations',stage=stage['id'],states=len(states))
        workers('reproductions.leact.builderbench.fixed_observations',['--root',str(args.root),
            '--corpus',str(args.corpus),'--stage',str(spec),'--output',str(output)],output/'logs')
        rows=[json.loads(line) for p in sorted(output.glob('shard-*.jsonl')) for line in p.read_text().splitlines()]
        if len(rows)!=len(states) or {r['state_id'] for r in rows}!={s['state_id'] for s in states}:
            raise ValueError('Fixed diagnostic has incomplete state coverage')
        if any(r['checkpoint_sha256']!=stage['checkpoint_sha256'] for r in rows):
            raise ValueError('Fixed diagnostic contains another checkpoint')
        result=dict(stage=stage['id'],label=stage['label'],states=len(rows),
            decision_seconds=stats(r['latency_seconds'] for r in rows),
            completion_tokens=stats(r['generated']['generated_tokens'] for r in rows),
            action_match_reward=sum(r['action_match_reward'] for r in rows)/len(rows),
            invalid=sum(not r['valid'] for r in rows),thinking=stage['method']!='bc',rows=rows)
        write_json(output/'results.json',result);results.append(result)
        baselines={r['thinking']:r for r in results if r['stage'].startswith('base')}
        comparisons=[paired(baselines[r['thinking']],r) for r in results if not r['stage'].startswith('base')]
        write_json(args.root/'fixed-report.json',dict(complete=len(results)==len(stages),
            states=len(states),cohort='Matched admitted in-domain warmstart states; timing diagnostic, not held-out task evaluation',
            reports=[{k:v for k,v in r.items() if k!='rows'} for r in results],comparisons=comparisons))
    write_json(args.root/'fixed-complete.json',dict(stages=len(stages),states=len(states)))
    phase(args.root,'fixed','complete')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['root','corpus']:parser.add_argument('--'+name,type=Path,required=True)
    for name in ['stage','output']:parser.add_argument('--'+name,type=Path)
    parser.add_argument('--driver',action='store_true')
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=8)
    args=parser.parse_args()
    (driver if args.driver else worker)(args)


if __name__=='__main__':main()
