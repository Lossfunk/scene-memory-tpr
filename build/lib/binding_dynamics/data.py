"""Matched factual, replacement and swap histories; no queried-answer input."""
from dataclasses import replace
import json
import numpy as np
from .episodes import generate_episode
from .features import history_record
from .interventions import flat_state

def collect_counterfactuals(target,path,scenes,seed,snapshot_steps=(34,64,99),minimum_observations=3):
    if path.exists():raise FileExistsError(path)
    rows=[];states=[];logits=[];ids=[];times=[];targets=[];nodes=[];rejected=[]
    episodes=[];metadata=[]
    def flush():
        if not episodes:return
        # One flush has a common prefix length.
        t=metadata[0][1]
        labels=np.stack([e.input_labels[:t+1] for e in episodes])
        actions=np.stack([e.actions[:t+1] for e in episodes])
        run=target.run(labels,actions)
        hs=flat_state(run.state);ys=run.logits[:,-1].cpu().numpy()
        for e,meta,h,y in zip(episodes,metadata,hs,ys):
            sid,step,pair=meta
            rows.append(history_record(e,step));states.append(h);logits.append(y)
            ids.append(sid);times.append(step);targets.append(e.targets[step]);nodes.append(pair)
        episodes.clear();metadata.clear()
    for t in snapshot_steps:
        for sid in range(scenes):
            e=generate_episode(sid,seed,unique_labels=True)
            h=history_record(e,t)
            eligible=np.setdiff1d(np.flatnonzero(h['counts']>=minimum_observations),e.visits[t:t+2])
            if len(eligible)<2:
                rejected.append([sid,t]);continue
            rng=np.random.default_rng(np.random.SeedSequence([seed,sid,t,311]))
            a,b=map(int,rng.choice(eligible,2,replace=False))
            new=int(rng.choice(np.setdiff1d(np.arange(26),e.labels)))
            single=e.labels.copy();single[a]=new
            swapped=e.labels.copy();swapped[a],swapped[b]=swapped[b],swapped[a]
            for labels in [e.labels,single,swapped]:
                episodes.append(replace(e,labels=labels));metadata.append((sid,t,[a,b]))
            if len(episodes)>=96:flush()
        flush()
        print('Counterfactual extraction',seed,'step',t,'finished',flush=True)
    data={k:np.stack([r[k] for r in rows]).reshape(-1,3,*rows[0][k].shape) for k in rows[0]}
    data.update(states=np.stack(states).reshape(-1,3,1536),logits=np.stack(logits).reshape(-1,3,26),
        scene_ids=np.asarray(ids).reshape(-1,3)[:,0],times=np.asarray(times).reshape(-1,3)[:,0],
        targets=np.asarray(targets).reshape(-1,3),edit_nodes=np.asarray(nodes).reshape(-1,3,2)[:,0])
    for k in ['q','positions','counts']:
        np.testing.assert_array_equal(data[k],np.repeat(data[k][:,:1],3,axis=1))
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**data)
    info={'scenes':scenes,'seed':seed,'groups':len(data['states']),'rejected_scene_steps':rejected,
          'variant_order':['original','single','swap'],'observations_only':True,
          'original_accuracy':float((data['logits'][:,0].argmax(-1)==data['targets'][:,0]).mean())}
    path.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
    return data


def flatten_groups(data):
    return {k:v.reshape(-1,*v.shape[2:]) for k,v in data.items() if k not in ['scene_ids','times','edit_nodes']}

import itertools

def matched_binding_records(target, scenes, seed, boundary=34, minimum_observations=3, **_):
    records, states, families, variants, node_rows, rejected = [], [], [], [], [], []
    chunk_episodes, chunk_meta = [], []
    def flush():
        if not chunk_episodes:
            return
        labels = np.stack([e.input_labels[:boundary+1] for e in chunk_episodes])
        actions = np.stack([e.actions[:boundary+1] for e in chunk_episodes])
        result = flat_state(target.run(labels, actions).state)
        for e, (family, variant, nodes), state in zip(chunk_episodes, chunk_meta, result):
            records.append(history_record(e, boundary))
            states.append(state)
            families.append(family); variants.append(variant); node_rows.append(nodes)
        chunk_episodes.clear(); chunk_meta.clear()
    for scene_id in range(scenes):
        e = generate_episode(scene_id, seed, unique_labels=True)
        h = history_record(e, boundary)
        nodes = np.setdiff1d(np.flatnonzero(h['counts']>=minimum_observations), e.visits[boundary:boundary+2])
        if len(nodes)<3:
            rejected.append(scene_id)
            continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, scene_id, 9472]))
        nodes = np.sort(rng.choice(nodes, 3, replace=False))
        for variant, assignment in enumerate(itertools.permutations(e.labels[nodes])):
            labels = e.labels.copy(); labels[nodes] = assignment
            chunk_episodes.append(replace(e, labels=labels))
            chunk_meta.append((scene_id, variant, nodes))
        if len(chunk_episodes)>=96:
            flush()
    flush()
    data = {k: np.stack([r[k] for r in records]) for k in records[0]}
    data.update(states=np.stack(states), scene_ids=np.asarray(families), variant_ids=np.asarray(variants), permuted_nodes=np.asarray(node_rows))
    # Verify that no instantaneous input or geometry can reveal the assignment.
    for key in ['q', 'positions', 'mask', 'counts']:
        grouped = data[key].reshape(-1, 6, *data[key].shape[1:])
        np.testing.assert_array_equal(grouped, np.repeat(grouped[:, :1], 6, axis=1))
    inv = np.sort(data['labels'].reshape(-1, 6, 6), axis=-1)
    np.testing.assert_array_equal(inv, np.repeat(inv[:, :1], 6, axis=1))
    return data, rejected
