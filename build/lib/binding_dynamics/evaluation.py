from types import SimpleNamespace
import numpy as np
from .episodes import generate_episode
from .interventions import probe_path,recurrent_state
from .data import flatten_groups
from .metrics import intervals


def boundary_data(data,boundary=34):
    keep=data['times']==boundary
    return {k:v[keep] for k,v in data.items()}


def variant_data(data,v):
    return {k:a[:,v] for k,a in data.items() if k not in ['scene_ids','times','edit_nodes']}


def model_edits(model,data,mode='raw_difference'):
    factual=variant_data(data,0);actual=data['states'][:,0]
    result=[]
    for variant in [1,2]:
        changed=variant_data(data,variant)
        if hasattr(model,'edit'):h=model.edit(actual,factual,changed,mode)
        else:h=actual+(model.predict(changed)-model.predict(factual))
        result.append(h)
    return np.stack(result,axis=1)


def make_probe_inputs(data,seed,delay):
    all_labels=[];all_actions=[];all_nodes=[];queries=[];answers=[];target_mask=[]
    for sid,pair,labels in zip(data['scene_ids'],data['edit_nodes'],data['labels']):
        e=generate_episode(int(sid),seed,unique_labels=True)
        qset=np.setdiff1d(np.arange(6),[e.visits[35]])
        rows=[]
        for variant in [1,2]:
            edited=list(pair[:variant]);paths=[]
            for q in qset:
                blocked=tuple(sorted(set([*edited,int(q)])))
                c=SimpleNamespace(episode=e,node=int(pair[0]),boundary=34,blocked_nodes=blocked)
                path=probe_path(c,delay,int(q),seed)
                if np.isin(path[2],blocked).any():raise AssertionError('Scored node observed')
                paths.append(path)
            rows.append(paths)
        all_labels.append([[p[0] for p in row] for row in rows]);all_actions.append([[p[1] for p in row] for row in rows])
        all_nodes.append([[p[2] for p in row] for row in rows]);queries.append(qset)
        answers.append(labels[1:,qset]);target_mask.append([np.isin(qset,pair[:v]) for v in [1,2]])
    return dict(labels=np.asarray(all_labels),actions=np.asarray(all_actions),nodes=np.asarray(all_nodes),
                queries=np.asarray(queries),answers=np.asarray(answers),target_mask=np.asarray(target_mask))


def score_logits(logits,probe):
    correct=(logits.argmax(-1)==probe['answers']).astype(float)
    target=probe['target_mask'];result={};arrays={}
    lp=logits.astype(np.float64);lp-=lp.max(-1,keepdims=True);prob=np.exp(lp);prob/=prob.sum(-1,keepdims=True)
    answer_p=np.take_along_axis(prob,probe['answers'][...,None],axis=-1)[...,0]
    for vi,name in enumerate(['single','swap']):
        values={'target_accuracy':(correct[:,vi]*target[:,vi]).sum(-1)/target[:,vi].sum(-1),
            'offtarget_accuracy':(correct[:,vi]*~target[:,vi]).sum(-1)/(~target[:,vi]).sum(-1),
            'all_five_correct':correct[:,vi].prod(-1),
            'target_probability':(answer_p[:,vi]*target[:,vi]).sum(-1)/target[:,vi].sum(-1),
            'offtarget_probability':(answer_p[:,vi]*~target[:,vi]).sum(-1)/(~target[:,vi]).sum(-1)}
        result[name]={k:intervals(v) for k,v in values.items()};arrays[name]=values
    both=.5*(arrays['single']['all_five_correct']+arrays['swap']['all_five_correct'])
    result['selection_joint_score']=intervals(both)
    return result,arrays


def run_probe(target,states,probe,batch_scenes=48):
    # states [scene, variant, hidden]; every query is independently cloned.
    result=[];steps=probe['labels'].shape[-1]
    for begin in range(0,len(states),batch_scenes):
        end=min(begin+batch_scenes,len(states))
        h=np.repeat(states[begin:end,:,None,:],5,axis=2).reshape(-1,1536)
        run=target.run(probe['labels'][begin:end].reshape(-1,steps),probe['actions'][begin:end].reshape(-1,steps,2),recurrent_state(h,target.device))
        result.append(run.logits[:,-1].cpu().numpy().reshape(end-begin,2,5,26))
    return np.concatenate(result)
