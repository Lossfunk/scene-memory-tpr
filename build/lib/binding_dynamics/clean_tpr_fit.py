import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import flatten_groups
from .clean_tpr import CleanTPR,clean_inputs


def loss_parts(model,x,y,spec,pair_weight):
    pred=model(x);state=F.mse_loss(pred,y)
    p=pred.reshape(-1,3,model.state_dim);t=y.reshape(-1,3,model.state_dim)
    pair=F.mse_loss(p[:,1:]-p[:,:1],t[:,1:]-t[:,:1])
    return state+(0 if spec.get('state_only') else pair_weight*pair),state,pair


def train_clean(train,val,init,spec,config,seed,device='mps'):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    tf=flatten_groups(train);vf=flatten_groups(val);tx=clean_inputs(tf,device);vx=clean_inputs(vf,device)
    pm=(tx['phi']*tx['mask'][...,None]).sum((0,1))/tx['mask'].sum()
    lm=(F.one_hot(tx['labels'],26).float()*tx['mask'][...,None]).sum((0,1))/tx['mask'].sum()
    m=CleanTPR(spec,pm.cpu(),lm.cpu());m.initialize(init)
    with torch.no_grad():
        m.fillers.weight.add_(torch.randn_like(m.fillers.weight)*.001)
        m.roles.weight.add_(torch.randn_like(m.roles.weight)*.001)
    m.to(device);y=torch.as_tensor(tf['states'],device=device);vy=torch.as_tensor(vf['states'],device=device)
    cfg=config['training'];opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    best=None;best_loss=float('inf');best_epoch=0;stale=0;history=[];start=time.perf_counter()
    for epoch in range(cfg['epochs']):
        m.train();groups=rng.permutation(len(train['states']))
        for begin in range(0,len(groups),cfg['batch_families']):
            ix=torch.as_tensor((groups[begin:begin+cfg['batch_families'],None]*3+np.arange(3)).ravel(),device=device)
            opt.zero_grad(set_to_none=True)
            obj,*_=loss_parts(m,{k:v[ix] for k,v in tx.items()},y[ix],spec,config['pair_loss_weight'])
            obj=obj+1e-5*(torch.linalg.vector_norm(m.fillers.weight,dim=0).sum()+torch.linalg.vector_norm(m.roles.weight,dim=1).sum())
            if not bool(torch.isfinite(obj)):raise RuntimeError('Nonfinite training loss')
            obj.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.0);opt.step()
        m.eval()
        with torch.no_grad():values=[float(v) for v in loss_parts(m,vx,vy,spec,config['pair_loss_weight'])]
        history.append({'epoch':epoch+1,'objective':values[0],'state_mse':values[1],'pair_mse':values[2]})
        if values[0]<best_loss-1e-7:
            best_loss=values[0];best_epoch=epoch+1;stale=0;best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        else:stale+=1
        if epoch==0 or (epoch+1)%20==0:print(spec['name'],seed,'epoch',epoch+1,history[-1],flush=True)
        if stale>=cfg['patience']:break
    m.load_state_dict(best);m.eval()
    return m,{'spec':spec,'seed':seed,'epochs':len(history),'best_epoch':best_epoch,'history':history,
              'parameter_count':sum(p.numel() for p in m.parameters()),'runtime_seconds':time.perf_counter()-start,
              'loss':'state MSE plus matched counterfactual-difference MSE, unless state_only; no output supervision',
              'model_inputs':['observed labels','observed positions','mask','original current-input/integrated-position context q']}
