"""Only a transformed TPR, optional marginal tensor terms, and a fixed monotone link."""
import numpy as np
import torch
from torch import nn
from .features import Q_DIM,POSITION_DIM,position_features


def clean_inputs(data,device):
    # Deliberately do not read query, ages, counts, next labels, or stable node IDs.
    return {'labels':torch.as_tensor(data['labels'],dtype=torch.long,device=device),
            'phi':torch.as_tensor(position_features(data['positions']),device=device),
            'mask':torch.as_tensor(data['mask'],device=device),
            'q':torch.as_tensor(data['q'],device=device)}


def link(z,name):
    if name=='identity':return z
    if name=='tanh':return torch.tanh(z)
    if name=='softsign':return z/(1+torch.abs(z))
    if name=='atan':return (2/np.pi)*torch.atan((np.pi/2)*z)
    raise ValueError(name)


def transport(actual,dz,name):
    """Transport a finite pre-link difference; preserve exact endpoints by limits."""
    h=np.clip(actual.astype(np.float64),-1+1e-12,1-1e-12)
    if name=='tanh':z=np.arctanh(h);out=np.tanh(z+dz)
    elif name=='softsign':z=h/(1-np.abs(h));z=z+dz;out=z/(1+np.abs(z))
    elif name=='atan':z=(2/np.pi)*np.tan((np.pi/2)*h);out=(2/np.pi)*np.arctan((np.pi/2)*(z+dz))
    else:raise ValueError(name)
    out=np.where((dz==0)|(np.abs(actual)>=1),actual,out)
    return out.astype(np.float32)


class CleanTPR(nn.Module):
    def __init__(self,spec,position_mean=None,label_mean=None,state_dim=1536):
        super().__init__();self.spec=spec;self.state_dim=state_dim
        self.fr=spec['filler_rank'];self.rr=spec['role_rank'];self.activation=spec['activation']
        self.register_buffer('position_mean',torch.zeros(POSITION_DIM) if position_mean is None else torch.as_tensor(position_mean,dtype=torch.float32))
        self.register_buffer('label_mean',torch.full((26,),1/26) if label_mean is None else torch.as_tensor(label_mean,dtype=torch.float32))
        self.fillers=nn.Embedding(26,self.fr);self.roles=nn.Linear(POSITION_DIM,self.rr,bias=False)
        self.output=nn.Linear(self.fr*self.rr,state_dim,bias=False)
        self.context=nn.Linear(Q_DIM,state_dim,bias=False);self.bias=nn.Parameter(torch.zeros(state_dim))
        if spec.get('inventory'):self.inventory=nn.Linear(26,state_dim,bias=False)
        if spec.get('geometry'):self.geometry=nn.Linear(POSITION_DIM,state_dim,bias=False)

    def components(self,x):
        p=x['phi'];f=self.fillers(x['labels']);mask=x['mask']
        context=self.context(x['q'])+self.bias
        inventory=torch.zeros_like(context);geometry=torch.zeros_like(context)
        if hasattr(self,'inventory'):
            p=p-self.position_mean
            counts=(torch.nn.functional.one_hot(x['labels'],26).float()*mask[...,None]).sum(1)
            inventory=self.inventory(counts)
        if hasattr(self,'geometry'):
            f=f-self.label_mean@self.fillers.weight
            geometry=self.geometry((p*mask[...,None]).sum(1))
        memory=self.output(torch.bmm((f*mask[...,None]).transpose(1,2),self.roles(p)).flatten(1))
        return {'context':context,'inventory':inventory,'geometry':geometry,'binding':memory}

    def preactivation(self,x):
        parts=self.components(x)
        return parts['context']+parts['inventory']+parts['geometry']+parts['binding']

    def forward(self,x):return link(self.preactivation(x),self.activation)

    @torch.no_grad()
    def initialize(self,linear):
        if linear.kind!='full_pair' or linear.frame!='absolute':raise ValueError('Absolute full-pair initialization required')
        a=linear.weight[Q_DIM:].reshape(26,POSITION_DIM,self.state_dim).astype(np.float64)
        _,uf=np.linalg.eigh(np.einsum('frd,grd->fg',a,a));_,ur=np.linalg.eigh(np.einsum('frd,fsd->rs',a,a))
        uf=uf[:,-self.fr:];ur=ur[:,-self.rr:]
        core=np.einsum('frd,fi,rj->dij',a,uf,ur,optimize=True)
        self.fillers.weight.copy_(torch.from_numpy(uf.astype(np.float32)))
        self.roles.weight.copy_(torch.from_numpy(ur.T.astype(np.float32)))
        self.output.weight.copy_(torch.from_numpy(core.reshape(self.state_dim,-1).astype(np.float32)))
        self.context.weight.copy_(torch.from_numpy(linear.weight[:Q_DIM].T.copy()));self.bias.copy_(torch.from_numpy(linear.bias))
        self.initialize_marginals()

    @torch.no_grad()
    def initialize_marginals(self):
        """Untying starts from an exactly equivalent uncentered tensor map."""
        if hasattr(self,'inventory'):
            r=self.roles(self.position_mean)
            self.inventory.weight.copy_(self.output((self.fillers.weight[:,:,None]*r[None,None,:]).flatten(1)).T)
        if hasattr(self,'geometry'):
            f=self.label_mean@self.fillers.weight
            self.geometry.weight.copy_(self.output((f[None,:,None]*self.roles.weight.T[:,None,:]).flatten(1)).T)

    @torch.inference_mode()
    def predict(self,data,preactivation=False,batch_size=512):
        x=clean_inputs(data,next(self.parameters()).device);parts=[]
        for start in range(0,len(data['labels']),batch_size):
            batch={k:v[start:start+batch_size] for k,v in x.items()}
            parts.append((self.preactivation(batch) if preactivation else self(batch)).cpu().numpy())
        return np.concatenate(parts)

    def edit(self,actual,data,changed,mode='clipped_difference'):
        if mode=='inverse_transport':
            dz=self.predict(changed,True)-self.predict(data,True)
            return transport(actual,dz,self.activation)
        edited=actual+(self.predict(changed)-self.predict(data))
        if mode=='raw_difference':return edited
        if mode=='clipped_difference':return np.clip(edited,-1,1)
        raise ValueError(mode)

    def save(self,path,info):
        torch.save({'spec':self.spec,'state_dim':self.state_dim,'state_dict':{k:v.detach().cpu() for k,v in self.state_dict().items()},'fit_info':info},path)

    @classmethod
    def load(cls,path,device='cpu'):
        s=torch.load(path,map_location='cpu',weights_only=True)
        m=cls(s['spec'],state_dim=s['state_dim']);m.load_state_dict(s['state_dict']);return m.to(device).eval()
