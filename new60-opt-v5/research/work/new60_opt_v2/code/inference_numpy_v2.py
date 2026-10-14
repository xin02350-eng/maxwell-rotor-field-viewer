"""Portable NumPy-only inference for both frozen V2 versions.

Implements saved forward equations, never reads training/validation/test labels.
Original entry-point parity must be verified by qa/check_cli_v2.py.
"""
from pathlib import Path
import hashlib,json
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def load_model(path,expected_sha256=None):
    path=Path(path)
    if expected_sha256 and digest(path)!=expected_sha256:raise ValueError('Model SHA256 mismatch')
    with np.load(path,allow_pickle=False) as z:m={k:z[k] for k in z.files}
    if 'representation_relative_path' in m:
        rep=(path.parent/str(m['representation_relative_path'])).resolve()
        if digest(rep)!=str(m['representation_sha256']):raise ValueError('Representation SHA256 mismatch')
        with np.load(rep,allow_pickle=False) as z:m.update({k:z[k] for k in z.files if k not in m})
    return m

def polynomial(x,degree):
    if degree<0:return np.zeros((len(x),0))
    return np.stack([x[:,0]**i*x[:,1]**(n-i) for n in range(degree+1) for i in range(n+1)],axis=-1)

def neural_features(c,kind,scale):
    r=c[:,0]/scale;ang=np.deg2rad(c[:,1]);x=r*np.cos(ang);y=r*np.sin(ang)
    if kind=='cartesian':return np.column_stack((x,y))
    if kind=='cartesian2':return np.column_stack((x,y,x*x,x*y,y*y))
    if kind=='mixed':return np.column_stack((r,c[:,1]/90,x,y,x*x,x*y,y*y))
    if kind=='native':return np.column_stack((r,c[:,1]/90))
    raise ValueError('Unknown neural features')

def kernel_coordinates(c,kind,scale):
    cur=c[:,0]/scale
    if kind=='polar_xy' or kind.startswith('saturated_'):
        if kind.startswith('saturated_'):
            factor=float(kind.split('_',1)[1]);cur=factor*np.tanh(cur/factor)
        ang=np.deg2rad(c[:,1]);return np.stack((cur*np.cos(ang),cur*np.sin(ang)),axis=-1)
    p={'native':1.,'sqrt_native':.5,'square_native':2.}[kind]
    return np.stack((cur**p,c[:,1]/90.),axis=-1)

def kernel(x,c,kind,length):
    r=np.sqrt(np.maximum(np.sum((x[:,None,:]-c[None,:,:])**2,axis=-1),0))/length
    if kind=='gaussian':return np.exp(-.5*r*r)
    if kind=='matern52':
        q=np.sqrt(5)*r;return (1+q+q*q/3)*np.exp(-q)
    if kind=='matern32':
        q=np.sqrt(3)*r;return (1+q)*np.exp(-q)
    if kind=='exponential':return np.exp(-r)
    if kind=='inverse_multiquadric':return 1/np.sqrt(1+r*r)
    if kind=='multiquadric':return -np.sqrt(1+r*r)
    if kind=='thin_plate':return r*r*np.log(np.maximum(r,1e-300))
    if kind=='cubic':return r**3
    raise ValueError('Unknown saved kernel')

def _kernel_field(m,c,meta):
    out=np.empty((len(c),meta['time_count']*meta['field_dim']))
    for bi,spec in enumerate(meta['blocks']):
        p=f'b{bi}_';x=(kernel_coordinates(c,spec['coordinate'],spec['current_scale'])-m[p+'center'])/m[p+'input_scale']
        x*=np.array([np.sqrt(spec['anisotropy']),1/np.sqrt(spec['anisotropy'])])
        if spec['kernel']=='polynomial_ridge':z=polynomial(x,spec['degree'])@m[p+'weights']
        else:z=kernel(x,m[p+'centers'],spec['kernel'],spec['lengthscale'])@m[p+'weights']+polynomial(x,spec['degree'])@m[p+'trend']
        out[:,m[p+'indices']]=m[p+'mean']+z@m[p+'basis'].T
    return out.reshape(len(c),meta['time_count'],meta['field_dim'])

def _frame_kernel_field(m,c):
    def embedding(cc):
        ang=np.deg2rad(cc[:,1]);cur=cc[:,0]
        if str(m['coordinate'])=='native':raw=cc
        else:
            if str(m['coordinate'])=='saturated':cur=float(m['cap'])*np.tanh(cur/float(m['cap']))
            raw=np.stack((cur*np.cos(ang),cur*np.sin(ang)),axis=1)
        x=(raw-m['input_mean'])/m['input_scale']
        h=np.tanh(np.einsum('nd,tdh->tnh',x,m['w1'])+m['b1'][:,None,:])
        w=np.tanh(np.einsum('tnh,thd->tnd',h,m['w2'])+m['b2'][:,None,:])
        return (x[None,:,:]+.5*w)/np.exp(m['log_length'])[:,None,:]
    x=embedding(c);center=embedding(m['train_conditions']);rr=np.sum((x[:,:,None,:]-center[:,None,:,:])**2,axis=-1)
    if str(m['kernel'])=='gaussian':kk=np.exp(-.5*rr)
    else:
        q=np.sqrt(5*np.maximum(rr,1e-24));kk=(1+q+q*q/3)*np.exp(-q)
    coef=np.einsum('tnc,tcr->ntr',kk,m['alpha'])*float(m['output_scale'])+m['cycle_coefficient_mean'][None,:,:]
    return m['field_mean']+coef@m['spatial_basis'].T

def _neural_field(m,c):
    scale=float(m['current_scale']);f=neural_features(c,str(m.get('feature_kind','mixed')),scale)
    x=(f-m['feature_mean'])/m['feature_scale']
    for i in range(int(m['layer_count'])):
        x=x@m[f'w{i}']+m[f'b{i}']
        if i+1<int(m['layer_count']):
            x=np.tanh(x) if str(m.get('activation','tanh'))=='tanh' else x/(1+np.exp(-np.clip(x,-700,700)))
    latent=x*m['output_scale']
    if 'base_weights' in m:latent+=polynomial(neural_features(c,'cartesian',scale),int(m['base_degree']))@m['base_weights']
    z=(m['cycle_mean']+latent@m['cycle_basis']).reshape(len(c),int(m['frame_count']),-1)
    return m['spatial_mean']+z@m['spatial_basis'].T if 'spatial_basis' in m and not int(m.get('direct_field',0)) else z

def predict(model,conditions,choice,frame_index=None):
    m=model;c=np.asarray(conditions,dtype=np.float64);single=c.ndim==1;c=np.atleast_2d(c)
    if c.ndim!=2 or c.shape[1]!=2 or not np.isfinite(c).all() or np.any(c[:,0]<0):raise ValueError('Expected finite native I_RMS>=0, Thet')
    meta=json.loads(str(m.get('metadata_json','{}')))
    if meta.get('format')=='new60_condition_kernel_v2':out=_kernel_field(m,c,meta)
    elif 'alpha' in m and m.get('w1',np.empty(0)).ndim==3:out=_frame_kernel_field(m,c)
    else:out=_neural_field(m,c)
    if choice.get('output_transform')=='hypot(Br,Bt)':out=np.hypot(out[...,0::2],out[...,1::2])
    if choice.get('predict_kwargs',{}).get('nonnegative',False) and choice['quantity']=='planar_magnitude':out=np.maximum(out,0)
    if frame_index is not None:
        if type(frame_index) is not int or not 0<=frame_index<out.shape[1]:raise ValueError('Frame must be a saved index0..59')
        out=out[:,frame_index]
    if not np.isfinite(out).all():raise ValueError('Nonfinite prediction')
    return out[0] if single else out
