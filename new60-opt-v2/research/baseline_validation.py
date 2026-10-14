"""Compare V1 fields and derived planar magnitudes on validation only."""
import json,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
V1=Path(__file__).resolve().parent.parent/'new60_20261014'
sys.path.insert(0,str(V1/'code'))
import hybrid_cycle_rbf as h
import experiment_core as core

def main():
    rows={}
    for region in ('rotor','stator','airgap'):
        a,s,m=h._load_cache(V1/'cache_v1'/f'{region}.npz',V1/'cache_v1/split.json')
        va=np.array(s['validation']); choice=json.loads((V1/'experiments'/region/'final_choice.json').read_text(encoding='utf8'))
        path=V1/choice['selected_model_relative_path']
        with np.load(path,allow_pickle=False) as z:
            fmt=json.loads(str(z['metadata_json']))['format']
        p=h.predict(path,a['conditions'][va]) if fmt.startswith('new60_mlp_cycle_rbf') else core.predict(path,a['features'][va])
        y=a['field'][va]; result=dict(vector_or_Br=h._metrics(p,y),source_model_sha256=h.digest(path),test_labels_accessed=False)
        if m['component_count']==2:
            pm=np.linalg.norm(p.reshape(9,60,-1,2),axis=-1);ym=np.linalg.norm(y.reshape(9,60,-1,2),axis=-1)
            result['planar_magnitude']=h._metrics(pm,ym)
            result['per_case_planar_magnitude']={str(cid):h._metrics(pm[i],ym[i]) for i,cid in enumerate(a['case_ids'][va])}
        rows[region]=result
    out=Path(__file__).parent/'baseline_validation.json'
    out.write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(rows,ensure_ascii=False,indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
