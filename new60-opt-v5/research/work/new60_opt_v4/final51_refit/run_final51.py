"""Two prespecified final51 neural trainings; no test or independent validation."""
from pathlib import Path
import json
import traceback
from regime_final51 import train,sha

HERE=Path(__file__).resolve().parent
PREPARED=HERE.parents[1]/"new60_opt_v3/joint_neural/point_prepared"
CACHE=HERE.parents[1]/"new60_20261014/cache_v1/rotor.npz"

if __name__=="__main__":
    protocol=json.loads((HERE/"protocol.json").read_text(encoding="utf-8"));protocol["protocol_sha256"]=sha(HERE/"protocol.json")
    if sha(HERE/"regime_final51.py")!=protocol["neural_source_sha256"]:raise ValueError("Frozen final51 source hash mismatch")
    completed=[]
    for config in protocol["neural_candidates"]:
        try:
            result=train(PREPARED,CACHE,HERE/config["name"],config,protocol);completed.append(result)
            (HERE/"progress.json").write_text(json.dumps(dict(completed=len(completed),planned=2,results=completed,test_labels_loaded=False),ensure_ascii=False,indent=2),encoding="utf-8")
        except Exception as e:
            (HERE/"failure.json").write_text(json.dumps(dict(candidate=config,exception=repr(e),traceback=traceback.format_exc()),ensure_ascii=False,indent=2),encoding="utf-8");raise
