"""NumPy wrapper: predict signed Br/Bt, then return planar hypot(Br,Bt).

This is a deterministic output transform of an existing vector model. It is
not a separately trained magnitude model and is never used for Br-only gap.
"""
from pathlib import Path
import json
import numpy as np
from kernel_model import predict as vector_predict


def predict(model,conditions,frame_index=None):
    if isinstance(model,(str,Path)):
        with np.load(model,allow_pickle=False) as archive:
            metadata=json.loads(str(archive["metadata_json"]))
    else:metadata=json.loads(str(model["metadata_json"]))
    if metadata.get("quantity")!="Br,Bt":raise ValueError("Planar magnitude requires a signed Br/Bt model")
    field=vector_predict(model,conditions,frame_index=frame_index)
    return np.hypot(field[...,0::2],field[...,1::2])
