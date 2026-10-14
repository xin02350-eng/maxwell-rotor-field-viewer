"""Derive a planar magnitude from the frozen vector neural predictor."""
import numpy as np
from cycle_neural import predict as vector_predict

def predict(model,conditions,frame_index=None):
    p=vector_predict(model,conditions,frame_index=frame_index)
    return np.hypot(p[...,0::2],p[...,1::2])
