"""Conservative visualization-only mesh of the sampled rotor half-section.

The mesh connects existing samples, not a finite-element/CAD reconstruction.
Scalar fields are linearly interpolated inside the retained local triangles.
"""
from pathlib import Path
import argparse, json
from collections import Counter
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent


def make_mesh(xy, radius_limit_mm=.4):
    xy = np.asarray(xy, dtype=np.float64)
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError('Duplicate coordinates need explicit aggregation before triangulation.')
    raw = Delaunay(xy).simplices
    points = xy[raw]
    uv, vv = points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]
    area = abs(uv[:, 0]*vv[:, 1]-uv[:, 1]*vv[:, 0])/2
    edges = np.stack([np.linalg.norm(points[:, a]-points[:, b], axis=1)
                      for a, b in [(0, 1), (1, 2), (2, 0)]], axis=1)
    radius = np.prod(edges, axis=1) / (4*np.maximum(area, 1e-15))
    # Short local triangles only. The circumradius test suppresses connections
    # across sparse/empty regions even when the triangle has short near-collinear edges.
    keep = (area > 1e-10) & (radius <= radius_limit_mm+1e-10)
    keep &= edges.max(1) <= 2*radius_limit_mm+1e-10
    keep &= (points[:, :, 1].min(1) >= 0)
    # The archive's geometric radii are shaft=8.4 and rotor-iron outer=18.95 mm.
    keep &= np.linalg.norm(points, axis=2).max(1) <= 18.95
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        start, vec = points[:, a], points[:, b]-points[:, a]
        f = np.clip(-np.sum(start*vec, axis=1)/np.maximum(np.sum(vec**2, axis=1), 1e-15), 0, 1)
        nearest = start+f[:, None]*vec
        keep &= np.linalg.norm(nearest, axis=1) >= 8.4
    triangles = raw[keep].astype(np.int32)
    pairs = np.sort(np.vstack([triangles[:, [0,1]], triangles[:, [1,2]], triangles[:, [2,0]]]), axis=1)
    edge_counts = Counter(map(tuple, pairs.tolist()))
    boundary = np.array([k for k,v in edge_counts.items() if v==1],dtype=np.int32)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:,0],pairs[:,1])),shape=(len(xy),len(xy))).tocsr()
    components, _ = connected_components(graph, directed=False)
    v, e, f = len(np.unique(triangles)), len(edge_counts), len(triangles)
    audit = {'method':'Delaunay with local circumradius cutoff and archive radial bounds',
             'purpose':'visualization only, not CAD or FE mesh',
             'circumradius_limit_mm':radius_limit_mm, 'max_edge_limit_mm':2*radius_limit_mm,
             'input_points':len(xy),'used_points':v,'raw_triangles':len(raw),
             'kept_triangles':len(triangles),'removed_triangles':int((~keep).sum()),
             'connected_components':int(components), 'interior_holes_euler_estimate':int(components-v+e-f),
             'rendered_area_mm2':float(area[keep].sum()),
             'max_kept_edge_mm':float(edges[keep].max()),
             'min_y_mm':float(xy[:,1].min()),'no_mirror_expansion':True,
             'fields_unchanged_at_vertices':True,
             'caveat':'Hole boundaries are inferred from sample spacing; exact CAD contours are unavailable.'}
    return triangles, boundary, audit


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=str(ROOT/'work/rotor_dataset.npz'))
    args=ap.parse_args();d=np.load(args.data);xy=d['xy']
    triangles,boundary,audit=make_mesh(xy)
    assert audit['used_points']==2500 and audit['connected_components']==1
    np.savez_compressed(OUT/'rotor_surface_mesh.npz',xy=xy,triangles=triangles,boundary_edges=boundary,point_id=d['point_id'])
    (OUT/'surface_mesh_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
