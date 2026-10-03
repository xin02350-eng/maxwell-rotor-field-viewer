"""Draw masked 2D clouds and magnetic-value-height 3D surfaces.

Uses only the supplied rotor_surface_mesh.npz triangles. The mesh is an
approximate sample-inferred boundary, not CAD. No remeshing or mirroring.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from make_figures import configure, save, map_axis

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent


def draw_cloud(ax, tri, xy, edges, vals, vmax, cmap='viridis'):
    artist = ax.tripcolor(tri, vals, shading='gouraud', cmap=cmap, vmin=0, vmax=vmax)
    ax.add_collection(LineCollection(xy[edges], colors='#334155', linewidths=.38, alpha=.72))
    map_axis(ax, xy)
    return artist


def draw_surface(ax, tri, xy, edges, vals, vmax, error=False, compact=False):
    artist = ax.plot_trisurf(tri, vals, cmap='inferno' if error else 'viridis',
        vmin=0, vmax=vmax, linewidth=0, antialiased=True, shade=False,
        edgecolors='none')
    points3d = np.column_stack([xy, vals])
    ax.add_collection3d(Line3DCollection(points3d[edges], colors='#334155', linewidths=.35, alpha=.75))
    # Gray outlines on z=0 identify the sampled support and its holes.
    footprint = np.column_stack([xy, np.zeros(len(xy))])
    ax.add_collection3d(Line3DCollection(footprint[edges], colors='#94A3B8', linewidths=.35, alpha=.50))
    ax.set_xlim(-20,20); ax.set_ylim(0,20); ax.set_zlim(0,vmax)
    ax.set_xticks([-20,0,20]);ax.set_yticks([0,10,20])
    ax.set_zticks(np.linspace(0,vmax,4))
    ax.zaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.set_xlabel('x_ref (mm)',labelpad=5,fontsize=9)
    ax.set_ylabel('y_ref (mm)',labelpad=5,fontsize=9)
    # Keep the height unit inside the panel: native 3D z-labels can be clipped
    # by Matplotlib's tight bounding box at the right edge of a multi-panel row.
    ax.set_zlabel('')
    ax.text2D(.96,.92,'z = |误差| (mT)' if error else 'z = |B| (T)',
        transform=ax.transAxes,ha='right',va='top',fontsize=9,color='#475569')
    ax.tick_params(axis='both',labelsize=8,pad=1)
    ax.view_init(elev=34,azim=-64)
    ax.set_box_aspect((2,1,1.0))
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.pane.set_facecolor((.97,.98,.99,1))
        axis.pane.set_edgecolor((.82,.87,.91,.8))
        axis._axinfo['grid'].update({'color':(.65,.72,.79,.25),'linewidth':.6})
    return artist


def horizontal_bar(fig, artist, rect, label):
    cb=fig.colorbar(artist,cax=fig.add_axes(rect),orientation='horizontal')
    cb.set_label(label,fontsize=10)
    cb.ax.tick_params(labelsize=9)
    return cb


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default=str(ROOT/'work/rotor_dataset.npz'))
    ap.add_argument('--predictions',default=str(ROOT/'work/predictions.npz'))
    ap.add_argument('--mesh',default=str(OUT/'rotor_surface_mesh.npz'))
    args=ap.parse_args()
    configure()
    d=np.load(args.data);p=np.load(args.predictions);mesh=np.load(args.mesh)
    xy=mesh['xy'];faces=mesh['triangles'];edges=mesh['boundary_edges']
    assert np.allclose(xy,d['xy'],atol=1e-12)
    if 'point_id' in mesh: assert np.array_equal(mesh['point_id'],d['point_id'])
    assert faces.min()==0 and faces.max()==len(xy)-1
    assert np.unique(faces).size==len(xy)
    tri=Triangulation(xy[:,0],xy[:,1],triangles=faces)
    c,t=4,44
    assert p['split'][t]==2
    truth=d['Bmag'][c,t].astype(float);pred=p['angle_pred'][c,t].astype(float)
    err=1000*np.abs(pred-truth)
    vmax=float(d['Bmag'].max())
    emax=float(err.max())
    rmse=float(np.sqrt(np.mean((pred-truth)**2)))
    desc=f'I_rms = {d["conditions"][c,0]:g} A   ·   电流角 = {d["conditions"][c,1]:g}°   ·   相对机械转角 θ = {d["theta"][t]:g}°   ·   留出测试帧'

    fig,axes=plt.subplots(1,3,figsize=(13.0,4.8))
    for i,(ax,values,title) in enumerate(zip(axes,[truth,pred,err],['Maxwell 真值','神经网络预测','绝对误差'])):
        artist=draw_cloud(ax,tri,xy,edges,values,emax if i==2 else vmax,'inferno' if i==2 else 'viridis')
        ax.set_title(title,pad=12,fontsize=12)
        cb=fig.colorbar(artist,ax=ax,orientation='horizontal',fraction=.065,pad=.23,shrink=.92)
        cb.set_label('|误差| (mT)' if i==2 else '|B| (T)')
    fig.suptitle('二维转子磁密云图 · 局部三角面插值',x=.065,ha='left',y=1.01,fontsize=17,fontweight='bold')
    fig.text(.065,.916,desc,fontsize=11)
    fig.text(.065,.015,f'本帧 RMSE = {rmse*1000:.4f} mT。仅在共享网格的 4561 个局部三角面内插值；4 个内部孔及外边界均保留。',fontsize=9,color='#475569')
    fig.text(.065,-.031,'边界由采样点推断，属于近似显示边界，不是 CAD。保留原始半转子采样范围；不补全另一半或孔洞。',fontsize=9,color='#475569')
    fig.subplots_adjust(left=.065,right=.99,top=.82,bottom=.18,wspace=.31)
    save(fig,'08_rotor_2d_cloud')

    fig=plt.figure(figsize=(15.4,6.1))
    for i,(values,title) in enumerate(zip([truth,pred,err],['Maxwell 真值','神经网络预测','绝对误差'])):
        left=.012+i*.325
        ax=fig.add_axes([left,.20,.315,.58],projection='3d')
        artist=draw_surface(ax,tri,xy,edges,values,emax if i==2 else vmax,error=i==2)
        ax.set_title(title,pad=0,fontsize=12)
        horizontal_bar(fig,artist,[left+.052,.155,.218,.022],'|误差| (mT)' if i==2 else '|B| (T)')
    fig.suptitle('三维高度曲面 · 高度表示磁密或绝对误差',x=.045,ha='left',y=.99,fontsize=18,fontweight='bold')
    fig.text(.045,.91,desc,fontsize=11)
    fig.text(.045,.015,'x、y 是转子参考坐标 (mm)；z 是磁密 |B| (T)，误差图的 z 为 |误差| (mT)。真值与预测采用相同高度与色阶。',fontsize=10,color='#475569')
    fig.text(.045,-.026,'这是一张二维采样场的高度表示，不是实体厚度或三维体积场。曲面只覆盖共享局部网格；灰线是 z = 0 上的采样边界投影。',fontsize=9,color='#475569')
    save(fig,'09_rotor_3d_surface')

    fig=plt.figure(figsize=(13.2,6.8))
    ax2=fig.add_axes([.068,.31,.405,.46])
    artist=draw_cloud(ax2,tri,xy,edges,pred,vmax)
    ax2.set_title('二维云图：颜色表示 |B|',fontsize=13,pad=16)
    ax3=fig.add_axes([.50,.235,.47,.58],projection='3d')
    draw_surface(ax3,tri,xy,edges,pred,vmax)
    ax3.set_title('三维高度曲面：高度与颜色均表示 |B|',fontsize=12,pad=0)
    horizontal_bar(fig,artist,[.15,.175,.70,.024],'预测磁密 |B| (T) · 两图共用色阶')
    fig.suptitle('同一预测场 · 二维与三维可视化',x=.065,ha='left',y=.995,fontsize=19,fontweight='bold')
    fig.text(.065,.916,desc,fontsize=11)
    fig.text(.065,.055,'输入工况与转角 → 神经网络预测 2500 点磁密 → 共享局部网格显示；孔洞、轴孔和原半转子范围保持为空。',fontsize=10,color='#475569')
    fig.text(.065,.011,'二维和三维显示同一组预测值。3D 高度单位为 T，不表示机械厚度；显示边界是基于采样点推断的近似边界。',fontsize=10,color='#475569')
    save(fig,'10_rotor_2d_3d_overview')
    print(json.dumps({'case_id':c,'frame_index':t,'theta_deg':float(d['theta'][t]),
        'triangles':int(len(faces)),'points':int(len(xy)),'B_scale_T':[0,vmax],
        'error_scale_mT':[0,emax],'rmse_T':rmse,
        'note':'Shared sample-derived local mesh only; 3D height is magnetic value, not a volume field.'},indent=2))


if __name__=='__main__':
    main()
