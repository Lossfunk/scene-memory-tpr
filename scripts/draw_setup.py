#!/usr/bin/env python3
"""Figure 1: original scene/task and TPR schematic, informed by the cited papers.

Coordinates and tile values illustrate the computation; they are not fitted
vectors or experimental observations. No external artwork is reused.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT=Path(__file__).resolve().parents[1]
BLUE='#405e88'; ORANGE='#b24a24'; INK='#242424'; GREEN='#39766a'


def main():
    fig=plt.figure(figsize=(12,8.7))
    ax=fig.add_axes([.02,.02,.96,.96]);ax.set(xlim=(0,12),ylim=(0,8.7));ax.axis('off')
    def txt(x,y,s,size=11,color=INK,ha='center',**kw):
        ax.text(x,y,s,fontsize=size,color=color,ha=ha,va='center',**kw)
    def arrow(a,b,color=INK):
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',lw=1.2,color=color,shrinkA=3,shrinkB=3))
    def box(x,y,w,h,title,sub,color=INK):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02,rounding_size=.06',ec=color,fc='white',lw=1.1))
        txt(x+w/2,y+h*.70,title,11,color,fontweight='bold');txt(x+w/2,y+h*.29,sub,10)
    def tiles(x,y,values,cell=.26):
        values=np.asarray(values); cmap=plt.get_cmap('RdBu_r')
        for j in range(values.shape[0]):
            for k in range(values.shape[1]):
                ax.add_patch(Rectangle((x+k*cell,y+(values.shape[0]-j-1)*cell),cell,cell,
                    facecolor=cmap((values[j,k]+1)/2),edgecolor='white',lw=.8))
        ax.add_patch(Rectangle((x,y),values.shape[1]*cell,values.shape[0]*cell,fill=False,ec='#aaaaaa',lw=.5))

    txt(.15,8.42,'Figure 1. The scene, its predictive GRU, and a TPR model of the GRU state',14,ha='left',fontweight='bold')
    txt(.18,7.93,'A  The GRU predicts observations in the scene',13,ORANGE,ha='left',fontweight='bold')
    txt(.18,7.56,'A six-letter scene; one step shown',10,ha='left')
    scene=fig.add_axes([.064,.55,.255,.285])
    scene.set(xlim=(-4,4),ylim=(-4,4),xticks=[-4,0,4],yticks=[-4,0,4],xlabel='x position',ylabel='y position')
    scene.set_aspect('equal');scene.tick_params(labelsize=8);scene.xaxis.label.set_size(9);scene.yaxis.label.set_size(9)
    for spine in scene.spines.values():spine.set_color('#bbbbbb')
    coords={'D':(0,0),'A':(2.5,2),'B':(-2,2.7),'C':(-2.8,-1.6),'E':(1.2,-2.7),'F':(3,-.8)}
    for letter,(x,y) in coords.items():
        c=ORANGE if letter=='D' else INK
        scene.scatter(x,y,s=380,facecolor='white',edgecolor=c,linewidth=1,zorder=3)
        scene.text(x,y,letter,color=c,ha='center',va='center',fontsize=12,zorder=4)
    scene.annotate('',xy=(2.5,2),xytext=(0,0),arrowprops=dict(arrowstyle='-|>',lw=1.6,color=ORANGE,shrinkA=16,shrinkB=16),zorder=2)
    scene.text(-.2,-.67,'current',fontsize=8,color=ORANGE,ha='center')
    scene.text(2.5,2.75,'next',fontsize=8,color=INK,ha='center')
    txt(1.78,4.23,'Arrow: intended displacement (+2.5, +2.0)',9)

    box(3.98,6.83,3.3,.68,'Current letter: D','26-entry one-hot input',ORANGE)
    box(3.98,5.73,3.3,.68,'Movement: (+2.5, +2.0)','2-entry displacement input',ORANGE)
    arrow((7.30,7.12),(8.0,6.38),ORANGE);arrow((7.30,6.05),(8.0,6.13),ORANGE)
    box(8.03,5.70,3.50,1.15,'Frozen GRU','3 layers; 512 units per layer',ORANGE)
    txt(9.78,7.44,'State carried from earlier inputs',10,ORANGE)
    arrow((9.78,7.20),(9.78,6.88),ORANGE)
    arrow((9.78,5.68),(9.78,5.31),ORANGE)
    box(8.03,4.38,3.50,.89,'Predict the next letter','Correct target here: A',ORANGE)
    txt(5.67,4.80,'Record GRU state H\nas a fitting target for the TPR',10,ORANGE)
    arrow((8.00,5.92),(6.02,5.23),ORANGE)
    txt(.18,3.96,'The map is shown for the reader. The GRU receives only the current letter and displacement; A is scored before it is supplied.',10,ha='left')
    ax.plot([.18,11.82],[3.69,3.69],color='#dddddd',lw=.8)

    txt(.18,3.35,'B  The TPR approximates the GRU state for an observed history',13,BLUE,ha='left',fontweight='bold')
    txt(.18,2.96,'Letter vector',10,BLUE,ha='left');txt(1.63,2.96,'Position vector',10,GREEN,ha='left')
    f=np.array([.8,-.4,.3,-.6]);r=np.array([.6,-.7,.2,.9])
    tiles(.56,1.60,f[:,None]);txt(.69,1.31,r'$f(\mathrm{D})$',12,BLUE)
    txt(1.25,2.10,r'$\otimes$',18)
    tiles(1.80,2.05,r[None,:]);txt(2.33,1.68,r'$r(0,0)$',12,GREEN)
    arrow((3.0,2.16),(3.56,2.16),BLUE)
    tiles(3.74,1.59,np.outer(f,r));txt(4.26,1.28,'One binding',10,BLUE)
    txt(5.12,2.1,'+',18)
    other=np.outer(np.roll(f,1),np.roll(r,2))
    tiles(5.54,1.59,other);txt(6.06,1.28,'Sum of other\nobserved bindings',10,BLUE)
    txt(7.26,2.1,'=',18)
    tiles(8.02,1.59,np.outer(f,r)+other)
    txt(8.54,1.28,r'Summed tensor $M$',10,BLUE)
    txt(10.59,2.64,'Predict the GRU state',10,BLUE,fontweight='bold')
    arrow((9.19,2.13),(9.68,2.13),BLUE)
    box(9.72,1.57,1.98,.91,'Learned map',r'$M,q\;\mapsto\;\widehat H$',BLUE)
    txt(10.69,1.05,'Fit '+r'$\widehat H$'+' to recorded '+r'$H$',10,BLUE)
    txt(.18,.67,'One term per observed location; repeated visits are counted once. Context q describes the current input and movement.',10,ha='left')
    txt(.18,.36,'Tiles illustrate signed feature values (blue negative, red positive), not fitted data; actual bindings are 26 × 16.',9,ha='left')
    txt(.18,.08,'Original schematic: scene task after Ventura et al. (2026), Fig. 1B; TPR construction after McCoy et al. (2026), Fig. 2.1.',9,color='#555555',ha='left')
    for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/f'setup.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
