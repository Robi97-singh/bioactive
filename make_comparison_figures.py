import os, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
R="/mnt/ssd8/bioactive/src/models/results"; O=R+"/comparison"; os.makedirs(O,exist_ok=True)
M=[("bioact_resnet50_v2_faithful","ResNet50","Supervised CNN",448),
   ("bioact_dinov2_base","DINOv2-Base","Self-supervised ViT",448),
   ("bioact_clip_vitl14","CLIP ViT-L/14","Vision-Language",224),
   ("bioact_biomedclip","BiomedCLIP","Vision-Language",224)]
C={"Supervised CNN":"#6a3d9a","Self-supervised ViT":"#1f78b4","Vision-Language":"#33a02c"}
BASE=0.702; PM=0.660; PS=0.094
fr={}
for md,nm,par,res in M:
    df=pd.read_csv(f"{R}/{md}/plots/per_assay_auc.csv")
    un=[c for c in df.columns if str(c).startswith("Unnamed")]
    df=df.rename(columns={un[0]:"assay"}) if un else df
    s=df[["assay","test_roc_auc"]].copy(); s["assay"]=s["assay"].astype(str)
    fr[nm]=s.set_index("assay")["test_roc_auc"]
mat=pd.concat(fr.values(),axis=1,keys=fr.keys()).dropna(how="all")
means=mat.mean(0);
n2p={nm:par for _,nm,par,_ in M}; n2r={nm:r for _,nm,_,r in M}
summ=pd.DataFrame({"mean":means.round(4),"median":mat.median(0).round(4),
  "ge0.70":(mat>=0.70).sum(0),"lt0.50":(mat<0.50).sum(0),
  "paradigm":[n2p[n] for n in means.index],"res":[n2r[n] for n in means.index],
  "delta_vs_resnet":(means-BASE).round(4),"beats_resnet":means>BASE}).sort_values("mean",ascending=False)
summ.to_csv(O+"/comparison_summary.csv"); print(summ.to_string())
# Fig 1: bar
order=means.sort_values(ascending=False).index.tolist()
fig,ax=plt.subplots(figsize=(9,6)); xs=np.arange(len(order))
ax.bar(xs,[means[n] for n in order],color=[C[n2p[n]] for n in order],width=0.62,zorder=3)
ax.axhline(BASE,color="#6a3d9a",lw=2,zorder=2,label=f"ResNet50 baseline ({BASE:.3f})")
ax.axhline(PM,color="#e31a1c",ls="--",lw=1.6,zorder=2,label=f"Paper 6-fold CV ({PM:.3f}\u00b1{PS:.3f})")
ax.axhspan(PM-PS,PM+PS,color="#e31a1c",alpha=0.08,zorder=1)
for i,n in enumerate(order):
    ax.text(i,means[n]+0.006,f"{means[n]:.3f}",ha="center",va="bottom",fontsize=11,fontweight="bold")
    ax.text(i,0.02,f"{n2r[n]}px",ha="center",va="bottom",fontsize=8,color="white",fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels(order,fontsize=10)
ax.set_ylabel("Mean test ROC-AUC (29 assays)"); ax.set_ylim(0,0.78)
ax.set_title("VLM benchmark on Cell Painting bioactivity (single fold, 29 assays)")
ph=[Patch(facecolor=c,label=p) for p,c in C.items()]
l1=ax.legend(handles=ph,title="Paradigm",loc="upper right",fontsize=9); ax.add_artist(l1)
ax.legend(loc="lower center",fontsize=9,bbox_to_anchor=(0.5,-0.16),ncol=2)
ax.grid(axis="y",alpha=0.25,zorder=0); plt.tight_layout()
plt.savefig(O+"/comparison_bar.png",dpi=150,bbox_inches="tight"); plt.close(); print("saved comparison_bar.png")
# Fig 2: heatmap
ao=mat["ResNet50"].sort_values(ascending=False).index.tolist()
H=mat.loc[ao,order].T
fig,ax=plt.subplots(figsize=(15,3.6))
im=ax.imshow(H.values,aspect="auto",cmap="RdYlGn",vmin=0.4,vmax=0.9)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order,fontsize=10)
ax.set_xticks(range(len(ao))); ax.set_xticklabels(ao,rotation=90,fontsize=6)
ax.set_xlabel("Assay (sorted by ResNet AUC)"); ax.set_title("Per-assay test ROC-AUC across models")
fig.colorbar(im,ax=ax,fraction=0.012,pad=0.01).set_label("ROC-AUC"); plt.tight_layout()
plt.savefig(O+"/comparison_heatmap.png",dpi=150,bbox_inches="tight"); plt.close(); print("saved comparison_heatmap.png")
# Fig 3: paired
oth=[n for n in order if n!="ResNet50"]; base=mat["ResNet50"]
fig,axes=plt.subplots(1,len(oth),figsize=(5*len(oth),5),sharex=True,sharey=True)
for ax,n in zip(axes,oth):
    cc=mat[[n]].join(base,how="inner").dropna(); x=cc["ResNet50"].values; y=cc[n].values
    ax.scatter(x,y,c=C[n2p[n]],s=42,alpha=0.75,zorder=3,edgecolor="white",linewidth=0.5)
    ax.plot([0.2,1],[0.2,1],"k--",lw=1,alpha=0.6,zorder=2)
    ax.set_title(f"{n} vs ResNet50\n{n} higher on {int((y>x).sum())}/{len(x)}",fontsize=10)
    ax.set_xlabel("ResNet50 AUC"); ax.set_ylabel("Model AUC")
    ax.set_xlim(0.25,1); ax.set_ylim(0.25,1); ax.grid(alpha=0.2,zorder=0); ax.set_aspect("equal")
fig.suptitle("Paired per-assay comparison vs ResNet50 (below y=x: ResNet wins)"); plt.tight_layout()
plt.savefig(O+"/comparison_paired.png",dpi=150,bbox_inches="tight"); plt.close(); print("saved comparison_paired.png")
print("DONE ->",O)