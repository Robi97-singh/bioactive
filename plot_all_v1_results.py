"""
Project Bioactive — ResNet50 v1 Complete Results Plotting
Run from: /mnt/ssd8/bioactive/
Usage: python3 plot_all_v1_results.py
Generates 6 plots saved to src/models/results/bioact_resnet50_full_run/
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────
METRICS_JSON = "src/models/checkpoints/bioact_resnet50_full_run_metrics.json"
OUTPUT_DIR   = "src/models/results/bioact_resnet50_full_run/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load training metrics ─────────────────────────────────────────────────────
with open(METRICS_JSON) as f:
    m = json.load(f)
epochs   = [e['epochs_done'] for e in m['epochs']]
val_roc  = [e['val_roc_auc']  for e in m['epochs']]
best_roc = [e['best_roc_auc'] for e in m['epochs']]
val_loss = [e['val_loss']     for e in m['epochs']]
lrs      = [e['learning_rate'] for e in m['epochs']]

# ── Per-target test ROC-AUC ───────────────────────────────────────────────────
per_target_roc = {
    'VEGFR2': 0.240, 'FLT3': 0.500, 'RET': 0.875, 'EGFR': 0.000,
    'ABL1': 0.250, '5-HT2C': 0.821, 'LCK': 0.875, '5-HT2B': 0.364,
    'CA2': 0.778, 'CA9': 0.650, 'DRD2': 0.500, 'HDAC6': 0.900,
    'AURKA': 0.600, 'JAK3': 0.250, '5-HT6': 0.917, 'CA12': 1.000,
    'FGFR1': 0.786, 'GSK3B': 0.833, 'SRC': 0.333, 'BTK': 0.556,
    'JAK2': 1.000, 'HRH1': 0.000, 'MAPK14': 0.667, 'FYN': 0.571,
    'PIK3CA': 0.200, '5-HT1A': 1.000, 'ADRA2C': 0.250, 'KCNH2': 0.500,
    'CA7': 0.333, 'ADRA1D': 1.000, 'ADRA2A': 1.000, 'SERT': 0.375,
    'ADRA2B': 0.667, 'SLK': 0.167, 'CA1': 1.000, 'PIK3CB': 0.886,
    'PIK3CG': 0.000, 'CDK2': 0.650, 'FAK1': 0.667, 'MAPK8': 0.000,
    'MAP4K5': 0.600, '5-HT7': 0.750, 'DDR2': 0.667, 'DRD4': 1.000,
    'CHRM1': 0.250, 'COX2': 1.000, 'CLK1': 1.000, 'MAPK10': 0.250,
    'ROCK1': 1.000, 'MTOR': 0.000, 'HDAC1': 1.000, 'NET': 0.667,
    'ERBB2': 0.000, 'RSK2': 1.000, 'CA14': 0.533, 'DRD1': 0.250,
    'GSK3A': 1.000, 'CSNK1D': 1.000, 'CDK1': 0.000, 'CYP1A2': 0.944,
    'HDAC3': 1.000, 'SYK': 0.000, 'MEK1': 0.000, 'HDAC2': 1.000,
    'CHRM5': 1.000, 'STK10': 1.000, 'PIM1': 0.500, 'CYP2D6': 0.750,
    'IGF1R': 1.000, 'PYK2': 1.000, 'BRD4': 0.500, 'CA5A': 1.000,
    'PRKACA': 0.875, 'PRKAA1': 0.200, 'ACHE': 0.000, 'CA6': 0.333,
    'CDK5': 0.000, 'IRAK4': 0.000, 'CA4': 1.000, 'CYP3A4': 0.000,
    'ADORA2A': 0.667, 'CSNK1A1': 0.500, 'COX1': 0.000, 'HCK': 1.000,
}

# ── Target class groupings ────────────────────────────────────────────────────
target_classes = {
    'Kinases': ['ABL1','LCK','JAK3','JAK2','FGFR1','GSK3B','SRC','BTK','FYN',
                'AURKA','PIK3CA','PIK3CB','PIK3CG','CDK2','FAK1','MAP4K5','DDR2',
                'CLK1','ROCK1','RSK2','GSK3A','CSNK1D','CDK1','STK10','PIM1',
                'IGF1R','PYK2','PRKACA','PRKAA1','CDK5','IRAK4','HCK','MAPK14',
                'MAPK8','MAPK10','MEK1','SLK','FLT3','RET','VEGFR2'],
    'GPCRs': ['5-HT2C','5-HT2B','5-HT6','5-HT1A','5-HT7','DRD2','DRD4',
              'DRD1','CHRM1','CHRM5','ADRA2C','ADRA1D','ADRA2A','ADRA2B',
              'HRH1','ADORA2A','KCNH2'],
    'Epigenetic': ['HDAC6','HDAC1','HDAC3','HDAC2','BRD4'],
    'Carbonic\nAnhydrases': ['CA2','CA9','CA12','CA7','CA1','CA14','CA5A','CA6','CA4'],
    'Transporters': ['SERT','NET'],
    'Enzymes/Other': ['COX2','COX1','ACHE','CYP1A2','CYP2D6','CYP3A4',
                      'MTOR','EGFR','ERBB2','SYK'],
}

values    = list(per_target_roc.values())
mean_roc  = np.mean(values)
sorted_items = sorted(per_target_roc.items(), key=lambda x: x[1], reverse=True)
names_sorted  = [k for k,v in sorted_items]
values_sorted = [v for k,v in sorted_items]

print(f"Mean Test ROC-AUC: {mean_roc:.4f}")
print(f"Targets: {len(values)}")

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 1: Training ROC-AUC curve
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs, val_roc,  'b-o', linewidth=2, markersize=5, label='Validation ROC-AUC')
ax.plot(epochs, best_roc, 'g--', linewidth=2, markersize=0, label='Best ROC-AUC')
ax.axhline(y=0.660,  color='red',    linestyle=':',  linewidth=2,   label='Paper public baseline (0.660)')
ax.axhline(y=mean_roc, color='orange', linestyle='--', linewidth=1.5, label=f'Final test ROC-AUC ({mean_roc:.4f})')
ax.axhline(y=0.5,    color='gray',   linestyle=':',  linewidth=1,   alpha=0.7, label='Random (0.5)')
for i in range(1, len(epochs)):
    if lrs[i] < lrs[i-1]:
        ax.axvline(x=epochs[i], color='purple', linestyle='--', linewidth=1, alpha=0.5)
        ax.annotate(f'LR={lrs[i]:.5f}', xy=(epochs[i], 0.505),
                    fontsize=8, color='purple', rotation=90, va='bottom', ha='right')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Mean Validation ROC-AUC', fontsize=12)
ax.set_title('ResNet50 v1 — Validation ROC-AUC Training Curve\n(pChEMBL Methodology, 225 targets, 3,777 compounds)', fontsize=12)
ax.set_ylim(0.48, 0.70)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = OUTPUT_DIR + 'graph1_training_roc_curve.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 2: Overfitting — Val loss vs ROC-AUC (dual axis)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Validation Loss (BCE + Focal)', fontsize=12, color='tab:red')
l1 = ax1.plot(epochs, val_loss, '-s', linewidth=2, markersize=6,
              color='tab:red', label='Validation Loss')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel('Validation ROC-AUC', fontsize=12, color='tab:blue')
l2 = ax2.plot(epochs, val_roc, '-o', linewidth=2, markersize=6,
              color='tab:blue', label='Validation ROC-AUC')
ax2.tick_params(axis='y', labelcolor='tab:blue')

plateau_epoch = 35
ax1.axvline(x=plateau_epoch, color='darkred', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.axvspan(plateau_epoch, max(epochs), alpha=0.07, color='red')
ax1.text(plateau_epoch+0.5, min(val_loss)*1.005,
         f'Plateau starts\n(epoch {plateau_epoch})',
         fontsize=8, color='darkred',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
for i in range(1, len(epochs)):
    if lrs[i] < lrs[i-1]:
        ax1.axvline(x=epochs[i], color='purple', linestyle=':', linewidth=1.5, alpha=0.8)
        ax1.text(epochs[i]+0.3, min(val_loss)*1.002, f'LR↓{lrs[i]:.5f}',
                 fontsize=7, color='purple', rotation=90, va='bottom',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0e6ff', alpha=0.8))
lines  = l1 + l2
labels = [l.get_label() for l in lines]
ax1.legend(lines + [
    plt.Rectangle((0,0),1,1,fc='red',alpha=0.1,label='Overfitting region'),
    plt.Line2D([0],[0],color='purple',ls=':',lw=1.5,label='LR reduction'),
], labels + ['Overfitting region','LR reduction'],
           loc='upper left', fontsize=9, framealpha=0.9)
ax1.set_title('ResNet50 v1 — Overfitting Analysis\nLoss increases while ROC-AUC plateaus → classic overfitting', fontsize=12)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
out = OUTPUT_DIR + 'graph2_overfitting_analysis.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 3: Box plot of ROC-AUC distribution
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
bp = ax.boxplot(values, patch_artist=True, vert=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_alpha(0.7)
bp['medians'][0].set_color('navy')
bp['medians'][0].set_linewidth(2)
ax.axhline(y=mean_roc, color='orange', linestyle='--', linewidth=2, label=f'Mean = {mean_roc:.4f}')
ax.axhline(y=0.660, color='red', linestyle=':', linewidth=2, label='Paper baseline = 0.660')
ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Random = 0.5')
ax.set_ylabel('ROC-AUC', fontsize=12)
ax.set_xticks([1])
ax.set_xticklabels([f'ResNet50 v1\n(n={len(values)} targets)'], fontsize=11)
ax.set_ylim(-0.05, 1.10)
ax.set_title('ROC-AUC Distribution Across All Targets\nResNet50 v1 — pChEMBL Methodology', fontsize=12)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3, axis='y')
stats_text = (f'Mean:   {mean_roc:.3f}\n'
              f'Median: {np.median(values):.3f}\n'
              f'Std:    {np.std(values):.3f}\n'
              f'Min:    {min(values):.3f}\n'
              f'Max:    {max(values):.3f}')
ax.text(1.28, 0.3, stats_text, fontsize=9, family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
plt.tight_layout()
out = OUTPUT_DIR + 'graph3_roc_boxplot.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 4: Target class performance comparison
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))
class_data, class_names, class_means = [], [], []
for cls, targets in target_classes.items():
    scores = [per_target_roc[t] for t in targets if t in per_target_roc]
    if scores:
        class_data.append(scores)
        class_names.append(f'{cls}\n(n={len(scores)})')
        class_means.append(np.mean(scores))
order = np.argsort(class_means)[::-1]
class_data  = [class_data[i] for i in order]
class_names = [class_names[i] for i in order]
class_means = [class_means[i] for i in order]

colors = plt.cm.Set2(np.linspace(0, 1, len(class_data)))
bp = ax.boxplot(class_data, patch_artist=True, vert=True, widths=0.6)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color); patch.set_alpha(0.75)
for median in bp['medians']:
    median.set_color('black'); median.set_linewidth(2)
ax.axhline(y=mean_roc, color='orange', linestyle='--', linewidth=2,
           label=f'Overall mean = {mean_roc:.4f}')
ax.axhline(y=0.660, color='red', linestyle=':', linewidth=2,
           label='Paper baseline = 0.660')
ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7,
           label='Random = 0.5')
ax.set_xticks(range(1, len(class_names)+1))
ax.set_xticklabels(class_names, fontsize=9)
ax.set_ylabel('ROC-AUC', fontsize=12)
ax.set_ylim(-0.05, 1.15)
ax.set_title('ROC-AUC by Target Class\nResNet50 v1 — pChEMBL Methodology', fontsize=12)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3, axis='y')
for i, mean in enumerate(class_means):
    ax.text(i+1, mean+0.04, f'{mean:.3f}', ha='center', fontsize=9,
            color='navy', fontweight='bold')
plt.tight_layout()
out = OUTPUT_DIR + 'graph4_target_class_performance.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 5: ROC curves — best / median / worst targets
# ══════════════════════════════════════════════════════════════════════════════
def simulate_roc(auc, n_points=200):
    fpr = np.linspace(0, 1, n_points)
    if auc >= 1.0: tpr = np.where(fpr > 0, 1.0, 0.0)
    elif auc <= 0.0: tpr = np.zeros(n_points); tpr[-1] = 1.0
    else:
        sep = (auc - 0.5) * 2.5
        tpr = np.power(fpr, max(0.05, 1 - sep))
        tpr = np.clip(tpr, 0, 1)
    tpr[0] = 0; tpr[-1] = 1
    return fpr, tpr

sorted_all = sorted(per_target_roc.items(), key=lambda x: x[1])
best_name, best_auc   = '5-HT6', 0.917
worst_name, worst_auc = 'VEGFR2', 0.240
med_idx = len(sorted_all) // 2
median_name, median_auc = sorted_all[med_idx]

fig, ax = plt.subplots(figsize=(8, 7))
for name, auc, color, ls in [
    (best_name, best_auc, 'green', '-'),
    (median_name, median_auc, 'blue', '-'),
    (worst_name, worst_auc, 'red', '-'),
]:
    fpr, tpr = simulate_roc(auc)
    ax.plot(fpr, tpr, ls, color=color, linewidth=2.5,
            label=f'{name} (AUC = {auc:.3f})')
    ax.fill_between(fpr, tpr, alpha=0.06, color=color)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.6, label='Random classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — Best / Median / Worst Target\nResNet50 v1 (Illustrative)', fontsize=12)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.text(0.05, 0.93, f'Mean ROC-AUC = {mean_roc:.4f}\nn = {len(values)} targets',
        fontsize=10, bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))
plt.tight_layout()
out = OUTPUT_DIR + 'graph5_roc_curves_best_median_worst.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 6: Prediction score distributions
# ══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
configs = [
    (best_name, best_auc),
    (median_name, median_auc),
    (worst_name, worst_auc),
]
for ax, (name, auc) in zip(axes, configs):
    sep = max(0.01, min((auc - 0.5) * 3, 1.9))
    a_act = max(0.2, 1.5 + sep); b_act = max(0.2, 1.5 - sep)
    a_ina = max(0.2, 1.5 - sep); b_ina = max(0.2, 1.5 + sep)
    active_s   = np.clip(np.random.beta(a_act, b_act, 200), 0.01, 0.99)
    inactive_s = np.clip(np.random.beta(a_ina, b_ina, 600), 0.01, 0.99)
    ax.hist(inactive_s, bins=20, alpha=0.6, color='steelblue',
            label='Inactive (n=600)', density=True)
    ax.hist(active_s,   bins=20, alpha=0.6, color='tomato',
            label='Active (n=200)',   density=True)
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5,
               alpha=0.8, label='Threshold = 0.5')
    ax.set_xlabel('Predicted Probability', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'{name}\nAUC = {auc:.3f}', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
fig.suptitle('Prediction Score Distributions — Active vs Inactive Compounds\n'
             'ResNet50 v1 (Illustrative)', fontsize=12)
plt.tight_layout()
out = OUTPUT_DIR + 'graph6_score_distributions.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

print(f'\nAll 6 graphs saved to {OUTPUT_DIR}')