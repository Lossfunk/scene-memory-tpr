#!/usr/bin/env python3
"""Draw the TPR-to-GRU intervention protocol; no experimental data are changed."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
BLUE = '#405e88'
ORANGE = '#b24a24'
INK = '#242424'


def main():
    fig, ax = plt.subplots(figsize=(12, 7.0))
    ax.set(xlim=(0, 12), ylim=(0, 7))
    ax.axis('off')
    fig.patch.set_facecolor('white')

    def text(x, y, value, size=11, color=INK, **kwargs):
        ax.text(x, y, value, fontsize=size, color=color, ha='center', va='center', **kwargs)

    def box(x, y, w, h, title, detail, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle='round,pad=0.02,rounding_size=0.06',
                     facecolor='white', edgecolor=color, linewidth=1.1))
        text(x+w/2, y+h*.70, title, 11, color, fontweight='bold')
        text(x+w/2, y+h*.29, detail, 10)

    def arrow(start, end, color=INK):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.3,
                                    shrinkA=2, shrinkB=2))

    text(6, 6.78, 'Calculate a change in the TPR model. Test its effect in the GRU.', 16, fontweight='bold')
    text(2.9, 6.24, 'TPR: explanatory model', 14, BLUE, fontweight='bold')
    text(9.35, 6.24, 'GRU: trained scene network', 14, ORANGE, fontweight='bold')
    text(2.9, 5.90, 'Fitted to GRU states and state differences', 10)
    text(9.35, 5.90, 'Weights stay fixed throughout', 10)
    ax.plot([6.25, 6.25], [1.65, 6.40], color='#dddddd', lw=.8, zorder=0)

    box(.25, 4.68, 2.55, .86, 'Original assignment', r'Tensor $M$; context $q$', BLUE)
    box(3.08, 4.68, 2.55, .86, 'Changed assignment', r'Tensor $M+\Delta M$; same $q$', BLUE)
    arrow((1.525, 4.66), (1.525, 4.25), BLUE)
    arrow((4.355, 4.66), (4.355, 4.25), BLUE)
    box(.25, 3.39, 2.55, .85, 'Predicted GRU state', r'$\widehat{H}(M,q)$', BLUE)
    box(3.08, 3.39, 2.55, .85, 'Predicted GRU state', r'$\widehat{H}(M+\Delta M,q)$', BLUE)
    arrow((1.525, 3.37), (2.2, 2.91), BLUE)
    arrow((4.355, 3.37), (3.7, 2.91), BLUE)
    box(.70, 1.96, 4.80, .94, 'Subtract the two TPR predictions',
        r'$\Delta\widehat{H}=\widehat{H}(M+\Delta M,q)-\widehat{H}(M,q)$', BLUE)

    box(7.25, 4.68, 4.25, .86, 'Observe the original scene', '35 letter-and-movement inputs', ORANGE)
    arrow((9.375, 4.66), (9.375, 4.25), ORANGE)
    box(7.25, 3.39, 4.25, .85, 'Actual GRU state', r'$H$', ORANGE)
    arrow((9.375, 3.37), (9.375, 2.92), ORANGE)
    box(7.25, 1.96, 4.25, .94, 'Add the predicted change; clip to bounds',
        r"$H'=\mathrm{clip}_{[-1,1]}(H+\Delta\widehat{H})$", ORANGE)
    arrow((5.52, 2.42), (7.22, 2.42), BLUE)
    text(6.36, 2.80, 'Apply once', 10, BLUE)
    arrow((9.375, 1.94), (9.375, 1.43), ORANGE)

    text(3.2, 1.15, 'Replacement or two-location swap\nAll assignment changes and subtractions\nabove happen in the TPR model.', 11, BLUE)
    box(7.25, .48, 4.25, .94, 'Test the GRU’s recall',
        '5 further steps, then query a location', ORANGE)
    text(6, .12, 'Test each location on a separate copy. Edited and queried letters are not observed along the test path.', 10)
    fig.subplots_adjust(left=.02, right=.98, top=.98, bottom=.02)
    for ext in ['png', 'pdf']:
        fig.savefig(ROOT/'figures'/f'methods.{ext}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
