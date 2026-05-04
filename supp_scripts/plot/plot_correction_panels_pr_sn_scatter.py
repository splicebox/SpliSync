#!/usr/bin/env python3
"""
Generates a 2x3 grid of Precision-Recall scatter plots for splice site correction tools.

This script compares the performance of SpliSync against baseline and alternative 
correction tools across multiple simulated and real long-read datasets.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ------------------------------------------------------------------
# 1. Configuration & Data Setup
# ------------------------------------------------------------------
PANEL_TITLES = [
    'Simulated, human, cDNA, R9',
    'SGNEX A549, human, dRNA, R9',
    'LRGASP ENCFF683TBO, mouse, cDNA, R9',
    'CRR1887708, Chinese cattle, cDNA, R9',
    'Simulated, mouse, cDNA, R10',
    '1000G ERR14734531, human, CapTrap, R10'
]

# Data structure: List of dicts matching the panel titles.
# Format -> 'Tool': (Precision, Recall)
PANEL_DATA = [
    {
        'No correction':   (0.598480237, 0.869214557),
        'SpliSync':        (0.963391588, 0.846406909),
        'TranscriptClean': (0.598480237, 0.869214557),
        'isONcorrect':     (0.597789043, 0.869193996),
        'LoRMA':           (0.936885935, 0.366644392),
        'deSALT':          (0.533752491, 0.854862753),
    },
    {
        'No correction':   (0.743201239, 0.404197658),
        'SpliSync':        (0.942971334, 0.389542198),
        'TranscriptClean': (0.743201239, 0.404197658),
        'isONcorrect':     (0.743081916, 0.404006900),
        'deSALT':          (0.695135927, 0.400981087),
    },
    {
        'No correction':   (0.278448842, 0.549271612),
        'SpliSync':        (0.818154939, 0.513792742),
        'TranscriptClean': (0.278448842, 0.549271612),
        'isONcorrect':     (0.305323562, 0.549168682),
        'deSALT':          (0.177542523, 0.547687284),
    },
    {
        'No correction':   (0.331544863, 0.473439042),
        'SpliSync':        (0.747757836, 0.461435511),
        'TranscriptClean': (0.331544863, 0.473439042),
        'isONcorrect':     (0.329388369, 0.474388168),
        'deSALT':          (0.288200104, 0.471080796),
    },
    {
        'No correction':   (0.537616954, 0.982260033),
        'SpliSync':        (0.883587647, 0.969953929),
        'TranscriptClean': (0.537616954, 0.982260033),
        'isONcorrect':     (0.434114257, 0.982004936),
        'deSALT':          (0.470554331, 0.978568816),
    },
    {
        'No correction':   (0.568963686, 0.264247222),
        'SpliSync':        (0.833273261, 0.247113555),
        'TranscriptClean': (0.568963686, 0.264247222),
        'isONcorrect':     (0.548753748, 0.264222555),
        'deSALT':          (0.528158421, 0.262719515),
    }
]

# Mapping tools to their visual styles: (marker_shape, fill_style, color)
TOOL_MARKERS = {
    'No correction':   ('o', 'none', 'red'),
    'SpliSync':        ('o', 'full', 'green'),
    'TranscriptClean': ('+', 'full', 'black'),
    'isONcorrect':     ('v', 'full', 'black'),
    'LoRMA':           ('^', 'full', 'black'),
    'deSALT':          ('s', 'full', 'black')
}


# ------------------------------------------------------------------
# 2. Plotting Logic
# ------------------------------------------------------------------
def plot_multipanel_correction():
    """Builds and displays the 2x3 grid of precision-recall scatter plots."""
    
    # Adjusted figsize to accommodate 3 columns and long titles
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), sharex=True, sharey=True)

    # Iterate over the flattened axes grid, titles, and data simultaneously
    for ax, title, panel_dict in zip(axes.ravel(), PANEL_TITLES, PANEL_DATA):
        
        for tool, (precision, recall) in panel_dict.items():
            marker_shape, fill_style, color = TOOL_MARKERS[tool]
            
            face_color = 'none' if fill_style == 'none' else color
            edge_color = color
            
            # Emphasize the baseline 'No correction' tool
            marker_size = 80 * 1.7 if tool == 'No correction' else 80

            ax.scatter(
                recall, 
                precision, 
                marker=marker_shape, 
                s=marker_size,
                facecolors=face_color, 
                edgecolors=edge_color, 
                linewidths=1.2,
                label=tool
            )

            # Visually highlight the core method with an annotation arrow
            if tool == 'SpliSync':
                ax.annotate(
                    '', 
                    xy=(recall, precision), 
                    xytext=(recall, precision + 0.08),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.2)
                )

        ax.set_title(title, fontsize=13)
        ax.grid(True, linestyle=':', linewidth=0.5)

    # ------------------------------------------------------------------
    # 3. Axes Formatting
    # ------------------------------------------------------------------
    # Apply Y-axis labels only to the first column
    for ax in axes[:, 0]:
        ax.set_ylabel('Precision', fontsize=14)
        
    # Apply X-axis labels only to the bottom row
    for ax in axes[-1, :]:
        ax.set_xlabel('Recall', fontsize=14)

    # Standardize ticks and limits across all panels
    for ax in axes.ravel():
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(axis='both', labelsize=12)

    # ------------------------------------------------------------------
    # 4. Global Legend & Layout Display
    # ------------------------------------------------------------------
    legend_handles = []
    
    # Reconstruct the legend handles manually to ensure all tools are present 
    # (even if a tool like LoRMA is missing from a specific panel)
    for tool, (marker_shape, fill_style, color) in TOOL_MARKERS.items():
        face_color = 'none' if fill_style == 'none' else color
        legend_handles.append(
            Line2D(,, marker=marker_shape, color=color,
                   markerfacecolor=face_color, markeredgecolor=color,
                   linewidth=0, markersize=8, label=tool)
        )

    # Adjust the bottom margin to create room for the global legend
    fig.subplots_adjust(bottom=0.20)

    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.05),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=12
    )

    plt.suptitle('Splice site correction', fontsize=18, y=0.98)
    
    # tight_layout with a rect ensures it respects the suptitle and bottom legend
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    plot_multipanel_correction()
