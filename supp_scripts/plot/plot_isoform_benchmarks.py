#!/usr/bin/env python3
"""
Plots benchmarking results for gene isoform counts.

Generates a 2x4 grid of bar charts comparing isoform discovery across 
various error correction tools and transcript assemblers, separated by 
de-novo and reference-guided (GTF) approaches.
"""

import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------
# 1. Configuration & Data
# ------------------------------------------------------------------
TOOLS = ['No correction', 'SpliSync', 'TranscriptClean', 'deSALT', 'isONcorrect']
ASSEMBLERS = ['StringTie3', 'Bambu', 'IsoQuant', 'FLAIR']

# Data structure: Each inner list corresponds to the ASSEMBLERS index
DENOVO_COUNTS = [
   ,  # StringTie3
   ,  # Bambu
    [ 5744,  5851,  5738,  6041,  5742],  # IsoQuant
   ,  # FLAIR
]

GUIDED_COUNTS = [
   ,  # StringTie3
   ,  # Bambu
    [ 7471,  7546,  7464,  7704,  7469],  # IsoQuant
   ,  # FLAIR
]


# ------------------------------------------------------------------
# 2. Plotting Logic
# ------------------------------------------------------------------
def plot_isoform_benchmarks():
    """Builds and displays the benchmarking figure."""
    
    # Setup the figure grid
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    colors = plt.cm.tab10.colors[:len(TOOLS)]
    x_positions = np.arange(len(TOOLS))
    
    # Formatting constants
    headroom = 0.15
    label_offset = 0.03

    # Iterate through assemblers and data simultaneously using zip()
    data_iterator = zip(ASSEMBLERS, DENOVO_COUNTS, GUIDED_COUNTS)
    
    for col_idx, (assembler, denovo, guided) in enumerate(data_iterator):
        
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]
        
        # Calculate shared Y-axis limit for the column to keep pairs visually comparable
        col_max = max(max(denovo), max(guided))
        y_limit = col_max * (1 + headroom)

        # ---------- Top Row : De-novo ----------
        ax_top.bar(x_positions, denovo, color=colors, edgecolor='black')
        ax_top.set_title(assembler, fontsize=14)
        ax_top.set_ylim(0, y_limit)
        ax_top.set_xticks([])  # Hide x-ticks on top row
        ax_top.tick_params(axis='y', labelsize=10)

        for x, y in enumerate(denovo):
            ax_top.text(
                x, y + (col_max * label_offset), f'{y:,}',
                ha='center', va='bottom', fontsize=8
            )

        # ---------- Bottom Row : Guided (GTF) ----------
        ax_bot.bar(x_positions, guided, color=colors, edgecolor='black')
        ax_bot.set_title(f'{assembler} | GTF', fontsize=14)
        ax_bot.set_ylim(0, y_limit)
        ax_bot.set_xticks(x_positions)
        ax_bot.set_xticklabels(TOOLS, rotation=30, ha='right', fontsize=10)
        ax_bot.tick_params(axis='y', labelsize=10)

        for x, y in enumerate(guided):
            ax_bot.text(
                x, y + (col_max * label_offset), f'{y:,}',
                ha='center', va='bottom', fontsize=8
            )

        # Apply y-axis labels only to the leftmost column
        if col_idx == 0:
            ax_top.set_ylabel('Gene Isoforms', fontsize=12)
            ax_bot.set_ylabel('Gene Isoforms', fontsize=12)

    # ------------------------------------------------------------------
    # 3. Layout Fine-tuning & Display
    # ------------------------------------------------------------------
    plt.tight_layout(h_pad=2.0, w_pad=2.0)
    plt.show()


if __name__ == '__main__':
    plot_isoform_benchmarks()
