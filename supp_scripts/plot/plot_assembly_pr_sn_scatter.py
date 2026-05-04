#!/usr/bin/env python3
"""
Generates a Precision-Recall scatter plot for transcriptomic assembly and correction tools.

This script evaluates simulated human cDNA (R9) data, comparing the performance 
of various error correction methods across different assembly pipelines.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ------------------------------------------------------------------
# 1. Configuration & Data Setup
# ------------------------------------------------------------------
# Data structure: 'Assembler Group': {'Correction Tool': (Precision, Recall)}
ASSEMBLY_DATA = {
    'No assembly': {
        'No correction'  : (0.598480237, 0.869214557),
        'TranscriptClean': (0.598480237, 0.869214557),
        'deSALT'         : (0.533752491, 0.854862753),
        'isONcorrect'    : (0.597789043, 0.869193996),
        'LoRMA'          : (0.936885935, 0.366644392),
        'SpliSync'       : (0.963391588, 0.846406909),
    },
    'StringTie3 | gtf': {
        'No correction'  : (0.988327083, 0.505294541),
        'SpliSync'       : (0.986716809, 0.524647887),
        'TranscriptClean': (0.987543095, 0.513868613),
        'deSALT'         : (0.989076465, 0.476138583),
        'isONcorrect'    : (0.988266050, 0.505232857),
        'LoRMA'          : (0.987407536, 0.320437956),
    },
    'StringTie3': {
        'No correction'  : (0.984746523, 0.699547651),
        'SpliSync'       : (0.981570518, 0.723049244),
        'TranscriptClean': (0.984782813, 0.699578493),
        'deSALT'         : (0.983039495, 0.633710291),
        'isONcorrect'    : (0.984672943, 0.699439704),
        'LoRMA'          : (0.985688469, 0.347661149),
    },
    'Bambu | gtf': {
        'No correction'  : (0.838662163, 0.778523697),
        'SpliSync'       : (0.839900676, 0.782414928),
        'TranscriptClean': (0.838662163, 0.778523697),
        'deSALT'         : (0.839579038, 0.778744731),
        'isONcorrect'    : (0.838752915, 0.778431171),
        'LoRMA'          : (0.836647438, 0.760440012),
    },
    'Bambu': {
        'No correction'  : (0.981961214, 0.350334122),
        'SpliSync'       : (0.987125288, 0.369682328),
        'TranscriptClean': (0.981961214, 0.350334122),
        'deSALT'         : (0.987303979, 0.356965149),
        'isONcorrect'    : (0.981960841, 0.350606559),
        'LoRMA'          : (0.985157736, 0.258281073),
    },
    'IsoQuant | gtf': {
        'No correction'  : (0.955842072, 0.428713889),
        'SpliSync'       : (0.954988980, 0.427629279),
        'TranscriptClean': (0.955093426, 0.428014804),
        'deSALT'         : (0.952614980, 0.439400637),
        'isONcorrect'    : (0.955745822, 0.428626504),
        'LoRMA'          : (0.965602106, 0.346890100),
    },
    'IsoQuant': {
        'No correction'  : (0.991249687, 0.285329495),
        'SpliSync'       : (0.989488312, 0.285483705),
        'TranscriptClean': (0.991264343, 0.284645831),
        'deSALT'         : (0.990769691, 0.306224941),
        'isONcorrect'    : (0.991336037, 0.285257531),
        'LoRMA'          : (0.991151242, 0.225701655),
    },
    'FLAIR | gtf': {
        'No correction'  : (0.742651238, 0.691806312),
        'SpliSync'       : (0.902650298, 0.718844454),
        'TranscriptClean': (0.742651238, 0.691806312),
        'deSALT'         : (0.697359509, 0.682723347),
        'isONcorrect'    : (0.742093799, 0.691760049),
        'LoRMA'          : (0.973525539, 0.361031150),
    },
    'FLAIR': {
        'No correction'  : (0.999983890, 0.638151537),
        'SpliSync'       : (0.999983847, 0.636429526),
        'TranscriptClean': (0.999983906, 0.638794078),
        'deSALT'         : (0.999991958, 0.639205305),
        'isONcorrect'    : (0.999983891, 0.638182379),
        'LoRMA'          : (1.000000000, 0.317657037),
    },
}

# Mapping specific tools to their marker styles (marker_shape, face_style)
TOOL_MARKERS = {
    'No correction'  : ('o', 'none'),
    'TranscriptClean': ('+', 'black'),
    'deSALT'         : ('s', 'black'),
    'isONcorrect'    : ('v', 'black'),
    'LoRMA'          : ('^', 'black'),
    'SpliSync'       : ('o', 'black'),
}

# Mapping assembly pipelines to their visual colors
GROUP_COLORS = {
    'No assembly'      : 'black',
    'FLAIR'            : 'lightgreen',
    'FLAIR | gtf'      : 'darkgreen',
    'StringTie3'       : 'lightcoral',
    'StringTie3 | gtf' : 'firebrick',
    'Bambu'            : 'skyblue',
    'Bambu | gtf'      : 'blue',
    'IsoQuant'         : 'gold',
    'IsoQuant | gtf'   : 'goldenrod',
}


# ------------------------------------------------------------------
# 2. Plotting Logic
# ------------------------------------------------------------------
def plot_precision_recall_scatter():
    """Builds and displays the scatter plot."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    base_marker_size = 50

    # Plot each data point
    for group, tools in ASSEMBLY_DATA.items():
        for tool, (precision, recall) in tools.items():
            
            marker_shape, face_style = TOOL_MARKERS[tool]
            edge_color = GROUP_COLORS[group]
            face_color = 'none' if face_style == 'none' else edge_color
            
            # Amplify the size of the baseline method for visibility
            marker_size = base_marker_size * 1.75 if tool == 'No correction' else base_marker_size

            ax.scatter(
                recall,
                precision,
                marker=marker_shape,
                s=marker_size,
                facecolors=face_color,
                edgecolors=edge_color,
                linewidths=1.0
            )
            
            # Visually highlight the core method with an annotation arrow
            if tool == 'SpliSync':
                ax.annotate(
                    '',
                    xy=(recall, precision),
                    xytext=(recall, precision + 0.02),
                    fontsize=8,
                    va='bottom',
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.8)
                )

    # ------------------------------------------------------------------
    # 3. Axes, Grid, & Layout Formatting
    # ------------------------------------------------------------------
    ax.set_xlabel('Recall', fontsize=16)
    ax.set_ylabel('Precision', fontsize=16)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.5, 1.05)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, linestyle=':', linewidth=0.6)
    ax.set_title('Simulated, human, cDNA, R9', fontsize=16)

    # Build custom legend handles
    tool_handles = [
        Line2D(,, marker=shape, color='black', 
               markerfacecolor='none' if face == 'none' else 'black', 
               linewidth=0, markeredgewidth=1.0 if face == 'none' else 0,
               label=tool_name, markersize=6)
        for tool_name, (shape, face) in TOOL_MARKERS.items()
    ]

    color_handles = [
        Line2D(,, marker='s', color=col, markerfacecolor=col, 
               linestyle='', label=grp) 
        for grp, col in GROUP_COLORS.items()
    ]

    # Adjust layout to fit legends at the bottom
    fig.subplots_adjust(bottom=0.25)
    
    # Render legends
    fig.legend(handles=tool_handles, title='Tool (shape)',
               loc='lower center', bbox_to_anchor=(0.5, 0.08),
               ncol=len(tool_handles), frameon=False, fontsize=10,
               title_fontsize=12)

    fig.legend(handles=color_handles, title='Assembler (color)',
               loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=5, frameon=False, fontsize=10, # Adjusted ncol for better wrapping
               title_fontsize=12)

    plt.show()

if __name__ == '__main__':
    plot_precision_recall_scatter()
