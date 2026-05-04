#!/usr/bin/env bash
#
# make_splicelist.sh
# ------------------
# Create a two-column “splicelist” file for a given sample:
#
#   sample   condition
#   <sample> 0
#   <sample> 0
#   <sample> 0
#   <sample> 1
#   <sample> 1
#   <sample> 1
#
# By default the script writes three rows for condition 0 (often “control”)
# and three rows for condition 1 (often “treatment”).  You can override those
# counts on the command line.
#
# Usage:
#   ./make_splicelist.sh SAMPLE_NAME [N_CTRL] [N_TREAT]
#
# Example:
#   ./make_splicelist.sh NA12878 2 4
#
# This will create a file `NA12878.splicelist` with:
#   2 rows of condition 0
#   4 rows of condition 1
set -euo pipefail

################################################################################
# Argument parsing                                                             #
################################################################################
usage() {
  echo "Usage: $0 SAMPLE_NAME [N_CTRL] [N_TREAT]" >&2
  exit 1
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage
fi

sample="$1"
n_ctrl="${2:-3}"     # default 3 control replicates
n_treat="${3:-3}"    # default 3 treatment replicates
outfile="${sample}.splicelist"

################################################################################
# Generate file                                                                #
################################################################################
{
  printf "sample\tcondition\n"           # header

  # Control (condition 0)
  for ((i = 0; i < n_ctrl; ++i)); do
    printf "%s\t0\n" "$sample"
  done

  # Treatment (condition 1)
  for ((i = 0; i < n_treat; ++i)); do
    printf "%s\t1\n" "$sample"
  done
} > "$outfile"

echo "Created: $outfile  (controls: $n_ctrl, treatments: $n_treat)"
