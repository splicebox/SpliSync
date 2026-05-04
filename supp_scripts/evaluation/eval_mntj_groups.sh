K_THRESHOLD=20
FILTERED_POSITIONS="k${K_THRESHOLD}.chr_pos"
SORTED_POSITIONS="${FILTERED_POSITIONS}.sorted"

python /data/lflorea1/wlui3/tools/SpliSync/mntj_groupfilter.py "$K_THRESHOLD"
sort "$FILTERED_POSITIONS" > "$SORTED_POSITIONS"

echo "# of group with header:"
wc -l "$SORTED_POSITIONS"

TP=$(comm -1 -2 "$SORTED_POSITIONS" "$1" | wc -l)
echo "TP: $TP"
FP=$(comm -2 -3 "$SORTED_POSITIONS" "$1" | wc -l)
echo "FP: $FP"
FN=$(comm -1 -3 "$SORTED_POSITIONS" "$1" | wc -l)
echo "FN: $FN"
echo "$TP $FP $FN"
