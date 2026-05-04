#!/usr/bin/env python3
"""

Convert junc file

    chr1  91629   92090 x  y       z
    ...

into

    chr1 91629 . 0 .
    chr1 92089 . 1 .

and deduplicate identical lines.

Usage
-----
python junc_to_splicesite.py input.tsv > output.txt
# or read from STDIN:
cat input.tsv | python junc_to_splicesite.py > output.txt
"""

import sys
import csv

def convert(handle):
    reader = csv.reader(handle, delimiter=' ')
    header = next(reader, None)           # skip header line

    seen = set()                          # store emitted lines for dedup

    for row in reader:
        if not row or row[0].startswith('#'):
            continue                      # skip blank/comment lines

        chrom, start, end = row[0], int(row[1])+1, int(row[2])-1

        # Two output lines: one for start (flag 0), one for end (flag 1)
        for pos, flag in ((start, '0'), (end, '1')):
            out_line = f"{chrom}\t{pos}\t.\t{flag}\t."
            if out_line not in seen:      # deduplicate
                print(out_line)
                seen.add(out_line)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            convert(f)
    else:
        convert(sys.stdin)
