import sys

PLUS_SUFFIX  = ';0'   # append when strand is '+'
MINUS_SUFFIX = ';16'   # append when strand is '-'

for line in sys.stdin:
    if not line.strip() or line.startswith('#'):
        sys.stdout.write(line)        # pass through blank / comment lines
        continue

    cols = line.rstrip('\n').split('\t')
    if cols[5] == '+':
        cols[3] += PLUS_SUFFIX
    elif cols[5] == '-':
        cols[3] += MINUS_SUFFIX
    cols[10] += ','
    cols[11] += ','
    sys.stdout.write('\t'.join(cols) + '\n')

