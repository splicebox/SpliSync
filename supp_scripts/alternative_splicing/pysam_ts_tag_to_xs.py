import pysam
import sys

input_bam = sys.argv[1]
output_bam = sys.argv[1]+".xs.bam"

with pysam.AlignmentFile(input_bam, "rb") as reader:
    with pysam.AlignmentFile(output_bam, "wb", template=reader) as writer:
        for read in reader:
            if read.has_tag("ts"):
                ts_val = read.get_tag("ts")
                
                # Check if read is mapped to reverse strand (0x10)
                is_reverse = read.is_reverse
                
                if is_reverse:
                    # Flip the tag
                    xs_val = "-" if ts_val == "+" else "+"
                else:
                    # Keep the tag as is
                    xs_val = ts_val
                
                read.set_tag("XS", xs_val)
            
            writer.write(read)
