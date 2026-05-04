"""Filter spliced groups"""

import sys
from typing import Tuple

import pandas as pd


GROUP_DATA_PATH = "group_data.txt"
INTRON_DATA_PATH = "intron_data.txt"
DIFF_SPLICED_GROUPS_PATH = "diff_spliced_groups.txt"


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Load the three input tables with `chrom` treated as text."""

	read_csv_kwargs = {"sep": "\t", "dtype": {"chrom": str}}
	group_data = pd.read_csv(GROUP_DATA_PATH, **read_csv_kwargs)
	intron_data = pd.read_csv(INTRON_DATA_PATH, **read_csv_kwargs)
	diff_spliced_groups = pd.read_csv(DIFF_SPLICED_GROUPS_PATH, **read_csv_kwargs)
	return group_data, intron_data, diff_spliced_groups


def main() -> None:
	group_data, intron_data, diff_spliced_groups = load_inputs()

	merged_data = group_data.merge(intron_data, on=["chrom", "start", "end"])
	merged_data["count"] = (
		merged_data["read_counts(0)"]
		.apply(lambda value: value.split(",")[0])
		.astype(int)
	)

	k_value = int(sys.argv[1])
	group_count_sum = merged_data.groupby(["group_id"])["count"].sum()
	supported_groups = merged_data[
		merged_data["group_id"].isin(group_count_sum[group_count_sum >= k_value].index)
	]

	removed_intron_indices = supported_groups[supported_groups["count"] == 1].index
	filtered_groups = merged_data[~merged_data.index.isin(removed_intron_indices)]
	filtered_groups = filtered_groups[filtered_groups.duplicated("group_id", keep=False)]

	output_group_ids = filtered_groups["group_id"].drop_duplicates()
	(
		diff_spliced_groups[diff_spliced_groups["group_id"].isin(output_group_ids)]
		[["chrom", "loc"]]
		.drop_duplicates()
		.to_csv(f"k{k_value}.chr_pos", index=False, header=False, sep=" ")
	)


if __name__ == "__main__":
	main()



