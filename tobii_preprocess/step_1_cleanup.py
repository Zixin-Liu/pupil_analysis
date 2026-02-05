
import polars as pl

from pathlib import Path


def remove_empty_columns_tobii(
    input_tsv,
    output_dir=None,
    suffix="_step_1",
    save_removed_list=True
):
    """
    Remove columns that are completely empty from a Tobii TSV file.

    Parameters
    ----------
    input_tsv : str or Path
        Path to the input Tobii .tsv file.
    output_dir : str or Path, optional
        Directory to save the cleaned TSV.
        If None, saves next to the input file.
    suffix : str, optional
        Suffix added to the output filename.
    save_removed_list : bool, optional
        Whether to save a text file listing removed columns.

    Returns
    -------
    output_tsv : Path
        Path to the cleaned TSV file.
    removed_columns : list of str
        Names of columns that were removed.
    """

    input_tsv = Path(input_tsv)

    if output_dir is None:
        output_dir = input_tsv.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    stem_slice = input_tsv.stem[-10:-5]
    output_tsv = output_dir / f"{stem_slice}{suffix}.tsv"


    # ---- load TSV with Polars ----
    df = pl.read_csv(input_tsv, separator="\t")
    
    # ---- find fully empty columns ----
    removed_columns = [col for col in df.columns if df[col].null_count() == df.height]

    
    # ---- drop empty columns ----
    df_clean = df.drop(removed_columns)
    
    # ---- save cleaned TSV ----
    df_clean.write_csv(output_tsv, separator="\t")

   

    print(
        f"[Step 1] {input_tsv.name}: "
    )

    return output_tsv



def extract_relevant_rows_1(
    input_tsv,
    output_dir=None,
    suffix="_step_2_1",
    event_column="Event",
    first_event_value="Baseline Colour black ID 5"
):
    """
    Step 2.1 of Tobii preprocessing (experiment-specific):
    - Remove all rows before the first occurrence of `first_event_value`
    - Remove Tobii metadata columns
    """

    input_tsv = Path(input_tsv)

    if output_dir is None:
        output_dir = input_tsv.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    output_tsv = output_dir / f"{input_tsv.stem}{suffix}.tsv"

    # ---- load TSV ----
    df = pl.read_csv(input_tsv, separator="\t")

    if event_column not in df.columns:
        raise KeyError(f"Event column '{event_column}' not found in TSV.")

    # ---- add a row index to simulate pandas' .index ----
    df = df.with_row_count("row_nr")

    # ---- find first occurrence of the baseline event ----
    filtered = df.filter(pl.col(event_column) == first_event_value)
    if filtered.height == 0:
        raise ValueError(f"Event value '{first_event_value}' not found in the file.")

    first_event_idx = filtered["row_nr"][0]  # first row number

    # ---- trim rows before first event ----
    df_trimmed = df.filter(pl.col("row_nr") >= first_event_idx).drop("row_nr")

    # ---- keep metadata columns ----
    metadata_cols = [
        "Recording timestamp [μs]",
        "Event",
        "Pupil diameter left [mm]",
        "Pupil diameter right [mm]",
        "Eye position left Z [DACS mm]",
        "Eye position right Z [DACS mm]",
        "Gaze point left X [DACS mm]",
        "Gaze point left Y [DACS mm]",
        "Gaze point right X [DACS mm]",
        "Gaze point right Y [DACS mm]"
    ]

    cols_to_keep = [c for c in df_trimmed.columns if c not in metadata_cols]
    df_trimmed = df_trimmed.loc[:, cols_to_keep]

    # ---- save cleaned TSV ----
    df_trimmed.write_csv(output_tsv, separator="\t")

    print(
        f"[Step 2] {input_tsv.name}: "
        f"rows before '{first_event_value}' removed | "
    )

    return output_tsv


def extract_relevant_rows_2(
    input_tsv,
    output_dir=None,
    suffix="_step_2_2",
    timestamp_col="Recording timestamp [μs]",
    event_col="Event",
    baseline_black="Baseline Colour black ID 5",
    baseline_gray="Baseline Colour gray ID 7",
    trial_start_marker="Trial 1 Event trialOnset ID 1",
    trial_end_marker="Trial 60 Event fix3 ID 4",
    pre_trial_buffer_us=200_000  # 200 ms
):
    """
    Step 2.2 of Tobii preprocessing (experiment-specific):
    - retain baeline measures
    - retain block data, with 200ms before the first trial onset
    """
    
    input_tsv = Path(input_tsv)
    stem_slice = input_tsv.stem[:5]
    
    if output_dir is None:
        output_dir = input_tsv.parent.parent / "preprocessed" / stem_slice
    else:
        output_dir = Path(output_dir) / "preprocessed" / stem_slice

    # create the folder if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

   
    
    # ---- load TSV ----
    df = pl.read_csv(input_tsv, separator="\t")

    if event_col not in df.columns or timestamp_col not in df.columns:
        raise KeyError(f"Columns '{event_col}' or '{timestamp_col}' not found in TSV.")

    # ---- find all baseline events ----
    black_times = df.filter(pl.col(event_col) == baseline_black)[timestamp_col]
    gray_times = df.filter(pl.col(event_col) == baseline_gray)[timestamp_col]

    if len(black_times) < 2 or len(gray_times) < 2:
        raise ValueError("Could not find two black and gray baseline events.")

    # ---- define baseline periods ----
    baseline_periods = [
        (black_times[0], gray_times[0] + 60_000_000),  # baseline1: 60s after gray
        (black_times[1], gray_times[1] + 30_000_000)   # baseline2: 30s after gray
    ]

    baseline_outputs = []
    for i, (start, end) in enumerate(baseline_periods, start=1):
        baseline_df = df.filter(
            (pl.col(timestamp_col) >= start) &
            (pl.col(timestamp_col) <= end)
        )
        output_path = output_dir / f"{stem_slice}_baseline{i}.tsv"
        baseline_df.write_csv(output_path, separator="\t")
        baseline_outputs.append(output_path)
        print(f"Saved Baseline {i} to {output_path}")

   # ---- find trial start/end timestamps for blocks ----
    trial_starts = df.filter(pl.col(event_col).str.contains(trial_start_marker))[timestamp_col]
    trial_ends = df.filter(pl.col(event_col).str.contains(trial_end_marker))[timestamp_col]

    if trial_starts.is_empty() or trial_ends.is_empty():
        raise ValueError("Could not find trial start or end events.")

    # ---- split into 4 blocks ----
    # Assume blocks are in order and continuous
    num_blocks = 4
    block_outputs = []
    trials_per_block = len(trial_starts) // num_blocks

    for b in range(num_blocks):
        # start timestamp for this block, include 200ms buffer for every block
        block_start_idx = b * trials_per_block
        block_end_idx = (b + 1) * trials_per_block - 1
        
        block_start = trial_starts[block_start_idx] - pre_trial_buffer_us  # always include 200ms before first trial
        block_end = trial_ends[block_end_idx]
        
        block_df = df.filter(
            (pl.col(timestamp_col) >= block_start) &
            (pl.col(timestamp_col) <= block_end)
        )
        output_path = output_dir / f"{stem_slice}_block{b+1}.tsv"
        block_df.write_csv(output_path, separator="\t")
        block_outputs.append(output_path)
        print(f"Saved Block {b+1} to {output_path}")



    return baseline_outputs, block_outputs

