import statistics
from math import log
import polars as pl
import numpy as np
from scipy.optimize import linear_sum_assignment

def matching_procedure(treatment: pl.DataFrame, control: pl.DataFrame, event_timestep: int, log_transform: bool):
    """
    Matches elements of control group to elements of treatment group
    based on the path of the outcome variable being measured.
    Args:
       treatment: DataFrame of treatment group and time-series data
       control: DataFrame of control group and time-series data
       event_timestep: timestep where the event occurs
       log_transform: transform rows by using ln()
    Returns:
        matches: dict object of treatment group and control matches ranked by correlation,
        contains only one match if singleton=True
    """
    assert event_timestep < treatment.height, f"invalid timestep for DataFrame of height {treatment.height}"
    num_treat = treatment.width
    df = pl.concat([treatment, control], how="horizontal")
    df_cor = df.corr()
    df_cor = df_cor.select(df_cor.columns[num_treat:]).head(num_treat)
    
    row_ind, col_ind = linear_sum_assignment(df_cor, maximize=True) # Solves the Linear Assignment Problem, maximizing correlations
    print(row_ind, col_ind)
    matches = {}
    for treated_idx, control_idx in zip(row_ind, col_ind):
        matches[int(treated_idx)] = int(control_idx)
    return matches

test_frame = pl.DataFrame({
    "col1": [1, 2, 3, 4, 5, 6, 7],
    "col2": [4, 27, 30, 41, 50, 60, 70],
    "col3": [5, 4, 22, 2, 1, 0, -1],
})

test_frame_2 = pl.DataFrame({
    "col4": [7, 12, 21, 22, 35, 42, 49],
    "col5": [100, 90, 80, 72, 60, 11, 40],
    "col6": [3, 6, 9, 14, 15, 14, 21],
    "col7": [3, 6, 4, 4, 4, 14, 21],
})

print(matching_procedure(test_frame, test_frame_2, 2, False))