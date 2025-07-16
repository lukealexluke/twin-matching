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

if __name__ == "__main__":
    pass