import polars as pl
from scipy.optimize import linear_sum_assignment
import numpy as np

def matching_procedure(treatment: pl.DataFrame, control: pl.DataFrame, log_transform: bool = False, event_timestep: int = None, num_matches: int = 1):
    """
    Matches elements of control group to elements of treatment group
    based on the path of the outcome variable being measured.
    Args:
       treatment: DataFrame of treatment group and time-series data
       control: DataFrame of control group and time-series data
       log_transform: whether or not to perform log-transformation on outcome variable for both DataFrames
       event_timestep: timestep where the event occurs
       num_matches: number of matches from treatment group to control group. (1 = injective, >1 = one-to-many)
    Returns:
        matches: dict object of treatment group and control matches ranked by correlation (highest to lowest)
    """
    assert num_matches >= 1, f"num_matches must be a positive integer ({num_matches} was given)"
    if event_timestep:
        assert event_timestep < treatment.height, f"Invalid timestep for DataFrame of height {treatment.height}"
    num_treat = treatment.width
    if num_matches == 1:
        assert num_treat <= control.width, f"There are not enough treatment units to ensure injectivity: {num_treat} -> {control.width}"

    if log_transform:
        treatment = treatment.with_columns([pl.col(col).log().alias(col) for col in treatment.columns])
        control = control.with_columns([pl.col(col).log().alias(col) for col in control.columns])
    
    df = pl.concat([treatment, control], how="horizontal")
    print("Calculating Correlations...")

    # remove columns with no variance
    unique_counts = df.select([pl.col(col).n_unique().alias(col) for col in df.columns]).row(0)
    non_constant_cols = [col for col, count in zip(df.columns, unique_counts) if count > 1]
    df = df.select(non_constant_cols)

    df_cor = df.corr()
    df_cor = df_cor.select(df_cor.columns[num_treat:]).head(num_treat) # consider only correlations between control and treatments
    print("Solving Assignment Problem...")
    matches = {}
    if num_matches == 1:
        # injective matching, solve linear assignment problem to determine matches
        row_ind, col_ind = linear_sum_assignment(df_cor, maximize=True)
        for treated_idx, control_idx in zip(row_ind, col_ind):
            matches[treatment.columns[int(treated_idx)]] = control.columns[int(control_idx)]
    else:
        # one-to-many matching, select n best matches per treatment unit
        for idx in range(len(treatment.columns)):
            row_vals = np.array(df_cor.row(idx))
            top_indices = row_vals.argsort()[-num_matches:][::-1]
            top_cols = [df_cor.columns[i] for i in top_indices]
            matches[treatment.columns[idx]] = top_cols
    return matches

if __name__ == "__main__":
    pass