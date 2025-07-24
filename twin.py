import polars as pl
from scipy.optimize import linear_sum_assignment
import string

def matching_procedure(treatment: pl.DataFrame, control: pl.DataFrame, log_transform: bool = False, event_timestep: int = None, injective: bool = False):
    """
    Matches elements of control group to elements of treatment group
    based on the path of the outcome variable being measured.
    Args:
       treatment: DataFrame of treatment group and time-series data
       control: DataFrame of control group and time-series data
       log_transform: whether or not to perform log-transformation on outcome variable for both DataFrames
       event_timestep: timestep where the event occurs
       injective: whether or not to match injectively (one-to-one) or one-to-many
    Returns:
        matches: dict object of treatment group and control matches ranked by correlation
    """
    if event_timestep:
        assert event_timestep < treatment.height, f"Invalid timestep for DataFrame of height {treatment.height}"
    num_treat = treatment.width
    if injective:
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
    row_ind, col_ind = linear_sum_assignment(df_cor, maximize=True) # Solves the Linear Assignment Problem on a cost matrix, selecting best twin matches
    matches = {}
    for treated_idx, control_idx in zip(row_ind, col_ind):
        matches[treatment.columns[int(treated_idx)]] = control.columns[int(control_idx)]
    return matches

if __name__ == "__main__":
    pass