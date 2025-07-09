import statistics
from math import log
import polars as pl
import numpy as np

def matching_procedure(treat: pl.DataFrame, control: pl.DataFrame, count: int):
    """
    Matches elements of control group to elements of treatment group
    based on the path of the outcome variable being measured.
    Args:
       control: DataFrame of control group and time-series data
       treat: DataFrame of treatment group and time-series data
       count: number of matches per treatment variable

    Returns:
        matches: dict object of treatment group and control matches ranked by correlation,
        contains only one match if singleton=True
    """
    if control.shape[1] > count:
        raise ValueError("Number of matches requested exceeds elements in control group")
    t_cols = treat.columns
    df = treat + control

    # evaluate Y_i,t and set up new data frame (D.3a-c), (D.4a,c)

    matches = {}
    for i, _ in enumerate(t_cols):
        rest = t_cols[:i] + t_cols[i+1:]
        holder = df.drop(rest)
        columns_sorted = sorted(holder.columns, key=lambda col: df[col][i], reverse=True)
        matches[t_cols[i]] = columns_sorted[1:count+1]

    return matches