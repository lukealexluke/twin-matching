import statistics
import pandas as pd

def matching_procedure(control: pd.DataFrame, treat: pd.DataFrame, singleton: bool, unique: bool):
    """
    Matches elements of control group to elements of treatment group
    based on the path of the outcome variable being measured.
    Args:
       control: DataFrame of control group and time-series data
       treat: DataFrame of treatment group and time-series data
       singleton: whether treatments should only match to one variable
       unique: whether treatments should match uniquely

    Returns:
        matches: dict object of treatment group and control matches ranked by correlation,
        contains only one match if singleton=True
    """
    pass