import polars as pl
import twin

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

print(twin.matching_procedure(test_frame, test_frame_2, 2, False))