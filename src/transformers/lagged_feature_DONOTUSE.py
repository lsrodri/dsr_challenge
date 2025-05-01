import pandas as pd
import os


def load_and_merge_lagged_variable(file_path, input_filename):
    """
    Load the labels data, create a lagged variable for total_cases,
    and merge it with the input data.

    Inputs:
        - file_path: Path to the directory containing the data files.
        - input_filename: Name of the input data file (e.g., features).

    """
    # Load labels data
    labels = pd.read_csv(os.path.join(file_path, "dengue_labels_train.csv"))

    # Sort for correct lagging
    labels.sort_values(by=["city", "year", "weekofyear"], inplace=True)

    # Create lagged variable
    labels["total_cases_lagged"] = labels.groupby(["city", "year"])[
        "total_cases"
    ].shift(1)

    # Handle transition from week 52 to 1
    def handle_week_transition(group):
        if group["weekofyear"].iloc[0] == 1:
            prev_year = group["year"].iloc[0] - 1
            prev_week_52 = labels[
                (labels["city"] == group["city"].iloc[0])
                & (labels["year"] == prev_year)
                & (labels["weekofyear"] == 52)
            ]
            if not prev_week_52.empty:
                group.loc[group.index[0], "total_cases_lagged"] = prev_week_52[
                    "total_cases"
                ].values[0]
        return group

    labels = (
        labels.groupby(["city", "year"])
        .apply(handle_week_transition)
        .reset_index(drop=True)
    )

    # Fill missing lag values with 0
    labels["total_cases_lagged"].fillna(0, inplace=True)

    # Drop total_cases before merging
    labels = labels.drop(columns=["total_cases"])

    # Load the input data (e.g., features)
    input_df = pd.read_csv(os.path.join(file_path, input_filename))

    # Merge using city, year, weekofyear
    merged_df = pd.merge(
        input_df, labels, on=["city", "year", "weekofyear"], how="left"
    )

    return merged_df
