def check_for_zero_streaks(df, streak_length):
    """
    Identifies material-plant pairs with at least a specified number
    of consecutive records where consumption equals zero.

    Args:
        df (pd.DataFrame): DataFrame containing material, plant,
            posting_date, and quantity columns.
        streak_length (int): Minimum number of consecutive zero-quantity
            records required to identify a streak.

    Returns:
        pd.DataFrame: DataFrame containing unique material-plant pairs
            that meet the zero-streak condition.
    """
    # Sort the data
    df = df.sort_values(["material", "plant", "posting_date"])

    # Flag records with zero consumption
    df["is_zero"] = df["quantity"].eq(0)

    # Check whether a streak of consecutive zeros exists
    df["zero_streak"] = (
        df.groupby(["material", "plant"])["is_zero"]
        .transform(lambda s: s.rolling(streak_length).sum().eq(streak_length))
    )

    # Get unique material-plant pairs
    result = (
        df.loc[df["zero_streak"], ["material", "plant", "zero_streak"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return result