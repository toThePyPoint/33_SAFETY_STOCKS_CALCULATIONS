import pandas as pd


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


def check_top_customers(input_df: pd.DataFrame, customers_df: pd.DataFrame, start_date, end_date):
    """
    Analyzes the share of the top customer in total sales for each material
    at plant level.

    For each plant/material combination, the function:
    - calculates the total sales quantity,
    - identifies the customer with the highest sales quantity,
    - calculates the percentage share of the top customer,
    - calculates the number of unique customers.

    Parameters
    ----------
    input_df : pd.DataFrame
        pd.DataFrame containing at least the following columns:
        'plant', 'material', 'customer_number', 'quantity'.

    customers_df : pd.DataFrame
        pd.DataFrame containing at least the following columns:
        'KUNNR', 'NAME1' from KNA1 SAP table

    start_date : str or pandas.Timestamp
        Start date of the analysis period (inclusive).

    end_date : str or pandas.Timestamp
        End date of the analysis period (inclusive).

    Returns
    -------
    pd.DataFrame
        pd.DataFrame containing:
        - plant
        - material
        - top_customer
        - top_customer_qty
        - total_qty
        - top_customer_pct
        - n_customers
    """

    # Create a copy of the input data and convert quantities to positive values
    df = input_df.loc[
        input_df["posting_date"].between(start_date, end_date)
    ].copy()
    df["quantity"] = df["quantity"].abs()

    # Aggregate sales quantity by customer, plant, and material
    customer_sales = (
        df.groupby(
            ["plant", "material", "customer_number"],
            as_index=False
        )["quantity"]
        .sum()
    )

    # Identify the customer with the highest sales quantity
    # for each plant/material combination
    top_customer = (
        customer_sales
        .sort_values("quantity", ascending=False)
        .drop_duplicates(["plant", "material"])
        .rename(columns={
            "customer_number": "top_customer",
            "quantity": "top_customer_qty"
        })
    )

    # Calculate total sales quantity for each material and plant
    total_sales = (
        df.groupby(
            ["plant", "material"],
            as_index=False
        )["quantity"]
        .sum()
        .rename(columns={"quantity": "total_qty"})
    )

    # Calculate the number of unique customers
    # for each material and plant
    n_customers = (
        df.groupby(
            ["plant", "material"]
        )["customer_number"]
        .nunique()
        .reset_index(name="n_customers")
    )

    # Merge top customer data with total sales
    result = top_customer.merge(
        total_sales,
        on=["plant", "material"]
    )

    # Add the number of unique customers
    result = result.merge(
        n_customers,
        on=["plant", "material"]
    )

    # Calculate the percentage share of the top customer
    # in total sales quantity
    result["top_customer_pct"] = (
        result["top_customer_qty"]
        / result["total_qty"]
        * 100
    ).round(1)

    # Select and sort the final columns
    result = result[
        [
            "plant",
            "material",
            "top_customer",
            "top_customer_qty",
            "total_qty",
            "top_customer_pct",
            "n_customers"
        ]
    ].sort_values(["plant", "material"])

    result["top_customer"] = result["top_customer"].astype(str).str.zfill(10)

    customers = customers_df.rename(columns={"KUNNR": "top_customer", "NAME1": "top_customer_name"}).copy()
    customers["top_customer"] = customers["top_customer"].astype(str).str.zfill(10)

    result = result.merge(
        customers,
        on="top_customer",
        how="left"
    )

    return result