import os
import pandas as pd


def aggregate_ss_summary_files(output_directory, prod_groups):
    """
    Łączy wszystkie pliki:
    <PRODUCT_GROUP>_ss_summary.xlsx

    do jednego pliku:
    all_items_ss_summary.xlsx
    """

    dataframes = []

    for product_group in prod_groups:
        file_name = f"{product_group}_ss_summary.xlsx"
        file_path = os.path.join(output_directory, file_name)

        if os.path.exists(file_path):
            df = pd.read_excel(file_path, engine="openpyxl", dtype={"plant": str, 'material': str})

            cols_to_skip = ['calculated_new_ss', 'calculated_new_ROP', 'is_dc_market_match', 'is_below_min_ss', 'is_no_ss_item', 'volatility_method']
            cols = [col for col in df.columns if col not in cols_to_skip]

            df = df[cols]

            mask = ((df["reorder_point"] > 0) &
                    (df["safety_stock_in_SAP"] > 0)
                    )
            df = df[mask]

            df["PRODUCT_GROUP"] = product_group
            dataframes.append(df)
        else:
            print(f"Plik nie istnieje: {file_path}")

    if not dataframes:
        raise ValueError("Nie znaleziono żadnych plików do agregacji.")

    combined_df = pd.concat(dataframes, ignore_index=True)

    output_file = os.path.join(
        output_directory,
        "all_items_ss_summary.xlsx"
    )

    combined_df.to_excel(
        output_file,
        index=False,
        engine="openpyxl"
    )

    print(f"Zapisano plik: {output_file}")

    return combined_df