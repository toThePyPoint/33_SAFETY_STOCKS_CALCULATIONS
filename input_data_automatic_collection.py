import pandas as pd
from sap_conn import get_conn
from sap_rtab import rfc_read_table


def chunks(values, size=30):
    for i in range(0, len(values), size):
        yield values[i:i + size]


def get_materials_by_mrp_controller(
    plant,
    mrp_controller,
    sap_system="K11",
):
    result_columns = [
        "MATNR",
        "WERKS",
        "DISPO",
    ]

    with get_conn(sap_system) as conn:
        rows = rfc_read_table(
            conn=conn,
            table="MARC",
            fields=result_columns,
            where=f"""
                WERKS = '{plant}'
                AND DISPO = '{mrp_controller}'
            """
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(columns=result_columns)

    return df[result_columns]


def get_safety_stocks(
    materials_list,
    plants,
    sap_system="K11",
):
    """
    Retrieve safety stock values for selected materials and plants from SAP.

    The function reads safety stock data (EISBE) from the MARC table.
    It returns all combinations of the provided materials and plants.
    If no safety stock value is found for a material/plant combination,
    EISBE is set to 0.

    Parameters
    ----------
    materials_list : list
        List of SAP material numbers. Material numbers may be provided
        with or without leading zeros.

    plants : list
        List of SAP plant codes, e.g. ["2101", "0301", "3701", "1201"].

    sap_system : str, optional
        SAP system used for the RFC connection.
        Default is "K11".

    Returns
    -------
    pandas.DataFrame
        DataFrame containing:
        - MATNR: SAP material number without leading zeros
        - WERKS: SAP plant
        - EISBE: Safety stock quantity

        All material/plant combinations are included in the result.
        Missing safety stock values are returned as 0.
    """

    result_columns = [
        "MATNR",
        "WERKS",
        "EISBE",
    ]

    # Normalize material numbers
    materials = [str(mat).lstrip("0") for mat in materials_list]
    materials_sap = [mat.zfill(18) for mat in materials]

    plants = [str(plant) for plant in plants]

    if not materials:
        return pd.DataFrame(columns=result_columns)

    all_rows = []

    plants_sql = "', '".join(plants)

    with get_conn(sap_system) as conn:
        for materials_batch in chunks(materials_sap, size=30):
            materials_sql = "', '".join(materials_batch)

            rows = rfc_read_table(
                conn=conn,
                table="MARC",
                fields=result_columns,
                where=f"""
                    MATNR IN ('{materials_sql}')
                    AND WERKS IN ('{plants_sql}')
                """
            )

            all_rows.extend(rows)

    df_sap = pd.DataFrame(all_rows)

    if not df_sap.empty:
        df_sap["MATNR"] = df_sap["MATNR"].str.lstrip("0")
        df_sap["EISBE"] = pd.to_numeric(
            df_sap["EISBE"],
            errors="coerce"
        )

    # Create all material x plant combinations
    df_full = pd.MultiIndex.from_product(
        [materials, plants],
        names=["MATNR", "WERKS"]
    ).to_frame(index=False)

    # Add safety stock values retrieved from SAP
    if not df_sap.empty:
        df_full = df_full.merge(
            df_sap[["MATNR", "WERKS", "EISBE"]],
            on=["MATNR", "WERKS"],
            how="left",
        )
    else:
        df_full["EISBE"] = 0

    # Treat missing safety stock values as zero
    df_full["EISBE"] = df_full["EISBE"].fillna(0)

    return df_full[result_columns]


def get_material_data(
    materials_list,
    plants,
    sap_system="K11",
):
    """
    Retrieve material master, safety stock, and valuation data for selected
    materials and plants from SAP.

    The function returns all combinations of the provided materials and plants.
    Safety stock is retrieved from MARC, material type from MARA, and valuation
    data from MBEW.

    Parameters
    ----------
    materials_list : list
        List of SAP material numbers. Material numbers may be provided
        with or without leading zeros.

    plants : list
        List of SAP plant codes, e.g. ["2101", "0301", "3701", "1201"].

    sap_system : str, optional
        SAP system used for the RFC connection.
        Default is "K11".

    Returns
    -------
    pandas.DataFrame
        DataFrame containing material and plant-level information.
    """

    result_columns = [
        "material",
        "material_description",
        "plant",
        "material_type",
        "safety_stock_in_SAP",
        # "VPRSV",
        # "STPRS",
        # "VERPR",
        # "PEINH",
        "unit_price",
        "price_currency"
    ]

    currencies = {
        "2101": "PLN",
        "0301": "EUR",
        "1201": "EUR",
        "3701": "CZK"
    }

    materials = [str(mat).lstrip("0") for mat in materials_list]
    materials_sap = [mat.zfill(18) for mat in materials]
    plants = [str(plant) for plant in plants]

    if not materials:
        return pd.DataFrame(columns=result_columns)

    marc_rows = []
    mara_rows = []
    mbew_rows = []
    makt_rows = []

    plants_sql = "', '".join(plants)

    with get_conn(sap_system) as conn:

        # ----------------------------------------------------------
        # MARC - plant-specific material data / safety stock
        # ----------------------------------------------------------
        for materials_batch in chunks(materials_sap, size=30):

            materials_sql = "', '".join(materials_batch)

            rows = rfc_read_table(
                conn=conn,
                table="MARC",
                fields=[
                    "MATNR",
                    "WERKS",
                    "EISBE",
                ],
                where=f"""
                    MATNR IN ('{materials_sql}')
                    AND WERKS IN ('{plants_sql}')
                """
            )

            marc_rows.extend(rows)

        # ----------------------------------------------------------
        # MARA - general material master / material type
        # ----------------------------------------------------------
        for materials_batch in chunks(materials_sap, size=30):

            materials_sql = "', '".join(materials_batch)

            rows = rfc_read_table(
                conn=conn,
                table="MARA",
                fields=[
                    "MATNR",
                    "MTART",
                ],
                where=f"""
                    MATNR IN ('{materials_sql}')
                """
            )

            mara_rows.extend(rows)

        # ----------------------------------------------------------
        # MBEW - material valuation / price
        # Assumption: valuation area (BWKEY) = plant.
        # ----------------------------------------------------------
        for materials_batch in chunks(materials_sap, size=30):

            materials_sql = "', '".join(materials_batch)

            rows = rfc_read_table(
                conn=conn,
                table="MBEW",
                fields=[
                    "MATNR",
                    "BWKEY",
                    "VPRSV",
                    "STPRS",
                    "VERPR",
                    "PEINH",
                ],
                where=f"""
                    MATNR IN ('{materials_sql}')
                    AND BWKEY IN ('{plants_sql}')
                """
            )

            mbew_rows.extend(rows)

    with get_conn(sap_system) as conn:
        # ----------------------------------------------------------
        # MAKT - material descriptions
        # ----------------------------------------------------------
        for materials_batch in chunks(materials_sap, size=30):
            materials_sql = "', '".join(materials_batch)

            rows = rfc_read_table(
                conn=conn,
                table="MAKT",
                fields=[
                    "MATNR",
                    "MAKTX",
                ],
                where=f"""
                    MATNR IN ('{materials_sql}')
                    AND SPRAS = 'E'
                """
            )

            makt_rows.extend(rows)

    # --------------------------------------------------------------
    # Build base DataFrame containing ALL material x plant combinations
    # --------------------------------------------------------------
    df = pd.MultiIndex.from_product(
        [materials, plants],
        names=["MATNR", "WERKS"]
    ).to_frame(index=False)

    # --------------------------------------------------------------
    # MARC
    # --------------------------------------------------------------
    df_marc = pd.DataFrame(marc_rows)

    if not df_marc.empty:

        df_marc["MATNR"] = df_marc["MATNR"].str.lstrip("0")

        df_marc["EISBE"] = pd.to_numeric(
            df_marc["EISBE"],
            errors="coerce"
        )

        df = df.merge(
            df_marc,
            on=["MATNR", "WERKS"],
            how="left",
        )

    else:
        df["EISBE"] = 0

    # --------------------------------------------------------------
    # MARA
    # --------------------------------------------------------------
    df_mara = pd.DataFrame(mara_rows)

    if not df_mara.empty:

        df_mara["MATNR"] = df_mara["MATNR"].str.lstrip("0")

        df_mara = df_mara.drop_duplicates("MATNR")

        df = df.merge(
            df_mara,
            on="MATNR",
            how="left",
        )

    else:
        df["MTART"] = pd.NA

    # --------------------------------------------------------------
    # MAKT
    # --------------------------------------------------------------
    df_makt = pd.DataFrame(makt_rows)

    if not df_makt.empty:

        df_makt["MATNR"] = df_makt["MATNR"].str.lstrip("0")

        df_makt = df_makt.drop_duplicates("MATNR")

        df = df.merge(
            df_makt,
            on="MATNR",
            how="left",
        )

    else:
        df["MAKTX"] = pd.NA

    # --------------------------------------------------------------
    # MBEW
    # --------------------------------------------------------------
    df_mbew = pd.DataFrame(mbew_rows)

    if not df_mbew.empty:

        df_mbew["MATNR"] = df_mbew["MATNR"].str.lstrip("0")

        df_mbew = df_mbew.rename(
            columns={"BWKEY": "WERKS"}
        )

        price_columns = [
            "STPRS",
            "VERPR",
            "PEINH",
        ]

        for column in price_columns:
            df_mbew[column] = pd.to_numeric(
                df_mbew[column],
                errors="coerce"
            )

        df = df.merge(
            df_mbew,
            on=["MATNR", "WERKS"],
            how="left",
        )

    else:
        df["VPRSV"] = pd.NA
        df["STPRS"] = pd.NA
        df["VERPR"] = pd.NA
        df["PEINH"] = pd.NA

    # --------------------------------------------------------------
    # Safety stock
    # --------------------------------------------------------------
    df["EISBE"] = df["EISBE"].fillna(0)

    # --------------------------------------------------------------
    # Calculate unit price
    #
    # VPRSV:
    # S = Standard price
    # V = Moving average price
    # --------------------------------------------------------------
    df["BASE_PRICE"] = df["STPRS"]

    df.loc[
        df["VPRSV"] == "V",
        "BASE_PRICE"
    ] = df["VERPR"]

    df["unit_price"] = (
        df["BASE_PRICE"]
        / df["PEINH"].replace(0, pd.NA)
    )

    df["price_currency"] = df['WERKS'].map(currencies)
    df = df.rename(columns={"WERKS": "plant", "MATNR": "material", "MTART": "material_type",
                            "EISBE": "safety_stock_in_SAP", "MAKTX": "material_description"})

    return (
        df[result_columns]
        .dropna(subset=["unit_price"])
        .reset_index(drop=True)
    )
