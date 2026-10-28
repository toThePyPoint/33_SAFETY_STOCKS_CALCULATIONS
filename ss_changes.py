import pandas as pd
from sap_conn import get_conn
from sap_rtab import rfc_read_table


def chunks(values, size=30):
    for i in range(0, len(values), size):
        yield values[i:i + size]


def get_ss_changes_records(
    materials_list,
    stocks_deleted_flag=False,
    sap_system="K11",
    date_from="20230101",
):
    result_columns = [
        "MATNR",
        "WERKS",
        "UDATE",
        "UTIME",
        "VALUE_OLD",
        "VALUE_NEW",
    ]

    materials_sap = [str(mat).zfill(18) for mat in materials_list]
    if not materials_sap:
        return pd.DataFrame(columns=result_columns)

    cdhdr_rows = []

    with get_conn(sap_system) as conn:
        for materials_batch in chunks(materials_sap, size=30):
            materials_sql = "', '".join(materials_batch)

            rows = rfc_read_table(
                conn=conn,
                table="CDHDR",
                fields=[
                    "OBJECTCLAS",
                    "OBJECTID",
                    "CHANGENR",
                    "USERNAME",
                    "UDATE",
                    "UTIME",
                    "TCODE",
                ],
                where=f"""
                    OBJECTCLAS = 'MATERIAL'
                    AND OBJECTID IN ('{materials_sql}')
                    AND UDATE >= '{date_from}'
                """
            )

            cdhdr_rows.extend(rows)

    df_cdhdr = pd.DataFrame(cdhdr_rows)
    if df_cdhdr.empty:
        return pd.DataFrame(columns=result_columns)

    df_cdhdr = df_cdhdr.drop_duplicates(
        ["OBJECTCLAS", "OBJECTID", "CHANGENR"]
    ).copy()

    all_cdpos = []

    with get_conn(sap_system) as conn:
        total = len(df_cdhdr)

        for i, (_, r) in enumerate(df_cdhdr.iterrows(), start=1):
            if i == 1 or i % 100 == 0 or i == total:
                print(f"CDPOS: {i}/{total}")
            rows = rfc_read_table(
                conn=conn,
                table="CDPOS",
                fields=[
                    "OBJECTCLAS",
                    "OBJECTID",
                    "CHANGENR",
                    "TABNAME",
                    "TABKEY",
                    "FNAME",
                    "CHNGIND",
                ],
                where=f"""
                    OBJECTCLAS = '{r["OBJECTCLAS"]}'
                    AND OBJECTID = '{r["OBJECTID"]}'
                    AND CHANGENR = '{r["CHANGENR"]}'
                """
            )

            all_cdpos.extend(rows)

    df_cdpos = pd.DataFrame(all_cdpos)
    if df_cdpos.empty:
        return pd.DataFrame(columns=result_columns)

    df_eisbe = (
        df_cdpos[
            (df_cdpos["TABNAME"].str.strip() == "MARC") &
            (df_cdpos["FNAME"].str.strip() == "EISBE")
            ]
        .drop_duplicates(
            ["OBJECTCLAS", "OBJECTID", "CHANGENR", "TABKEY"]
        )
        .copy()
    )
    if df_eisbe.empty:
        return pd.DataFrame(columns=result_columns)

    old_values = []
    new_values = []

    with get_conn(sap_system) as conn:

        for _, r in df_eisbe.iterrows():
            where = f"""
                OBJECTCLAS = '{r["OBJECTCLAS"]}'
                AND OBJECTID = '{r["OBJECTID"]}'
                AND CHANGENR = '{r["CHANGENR"]}'
                AND TABNAME = 'MARC'
                AND TABKEY = '{r["TABKEY"]}'
                AND FNAME = 'EISBE'
            """

            rows_old = rfc_read_table(
                conn=conn,
                table="CDPOS",
                fields=[
                    "OBJECTCLAS",
                    "OBJECTID",
                    "CHANGENR",
                    "TABKEY",
                    "VALUE_OLD",
                ],
                where=where,
            )

            rows_new = rfc_read_table(
                conn=conn,
                table="CDPOS",
                fields=[
                    "OBJECTCLAS",
                    "OBJECTID",
                    "CHANGENR",
                    "TABKEY",
                    "VALUE_NEW",
                ],
                where=where,
            )

            old_values.extend(rows_old)
            new_values.extend(rows_new)

    keys = [
        "OBJECTCLAS",
        "OBJECTID",
        "CHANGENR",
        "TABKEY",
    ]

    if not old_values or not new_values:
        return pd.DataFrame(columns=result_columns)

    df_old = pd.DataFrame(old_values)
    df_new = pd.DataFrame(new_values)

    df_values = df_old.merge(
        df_new,
        on=keys,
        how="outer",
    )

    df_result = (
        df_eisbe
        .merge(df_values, on=keys, how="left")
        .merge(
            df_cdhdr[
                [
                    "OBJECTCLAS",
                    "OBJECTID",
                    "CHANGENR",
                    "UDATE",
                    "UTIME",
                    "USERNAME",
                    "TCODE",
                ]
            ],
            on=["OBJECTCLAS", "OBJECTID", "CHANGENR"],
            how="left",
        )
        .sort_values(["UDATE", "UTIME"])
    )

    # df_result["MANDT"] = df_result["TABKEY"].str[:3]
    df_result["MATNR"] = df_result["TABKEY"].str[3:21].str.lstrip("0")
    df_result["WERKS"] = df_result["TABKEY"].str[21:25]

    df_result["UDATE"] = pd.to_datetime(
        df_result["UDATE"].str.strip(),
        format="%Y%m%d",
        errors="coerce"
    )

    df_result["VALUE_OLD"] = pd.to_numeric(
        df_result["VALUE_OLD"].str.strip(),
        errors="coerce"
    )

    df_result["VALUE_NEW"] = pd.to_numeric(
        df_result["VALUE_NEW"].str.strip(),
        errors="coerce"
    )

    if stocks_deleted_flag:
        mask = (df_result["VALUE_OLD"] > 0) & (df_result["VALUE_NEW"] == 0)
    else:
        mask = (df_result["VALUE_OLD"] == 0) & (df_result["VALUE_NEW"] > 0)

    df_ss_change = (
        df_result[mask]
        .sort_values(["MATNR", "WERKS", "UDATE", "UTIME"], ascending=[True, True, False, False])
        .drop_duplicates(["MATNR", "WERKS"], keep="first")
        .copy()
    )

    return df_ss_change[result_columns]