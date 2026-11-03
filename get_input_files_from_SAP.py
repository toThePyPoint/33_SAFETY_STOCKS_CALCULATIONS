import os
import time
import pandas as pd
from input_data_automatic_collection import get_materials_by_mrp_controller, get_material_data, get_material_movements


def get_input_files(prod_groups: dict, sap_system: str, start: str, end: str, doc_chunk_size: int, mat_chunk_size: int,
                    input_dir: str):

    total_start = time.perf_counter()

    group_materials = {}
    all_materials = set()
    # --------------------------------------------------------------
    # 1. Retrieve materials for all product groups
    # --------------------------------------------------------------
    for prod_group, (
        mrp_list,
        movement_types_list,
        plants_list,
        production_plant,
    ) in prod_groups.items():

        df_materials = get_materials_by_mrp_controller(
            plant=production_plant,
            mrp_controllers=mrp_list,
            sap_system=sap_system,
        )

        materials = (
            df_materials["MATNR"]
            .astype(str)
            .str.lstrip("0")
            .drop_duplicates()
            .tolist()
        )

        group_materials[prod_group] = materials
        all_materials.update(materials)

        print(
            f"{prod_group}: {len(materials):,} materials"
        )


    all_materials = sorted(all_materials)

    print(
        f"\nUnique materials in all groups: "
        f"{len(all_materials):,}"
    )

    # --------------------------------------------------------------
    # 2. Collect common parameters
    # --------------------------------------------------------------
    all_plants = sorted({
        plant
        for config in prod_groups.values()
        for plant in config[2]
    })

    all_movement_types = sorted({
        movement_type
        for config in prod_groups.values()
        for movement_type in config[1]
    })

    # --------------------------------------------------------------
    # 3. Retrieve material data once
    # --------------------------------------------------------------
    material_data_start = time.perf_counter()

    df_all_data = get_material_data(
        materials_list=all_materials,
        plants=all_plants,
        sap_system=sap_system,
    )

    print(
        f"\nMaterial data retrieved: "
        f"{len(df_all_data):,} rows | "
        f"{time.perf_counter() - material_data_start:.2f}s"
    )

    # --------------------------------------------------------------
    # 4. Retrieve movements once
    # --------------------------------------------------------------
    movements_start = time.perf_counter()

    df_all_movements = get_material_movements(
        materials_list=all_materials,
        plants=all_plants,
        movement_types=all_movement_types,
        date_from=start,
        date_to=end,
        document_chunk_size=doc_chunk_size,
        material_chunk_size=mat_chunk_size,
        sap_system=sap_system,
    )

    print(
        f"\nMovements retrieved: "
        f"{len(df_all_movements):,} rows | "
        f"{time.perf_counter() - movements_start:.2f}s"
    )

    # --------------------------------------------------------------
    # 5. Split results locally by product group
    # --------------------------------------------------------------
    for prod_group, (
        mrp_list,
        movement_types_list,
        plants_list,
        production_plant,
    ) in prod_groups.items():

        group_start = time.perf_counter()

        materials_set = set(group_materials[prod_group])

        df_data = df_all_data[
            df_all_data["material"].isin(materials_set)
            & df_all_data["plant"].isin(plants_list)
        ].copy()

        df_movements = df_all_movements[
            df_all_movements["material"].isin(materials_set)
            & df_all_movements["plant"].isin(plants_list)
            & df_all_movements["movement_type"].isin(
                movement_types_list
            )
        ].copy()

        data_path = os.path.join(
            input_dir,
            f"{prod_group}_items_and_parameters.xlsx",
        )

        movements_path = os.path.join(
            input_dir,
            f"{prod_group}_Consumption.xlsx",
        )

        df_data.to_excel(
            data_path,
            index=False,
        )

        df_movements.to_excel(
            movements_path,
            index=False,
        )

        print(
            f"{prod_group} done | "
            f"materials={len(materials_set):,} | "
            f"data rows={len(df_data):,} | "
            f"movement rows={len(df_movements):,} | "
            f"{time.perf_counter() - group_start:.2f}s"
        )

    print(
        f"\nAll product groups finished | "
        f"{time.perf_counter() - total_start:.2f}s"
    )