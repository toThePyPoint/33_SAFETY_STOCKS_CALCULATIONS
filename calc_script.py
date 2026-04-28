import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_stats_df(mb51_path, zsbe_path, prd_plant, get_all_dates, start_date, end_date, k_parameter,
                    ex_rates):
    # Mapping for mb51_df (Snake Case)
    mb51_rename = {
        'Zakład': 'plant',
        'Materiał': 'material',
        'Opis materiału': 'material_description',
        'Data księgowania': 'posting_date',
        'Ilość': 'quantity',
        'Podst. jedn. miary': 'base_uom',
        'Rodzaj ruchu': 'movement_type'
    }

    # Mapping for zsbe_df (Snake Case + Unit Price)
    zsbe_rename = {
        'Materiał': 'material',
        'Opis materiału': 'material_description',
        'Rodzaj materiału': 'material_type',
        'Zakład': 'plant',
        'Planow. czas dostawy': 'planned_delivery_time',
        'Całk. czas uzupełn.': 'total_replenishment_time',
        'Unnamed: 6': 'unit_price',  # Fix according to your information
        'Waluta': 'currency',
        'Jednostka ceny': 'price_unit',
        'dowolne użycie': 'unrestricted_use',
        'Podst. jedn. miary': 'base_uom',
        'pokrycie/M': 'coverage_month',
        'przec.ilość/MM': 'avg_qty_month',
        'zapas bezpieczeństwa': 'safety_stock_in_SAP',
        'Kontroler MRP': 'mrp_controller'
    }

    mb51_df = pd.read_excel(mb51_path, dtype={'Materiał': str, 'Zakład': str})
    zsbe_df = pd.read_excel(zsbe_path, dtype={'Materiał': str, 'Zakład': str})
    # Renaming
    mb51_df.rename(columns=mb51_rename, inplace=True)
    zsbe_df.rename(columns=zsbe_rename, inplace=True)
    # Drop confi items
    mb51_df = mb51_df[~mb51_df['material'].str.startswith('99')]
    zsbe_df = zsbe_df[~zsbe_df['material'].str.startswith('99')]

    # Prepare a unique list of materials and their types
    unique_materials = zsbe_df[zsbe_df['plant'] == prd_plant][['material', 'material_type']]

    mb51_df = pd.merge(left=mb51_df, right=unique_materials, on='material', how='left')

    # Keep rows that do NOT meet both conditions at the same time
    mb51_df = mb51_df[~((mb51_df['movement_type'] == 261) & (mb51_df['material_type'] == 'FERT'))]
    # Convert to datetime (if not already done)
    mb51_df['posting_date'] = pd.to_datetime(mb51_df['posting_date'])

    # Optional: convert to positive values if quantity represents issues (negative)
    mb51_df['quantity'] = mb51_df['quantity'].abs()

    # Get unique Material-Plant pairs
    unique_pairs = zsbe_df[['material', 'plant']].drop_duplicates()

    # Get unique dates depending on GET_ALL_DATES_FROM_MB51 flag
    if get_all_dates:
        all_dates = mb51_df['posting_date'].unique()
    else:
        all_dates = pd.date_range(start=start_date, end=end_date, freq='B')

    print("Dates for calculations", all_dates.sort_values())

    # Create the structure directly from pairs and dates
    # Create a list of tuples (material, plant, date)
    from itertools import product

    structure = [
        (m, p, d) for (m, p), d in product(unique_pairs.values, all_dates)
    ]

    # Convert to DataFrame
    full_frame = pd.DataFrame(structure, columns=['material', 'plant', 'posting_date'])

    # Aggregate MB51 to have one total per day/material/plant
    daily_actual = mb51_df.groupby(['material', 'plant', 'posting_date'])['quantity'].sum().reset_index()

    # Merge
    final_df = pd.merge(
        full_frame,
        daily_actual,
        on=['material', 'plant', 'posting_date'],
        how='left'
    ).fillna(0)

    # Group by material and plant to calculate statistics
    stats_df = final_df.groupby(['material', 'plant'])['quantity'].agg(['mean', 'std']).reset_index()

    # Rename columns to more descriptive names (Snake Case)
    stats_df.rename(columns={
        'mean': 'daily_avg_consumption',
        'std': 'daily_std_dev'
    }, inplace=True)

    # Handle missing values (if std could not be calculated, e.g. no variability)
    stats_df['daily_std_dev'] = stats_df['daily_std_dev'].fillna(0)

    # Optional: Round results to 2 decimal places for better readability
    stats_df['daily_avg_consumption'] = stats_df['daily_avg_consumption'].round(4)
    stats_df['daily_std_dev'] = stats_df['daily_std_dev'].round(4)

    # Check the result
    # print(f"Number of resulting rows: {len(stats_df)}") # Should be 3581
    stats_df.sort_values(by='daily_avg_consumption', ascending=False, inplace=True)
    stats_df.reset_index(drop=True, inplace=True)

    lead_times = zsbe_df[['material', 'plant', 'planned_delivery_time', 'total_replenishment_time']].drop_duplicates()

    # Create the 'lead_time' column based on a condition
    lead_times['lead_time'] = np.where(
        lead_times['plant'] == prd_plant,  # Condition: whether the plant is 2101
        lead_times['total_replenishment_time'],  # If true: take total replenishment time
        lead_times['planned_delivery_time']  # If false: take planned delivery time
    )

    # Optional: If you have NaN (missing) values in the data, it is worth filling them with zero,
    # so that the square root in SS calculations does not throw an error
    lead_times['lead_time'] = lead_times['lead_time'].fillna(0)

    # Now you can merge this with your statistics table
    stats_df = pd.merge(stats_df, lead_times[['material', 'plant', 'lead_time']], on=['material', 'plant'], how='left')
    stats_df['new_safety_stock'] = (
            k_parameter * stats_df['daily_std_dev'] * np.sqrt(stats_df['lead_time'])
    )

    # Round up (since you cannot have 0.5 units in stock)
    stats_df['new_safety_stock'] = np.ceil(stats_df['new_safety_stock']).astype(int)

    # Calculate ROP
    stats_df['reorder_point'] = (stats_df['daily_avg_consumption'] * stats_df['lead_time']
                                ) + stats_df['new_safety_stock']

    # Round up
    stats_df['reorder_point'] = np.ceil(stats_df['reorder_point']).astype(int)

    # Create a new column with converted price
    # .map() matches the currency to the exchange rate, then we multiply it by the unit price
    zsbe_df['unit_price_eur'] = (
            zsbe_df['unit_price'] * zsbe_df['currency'].map(ex_rates)
    )

    # Get price AND price unit from zsbe_df
    price_data = zsbe_df[['material', 'plant', 'unit_price_eur', 'price_unit']].drop_duplicates()

    # Merge with statistics
    stats_df = pd.merge(stats_df, price_data, on=['material', 'plant'], how='left')

    # Calculate value
    stats_df['safety_stock_value'] = (
                                             stats_df['new_safety_stock'] * stats_df['unit_price_eur']
                                     ) / stats_df['price_unit']

    stats_df['ROP_value'] = (
                                    stats_df['reorder_point'] * stats_df['unit_price_eur']
                            ) / stats_df['price_unit']

    # Round to 2 decimal places (monetary value)
    stats_df['safety_stock_value'] = stats_df['safety_stock_value'].round(2)
    stats_df['ROP_value'] = stats_df['ROP_value'].round(2)

    # Attach old safety stock for comparison
    old_ss = zsbe_df[['material', 'plant', 'safety_stock_in_SAP']]
    stats_df = pd.merge(stats_df, old_ss, on=['material', 'plant'], how='left')

    # Check the difference
    stats_df['ss_diff'] = stats_df['new_safety_stock'] - stats_df['safety_stock_in_SAP']

    return stats_df

def create_plant_summary(stats_df):
    # 1. Create a new DataFrame with summary per plant
    plant_summary = stats_df.groupby('plant').agg({
        'new_safety_stock': 'sum',
        'safety_stock_in_SAP': 'sum',
        'ss_diff': 'sum',
        'safety_stock_value': 'sum'
    }).reset_index()

    # 2. Calculate the value of the previous safety stock (earlier)
    # Assuming the value is calculated using the same unit price:
    stats_df['old_ss_value'] = (stats_df['safety_stock_in_SAP'] * stats_df['unit_price_eur']) / stats_df['price_unit']

    # Attach the sum of the old value to the summary
    old_value_sum = stats_df.groupby('plant')['old_ss_value'].sum().reset_index()
    plant_summary = pd.merge(plant_summary, old_value_sum, on='plant')

    # 3. Add a column with value difference (how much capital is released or tied up)
    plant_summary['value_diff'] = plant_summary['safety_stock_value'] - plant_summary['old_ss_value']

    # 4. Rename columns to be more readable for reporting
    plant_summary.rename(columns={
        'new_safety_stock': 'Total New SS (Qty)',
        'safety_stock_in_SAP': 'Total Old SS (Qty)',
        'ss_diff': 'Total Qty Diff',
        'safety_stock_value': 'Total New SS Value [EUR]',
        'old_ss_value': 'Total Old SS Value [EUR]',
        'value_diff': 'Value Difference [EUR]'
    }, inplace=True)

    # Round monetary values
    cols_to_round = ['Total New SS Value [EUR]', 'Total Old SS Value [EUR]', 'Value Difference [EUR]']
    plant_summary[cols_to_round] = plant_summary[cols_to_round].round(2)

    return plant_summary

def create_a_summary_plot(plant_summary, save_path=None):
    # Prepare data
    plot_data = plant_summary[plant_summary['plant'] != 'TOTAL'].copy()
    labels = plot_data['plant'].astype(str)
    x = np.arange(len(labels))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    def autolabel(rects, ax, is_value=False):
        for rect in rects:
            height = rect.get_height()
            # Format: thousands separator, 0 decimals for Qty, 2 for Value
            label_text = f'{height:,.0f}' if not is_value else f'{height:,.2f}'

            # Determine position (above for positive, below for negative)
            va_pos = 'bottom' if height >= 0 else 'top'
            offset = 3 if height >= 0 else -3

            ax.annotate(label_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, offset),
                        textcoords="offset points",
                        ha='center', va=va_pos,
                        fontsize=8, fontweight='bold', rotation=0)

    # --- PLOT 1: QUANTITIES ---
    rects1 = ax1.bar(x - width, plot_data['Total Old SS (Qty)'], width, label='Old SS (Qty)', color='lightgrey')
    rects2 = ax1.bar(x, plot_data['Total New SS (Qty)'], width, label='New SS (Qty)', color='skyblue')
    rects3 = ax1.bar(x + width, plot_data['Total Qty Diff'], width, label='Qty Diff', color='orange')

    ax1.set_ylabel('Quantity (pcs)')
    ax1.set_title('Safety Stock Comparison - Quantities')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # Expand Y-axis to fit labels (min/max with padding)
    all_qty = plot_data[['Total Old SS (Qty)', 'Total New SS (Qty)', 'Total Qty Diff']]
    ax1.set_ylim(all_qty.min().min() * 1.2, all_qty.max().max() * 1.2)

    autolabel(rects1, ax1)
    autolabel(rects2, ax1)
    autolabel(rects3, ax1)

    # --- PLOT 2: VALUES (Value in EUR) ---
    rects4 = ax2.bar(x - width, plot_data['Total Old SS Value [EUR]'], width, label='Old SS Value (EUR)',
                     color='#762a83')
    rects5 = ax2.bar(x, plot_data['Total New SS Value [EUR]'], width, label='New SS Value (EUR)', color='#1b7837')
    rects6 = ax2.bar(x + width, plot_data['Value Difference [EUR]'], width, label='Value Difference (EUR)',
                     color='#d73027')

    ax2.set_ylabel('Value (EUR)')
    ax2.set_title('Safety Stock Value Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    # Expand Y-axis for values
    all_val = plot_data[['Total Old SS Value [EUR]', 'Total New SS Value [EUR]', 'Value Difference [EUR]']]
    ax2.set_ylim(all_val.min().min() * 1.2, all_val.max().max() * 1.2)

    autolabel(rects4, ax2, is_value=True)
    autolabel(rects5, ax2, is_value=True)
    autolabel(rects6, ax2, is_value=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    # plt.show()
    plt.close(fig)

    return fig