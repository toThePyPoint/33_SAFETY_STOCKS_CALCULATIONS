import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


DEFAULT_CHART_STYLE = {
    'title_fontsize': 11,
    'title_fontweight': 'normal',
    'axis_label_fontsize': 11,
    'tick_label_fontsize': 11,
    'legend_fontsize': 7,
    'bar_label_fontsize': 8,
    'bar_label_fontweight': 'bold',
    'x_tick_rotation': 0,
    'x_tick_ha': 'center',
    'grid_alpha': 0.3,
}


def _get_chart_style(chart_style=None):
    style = DEFAULT_CHART_STYLE.copy()
    if chart_style:
        style.update(chart_style)
    return style


def _format_chart_axis(ax, title, ylabel, x, labels, style):
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, fontsize=style['axis_label_fontsize'])
    ax.set_title(title, fontsize=style['title_fontsize'], fontweight=style['title_fontweight'])
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        fontsize=style['tick_label_fontsize'],
        rotation=style['x_tick_rotation'],
        ha=style['x_tick_ha']
    )
    ax.tick_params(axis='y', labelsize=style['tick_label_fontsize'])
    ax.legend(fontsize=style['legend_fontsize'])
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, linestyle='--', alpha=style['grid_alpha'], zorder=0)


def _set_padded_ylim(ax, data, padding=0.1, fallback_range=10):
    y_min, y_max = data.min().min(), data.max().max()
    y_range = y_max - y_min if y_max != y_min else fallback_range
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)


def create_stats_df(mb51_path, zsbe_path, no_ss_items_path, prd_plant, get_all_dates, start_date, end_date, k_parameter,
                    ex_rates, std_mad_treshold, min_value_for_new_ss=0):
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
    no_ss_items_df = pd.read_excel(no_ss_items_path, dtype={'material': str})
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

    # print("Dates for calculations", all_dates.sort_values())

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
    stats_df = final_df.groupby(['material', 'plant'])['quantity'].agg(
        daily_avg_consumption='mean',
        daily_std_dev='std',
        daily_mad=lambda x: np.mean(np.abs(x - np.mean(x)))
    ).reset_index()

    # Attach old safety stock for comparison
    old_ss = zsbe_df[['material', 'plant', 'safety_stock_in_SAP']]
    stats_df = pd.merge(stats_df, old_ss, on=['material', 'plant'], how='left')

    stats_df['std_mad_ratio'] = stats_df['daily_std_dev'] / stats_df['daily_mad']
    stats_df['daily_mad'] = stats_df['daily_mad'].replace(0, np.nan)
    stats_df['std_mad_ratio'] = stats_df['daily_std_dev'] / stats_df['daily_mad']

    stats_df['volatility_measure'] = np.where(
        (stats_df['std_mad_ratio'] > std_mad_treshold) & (stats_df['safety_stock_in_SAP'].fillna(0) == 0),
        stats_df['daily_mad'],  # outlier case
        stats_df['daily_std_dev']  # normal case
    )

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
            k_parameter * stats_df['volatility_measure'] * np.sqrt(stats_df['lead_time'])
    )
    stats_df['volatility_method'] = np.where(
        (stats_df['std_mad_ratio'] > std_mad_treshold) &
        (stats_df['safety_stock_in_SAP'].fillna(0) == 0),
        'MAD',
        'STD'
    )

    # Round up (since you cannot have 0.5 units in stock)
    stats_df['new_safety_stock'] = np.ceil(stats_df['new_safety_stock']).astype(int)

    # Calculate ROP
    stats_df['reorder_point'] = (stats_df['daily_avg_consumption'] * stats_df['lead_time']
                                ) + stats_df['new_safety_stock']

    # Round up
    stats_df['reorder_point'] = np.ceil(stats_df['reorder_point']).astype(int)

    # Create a boolean mask for materials present in the exclusion list
    mask = stats_df['material'].isin(no_ss_items_df['material'])

    # Initialize the 'no_ss_item' column with False as the default value
    stats_df['is_no_ss_item'] = False
    stats_df['is_below_min_ss'] = False

    # save calculated values in separate columns
    stats_df['calculated_new_ss'] = stats_df['new_safety_stock']
    stats_df['calculated_new_ROP'] = stats_df['reorder_point']

    small_new_ss_mask = (
            (stats_df['new_safety_stock'] < min_value_for_new_ss) &
            (stats_df['safety_stock_in_SAP'].fillna(0) == 0)
    )
    stats_df.loc[small_new_ss_mask, ['new_safety_stock', 'reorder_point', 'is_below_min_ss']] = [0, 0, True]

    # For matching rows, update the stock parameters to zero and set the flag to True
    stats_df.loc[mask, ['new_safety_stock', 'reorder_point', 'is_no_ss_item']] = [0, 0, True]

    # Create a new column with converted price
    # .map() matches the currency to the exchange rate, then we multiply it by the unit price
    zsbe_df['unit_price_eur'] = (
            zsbe_df['unit_price'] * zsbe_df['currency'].map(ex_rates)
    )

    # Get price AND price unit from zsbe_df
    price_data = zsbe_df[['material', 'plant', 'unit_price_eur', 'price_unit', 'material_description']].drop_duplicates()

    # Merge with statistics
    stats_df = pd.merge(stats_df, price_data, on=['material', 'plant'], how='left')

    # Calculate value
    # 1. Calculate old_ss_value first to include it in the main aggregation
    stats_df['old_ss_value'] = (stats_df['safety_stock_in_SAP'] * stats_df['unit_price_eur']) / stats_df['price_unit']

    stats_df['new_safety_stock_value'] = (
                                             stats_df['new_safety_stock'] * stats_df['unit_price_eur']
                                     ) / stats_df['price_unit']

    stats_df['ROP_value'] = (
                                    stats_df['reorder_point'] * stats_df['unit_price_eur']
                            ) / stats_df['price_unit']

    # Round to 2 decimal places (monetary value)
    stats_df['new_safety_stock_value'] = stats_df['new_safety_stock_value'].round(2)
    stats_df['ROP_value'] = stats_df['ROP_value'].round(2)
    stats_df['old_ss_value'] = stats_df['old_ss_value'].round(2)

    # Check the difference
    stats_df['ss_diff'] = stats_df['new_safety_stock'] - stats_df['safety_stock_in_SAP']
    stats_df['rop_ss_diff'] = stats_df['reorder_point'] - stats_df['safety_stock_in_SAP']
    

    stats_df['new_ss_range'] = stats_df['new_safety_stock'] / stats_df['daily_avg_consumption']
    stats_df['new_ss_range'] = stats_df['new_ss_range'].round(2)

    return stats_df


def create_plant_summary(stats_df):
    # 2. Aggregate everything in ONE step to avoid merge conflicts/duplicates
    plant_summary = stats_df.groupby('plant').agg({
        'new_safety_stock': 'sum',
        'safety_stock_in_SAP': 'sum',
        'ss_diff': 'sum',
        'rop_ss_diff': 'sum',
        'new_safety_stock_value': 'sum',
        'ROP_value': 'sum',
        'old_ss_value': 'sum',  # Aggregated here directly
        'reorder_point': 'sum'
    }).reset_index()

    # 3. Add columns with value difference
    plant_summary['ss_value_diff'] = plant_summary['new_safety_stock_value'] - plant_summary['old_ss_value']
    plant_summary['rop_ss_value_diff'] = plant_summary['ROP_value'] - plant_summary['old_ss_value']

    # 4. Rename columns
    plant_summary.rename(columns={
        'new_safety_stock': 'Total New SS (Qty)',
        'reorder_point': 'Total Reorder Point (Qty)',
        'safety_stock_in_SAP': 'Total Old SS (Qty)',
        'ss_diff': 'Total SS - SS Qty Diff',
        'rop_ss_diff': 'Total ROP - SS Qty Diff',
        'new_safety_stock_value': 'Total New SS Value [EUR]',
        'ROP_value': 'Total ROP Value [EUR]',
        'old_ss_value': 'Total Old SS Value [EUR]',
        'ss_value_diff': 'Value Difference SS - SS [EUR]',
        'rop_ss_value_diff': 'Value Difference ROP - SS [EUR]'
    }, inplace=True)

    # 5. Round monetary values - using a safer list to ensure no duplicates are missed
    cols_to_round = [
        'Total New SS Value [EUR]',
        'Total Old SS Value [EUR]',
        'Value Difference SS - SS [EUR]',
        'Value Difference ROP - SS [EUR]',
        'Total ROP Value [EUR]'  # Added this as it is also a monetary value
    ]

    # Ensure columns exist before rounding to avoid errors
    existing_cols = [c for c in cols_to_round if c in plant_summary.columns]
    plant_summary[existing_cols] = plant_summary[existing_cols].round(0).astype(int)

    return plant_summary


def create_a_summary_plot_ss_to_ss_comparison(plant_summary, save_path=None, chart_style=None):
    # Prepare data
    style = _get_chart_style(chart_style)
    plot_data = plant_summary[plant_summary['plant'] != 'TOTAL'].copy()
    labels = plot_data['plant'].astype(str)
    x = np.arange(len(labels))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

    def autolabel(rects, ax, is_value=False, position='center'):
        for rect in rects:
            height = rect.get_height()
            label_text = f'{height:,.0f}'

            va_pos = 'bottom' if height >= 0 else 'top'
            offset_y = 5 if height >= 0 else -5

            # --- PRECYZYJNE ROZPYCHANIE (ha='center' + większy offset) ---
            if position == 'left':
                ha_align = 'center'
                offset_x = -4  # Przesunięcie o 8 punktów w lewo
            elif position == 'right':
                ha_align = 'center'
                offset_x = 4  # Przesunięcie o 8 punktów w prawo
            else:
                ha_align = 'center'
                offset_x = 0

            ax.annotate(label_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(offset_x, offset_y),
                        textcoords="offset points",
                        ha=ha_align, va=va_pos,
                        fontsize=style['bar_label_fontsize'], fontweight=style['bar_label_fontweight'])

    # --- PLOT 1: QUANTITIES ---
    rects1 = ax1.bar(x - width, plot_data['Total Old SS (Qty)'], width, label='Old SS (Qty)', color='lightgrey',
                     edgecolor='none', linewidth=0, zorder=3)
    rects2 = ax1.bar(x, plot_data['Total New SS (Qty)'], width, label='New SS (Qty)', color='skyblue',
                     edgecolor='none', linewidth=0, zorder=3)
    rects3 = ax1.bar(x + width, plot_data['Total SS - SS Qty Diff'], width, label='Qty SS - SS Diff', color='orange',
                     edgecolor='none', linewidth=0, zorder=3)

    _format_chart_axis(ax1, 'Safety Stock Comparison - Quantities', 'Quantity (pcs)', x, labels, style)

    # Expand Y-axis to fit labels (min/max with padding)
    all_qty = plot_data[['Total Old SS (Qty)', 'Total New SS (Qty)', 'Total SS - SS Qty Diff']]
    ax1.set_ylim(all_qty.min().min() * 1.3, all_qty.max().max() * 1.3)

    autolabel(rects1, ax1, position='left')
    autolabel(rects2, ax1, position='center')
    autolabel(rects3, ax1, position='right')

    # --- PLOT 2: VALUES (Value in EUR) ---
    rects4 = ax2.bar(x - width, plot_data['Total Old SS Value [EUR]'], width, label='Old SS Value (EUR)',
                     color='#762a83', edgecolor='none', linewidth=0, zorder=3)
    rects5 = ax2.bar(x, plot_data['Total New SS Value [EUR]'], width, label='New SS Value (EUR)', color='#1b7837',
                     edgecolor='none', linewidth=0, zorder=3)
    rects6 = ax2.bar(x + width, plot_data['Value Difference SS - SS [EUR]'], width, label='Value Difference SS - SS (EUR)',
                     color='#d73027', edgecolor='none', linewidth=0, zorder=3)

    _format_chart_axis(ax2, 'Safety Stock Value Comparison', 'Value (EUR)', x, labels, style)

    # Expand Y-axis for values
    all_val = plot_data[['Total Old SS Value [EUR]', 'Total New SS Value [EUR]', 'Value Difference SS - SS [EUR]']]
    ax2.set_ylim(all_val.min().min() * 1.4, all_val.max().max() * 1.4)

    autolabel(rects4, ax2, is_value=True, position='left')
    autolabel(rects5, ax2, is_value=True, position='center')
    autolabel(rects6, ax2, is_value=True, position='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    # plt.show()
    plt.close(fig)

    return fig


def create_a_summary_plot_rop_to_ss_comparison(plant_summary, save_path=None, chart_style=None):
    # Prepare data
    style = _get_chart_style(chart_style)
    plot_data = plant_summary[plant_summary['plant'] != 'TOTAL'].copy()
    labels = plot_data['plant'].astype(str)
    x = np.arange(len(labels))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

    def autolabel(rects, ax, is_value=False, position='center'):
        for rect in rects:
            height = rect.get_height()
            label_text = f'{height:,.0f}'

            va_pos = 'bottom' if height >= 0 else 'top'
            offset_y = 5 if height >= 0 else -5

            # --- PRECYZYJNE ROZPYCHANIE (ha='center' + większy offset) ---
            if position == 'left':
                ha_align = 'center'
                offset_x = -4  # Przesunięcie o 8 punktów w lewo
            elif position == 'right':
                ha_align = 'center'
                offset_x = 4  # Przesunięcie o 8 punktów w prawo
            else:
                ha_align = 'center'
                offset_x = 0

            ax.annotate(label_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(offset_x, offset_y),
                        textcoords="offset points",
                        ha=ha_align, va=va_pos,
                        fontsize=style['bar_label_fontsize'], fontweight=style['bar_label_fontweight'])

    # --- PLOT 1: QUANTITIES ---
    rects7 = ax1.bar(x - width, plot_data['Total Old SS (Qty)'], width, label='Old SS (Qty)', color='lightgrey',
                     edgecolor='none', linewidth=0, zorder=3)
    rects8 = ax1.bar(x, plot_data['Total Reorder Point (Qty)'], width, label='New ROP (Qty)', color='#a6cee3',
                     edgecolor='none', linewidth=0, zorder=3)
    rects9 = ax1.bar(x + width, plot_data['Total ROP - SS Qty Diff'], width, label='Qty ROP - SS Diff', color='#fb9a99',
                     edgecolor='none', linewidth=0, zorder=3)

    _format_chart_axis(ax1, 'ROP vs Current SS Comparison - Quantities', 'Quantity (pcs)', x, labels, style)

    # --- FIX: Gwarantowany odstęp od ramki ---
    all_qty = plot_data[['Total Old SS (Qty)', 'Total Reorder Point (Qty)', 'Total ROP - SS Qty Diff']]
    y_min, y_max = all_qty.min().min(), all_qty.max().max()

    # Obliczamy całkowitą wysokość danych
    y_range = y_max - y_min if y_max != y_min else 10

    # Ustawiamy limity z dużym zapasem (np. 30% zakresu w każdą stronę)
    ax1.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    ax1.margins(y=0.4)
    ax2.margins(y=0.4)

    autolabel(rects7, ax1, position='left')
    autolabel(rects8, ax1, position='center')
    autolabel(rects9, ax1, position='right')

    # --- PLOT 2: VALUES ---
    rects10 = ax2.bar(x - width, plot_data['Total Old SS Value [EUR]'], width, label='Old SS Value (EUR)',
                      color='#762a83', edgecolor='none', linewidth=0, zorder=3)
    rects11 = ax2.bar(x, plot_data['Total ROP Value [EUR]'], width, label='Total ROP Value (EUR)', color='#33a02c',
                      edgecolor='none', linewidth=0, zorder=3)
    rects12 = ax2.bar(x + width, plot_data['Value Difference ROP - SS [EUR]'], width, label='Value Diff ROP-SS (EUR)',
                      color='#e31a1c', edgecolor='none', linewidth=0, zorder=3)

    _format_chart_axis(ax2, 'ROP vs Current SS Value Comparison', 'Value (EUR)', x, labels, style)

    # --- FIX: Gwarantowany odstęp od ramki ---
    all_val = plot_data[['Total Old SS Value [EUR]', 'Total ROP Value [EUR]', 'Value Difference ROP - SS [EUR]']]
    y_min_v, y_max_v = all_val.min().min(), all_val.max().max()

    # Obliczamy całkowitą wysokość danych
    y_range_v = y_max_v - y_min_v if y_max_v != y_min_v else 100

    # Ustawiamy limity z dużym zapasem
    ax2.set_ylim(y_min_v - 0.1 * y_range_v, y_max_v + 0.1 * y_range_v)

    autolabel(rects10, ax2, is_value=True, position='left')
    autolabel(rects11, ax2, is_value=True, position='center')
    autolabel(rects12, ax2, is_value=True, position='right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close(fig)

    return fig


def create_all_products_summary_plot_ss_to_ss_comparison(all_products_summary, save_path=None, chart_style=None):
    style = _get_chart_style(chart_style)
    plot_data = all_products_summary[all_products_summary['product_group'] != 'TOTAL'].copy()
    labels = plot_data['product_group'].astype(str)
    x = np.arange(len(labels))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    def autolabel(rects, ax, position='center'):
        for rect in rects:
            height = rect.get_height()
            label_text = f'{height:,.0f}'
            va_pos = 'bottom' if height >= 0 else 'top'
            offset_y = 5 if height >= 0 else -5

            if position == 'left':
                offset_x = -4
            elif position == 'right':
                offset_x = 4
            else:
                offset_x = 0

            ax.annotate(label_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(offset_x, offset_y),
                        textcoords="offset points",
                        ha='center', va=va_pos,
                        fontsize=style['bar_label_fontsize'], fontweight=style['bar_label_fontweight'])

    rects1 = ax1.bar(x - width, plot_data['Total Old SS (Qty)'], width, label='Old SS (Qty)', color='lightgrey',
                     edgecolor='none', linewidth=0, zorder=3)
    rects2 = ax1.bar(x, plot_data['Total New SS (Qty)'], width, label='New SS (Qty)', color='skyblue',
                     edgecolor='none', linewidth=0, zorder=3)
    rects3 = ax1.bar(x + width, plot_data['Total SS - SS Qty Diff'], width, label='Qty SS - SS Diff', color='orange',
                     edgecolor='none', linewidth=0, zorder=3)

    _format_chart_axis(ax1, 'Safety Stock Comparison by Product Group - Quantities', 'Quantity (pcs)', x, labels, style)

    all_qty = plot_data[['Total Old SS (Qty)', 'Total New SS (Qty)', 'Total SS - SS Qty Diff']]
    _set_padded_ylim(ax1, all_qty)

    autolabel(rects1, ax1, position='left')
    autolabel(rects2, ax1, position='right')
    autolabel(rects3, ax1, position='center')

    rects4 = ax2.bar(x - width, plot_data['Total Old SS Value [EUR]'], width, label='Old SS Value (EUR)',
                     color='#762a83', edgecolor='none', linewidth=0, zorder=3)
    rects5 = ax2.bar(x, plot_data['Total New SS Value [EUR]'], width, label='New SS Value (EUR)', color='#1b7837',
                     edgecolor='none', linewidth=0, zorder=3)
    rects6 = ax2.bar(x + width, plot_data['Value Difference SS - SS [EUR]'], width, label='Value Difference SS - SS (EUR)',
                     color='#d73027', edgecolor='none', linewidth=0, zorder=3)

    _format_chart_axis(ax2, 'Safety Stock Comparison by Product Group - Values', 'Value (EUR)', x, labels, style)

    all_val = plot_data[['Total Old SS Value [EUR]', 'Total New SS Value [EUR]', 'Value Difference SS - SS [EUR]']]
    _set_padded_ylim(ax2, all_val, fallback_range=100)

    autolabel(rects4, ax2, position='left')
    autolabel(rects5, ax2, position='right')
    autolabel(rects6, ax2, position='center')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close(fig)

    return fig


def create_all_products_summary_plot_rop_to_ss_comparison(all_products_summary, save_path=None, chart_style=None):
    style = _get_chart_style(chart_style)
    plot_data = all_products_summary[all_products_summary['product_group'] != 'TOTAL'].copy()
    labels = plot_data['product_group'].astype(str)
    x = np.arange(len(labels))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    def autolabel(rects, ax, position='center'):
        for rect in rects:
            height = rect.get_height()
            label_text = f'{height:,.0f}'
            va_pos = 'bottom' if height >= 0 else 'top'
            offset_y = 5 if height >= 0 else -5

            if position == 'left':
                offset_x = -4
            elif position == 'right':
                offset_x = 4
            else:
                offset_x = 0

            ax.annotate(label_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(offset_x, offset_y),
                        textcoords="offset points",
                        ha='center', va=va_pos,
                        fontsize=style['bar_label_fontsize'], fontweight=style['bar_label_fontweight'])

    rects7 = ax1.bar(x - width, plot_data['Total Old SS (Qty)'], width, label='Old SS (Qty)', color='lightgrey',
                     edgecolor='none', linewidth=0, zorder=3)
    rects8 = ax1.bar(x, plot_data['Total Reorder Point (Qty)'], width, label='New ROP (Qty)', color='#a6cee3',
                     edgecolor='none', linewidth=0, zorder=3)
    rects9 = ax1.bar(x + width, plot_data['Total ROP - SS Qty Diff'], width, label='Qty ROP - SS Diff', color='#fb9a99',
                     edgecolor='none', linewidth=0, zorder=3)

    _format_chart_axis(ax1, 'ROP vs Current SS by Product Group - Quantities', 'Quantity (pcs)', x, labels, style)

    all_qty = plot_data[['Total Old SS (Qty)', 'Total Reorder Point (Qty)', 'Total ROP - SS Qty Diff']]
    _set_padded_ylim(ax1, all_qty)

    autolabel(rects7, ax1, position='left')
    autolabel(rects8, ax1, position='right')
    autolabel(rects9, ax1, position='center')

    rects10 = ax2.bar(x - width, plot_data['Total Old SS Value [EUR]'], width, label='Old SS Value (EUR)',
                      color='#762a83', edgecolor='none', linewidth=0, zorder=3)
    rects11 = ax2.bar(x, plot_data['Total ROP Value [EUR]'], width, label='Total ROP Value (EUR)', color='#33a02c',
                      edgecolor='none', linewidth=0, zorder=3)
    rects12 = ax2.bar(x + width, plot_data['Value Difference ROP - SS [EUR]'], width, label='Value Diff ROP-SS (EUR)',
                      color='#e31a1c', edgecolor='none', linewidth=0, zorder=3)

    _format_chart_axis(ax2, 'ROP vs Current SS by Product Group - Values', 'Value (EUR)', x, labels, style)

    all_val = plot_data[['Total Old SS Value [EUR]', 'Total ROP Value [EUR]', 'Value Difference ROP - SS [EUR]']]
    _set_padded_ylim(ax2, all_val, fallback_range=100)

    autolabel(rects10, ax2, position='left')
    autolabel(rects11, ax2, position='right')
    autolabel(rects12, ax2, position='center')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close(fig)

    return fig


def export_df_to_excel_file(df, file_path):
    df = df[[
        'plant', 'material', 'material_description', 'lead_time',
        'daily_avg_consumption', 'new_safety_stock', 'new_ss_range', 'reorder_point', 'safety_stock_in_SAP', 'ss_diff',
        'rop_ss_diff', 'volatility_method', 'is_no_ss_item', 'is_below_min_ss', 'calculated_new_ss', 'calculated_new_ROP'
    ]]

    df.to_excel(file_path, index=False)


def get_input_files(directory, prd_groups):
    directory = Path(directory)
    return {
        prd_group: (
            str(directory / f"{prd_group}_Consumption.XLSX"),
            str(directory / f"{prd_group}_items_and_parameters.XLSX"),
        )
        for prd_group in prd_groups
    }


def create_product_group_summary_row(plant_summary, product_group):
    product_summary = plant_summary.select_dtypes(include='number').sum().to_dict()
    product_summary['product_group'] = product_group
    return product_summary


def create_all_products_summary(product_summary_rows, add_total=True, total_label='TOTAL'):
    all_products_summary = pd.DataFrame(product_summary_rows)

    if all_products_summary.empty:
        return all_products_summary

    all_products_summary = all_products_summary[
        ['product_group'] + [col for col in all_products_summary.columns if col != 'product_group']
    ]

    if add_total:
        total_row = all_products_summary.select_dtypes(include='number').sum().to_dict()
        total_row['product_group'] = total_label
        all_products_summary = pd.concat(
            [all_products_summary, pd.DataFrame([total_row])],
            ignore_index=True
        )

    return all_products_summary


def create_all_product_groups_plant_summary(plant_summaries):
    if not plant_summaries:
        return pd.DataFrame()

    all_product_groups_plant_summary = pd.concat(plant_summaries, ignore_index=True)
    all_product_groups_plant_summary = all_product_groups_plant_summary.groupby(
        'plant',
        as_index=False
    ).sum(numeric_only=True)

    return all_product_groups_plant_summary[plant_summaries[-1].columns]


def create_new_safety_stocks_df(stats_by_product_group):
    columns = [
        'product_group',
        'plant',
        'material',
        'material_description',
        'daily_avg_consumption',
        'reorder_point',
        'new_safety_stock',
    ]
    new_safety_stocks = []

    for product_group, stats_df in stats_by_product_group.items():
        mask = (
                (stats_df['new_safety_stock'] > 0) &
                (stats_df['safety_stock_in_SAP'].fillna(0) == 0)
        )
        product_group_new_safety_stocks = stats_df.loc[mask, columns[1:]].copy()
        product_group_new_safety_stocks.insert(0, 'product_group', product_group)
        new_safety_stocks.append(product_group_new_safety_stocks)

    if not new_safety_stocks:
        return pd.DataFrame(columns=columns)

    return pd.concat(new_safety_stocks, ignore_index=True)[columns]


def create_safety_stocks_to_be_deleted_df(stats_by_product_group):
    columns = [
        'product_group',
        'plant',
        'material',
        'material_description',
        'daily_avg_consumption',
        'reorder_point',
        'new_safety_stock',
        'safety_stock_in_SAP',
        'old_ss_value'
    ]
    safety_stocks_to_be_deleted = []

    for product_group, stats_df in stats_by_product_group.items():
        mask = (
                (stats_df['new_safety_stock'] == 0) &
                (stats_df['safety_stock_in_SAP'].fillna(0) > 0)
        )
        product_group_safety_stocks_to_be_deleted = stats_df.loc[mask, columns[1:]].copy()
        product_group_safety_stocks_to_be_deleted.insert(0, 'product_group', product_group)
        safety_stocks_to_be_deleted.append(product_group_safety_stocks_to_be_deleted)

    if not safety_stocks_to_be_deleted:
        return pd.DataFrame(columns=columns)

    return pd.concat(safety_stocks_to_be_deleted, ignore_index=True)[columns]


def _display_report_item(display_output, item):
    if not display_output:
        return

    try:
        from IPython.display import display
    except ImportError:
        return

    display(item)


def _display_report_header(display_output, text):
    if not display_output:
        return

    try:
        from IPython.display import Markdown
    except ImportError:
        return

    _display_report_item(display_output, Markdown(text))


def create_many_product_groups_report(
        input_directory,
        product_groups,
        no_ss_items_path,
        prd_plant,
        get_all_dates,
        start_date,
        end_date,
        k_parameter,
        ex_rates,
        std_mad_treshold,
        min_value_for_new_ss=0,
        output_directory=None,
        new_safety_stocks_file_name=None,
        safety_stocks_to_be_deleted_file_name=None,
        display_output=True,
        show_group_charts=True,
        show_final_charts=True,
        chart_style=None
):
    all_files = get_input_files(input_directory, product_groups)
    product_summary_rows = []
    plant_summaries = []
    stats_by_product_group = {}
    plant_summary_by_product_group = {}
    group_figures = {}

    for product_group, (mb51_f_path, zsbe_f_path) in all_files.items():
        _display_report_header(display_output, f'## Calculating: {product_group}')

        stats_df = create_stats_df(
            mb51_f_path,
            zsbe_f_path,
            no_ss_items_path,
            prd_plant,
            get_all_dates,
            start_date,
            end_date,
            k_parameter,
            ex_rates,
            std_mad_treshold,
            min_value_for_new_ss
        )
        plant_summary = create_plant_summary(stats_df)

        stats_by_product_group[product_group] = stats_df
        plant_summary_by_product_group[product_group] = plant_summary
        plant_summaries.append(plant_summary)
        product_summary_rows.append(create_product_group_summary_row(plant_summary, product_group))

        if show_group_charts:
            fig_rop = create_a_summary_plot_rop_to_ss_comparison(plant_summary, chart_style=chart_style)
            group_figures[product_group] = fig_rop
            _display_report_item(display_output, fig_rop)

        if output_directory:
            output_file_path = Path(output_directory) / f'{product_group}_ss_summary.xlsx'
            export_df_to_excel_file(stats_df, output_file_path)

    all_products_summary = create_all_products_summary(product_summary_rows)
    all_product_groups_plant_summary = create_all_product_groups_plant_summary(plant_summaries)
    new_safety_stocks_df = create_new_safety_stocks_df(stats_by_product_group)
    safety_stocks_to_be_deleted_df = create_safety_stocks_to_be_deleted_df(stats_by_product_group)

    if output_directory and new_safety_stocks_file_name:
        new_safety_stocks_file_path = Path(output_directory) / new_safety_stocks_file_name
        new_safety_stocks_df.to_excel(new_safety_stocks_file_path, index=False)

    if output_directory and safety_stocks_to_be_deleted_file_name:
        safety_stocks_to_be_deleted_file_path = Path(output_directory) / safety_stocks_to_be_deleted_file_name
        safety_stocks_to_be_deleted_df.to_excel(safety_stocks_to_be_deleted_file_path, index=False)

    final_figures = {}

    if show_final_charts:
        _display_report_header(
            display_output,
            f'## New safety stocks to be created: {len(new_safety_stocks_df)}'
        )
        _display_report_header(
            display_output,
            f'## Safety stocks to be deleted: {len(safety_stocks_to_be_deleted_df)}'
        )

        _display_report_header(display_output, '## All product groups plant summary')
        _display_report_item(display_output, all_product_groups_plant_summary)

        fig_all_products_plant_summary = create_a_summary_plot_rop_to_ss_comparison(
            all_product_groups_plant_summary,
            chart_style=chart_style
        )
        final_figures['all_product_groups_plant_summary_rop'] = fig_all_products_plant_summary
        _display_report_item(display_output, fig_all_products_plant_summary)

        _display_report_header(display_output, '## All products summary')
        _display_report_item(display_output, all_products_summary)

        fig_all_products = create_all_products_summary_plot_rop_to_ss_comparison(
            all_products_summary,
            chart_style=chart_style
        )
        final_figures['all_products_summary_rop'] = fig_all_products
        _display_report_item(display_output, fig_all_products)

    return {
        'all_files': all_files,
        'all_products_summary': all_products_summary,
        'all_product_groups_plant_summary': all_product_groups_plant_summary,
        'new_safety_stocks_df': new_safety_stocks_df,
        'safety_stocks_to_be_deleted_df': safety_stocks_to_be_deleted_df,
        'stats_by_product_group': stats_by_product_group,
        'plant_summary_by_product_group': plant_summary_by_product_group,
        'group_figures': group_figures,
        'final_figures': final_figures,
    }
