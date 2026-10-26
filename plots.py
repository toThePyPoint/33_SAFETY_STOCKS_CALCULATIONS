import matplotlib.pyplot as plt
import numpy as np

from calc_script import DEFAULT_CHART_STYLE


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

