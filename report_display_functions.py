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