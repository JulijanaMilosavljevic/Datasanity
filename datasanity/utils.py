def truncate_dict(d: dict, max_items: int = 50) -> dict:
    """
    Truncate large dicts for nicer display in UI/report.
    Keeps first max_items items by insertion order.
    """
    if not isinstance(d, dict):
        return d
    if len(d) <= max_items:
        return d
    items = list(d.items())[:max_items]
    out = dict(items)
    out["..."] = f"truncated (showing first {max_items} of {len(d)})"
    return out
