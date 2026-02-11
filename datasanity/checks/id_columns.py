def check_id_like_columns(df):
    if len(df) == 0:
        return {"id_like_columns": []}

    id_like = [col for col in df.columns if df[col].nunique(dropna=False) > 0.98 * len(df)]

    return {
        "id_like_columns": id_like,
        "warning": "ID-like columns detected." if id_like else None
    }
