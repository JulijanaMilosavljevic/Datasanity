def check_constant_columns(df):
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]

    return {
        "constant_columns": constant_cols,
        "warning": "Constant columns detected." if constant_cols else None
    }
