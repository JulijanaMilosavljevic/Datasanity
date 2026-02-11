def check_missing_values(df):
    missing = df.isnull().mean()
    high_missing = missing[missing > 0.3].sort_values(ascending=False)

    return {
        "high_missing_columns": high_missing.to_dict(),
        "warning": "Columns with >30% missing values detected." if len(high_missing) > 0 else None
    }
