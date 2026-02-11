def check_duplicates(df):
    num_duplicates = int(df.duplicated().sum())

    return {
        "num_duplicates": num_duplicates,
        "warning": "Duplicate rows found." if num_duplicates > 0 else None
    }
