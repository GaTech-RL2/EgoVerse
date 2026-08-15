from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

engine = create_default_engine()
df = episode_table_to_df(engine)

print("\nSHAPE")
print(df.shape)

print("\nCOLUMNS")
print("\n".join(df.columns.astype(str)))

task_cols = [
    c for c in df.columns
    if any(x in c.lower() for x in ("task", "activity", "label", "instruction"))
]
print("\nPOSSIBLE TASK COLUMNS")
print(task_cols)

for col in task_cols:
    values = (
        df[col]
        .dropna()
        .astype(str)
        .loc[lambda x: x.str.contains("fold", case=False, na=False)]
        .drop_duplicates()
        .sort_values()
    )
    print(f"\nFOLD-RELATED VALUES IN {col}:")
    print("\n".join(values.tolist()[:200]) or "(none)")
