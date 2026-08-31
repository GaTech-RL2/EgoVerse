from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

engine = create_default_engine()
df = episode_table_to_df(engine)

print("\nSHAPE")
print(df.shape)

print("\nCOLUMNS")
print(df.columns.tolist())

print("\nDTYPES")
print(df.dtypes.astype(str).to_string())

for col in ["lab", "task", "scene", "embodiment", "duration", "fps", "status"]:
    if col in df.columns:
        print(f"\nTOP VALUES: {col}")
        print(df[col].value_counts(dropna=False).head(30).to_string())

print("\nSAMPLE ROWS (safe columns only)")
safe = [
    c for c in [
        "episode_hash", "lab", "task", "scene", "embodiment",
        "duration", "num_frames", "fps", "is_eval", "eval_success",
        "eval_score", "language",
    ]
    if c in df.columns
]
print(df[safe].head(10).to_string(index=False))
