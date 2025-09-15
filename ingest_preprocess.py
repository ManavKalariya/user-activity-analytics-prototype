import dask.dataframe as dd
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from dask.distributed import Client, progress

# ========== CONFIG ==========
file_path = "raw-dataset/2019-Nov.csv"   # <-- adjust path
use_sampling = True     # Set False for full run
sample_frac = 0.01      # 1% of users for dev runs
block_size = "128MB"    # partition size (controls memory usage)
# ============================

def main():
    print("🚀 Starting preprocessing...")

    # 1. Start Dask client (optimized for i5-1235U, 16 GB RAM)
    client = Client(
        n_workers=4,              # 4 worker processes
        threads_per_worker=2,     # 2 threads per worker → 8 active threads total
        memory_limit="3GB",       # per-worker memory cap
        local_directory="dask-temp"
    )
    print(f"✅ Dask client started. Dashboard: {client.dashboard_link}")

    # 2. Load dataset with tuned blocksize
    print("📂 Loading dataset...")
    df = dd.read_csv(file_path, blocksize=block_size)

    print("📋 Raw Data Sample:")
    print(df.head())   # triggers only a small preview

    # 3. OPTIONAL SAMPLING (for dev runs)
    if use_sampling:
        print("🎯 Sampling users for dev run...")
        unique_users = df["user_id"].dropna().unique().compute()
        sampled_users = pd.Series(unique_users).sample(frac=sample_frac, random_state=42).tolist()
        df = df[df["user_id"].isin(sampled_users)]
        print(f"✅ Using {len(sampled_users)} sampled users out of {len(unique_users)} total.")

    # 4. Handle missing values
    print("🧹 Cleaning missing values...")
    df["brand"] = df["brand"].fillna("unknown")
    df["category_code"] = df["category_code"].fillna("unknown")

    # 5. Parse timestamps
    print("⏱️ Parsing timestamps...")
    df["event_time"] = dd.to_datetime(df["event_time"], errors="coerce")

    # 6. Encode categorical variables
    print("🔢 Encoding categorical variables...")

    def encode_with_le(col):
        le = LabelEncoder()
        values = df[col].dropna().unique().compute()   # unique values in memory
        le.fit(values)
        return df[col].map_partitions(lambda s: le.transform(s.astype(str)), meta=("x", "int"))

    df["event_type_enc"] = encode_with_le("event_type")
    df["brand_enc"] = encode_with_le("brand")
    df["category_code_enc"] = encode_with_le("category_code")

    # 7. Sort by user_id + session_id + event_time
    print("📑 Sorting sessions...")
    df = df.set_index("event_time")
    df = df.map_partitions(lambda pdf: pdf.sort_values(["user_id", "user_session", "event_time"]))

    print("\n🔎 Processed Session Sample:")
    print(df.head(10))

    # 8. Persist sessions for future use
    print("💾 Saving sessions to disk...")
    output_path = "pre-processed/sessions_sampled.parquet" if use_sampling else "pre-processed/sessions.parquet"

    write_future = df.to_parquet(output_path, engine="pyarrow", overwrite=True, compute=False)
    progress(write_future)  # show progress bar
    write_future.compute()

    print(f"✅ Sessions saved to {output_path}")

if __name__ == "__main__":
    main()
