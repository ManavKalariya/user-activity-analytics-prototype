import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Input parquet directory
INPUT_PATH = "pre-processed/sessions_sampled.parquet/"
OUTPUT_PATH = "pre-processed/session_sequences_embedded.parquet"

# Ensure output dir exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

print("📥 Loading pre-processed sessions...")
df = pd.read_parquet(INPUT_PATH)
df = df.reset_index() # <-- Add this line

# Ensure timestamp is datetime
df['event_time'] = pd.to_datetime(df['event_time'])

# Sort for consistency
df = df.sort_values(by=['user_id', 'user_session', 'event_time'])

print("🔄 Encoding categorical attributes...")

# Encode categorical fields
event_encoder = LabelEncoder()
df['event_type_enc'] = event_encoder.fit_transform(df['event_type'].astype(str))

cat_encoder = LabelEncoder()
df['category_code_enc'] = cat_encoder.fit_transform(df['category_code'].astype(str))

brand_encoder = LabelEncoder()
df['brand_enc'] = brand_encoder.fit_transform(df['brand'].astype(str))

# Normalize price
scaler = MinMaxScaler()
df['price_norm'] = scaler.fit_transform(df[['price']])

# Optional: encode product_id (useful if you want embeddings later in ML model)
product_encoder = LabelEncoder()
df['product_id_enc'] = product_encoder.fit_transform(df['product_id'].astype(str))

print("🔄 Constructing session sequences...")

# Group by user + session
sessions = df.groupby(['user_id', 'user_session']).agg({
    'event_type': list,
    'event_type_enc': list,
    'product_id': list,
    'product_id_enc': list,
    'category_id': list,
    'category_code': list,
    'category_code_enc': list,
    'brand': list,
    'brand_enc': list,
    'price': list,
    'price_norm': list,
    'event_time': list
}).reset_index()

# Compute time deltas
def compute_time_deltas(timestamps):
    deltas = [0]
    for i in range(1, len(timestamps)):
        deltas.append((timestamps[i] - timestamps[i-1]).total_seconds())
    return deltas

sessions['time_deltas'] = sessions['event_time'].apply(compute_time_deltas)
sessions['start_time'] = sessions['event_time'].apply(lambda x: x[0])
sessions['end_time'] = sessions['event_time'].apply(lambda x: x[-1])

# Drop raw event_time list
sessions = sessions.drop(columns=['event_time'])

print("💾 Saving sequences with embeddings to parquet...")
sessions.to_parquet(OUTPUT_PATH, index=False)

print(f"✅ Session sequences with embeddings saved at: {OUTPUT_PATH}")

import numpy as np

print("📦 Converting sessions into numeric arrays...")

def build_event_matrix(row):
    events = np.array(list(zip(
        row['event_type_enc'],
        row['product_id_enc'],
        row['category_code_enc'],
        row['brand_enc'],
        row['price_norm'],
        row['time_deltas']
    )), dtype=np.float32)
    return events

sessions['event_matrix'] = sessions.apply(build_event_matrix, axis=1)

# Save as npz for fast model loading
npz_path = "pre-processed/session_sequences_arrays.npz"
np.savez_compressed(
    npz_path,
    user_id=sessions['user_id'].values,
    user_session=sessions['user_session'].values,
    start_time=sessions['start_time'].values.astype("datetime64[s]"),
    end_time=sessions['end_time'].values.astype("datetime64[s]"),
    event_matrices=sessions['event_matrix'].values
)

print(f"✅ Saved session arrays at: {npz_path}")

import numpy as np

MAX_LEN = 100  # 👈 tune this based on dataset (e.g., 50 events per session covers most cases)

print(f"📦 Converting sessions into padded arrays (max length = {MAX_LEN})...")

def build_event_matrix(row):
    events = np.array(list(zip(
        row['event_type_enc'],
        row['product_id_enc'],
        row['category_code_enc'],
        row['brand_enc'],
        row['price_norm'],
        row['time_deltas']
    )), dtype=np.float32)
    return events

sessions['event_matrix'] = sessions.apply(build_event_matrix, axis=1)

# Pad sequences
def pad_sequence(matrix, max_len=MAX_LEN, feature_dim=6):
    padded = np.zeros((max_len, feature_dim), dtype=np.float32)
    mask = np.zeros((max_len,), dtype=np.int32)
    length = min(len(matrix), max_len)
    padded[:length] = matrix[:length]
    mask[:length] = 1
    return padded, mask

padded_matrices = []
masks = []

for mat in sessions['event_matrix']:
    padded, mask = pad_sequence(mat, MAX_LEN)
    padded_matrices.append(padded)
    masks.append(mask)

padded_matrices = np.array(padded_matrices)
masks = np.array(masks)

# Save npz
npz_path = "pre-processed/session_sequences_padded.npz"
np.savez_compressed(
    npz_path,
    user_id=sessions['user_id'].values,
    user_session=sessions['user_session'].values,
    start_time=sessions['start_time'].values.astype("datetime64[s]"),
    end_time=sessions['end_time'].values.astype("datetime64[s]"),
    padded_matrices=padded_matrices,
    masks=masks
)

print(f"✅ Padded session arrays saved at: {npz_path}")
print("Shapes:")
print("  padded_matrices:", padded_matrices.shape)
print("  masks:", masks.shape)
