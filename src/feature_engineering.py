import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# Paths
PARQUET_PATH = "pre-processed/session_sequences_embedded.parquet"
NPZ_PATH = "pre-processed/session_sequences_padded.npz"
OUTPUT_PATH = "pre-processed/features_per_user_session.npz"

print("📥 Loading session parquet...")
sessions = pd.read_parquet(PARQUET_PATH)

print("📥 Loading padded arrays...")
data = np.load(NPZ_PATH, allow_pickle=True)
padded_matrices = data['padded_matrices']
masks = data['masks']

# -------------------------------
# 🔹 User-level RFM Features
# -------------------------------
print("🔄 Computing RFM features...")

# Recency: days since last session
sessions['end_time'] = pd.to_datetime(sessions['end_time'])
max_time = sessions['end_time'].max()
# Compute total purchase price per session
def purchase_sum(row):
    # Sum only prices where event_type == 'purchase'
    return np.sum([p for p, e in zip(row['price'], row['event_type']) if e == 'purchase'])
sessions['total_purchase_price'] = sessions.apply(purchase_sum, axis=1)
rfm = sessions.groupby('user_id').agg({
    'end_time': lambda x: (max_time - x.max()).days,
    'user_session': 'count',
    'total_purchase_price': 'sum'
}).reset_index()

rfm.columns = ['user_id', 'Recency', 'Frequency', 'Monetary']

# Normalize RFM
scaler = MinMaxScaler()
rfm[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(
    rfm[['Recency', 'Frequency', 'Monetary']]
)

# -------------------------------
# 🔹 Session Embeddings
# -------------------------------
print("🔄 Building session embeddings (Word2Vec + TF-IDF)...")

# Prepare sequences of product_ids
product_seqs = sessions['product_id'].apply(lambda x: [str(p) for p in x])

# Word2Vec embeddings
w2v = Word2Vec(sentences=product_seqs, vector_size=50, window=5, min_count=1, workers=4)
def get_w2v_embedding(seq):
    vecs = [w2v.wv[p] for p in seq if p in w2v.wv]
    return np.mean(vecs, axis=0) if len(vecs) > 0 else np.zeros(50)

sessions['w2v_embed'] = product_seqs.apply(get_w2v_embedding)

# TF-IDF embeddings (over category_code)
tfidf = TfidfVectorizer()
cat_texts = sessions['category_code'].apply(lambda x: " ".join([str(c) for c in x]))
tfidf_matrix = tfidf.fit_transform(cat_texts)

# -------------------------------
# 🔹 Construct Feature Dict
# -------------------------------
print("📦 Constructing feature dict per session...")

features = []
labels_next_event = []
labels_next_product = []

for i, row in sessions.iterrows():
    feat_dict = {
        "user_id": row['user_id'],
        "user_session": row['user_session'],
        "rfm": rfm[rfm['user_id'] == row['user_id']][['Recency','Frequency','Monetary']].values[0],
        "w2v_embed": row['w2v_embed'],
        "tfidf_embed": tfidf_matrix[i].toarray()[0]
    }
    features.append(feat_dict)

    # Generate labels (next-event and next-product)
    if len(row['event_type']) > 1:
        labels_next_event.append(row['event_type_enc'][-1])   # last event
        labels_next_product.append(row['product_id_enc'][-1]) # last product
    else:
        labels_next_event.append(-1)  # placeholder for sessions with single event
        labels_next_product.append(-1)

# Save features & labels
np.savez_compressed(
    OUTPUT_PATH,
    features=features,
    labels_next_event=np.array(labels_next_event),
    labels_next_product=np.array(labels_next_product),
    padded_matrices=padded_matrices,
    masks=masks
)

print(f"✅ Features + labels saved at: {OUTPUT_PATH}")
print("✅ Feature engineering complete.")

# #verifying saved output
# import numpy as np
# data = np.load("pre-processed/features_per_user_session.npz", allow_pickle=True)

# features = data['features']
# print("Total sessions:", len(features))
# print("First session keys:", features[0].keys())
# print("RFM vector:", features[0]['rfm'])
# print("W2V vector shape:", features[0]['w2v_embed'].shape)
# print("TF-IDF vector length:", len(features[0]['tfidf_embed']))

# print("Label (next-event):", data['labels_next_event'][0])
# print("Label (next-product):", data['labels_next_product'][0])
