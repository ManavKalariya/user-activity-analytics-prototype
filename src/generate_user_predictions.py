import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import os
from tqdm import tqdm
import warnings

# Suppress the specific UserWarning from sklearn
# We can do this because we are consistently building the feature set in the same order
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names, but MinMaxScaler was fitted with feature names")

# --------------------------
# Device Setup
# --------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("[INFO] Using Apple MPS")
else:
    DEVICE = torch.device("cpu")
    print("[INFO] Using CPU")

# --------------------------
# Model Definitions (Copied from training scripts)
# --------------------------
MAX_LEN = 100
FEATURE_DIM = 6
EMBED_DIM = 64
TRANSFORMER_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2

class SessionTower(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, embed_dim=EMBED_DIM, transformer_dim=TRANSFORMER_DIM,
                 num_heads=NUM_HEADS, num_layers=NUM_LAYERS):
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_proj = nn.Linear(feature_dim, embed_dim)
        self.pos_emb = nn.Embedding(MAX_LEN, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=transformer_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask):
        b, l, f = x.shape
        x = self.input_norm(x)
        x_proj = self.input_proj(x)
        pos_ids = torch.arange(l, device=x.device).unsqueeze(0).expand(b, l)
        pos_ids = pos_ids.clamp(0, self.pos_emb.num_embeddings - 1)
        x_proj = x_proj + self.pos_emb(pos_ids)
        src_key_padding_mask = ~mask
        out = self.transformer(x_proj, src_key_padding_mask=src_key_padding_mask)
        mask_f = mask.unsqueeze(-1).float()
        sum_out = (out * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = sum_out / denom
        return self.output_proj(pooled)

class ProductTower(nn.Module):
    def __init__(self, product_vocab_size, embed_dim=EMBED_DIM):
        super().__init__()
        self.product_embedding = nn.Embedding(product_vocab_size, embed_dim)
    
    def forward(self, product_ids):
        return self.product_embedding(product_ids)

class TwoTowerModel(nn.Module):
    def __init__(self, session_tower, product_tower, event_vocab_size, embed_dim=EMBED_DIM):
        super().__init__()
        self.session_tower = session_tower
        self.product_tower = product_tower
        self.event_head = nn.Linear(embed_dim, event_vocab_size)

    def forward(self, session_x, session_mask, product_ids=None):
        session_embedding = self.session_tower(session_x, session_mask)
        event_logits = self.event_head(session_embedding)
        
        if product_ids is None:
            # Inference for event prediction only
            return None, event_logits

        # Training/Inference for product retrieval
        product_embedding = self.product_tower(product_ids)
        session_embedding_norm = F.normalize(session_embedding, p=2, dim=1)
        product_embedding_norm = F.normalize(product_embedding, p=2, dim=2)
        product_scores = torch.bmm(session_embedding_norm.unsqueeze(1), product_embedding_norm.transpose(1, 2)).squeeze(1)
        return product_scores, event_logits

print("✅ Model classes defined.")

# --------------------------
# Load All Models and Data
# --------------------------
print("\n--- Step 1: Loading all models, scalers, and data ---")

# Paths
SESSIONS_PATH = "pre-processed/session_sequences_embedded.parquet"
PADDED_DATA_PATH = "pre-processed/session_sequences_padded.npz"
FEATURES_PATH = "pre-processed/features_per_user_session.npz"
TWO_TOWER_MODEL_PATH = "pre-processed/two_tower_model_best.pt"
SURVIVAL_MODEL_PATH = "pre-processed/survival_models/wft_model.joblib"
SCALER_PATH = "pre-processed/survival_models/scaler.joblib"
OUTPUT_PATH = "analytics/predictions/combined_user_predictions.json"

# Load data
sessions_df = pd.read_parquet(SESSIONS_PATH)
sessions_df['end_time'] = pd.to_datetime(sessions_df['end_time'])
padded_data = np.load(PADDED_DATA_PATH, allow_pickle=True)
padded_matrices = torch.tensor(padded_data['padded_matrices'], dtype=torch.float32)
masks = torch.tensor(padded_data['masks'], dtype=torch.bool)
features_data = np.load(FEATURES_PATH, allow_pickle=True)

# Recreate product vocabulary mapping from training
labels_product_np = features_data["labels_next_product"]
valid_idx = labels_product_np >= 0
labels_product_np = labels_product_np[valid_idx]
(unique, counts) = np.unique(labels_product_np, return_counts=True)
sorted_idx = np.argsort(-counts)
top_k = 200
top_products = unique[sorted_idx[:top_k]]
product_id_map = {i: int(prod) for i, prod in enumerate(top_products)}
product_id_map[top_k] = 'other' # Add 'other' category
product_vocab_size = top_k + 1

# Event type mapping (from feature engineering)
event_map = {'view': 0, 'cart': 1, 'purchase': 2}
event_map_inv = {v: k for k, v in event_map.items()}
event_vocab_size = len(event_map)

# Load Two-Tower Model
two_tower_model = TwoTowerModel(SessionTower(), ProductTower(product_vocab_size), event_vocab_size).to(DEVICE)
two_tower_model.load_state_dict(torch.load(TWO_TOWER_MODEL_PATH, map_location=DEVICE))
two_tower_model.eval()
print("✅ Two-Tower model loaded.")

# Load Survival Model and Scaler
survival_model = joblib.load(SURVIVAL_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Survival model and scaler loaded.")

# Create combined features_df for survival model prediction
user_features = {f['user_id']: f['rfm'] for f in features_data['features'] if 'user_id' in f and 'rfm' in f}
rfm_df = pd.DataFrame.from_dict(user_features, orient='index', columns=['Recency', 'Frequency', 'Monetary'])
rfm_df.index.name = 'user_id'

print("🔄 Generating session embeddings in batches to conserve memory...")
inference_batch_size = 512 # Process in smaller chunks
num_batches = int(np.ceil(len(padded_matrices) / inference_batch_size))
all_embeddings = []
with torch.no_grad():
    for i in tqdm(range(num_batches), desc="Generating All Session Embeddings"):
        start = i * inference_batch_size
        end = (i + 1) * inference_batch_size
        batch_padded = padded_matrices[start:end].to(DEVICE)
        batch_masks = masks[start:end].to(DEVICE)
        
        embeddings = two_tower_model.session_tower(batch_padded, batch_masks)
        all_embeddings.append(embeddings.cpu().numpy())
session_embeddings_np = np.vstack(all_embeddings)

embedding_cols = [f'embed_{i}' for i in range(session_embeddings_np.shape[1])]
session_embeddings_df = pd.DataFrame(session_embeddings_np, columns=embedding_cols, index=sessions_df.index)
features_df = pd.concat([sessions_df[['user_id', 'user_session', 'end_time']], session_embeddings_df], axis=1)
features_df = features_df.merge(rfm_df, on='user_id', how='left')
print("✅ All data prepared for prediction.")

# --------------------------
# Prediction Functions
# --------------------------

def predict_next_action(model, session_tensor, mask_tensor, all_product_ids):
    """Predicts next event and product using the Two-Tower model."""
    with torch.no_grad():
        session_tensor = session_tensor.unsqueeze(0).to(DEVICE)
        mask_tensor = mask_tensor.unsqueeze(0).to(DEVICE)
        all_product_ids = all_product_ids.to(DEVICE)
        
        # We need to pass all possible products to get scores for each
        product_scores, event_logits = model(session_tensor, mask_tensor, all_product_ids.unsqueeze(0))
        
        # Event prediction
        event_probs = F.softmax(event_logits, dim=1).squeeze().cpu().numpy()
        
        # Product prediction
        best_product_idx = product_scores.argmax().item()
        
        return event_probs, best_product_idx

def predict_purchase_timing(model, scaler, features):
    """Predicts time-to-next-purchase using the Survival model."""
    features_to_scale = features.drop(['user_id']).index
    scaled_features = scaler.transform(features.drop(['user_id']).values.reshape(1, -1))
    
    scaled_df = pd.DataFrame(scaled_features, columns=features_to_scale)
    
    median_time = model.predict_median(scaled_df)
    
    return median_time.iloc[0]

# --------------------------
# Main Prediction Loop
# --------------------------
print("\n--- Step 2: Generating combined predictions for all users ---")

# Get all unique user IDs
all_user_ids = sessions_df['user_id'].unique()

# Pre-calculate the set of purchase session IDs for efficient lookup
purchase_session_ids = set(sessions_df[sessions_df['event_type'].apply(lambda events: 'purchase' in events)]['user_session'])

all_predictions = {}
all_product_ids_tensor = torch.tensor(list(range(product_vocab_size)), dtype=torch.long)

for user_id in tqdm(all_user_ids, desc="Processing All Users"):
    user_predictions = {}
    
    # --- Part A: Predict Next Action (from latest session) ---
    user_sessions = sessions_df[sessions_df['user_id'] == user_id].sort_values('end_time', ascending=False)
    if user_sessions.empty:
        continue # Skip users with no sessions
        
    latest_session_idx = user_sessions.index[0]
    
    session_tensor = padded_matrices[latest_session_idx]
    mask_tensor = masks[latest_session_idx]
    
    event_probs, best_product_idx = predict_next_action(two_tower_model, session_tensor, mask_tensor, all_product_ids_tensor)
    
    predicted_event = event_map_inv[event_probs.argmax()]
    predicted_product_id = product_id_map.get(best_product_idx, 'unknown')
    
    user_predictions['next_action_prediction'] = {
        'predicted_event': predicted_event,
        'event_probabilities': {event_map_inv[i]: float(p) for i, p in enumerate(event_probs)},
        'predicted_product_id': predicted_product_id
    }
    
    # --- Part B: Predict Next Purchase Timing (from last purchase session) ---
    user_purchase_sessions = features_df[
        (features_df['user_id'] == user_id) &
        (features_df['user_session'].isin(purchase_session_ids))
    ].sort_values('end_time', ascending=False)
    
    if not user_purchase_sessions.empty:
        last_purchase_features = user_purchase_sessions.iloc[0].drop(['user_session', 'end_time'])
        
        expected_days = predict_purchase_timing(survival_model, scaler, last_purchase_features)
        
        user_predictions['next_purchase_prediction'] = {
            'status': 'Prediction generated',
            'expected_median_days_to_next_purchase': float(expected_days)
        }
    else:
        user_predictions['next_purchase_prediction'] = {
            'status': 'No purchase history found for this user',
            'expected_median_days_to_next_purchase': None
        }
        
    all_predictions[int(user_id)] = user_predictions

# --------------------------
# Save Output
# --------------------------
print("\n--- Step 3: Saving combined predictions ---")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

import json
with open(OUTPUT_PATH, 'w') as f:
    json.dump(all_predictions, f, indent=4)

print(f"✅ Combined predictions saved to: {OUTPUT_PATH}")
print("\n--- Script Finished ---")
print("\nSample Output:")
print(json.dumps(list(all_predictions.values())[0], indent=4))
