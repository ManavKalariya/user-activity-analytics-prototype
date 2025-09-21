import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines import CoxPHFitter, WeibullAFTFitter
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os
import joblib

print("--- Step 1: Environment Setup ---")
# Ensure lifelines is installed
try:
    import lifelines
    print("✅ lifelines library is installed.")
except ImportError:
    print("❌ lifelines library not found. Please install it using: pip install lifelines")
    exit()

# Define model classes here to load the saved state_dict
# (Copied from train_two_tower_model.py)
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

    def forward(self, session_x, session_mask, product_ids):
        session_embedding = self.session_tower(session_x, session_mask)
        product_embedding = self.product_tower(product_ids)
        session_embedding_norm = F.normalize(session_embedding, p=2, dim=1)
        product_embedding_norm = F.normalize(product_embedding, p=2, dim=2)
        product_scores = torch.bmm(session_embedding_norm.unsqueeze(1), product_embedding_norm.transpose(1, 2)).squeeze(1)
        event_logits = self.event_head(session_embedding)
        return product_scores, event_logits

print("✅ Model classes defined.")

print("\n--- Step 2: Load Data and Pre-trained Model ---")

# Paths
SESSIONS_PATH = "pre-processed/session_sequences_embedded.parquet"
PADDED_DATA_PATH = "pre-processed/session_sequences_padded.npz"
FEATURES_PATH = "pre-processed/features_per_user_session.npz" # <-- Path to pre-computed features
MODEL_PATH = "pre-processed/two_tower_model_best.pt"

# Load session data
if os.path.exists(SESSIONS_PATH):
    sessions_df = pd.read_parquet(SESSIONS_PATH)
    sessions_df['end_time'] = pd.to_datetime(sessions_df['end_time'])
    print(f"✅ Loaded {len(sessions_df)} sessions from parquet.")
else:
    print(f"❌ Error: Session data not found at {SESSIONS_PATH}")
    exit()

# Load pre-computed features (for RFM)
if os.path.exists(FEATURES_PATH):
    features_data = np.load(FEATURES_PATH, allow_pickle=True)
    # This file contains RFM for each session, let's extract it per user
    user_features = {}
    for f in features_data['features']:
        user_id = f['user_id']
        if user_id not in user_features:
            user_features[user_id] = f['rfm']
    
    rfm_df = pd.DataFrame.from_dict(user_features, orient='index', columns=['Recency', 'Frequency', 'Monetary'])
    rfm_df.index.name = 'user_id'
    print("✅ Loaded and processed pre-computed RFM features.")
else:
    print(f"❌ Error: Pre-computed features not found at {FEATURES_PATH}")
    exit()

# Load padded sequences and masks
if os.path.exists(PADDED_DATA_PATH):
    padded_data = np.load(PADDED_DATA_PATH, allow_pickle=True)
    padded_matrices = torch.tensor(padded_data['padded_matrices'], dtype=torch.float32)
    masks = torch.tensor(padded_data['masks'], dtype=torch.bool)
    print(f"✅ Loaded padded matrices with shape: {padded_matrices.shape}")
else:
    print(f"❌ Error: Padded data not found at {PADDED_DATA_PATH}")
    exit()

# Load the pre-trained Two-Tower model to extract the Session Tower
if os.path.exists(MODEL_PATH):
    # We need to know the vocab sizes to initialize the model before loading the state_dict
    # This is a bit of a chicken-and-egg problem. We'll use estimated values.
    # These specific values don't matter for the session tower, only for the product tower and event head.
    product_vocab_size = 201 # From previous script output
    event_vocab_size = 3    # From previous script output

    session_tower = SessionTower()
    product_tower = ProductTower(product_vocab_size)
    
    model = TwoTowerModel(session_tower, product_tower, event_vocab_size)
    model.load_state_dict(torch.load(MODEL_PATH))
    session_tower = model.session_tower
    session_tower.eval() # Set to evaluation mode
    print("✅ Pre-trained session tower loaded and set to evaluation mode.")
else:
    print(f"❌ Error: Pre-trained model not found at {MODEL_PATH}")
    exit()

print("\n--- Step 3: Feature Engineering (Session Embeddings) ---")

# RFM features are pre-loaded, so we only need to generate session embeddings.
print("🔄 Generating session embeddings using the pre-trained Session Tower...")
# Use a smaller batch size for inference if memory is a concern
inference_batch_size = 256 
num_batches = int(np.ceil(len(padded_matrices) / inference_batch_size))
all_embeddings = []

with torch.no_grad():
    for i in tqdm(range(num_batches), desc="Generating Embeddings"):
        start = i * inference_batch_size
        end = (i + 1) * inference_batch_size
        batch_padded = padded_matrices[start:end]
        batch_masks = masks[start:end]
        
        embeddings = session_tower(batch_padded, batch_masks)
        all_embeddings.append(embeddings.cpu().numpy())

session_embeddings_np = np.vstack(all_embeddings)
embedding_cols = [f'embed_{i}' for i in range(session_embeddings_np.shape[1])]
session_embeddings_df = pd.DataFrame(session_embeddings_np, columns=embedding_cols, index=sessions_df.index)

print(f"✅ Session embeddings generated with shape: {session_embeddings_np.shape}")

# Combine all features into one DataFrame
features_df = pd.concat([sessions_df[['user_id', 'user_session', 'end_time']], session_embeddings_df], axis=1)
features_df = features_df.merge(rfm_df, on='user_id', how='left')

print("✅ Combined DataFrame with session embeddings and pre-computed RFM features created.")
print("Columns:", features_df.columns)

print("\n--- Step 4: Prepare Data for Survival Analysis ---")

# We need to define the 'event' (a purchase) and the 'duration' to that event.
# Let's consider a user's journey from one purchase to the next.

# To identify purchase sessions, we need to recalculate monetary value per session
def purchase_sum(row):
    return np.sum([p for p, e in zip(row['price'], row['event_type']) if e == 'purchase'])
sessions_df['monetary_value'] = sessions_df.apply(purchase_sum, axis=1)

# Filter for sessions that contain a purchase
purchase_sessions = features_df[features_df['user_session'].isin(
    sessions_df[sessions_df['monetary_value'] > 0]['user_session']
)].copy()

# Sort by user and time to track the journey
purchase_sessions = purchase_sessions.sort_values(by=['user_id', 'end_time'])

# Calculate time to next purchase
purchase_sessions['next_purchase_time'] = purchase_sessions.groupby('user_id')['end_time'].shift(-1)

# Duration is the time difference in days
purchase_sessions['duration'] = (purchase_sessions['next_purchase_time'] - purchase_sessions['end_time']).dt.total_seconds() / (24 * 3600)

# The 'event' is observed if there is a next purchase
purchase_sessions['event_observed'] = purchase_sessions['duration'].notna().astype(int)

# For users with no next purchase (censored data), fill duration with time to end of study
last_observation_date = features_df['end_time'].max()
purchase_sessions['duration'].fillna(
    (last_observation_date - purchase_sessions['end_time']).dt.total_seconds() / (24 * 3600),
    inplace=True
)

# Clean up and select final columns for the model
survival_df = purchase_sessions.drop(columns=['user_session', 'end_time', 'next_purchase_time']).dropna()

# Ensure no zero or negative durations, which can cause issues with some models
survival_df = survival_df[survival_df['duration'] > 0]

print(f"✅ Prepared survival dataset with {len(survival_df)} samples.")
print(f"Censored (no next purchase): {len(survival_df[survival_df['event_observed'] == 0]) / len(survival_df):.2%}")
print("Survival data columns:", survival_df.columns)
print(survival_df.head())

print("\n--- Step 5: Train Survival Models ---")

# Scale the features to help with model convergence
features_to_scale = survival_df.columns.drop(['user_id', 'duration', 'event_observed'])
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(survival_df[features_to_scale])
model_df = pd.DataFrame(scaled_features, columns=features_to_scale, index=survival_df.index)

# Add back the non-scaled columns
model_df[['duration', 'event_observed']] = survival_df[['duration', 'event_observed']]

print("✅ Features scaled using MinMaxScaler.")

# --- Cox Proportional Hazards Model ---
print("🔄 Training Cox Proportional Hazards model...")
cph = CoxPHFitter(penalizer=0.1) # A little regularization
cph.fit(model_df, duration_col='duration', event_col='event_observed')

print("✅ CoxPH model training complete.")
print("--- CoxPH Model Summary ---")
cph.print_summary(model="Cox Proportional Hazards Model", decimals=3)
# The 'exp(coef)' column shows the hazard ratio. >1 means increased risk of purchase, <1 means decreased risk.

# --- Weibull AFT Model ---
print("\n🔄 Training Weibull AFT model...")
wft = WeibullAFTFitter(penalizer=0.1) # Add penalizer to aid convergence
wft.fit(model_df, duration_col='duration', event_col='event_observed')

print("✅ Weibull AFT model training complete.")
print("--- Weibull AFT Model Summary ---")
wft.print_summary(model="Weibull AFT Model", decimals=3)

# --- SAVE THE TRAINED MODELS AND SCALER ---
output_dir = "pre-processed/survival_models"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(cph, os.path.join(output_dir, "cph_model.joblib"))
joblib.dump(wft, os.path.join(output_dir, "wft_model.joblib"))
joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
print(f"\n✅ Survival models and scaler saved to {output_dir}")
# -----------------------------------------

print("\n--- Step 6: Predict and Visualize Probability Distribution ---")

# Select a few interesting users to predict for
# For example, let's pick 5 users from our survival dataset
X_new = model_df.drop(columns=['duration', 'event_observed']).iloc[5:10]

print(f"📊 Predicting time-to-purchase distribution for {len(X_new)} sample users...")

# Predict the survival function using the Weibull model
# This gives the probability of *not* having purchased by time `t`
survival_prob = wft.predict_survival_function(X_new)

# The probability of purchase is 1 - survival probability
purchase_prob = 1 - survival_prob

# Plotting the results
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")
plt.plot(purchase_prob)
plt.title('Predicted Probability of Next Purchase Over Time', fontsize=16)
plt.xlabel('Days Since Last Purchase', fontsize=12)
plt.ylabel('Probability of Having Purchased', fontsize=12)
plt.legend(title='Sample User Index', labels=purchase_prob.columns)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save the plot
output_plot_path = "analytics/plots/time_to_next_purchase_prob.png"
os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
plt.savefig(output_plot_path)
print(f"\n✅ Probability distribution plot saved to: {output_plot_path}")

# You can also predict the median time to next purchase
median_time_to_purchase = wft.predict_median(X_new)
print("\nPredicted Median Time to Next Purchase (in days):")
print(median_time_to_purchase)

# --- SAVE PREDICTIONS TO CSV ---
output_predictions_path = "analytics/predictions/time_to_next_purchase_predictions.csv"
os.makedirs(os.path.dirname(output_predictions_path), exist_ok=True)
# Add user_ids to the predictions for context
predictions_df = median_time_to_purchase.to_frame(name='predicted_median_days')
predictions_df['user_id'] = survival_df.iloc[5:10]['user_id'].values
predictions_df.to_csv(output_predictions_path)
print(f"✅ Numerical predictions saved to: {output_predictions_path}")
# --------------------------------

print("\n--- Script Finished ---")
