import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# --------------------------
# Device Setup (robust CUDA/MPS/CPU selection + AMP)
# --------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    use_amp = True
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # For Apple Silicon
    print("[INFO] Using Apple MPS")
    use_amp = False  # AMP not supported on MPS
else:
    DEVICE = torch.device("cpu")
    print("[INFO] Using CPU")
    use_amp = False

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
np.random.seed(SEED)

# --------------------------
# Config
# --------------------------
DATA_PADDDED_PATH = "pre-processed/session_sequences_padded.npz"
DATA_FEATURES_PATH = "pre-processed/features_per_user_session.npz"
MAX_LEN = 100
FEATURE_DIM = 6  # [event_type_enc, product_id_enc, category_code_enc, brand_enc, price_norm, time_delta]
BATCH_SIZE = 64
EPOCHS = 30
EMBED_DIM = 64 # embedding dim for event/product learned projection
TRANSFORMER_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
EARLY_STOPPING_PATIENCE = 5

# --------------------------
# Utilities: load or make dummy
# --------------------------
def load_data_or_dummy():
    if os.path.exists(DATA_PADDDED_PATH) and os.path.exists(DATA_FEATURES_PATH):
        print("Loading padded session arrays and features...")
        p = np.load(DATA_PADDDED_PATH, allow_pickle=True)
        features_npz = np.load(DATA_FEATURES_PATH, allow_pickle=True)

        padded = p["padded_matrices"] # shape (N, MAX_LEN, FEATURE_DIM)
        masks = p["masks"] # shape (N, MAX_LEN)
        labels_event = features_npz["labels_next_event"]  # shape (N,)
        labels_product = features_npz["labels_next_product"]
        # convert -1 labels (no next) to a special class (we will ignore them in loss)
        # Check for inf/nan in input features before training
        if np.any(np.isnan(padded)) or np.any(np.isinf(padded)):
            print("Warning: Found NaN or Inf in input features! Filtering...")
            valid_feat_idx = ~(np.isnan(padded).any(axis=(1,2)) | np.isinf(padded).any(axis=(1,2)))
            padded = padded[valid_feat_idx]
            masks = masks[valid_feat_idx]
            labels_event = labels_event[valid_feat_idx]
            labels_product = labels_product[valid_feat_idx]
        return padded, masks, labels_event, labels_product
    else:
        print("Padded file not found — creating dummy data for demo...")
        N = 2000
        vocab_event = 5 # number of event types
        vocab_product = 200 # number of distinct products (for product prediction)
        # Create padded random sequences
        padded = np.zeros((N, MAX_LEN, FEATURE_DIM), dtype=np.float32)
        masks = np.zeros((N, MAX_LEN), dtype=np.int32)
        labels_event = np.zeros((N,), dtype=np.int64)
        labels_product = np.zeros((N,), dtype=np.int64)
        for i in range(N):
            seq_len = np.random.randint(3, min(20, MAX_LEN))
            masks[i, :seq_len] = 1
            # event_type_enc, product_id_enc, category_code_enc, brand_enc, price_norm, time_deltas
            padded[i, :seq_len, 0] = np.random.randint(0, vocab_event-1, size=(seq_len,))
            padded[i, :seq_len, 1] = np.random.randint(0, vocab_product-1, size=(seq_len,))
            padded[i, :seq_len, 2] = np.random.randint(0, 20, size=(seq_len,))
            padded[i, :seq_len, 3] = np.random.randint(0, 30, size=(seq_len,))
            padded[i, :seq_len, 4] = np.random.rand(seq_len)
            padded[i, :seq_len, 5] = np.clip(np.random.exponential(scale=60.0, size=(seq_len,)), 0, 10000)
            # labels: choose the last actual event/product of the sequence as "next" (for demo)
            labels_event[i] = int(padded[i, min(seq_len-1, MAX_LEN-1), 0])
            labels_product[i] = int(padded[i, min(seq_len-1, MAX_LEN-1), 1])
        return padded, masks, labels_event, labels_product

padded_np, mask_np, labels_event_np, labels_product_np = load_data_or_dummy()
N = padded_np.shape[0]

print("Data shapes:", padded_np.shape, mask_np.shape, labels_event_np.shape, labels_product_np.shape)
# --------------------------

# Prepare vocab sizes (derive from data)

# Filter out invalid product labels
valid_idx = labels_product_np >= 0
padded_np = padded_np[valid_idx]
mask_np = mask_np[valid_idx]
labels_event_np = labels_event_np[valid_idx]
labels_product_np = labels_product_np[valid_idx]
print("Filtered samples with valid product labels:", padded_np.shape[0])


# Count product frequencies and group rare products
(unique, counts) = np.unique(labels_product_np, return_counts=True)
# 1) Use lower threshold for rare products (more products included)
rare_threshold = 5  # Lower threshold for rare
rare_products = unique[counts < rare_threshold]

# 2) Use top 2000 products + 'other' for test run
sorted_idx = np.argsort(-counts)
top_k = 200  # Reduce product vocab size to top 200
top_products = unique[sorted_idx[:top_k]]
other_class = top_k  # index for 'other'

# Map products not in top_k to 'other'
labels_product_np_mapped = np.full_like(labels_product_np, other_class)
for i, prod in enumerate(top_products):
    labels_product_np_mapped[labels_product_np == prod] = i
labels_product_np = labels_product_np_mapped
product_vocab = top_k + 1

# 1b) Optionally, filter out 'other' samples for debugging
filter_other = False  # Set True to only train on top_k products
if filter_other:
    valid_idx2 = labels_product_np < top_k
    padded_np = padded_np[valid_idx2]
    mask_np = mask_np[valid_idx2]
    labels_event_np = labels_event_np[valid_idx2]
    labels_product_np = labels_product_np[valid_idx2]
    print("Filtered to only top products:", padded_np.shape[0])

event_vocab = int(padded_np[..., 0].max()) + 1
print("Detected event_vocab =", event_vocab, "product_vocab =", product_vocab)

print("Product label stats: min", labels_product_np.min(), "max", labels_product_np.max(), "mean", labels_product_np.mean())
print("Event label stats: min", labels_event_np.min(), "max", labels_event_np.max(), "mean", labels_event_np.mean())
print("Padded feature stats: min", padded_np.min(), "max", padded_np.max(), "mean", padded_np.mean())

print("Min product label:", labels_product_np.min())
print("Max product label:", labels_product_np.max())
print("Product vocab size:", product_vocab)
assert labels_product_np.min() >= 0
assert labels_product_np.max() < product_vocab



# ...continue with dataset creation...

# --------------------------
# Dataset for Retrieval (with Negative Sampling)
# --------------------------
class RetrievalSessionDataset(Dataset):
    def __init__(self, padded, masks, labels_product, labels_event, product_vocab_size, num_neg_samples=5):
        self.padded = torch.tensor(padded, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.bool)
        self.labels_product = torch.tensor(labels_product, dtype=torch.long)
        self.labels_event = torch.tensor(labels_event, dtype=torch.long)
        self.product_vocab_size = product_vocab_size
        self.num_neg_samples = num_neg_samples

    def __len__(self):
        return self.padded.shape[0]

    def __getitem__(self, idx):
        positive_product = self.labels_product[idx]
        event_label = self.labels_event[idx] # <-- ADD THIS
        
        # Sample negative products
        negative_products = torch.randint(0, self.product_vocab_size, (self.num_neg_samples,))
        # Ensure we don't accidentally sample the positive product
        for i in range(self.num_neg_samples):
            while negative_products[i] == positive_product:
                negative_products[i] = torch.randint(0, self.product_vocab_size, (1,))

        return {
            "x": self.padded[idx],
            "mask": self.masks[idx],
            "positive_product": positive_product,
            "negative_products": negative_products,
            "event_label": event_label # <-- ADD THIS
        }

# Stratified split using product labels
from sklearn.model_selection import train_test_split
train_idx, val_idx = train_test_split(
    np.arange(len(padded_np)),
    test_size=0.2,
    random_state=SEED,
    stratify=labels_product_np
)

train_ds = RetrievalSessionDataset(
    padded_np[train_idx],
    mask_np[train_idx],
    labels_product_np[train_idx],
    labels_event_np[train_idx], # <-- PASS EVENT LABELS
    product_vocab
)
val_ds = RetrievalSessionDataset(
    padded_np[val_idx],
    mask_np[val_idx],
    labels_product_np[val_idx],
    labels_event_np[val_idx], # <-- PASS EVENT LABELS
    product_vocab
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# --------------------------
# Model: Two-Tower Architecture
# --------------------------
class SessionTower(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, embed_dim=EMBED_DIM, transformer_dim=TRANSFORMER_DIM,
                 num_heads=NUM_HEADS, num_layers=NUM_LAYERS):
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim) # Normalize input features
        self.input_proj = nn.Linear(feature_dim, embed_dim)
        self.pos_emb = nn.Embedding(MAX_LEN, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=transformer_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, embed_dim) # Project to final embedding space

    def forward(self, x, mask):
        b, l, f = x.shape
        x = self.input_norm(x) # Apply normalization
        x_proj = self.input_proj(x)
        pos_ids = torch.arange(l, device=x.device).unsqueeze(0).expand(b, l)
        pos_ids = pos_ids.clamp(0, self.pos_emb.num_embeddings - 1)
        x_proj = x_proj + self.pos_emb(pos_ids)
        
        x_t = x_proj.permute(1, 0, 2)
        src_key_padding_mask = ~mask
        out = self.transformer(x_t, src_key_padding_mask=src_key_padding_mask)
        out = out.permute(1, 0, 2)
        
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
        # --- Session Tower Path ---
        session_embedding = self.session_tower(session_x, session_mask) # (batch_size, embed_dim)

        # --- Product Tower Path ---
        product_embedding = self.product_tower(product_ids) # (batch_size, num_samples, embed_dim)

        # --- Product Retrieval Path (using normalized embeddings) ---
        session_embedding_norm = F.normalize(session_embedding, p=2, dim=1)
        product_embedding_norm = F.normalize(product_embedding, p=2, dim=2)
        # Calculate dot product for similarity
        product_scores = torch.bmm(session_embedding_norm.unsqueeze(1), product_embedding_norm.transpose(1, 2)).squeeze(1)

        # --- Event Classification Path (using raw session embedding) ---
        event_logits = self.event_head(session_embedding)

        return product_scores, event_logits

model = TwoTowerModel(SessionTower(), ProductTower(product_vocab), event_vocab_size=event_vocab).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
criterion_product = nn.BCEWithLogitsLoss()
criterion_event = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# --------------------------
# Metrics for Retrieval
# --------------------------
def compute_retrieval_accuracy(positive_scores, negative_scores):
    """
    Computes the accuracy of the retrieval model.
    Accuracy is defined as the proportion of times the positive item's score
    is higher than all negative items' scores.
    """
    # positive_scores: (batch_size, 1)
    # negative_scores: (batch_size, num_neg_samples)
    correct = (positive_scores > negative_scores).all(dim=1).sum().item()
    return correct / positive_scores.size(0)

# --------------------------
# Training loop for Two-Tower Model
# --------------------------
def train_one_epoch(model, loader, optimizer, criterion_product, criterion_event, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    total_prod_acc = 0.0
    total_event_acc = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        positive_product = batch["positive_product"].to(device)
        negative_products = batch["negative_products"].to(device)
        event_labels = batch["event_label"].to(device)

        all_products = torch.cat([positive_product.unsqueeze(1), negative_products], dim=1)

        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            product_scores, event_logits = model(x, mask, all_products)
            
            # --- Product Loss ---
            # Targets: 1 for positive, 0 for negatives
            targets_product = torch.zeros_like(product_scores)
            targets_product[:, 0] = 1
            loss_product = criterion_product(product_scores, targets_product)

            # --- Event Loss ---
            loss_event = criterion_event(event_logits, event_labels)

            # --- Combined Loss ---
            loss = loss_product + loss_event # Simple sum, can be weighted

            if torch.isnan(loss):
                print("\n[DEBUG] NaN loss detected!")
                print(f"[DEBUG] loss_product: {loss_product.item()}, loss_event: {loss_event.item()}")
                raise ValueError("NaN loss detected. Stopping training.")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        
        # --- Calculate Accuracies ---
        # Product retrieval accuracy
        positive_scores = product_scores[:, 0].unsqueeze(1)
        negative_scores = product_scores[:, 1:]
        total_prod_acc += compute_retrieval_accuracy(positive_scores, negative_scores) * x.size(0)
        
        # Event classification accuracy
        event_preds = event_logits.argmax(dim=1)
        total_event_acc += (event_preds == event_labels).sum().item()

        total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    avg_prod_acc = total_prod_acc / total_samples
    avg_event_acc = total_event_acc / total_samples
    return avg_loss, avg_prod_acc, avg_event_acc

def eval_model(model, loader, criterion_product, criterion_event, device, use_amp):
    model.eval()
    total_loss = 0.0
    total_prod_acc = 0.0
    total_event_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            positive_product = batch["positive_product"].to(device)
            negative_products = batch["negative_products"].to(device)
            event_labels = batch["event_label"].to(device)

            all_products = torch.cat([positive_product.unsqueeze(1), negative_products], dim=1)

            with torch.cuda.amp.autocast(enabled=use_amp):
                product_scores, event_logits = model(x, mask, all_products)

                # --- Product Loss ---
                targets_product = torch.zeros_like(product_scores)
                targets_product[:, 0] = 1
                loss_product = criterion_product(product_scores, targets_product)

                # --- Event Loss ---
                loss_event = criterion_event(event_logits, event_labels)
                
                loss = loss_product + loss_event

            total_loss += loss.item() * x.size(0)
            
            # --- Accuracies ---
            positive_scores = product_scores[:, 0].unsqueeze(1)
            negative_scores = product_scores[:, 1:]
            total_prod_acc += compute_retrieval_accuracy(positive_scores, negative_scores) * x.size(0)
            
            event_preds = event_logits.argmax(dim=1)
            total_event_acc += (event_preds == event_labels).sum().item()
            
            total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    avg_prod_acc = total_prod_acc / total_samples
    avg_event_acc = total_event_acc / total_samples
    return avg_loss, avg_prod_acc, avg_event_acc

# --------------------------
# Run training with early stopping and checkpointing
# --------------------------
best_val_metric = 0
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_prod_acc, train_event_acc = train_one_epoch(
        model, train_loader, optimizer, criterion_product, criterion_event, scaler, DEVICE, use_amp
    )
    val_loss, val_prod_acc, val_event_acc = eval_model(
        model, val_loader, criterion_product, criterion_event, DEVICE, use_amp
    )
    
    # Use a combined metric for scheduler and early stopping
    # For example, the average of the two accuracies
    val_metric = (val_prod_acc + val_event_acc) / 2.0
    scheduler.step(val_metric)
    
    print(f"Epoch {epoch}/{EPOCHS} | Loss: {train_loss:.4f}/{val_loss:.4f} | "
          f"Prod Acc: {train_prod_acc:.4f}/{val_prod_acc:.4f} | "
          f"Event Acc: {train_event_acc:.4f}/{val_event_acc:.4f}")

    if val_metric > best_val_metric:
        best_val_metric = val_metric
        epochs_no_improve = 0
        torch.save(model.state_dict(), "pre-processed/two_tower_model_best.pt")
        print(f"  -> Saved best model (val_metric={best_val_metric:.4f}).")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

# Save last epoch model
torch.save(model.state_dict(), "pre-processed/two_tower_model_last.pt")
print("Saved last epoch model.")
print(f"Training finished. Best val_metric: {best_val_metric:.4f}")
