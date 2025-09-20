import os
import numpy as np
import torch
import torch.nn as nn
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
    use_amp = False  # AMP not supported on CPU

SEED = 42
torch.manual_seed(SEED)
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
# Dataset & DataLoader (Stratified Split)
# --------------------------
class SessionDataset(Dataset):
    def __init__(self, padded, masks, labels_event, labels_product):
        self.padded = torch.tensor(padded, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.bool) # shape (N, L)
        self.labels_event = torch.tensor(labels_event, dtype=torch.long)
        self.labels_product = torch.tensor(labels_product, dtype=torch.long)
    def __len__(self):
        return self.padded.shape[0]
    def __getitem__(self, idx):
        return {
            "x": self.padded[idx], # (MAX_LEN, FEATURE_DIM)
            "mask": self.masks[idx], # (MAX_LEN,)
            "y_event": self.labels_event[idx], # scalar target (we'll define next-event as last real event)
            "y_product": self.labels_product[idx]
        }

# Stratified split using product labels
from sklearn.model_selection import train_test_split
train_idx, val_idx = train_test_split(
    np.arange(len(padded_np)),
    test_size=0.2,
    random_state=SEED,
    stratify=labels_product_np
)

# --- Oversample rare product classes in training set ---
def oversample_indices(labels, rare_threshold=5):
    # Find rare classes
    unique, counts = np.unique(labels, return_counts=True)
    rare_classes = unique[counts < rare_threshold]
    # For each rare class, duplicate its indices to match rare_threshold
    oversampled_idx = list(train_idx)
    for cls in rare_classes:
        cls_idx = train_idx[labels[train_idx] == cls]
        n_needed = rare_threshold - len(cls_idx)
        if n_needed > 0 and len(cls_idx) > 0:
            oversampled_idx += list(np.random.choice(cls_idx, n_needed, replace=True))
    return np.array(oversampled_idx)

oversampled_train_idx = oversample_indices(labels_product_np, rare_threshold=5)

train_ds = SessionDataset(
    padded_np[oversampled_train_idx],
    mask_np[oversampled_train_idx],
    labels_event_np[oversampled_train_idx],
    labels_product_np[oversampled_train_idx]
)
val_ds = SessionDataset(
    padded_np[val_idx],
    mask_np[val_idx],
    labels_event_np[val_idx],
    labels_product_np[val_idx]
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# --------------------------
# Model: small Transformer encoder + heads for event & product
# --------------------------
class SimpleSessionTransformer(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, embed_dim=EMBED_DIM, transformer_dim=TRANSFORMER_DIM,
                 num_heads=NUM_HEADS, num_layers=NUM_LAYERS, event_vocab=event_vocab, product_vocab=product_vocab):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, embed_dim)
        self.pos_emb = nn.Embedding(MAX_LEN, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=transformer_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.event_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, event_vocab)
        )
        # Use LogSoftmax for product prediction for stability
        self.product_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, product_vocab),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x, mask):
        b, l, f = x.shape
        x_proj = self.input_proj(x)
        pos_ids = torch.arange(l, device=x.device).unsqueeze(0).expand(b, l)
        x_proj = x_proj + self.pos_emb(pos_ids)
        # Clamp to valid range
        pos_ids = pos_ids.clamp(0, self.pos_emb.num_embeddings - 1)
        x_t = x_proj.permute(1, 0, 2)
        src_key_padding_mask = ~mask
        out = self.transformer(x_t, src_key_padding_mask=src_key_padding_mask)
        out = out.permute(1, 0, 2)
        mask_f = mask.unsqueeze(-1).float()
        sum_out = (out * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = sum_out / denom
        event_logits = self.event_head(pooled)
        product_logits = self.product_head(pooled)
        return event_logits, product_logits

model = SimpleSessionTransformer().to(DEVICE)
# 4) Lower learning rate further
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
criterion_event = nn.CrossEntropyLoss()
# 6) Use NLLLoss for product prediction
# --- Focal Loss for Product Prediction ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    def forward(self, input, target):
        # input: log-probabilities (N, C), target: (N,)
        logpt = input.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        focal_term = (1 - pt) ** self.gamma
        loss = -focal_term * logpt
        if self.weight is not None:
            loss = loss * self.weight[target]
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Use FocalLoss for product prediction
criterion_product = FocalLoss(gamma=2.0)
# AMP scaler (only on CUDA)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# --------------------------
# Metrics
# --------------------------
def compute_accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

# --------------------------
# Training loop (with AMP, logging, early stopping, validation, checkpointing)
# --------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_event_correct = 0
    total_product_correct = 0
    total_samples = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        y_event = batch["y_event"].to(device)
        y_product = batch["y_product"].to(device)

        optimizer.zero_grad()
        # Forward pass with AMP
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits_event, logits_product = model(x, mask)
            loss_event = criterion_event(logits_event, y_event)
            if epoch == 1:  # Only print for the first epoch to avoid spam
                print("Sample y_product:", y_product[:10].cpu().numpy())
                print("Sample logits_product:", logits_product[:10].cpu().detach().numpy())
            loss_product = criterion_product(logits_product, y_product)
            loss = loss_event + loss_product
        # Backward + Optimizer Step with AMP
    scaler.scale(loss).backward()
    # 5) Ensure gradient clipping is working
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    total_loss += loss.item() * x.size(0)
    total_event_correct += (logits_event.argmax(dim=1) == y_event).sum().item()
    total_product_correct += (logits_product.argmax(dim=1) == y_product).sum().item()
    total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    acc_event = total_event_correct / total_samples
    acc_product = total_product_correct / total_samples
    return avg_loss, acc_event, acc_product

def eval_model(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_event_correct = 0
    total_product_correct = 0
    total_samples = 0
    val_labels = []
    val_logits = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            y_event = batch["y_event"].to(device)
            y_product = batch["y_product"].to(device)

            # Diagnostics: Check for NaN/Inf in validation data
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("[VAL DIAG] Found NaN/Inf in validation features! Filtering batch...")
                valid_idx = ~(torch.isnan(x).any(dim=(1,2)) | torch.isinf(x).any(dim=(1,2)))
                x = x[valid_idx]
                mask = mask[valid_idx]
                y_event = y_event[valid_idx]
                y_product = y_product[valid_idx]
                if x.shape[0] == 0:
                    continue

            # Diagnostics: Check for NaN/Inf in validation labels
            if torch.isnan(y_product).any() or torch.isinf(y_product).any():
                print("[VAL DIAG] Found NaN/Inf in validation product labels! Filtering batch...")
                valid_idx = ~(torch.isnan(y_product) | torch.isinf(y_product))
                x = x[valid_idx]
                mask = mask[valid_idx]
                y_event = y_event[valid_idx]
                y_product = y_product[valid_idx]
                if x.shape[0] == 0:
                    continue

            # Diagnostics: Ensure validation labels are in range
            out_of_range = (y_product < 0) | (y_product >= product_vocab)
            if out_of_range.any():
                print(f"[VAL DIAG] Found out-of-range product labels in validation! Filtering batch...")
                valid_idx = ~out_of_range
                x = x[valid_idx]
                mask = mask[valid_idx]
                y_event = y_event[valid_idx]
                y_product = y_product[valid_idx]
                if x.shape[0] == 0:
                    continue

            # Optionally filter out 'other' class from validation for debugging
            filter_other_val = True  # Set True to exclude 'other' class
            if filter_other_val:
                valid_idx = y_product < (product_vocab - 1)
                x = x[valid_idx]
                mask = mask[valid_idx]
                y_event = y_event[valid_idx]
                y_product = y_product[valid_idx]
                if x.shape[0] == 0:
                    continue

            # Forward pass with AMP (safe to use in eval)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits_event, logits_product = model(x, mask)
                loss_event = criterion_event(logits_event, y_event)
                loss_product = criterion_product(logits_product, y_product)
                loss = loss_event + loss_product

            total_loss += loss.item() * x.size(0)
            total_event_correct += (logits_event.argmax(dim=1) == y_event).sum().item()
            total_product_correct += (logits_product.argmax(dim=1) == y_product).sum().item()
            total_samples += x.size(0)

            val_labels.append(y_product.cpu().numpy())
            val_logits.append(logits_product.cpu().detach().numpy())

    import numpy as np
    if len(val_labels) > 0:
        val_labels_np = np.concatenate(val_labels)
        print("[VAL DIAG] Validation product label distribution:")
        print("  min:", val_labels_np.min(), "max:", val_labels_np.max(), "mean:", val_labels_np.mean())
        print("  bincount:", np.bincount(val_labels_np, minlength=product_vocab))
    else:
        print("[VAL DIAG] No valid samples in validation set after filtering!")
    if len(val_logits) > 0:
        val_logits_np = np.concatenate(val_logits)
        print("[VAL DIAG] Validation product logits (first 10 samples):")
        print(val_logits_np[:10])
    else:
        print("[VAL DIAG] No logits to display (no valid samples).")

    if total_samples == 0:
        return float('nan'), 0.0, 0.0

    avg_loss = total_loss / total_samples
    acc_event = total_event_correct / total_samples
    acc_product = total_product_correct / total_samples
    return avg_loss, acc_event, acc_product

# --------------------------
# Run training with early stopping and checkpointing
# --------------------------
best_val_loss = float("inf")
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_event_acc, train_product_acc = train_one_epoch(model, train_loader, optimizer, DEVICE)
    val_loss, val_event_acc, val_product_acc = eval_model(model, val_loader, DEVICE)
    scheduler.step(val_loss)
    print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f}, train_event_acc={train_event_acc:.4f}, train_prod_acc={train_product_acc:.4f} | val_loss={val_loss:.4f}, val_event_acc={val_event_acc:.4f}, val_prod_acc={val_product_acc:.4f}")

    # Logging metrics
    print(f"  [Metrics] Train Event Acc: {train_event_acc:.4f}, Train Product Acc: {train_product_acc:.4f}, Val Event Acc: {val_event_acc:.4f}, Val Product Acc: {val_product_acc:.4f}")

    # Early stopping and checkpointing
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "pre-processed/transformer_next_event_product_best.pt")
        print("  -> Saved best model.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

# Save last epoch model
torch.save(model.state_dict(), "pre-processed/transformer_next_event_product_last.pt")
print("Saved last epoch model.")
print("Training finished. Best val_loss:", best_val_loss)