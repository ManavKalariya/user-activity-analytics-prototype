import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Ensure CUDA error surfaces at the right op
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
EMBED_DIM = 64      # embedding dim for learned projection
TRANSFORMER_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
EARLY_STOPPING_PATIENCE = 5
K_EVAL = 20  # Recall/MRR@K for product ranking

# --------------------------
# Utilities: load or make dummy
# --------------------------
def load_data_or_dummy():
    if os.path.exists(DATA_PADDDED_PATH) and os.path.exists(DATA_FEATURES_PATH):
        print("Loading padded session arrays and features...")
        p = np.load(DATA_PADDDED_PATH, allow_pickle=True)
        features_npz = np.load(DATA_FEATURES_PATH, allow_pickle=True)

        padded = p["padded_matrices"]  # shape (N, MAX_LEN, FEATURE_DIM)
        masks = p["masks"]             # shape (N, MAX_LEN)
        labels_event = features_npz["labels_next_event"]  # shape (N,)
        labels_product = features_npz["labels_next_product"]

        # Filter rows with NaN/Inf anywhere in the features
        if np.any(np.isnan(padded)) or np.any(np.isinf(padded)):
            print("Warning: Found NaN or Inf in input features! Filtering...")
            valid_feat_idx = ~(np.isnan(padded).any(axis=(1, 2)) | np.isinf(padded).any(axis=(1, 2)))
            padded = padded[valid_feat_idx]
            masks = masks[valid_feat_idx]
            labels_event = labels_event[valid_feat_idx]
            labels_product = labels_product[valid_feat_idx]
        return padded, masks, labels_event, labels_product
    else:
        print("Padded file not found — creating dummy data for demo...")
        N = 2000
        vocab_event = 5   # number of event types
        vocab_product = 200  # number of distinct products (for product prediction)
        # Create padded random sequences
        padded = np.zeros((N, MAX_LEN, FEATURE_DIM), dtype=np.float32)
        masks = np.zeros((N, MAX_LEN), dtype=np.int32)
        labels_event = np.zeros((N,), dtype=np.int64)
        labels_product = np.zeros((N,), dtype=np.int64)
        for i in range(N):
            seq_len = np.random.randint(3, min(20, MAX_LEN))
            masks[i, :seq_len] = 1
            # event_type_enc, product_id_enc, category_code_enc, brand_enc, price_norm, time_deltas
            padded[i, :seq_len, 0] = np.random.randint(0, vocab_event - 1, size=(seq_len,))
            padded[i, :seq_len, 1] = np.random.randint(0, vocab_product - 1, size=(seq_len,))
            padded[i, :seq_len, 2] = np.random.randint(0, 20, size=(seq_len,))
            padded[i, :seq_len, 3] = np.random.randint(0, 30, size=(seq_len,))
            padded[i, :seq_len, 4] = np.random.rand(seq_len)
            padded[i, :seq_len, 5] = np.clip(np.random.exponential(scale=60.0, size=(seq_len,)), 0, 10000)
            # labels: choose the last actual event/product of the sequence as "next" (for demo)
            labels_event[i] = int(padded[i, min(seq_len - 1, MAX_LEN - 1), 0])
            labels_product[i] = int(padded[i, min(seq_len - 1, MAX_LEN - 1), 1])
        return padded, masks, labels_event, labels_product

padded_np, mask_np, labels_event_np, labels_product_np = load_data_or_dummy()
N = padded_np.shape[0]

print("Data shapes:", padded_np.shape, mask_np.shape, labels_event_np.shape, labels_product_np.shape)

# --------------------------
# Prepare vocab sizes (derive from data) + reduce extreme product space via top-K + 'other'
# --------------------------

# Filter out invalid product labels
valid_idx = labels_product_np >= 0
padded_np = padded_np[valid_idx]
mask_np = mask_np[valid_idx]
labels_event_np = labels_event_np[valid_idx]
labels_product_np = labels_product_np[valid_idx]
print("Filtered samples with valid product labels:", padded_np.shape[0])

# Count product frequencies and group rare products
unique, counts = np.unique(labels_product_np, return_counts=True)
sorted_idx = np.argsort(-counts)

top_k = 500  # keep top-K products + 1 'other'
top_products = unique[sorted_idx[:top_k]]
other_class = top_k  # index for 'other'

# Map products not in top_k to 'other'
labels_product_np_mapped = np.full_like(labels_product_np, other_class)
for i, prod in enumerate(top_products):
    labels_product_np_mapped[labels_product_np == prod] = i
labels_product_np = labels_product_np_mapped
product_vocab = top_k + 1

# Optionally train only on top products for quick debugging
filter_other = False
if filter_other:
    valid_idx2 = labels_product_np < top_k
    padded_np = padded_np[valid_idx2]
    mask_np = mask_np[valid_idx2]
    labels_event_np = labels_event_np[valid_idx2]
    labels_product_np = labels_product_np[valid_idx2]
    product_vocab = top_k  # no 'other'
    print("Filtered to only top products:", padded_np.shape[0])

event_vocab = int(padded_np[..., 0].max()) + 1
print("Detected event_vocab =", event_vocab, "product_vocab =", product_vocab)

print("Product label stats: min", labels_product_np.min(), "max", labels_product_np.max(), "mean", labels_product_np.mean())
print("Event label stats: min", labels_event_np.min(), "max", labels_event_np.max(), "mean", labels_event_np.mean())
print("Padded feature stats: min", padded_np.min(), "max", padded_np.max(), "mean", padded_np.mean())

assert labels_product_np.min() >= 0
assert labels_product_np.max() < product_vocab

# --------------------------
# Dataset & DataLoader (Stratified Split + optional oversampling)
# --------------------------
class SessionDataset(Dataset):
    def __init__(self, padded, masks, labels_event, labels_product):
        self.padded = torch.tensor(padded, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.bool)  # shape (N, L)
        self.labels_event = torch.tensor(labels_event, dtype=torch.long)
        self.labels_product = torch.tensor(labels_product, dtype=torch.long)

    def __len__(self):
        return self.padded.shape[0]

    def __getitem__(self, idx):
        return {
            "x": self.padded[idx],     # (MAX_LEN, FEATURE_DIM)
            "mask": self.masks[idx],   # (MAX_LEN,)
            "y_event": self.labels_event[idx],
            "y_product": self.labels_product[idx],
        }

# Stratified split using product labels
all_indices = np.arange(len(padded_np))
train_idx, val_idx = train_test_split(
    all_indices,
    test_size=0.2,
    random_state=SEED,
    stratify=labels_product_np
)

def oversample_indices(base_indices: np.ndarray, labels: np.ndarray, rare_threshold: int = 5) -> np.ndarray:
    """
    For classes in `base_indices` with count < rare_threshold, duplicate indices up to rare_threshold.
    """
    base_labels = labels[base_indices]
    unique_b, counts_b = np.unique(base_labels, return_counts=True)
    oversampled = list(base_indices)
    for cls, cnt in zip(unique_b, counts_b):
        if cnt < rare_threshold and cnt > 0:
            cls_idx = base_indices[base_labels == cls]
            n_needed = rare_threshold - cnt
            oversampled += list(np.random.choice(cls_idx, size=n_needed, replace=True))
    return np.array(oversampled, dtype=int)

oversampled_train_idx = oversample_indices(train_idx, labels_product_np, rare_threshold=5)

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

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# --------------------------
# Model: Transformer encoder + event head + Adaptive Softmax product head
# --------------------------
def build_adaptive_cutoffs(vocab_size: int) -> list:
    # Create reasonable cutoffs < vocab_size for AdaptiveLogSoftmaxWithLoss
    # Use up to three buckets distributed by quantiles.
    raw = [min(1000, vocab_size // 8), min(10000, vocab_size // 2), min(50000, vocab_size - 1)]
    cutoffs = sorted(set([c for c in raw if 1 <= c < vocab_size]))
    return cutoffs

class AdaptiveSoftmaxProductHead(nn.Module):
    """
    Product head using AdaptiveLogSoftmaxWithLoss for scalable training.
    Use .loss(...) for training and .topk(...) for retrieval-style evaluation.
    """
    def __init__(self, embed_dim: int, product_vocab: int, cutoffs=None, div_value: float = 4.0):
        super().__init__()
        if cutoffs is None:
            cutoffs = build_adaptive_cutoffs(product_vocab)
        self.adaptive = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=embed_dim,
            n_classes=product_vocab,
            cutoffs=cutoffs,
            div_value=div_value
        )

    def loss(self, user_repr: torch.Tensor, targets: torch.Tensor):
        out = self.adaptive(user_repr, targets)
        return out.output  # scalar loss

    @torch.no_grad()
    def topk(self, user_repr: torch.Tensor, k: int = 20):
        # Compute log-probabilities for all classes and take top-k
        log_probs = self.adaptive.log_prob(user_repr)  # (B, V)
        scores, indices = torch.topk(log_probs, k=k, dim=1)
        return indices, scores

class SimpleSessionTransformer(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, embed_dim=EMBED_DIM, transformer_dim=TRANSFORMER_DIM,
                 num_heads=NUM_HEADS, num_layers=NUM_LAYERS, event_vocab=0, product_vocab=0):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, embed_dim)
        self.pos_emb = nn.Embedding(MAX_LEN, embed_dim)
        # Use batch_first=True for clarity
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=transformer_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.event_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, event_vocab)
        )
        self.product_head = AdaptiveSoftmaxProductHead(embed_dim=embed_dim, product_vocab=product_vocab)

    def forward_repr(self, x, mask):
        # x: (B, L, F); mask: (B, L) with True for valid tokens
        b, l, _ = x.shape
        pos_ids = torch.arange(l, device=x.device).unsqueeze(0).expand(b, l)
        pos_ids = pos_ids.clamp(0, self.pos_emb.num_embeddings - 1)
        h = self.input_proj(x) + self.pos_emb(pos_ids)
        # Transformer expects src_key_padding_mask True for PAD positions
        out = self.transformer(h, src_key_padding_mask=~mask)  # (B, L, D)
        # Masked mean pooling over time
        mask_f = mask.unsqueeze(-1).float()
        pooled = (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return pooled

    def forward_event_logits(self, x, mask):
        pooled = self.forward_repr(x, mask)
        logits_event = self.event_head(pooled)
        return logits_event, pooled

model = SimpleSessionTransformer(
    feature_dim=FEATURE_DIM,
    embed_dim=EMBED_DIM,
    transformer_dim=TRANSFORMER_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    event_vocab=event_vocab,
    product_vocab=product_vocab
).to(DEVICE)

# Optimizer/scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
criterion_event = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# --------------------------
# Metrics
# --------------------------
def compute_accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def recall_at_k(topk_idx: torch.Tensor, targets: torch.Tensor) -> float:
    # topk_idx: (B, K), targets: (B,)
    return (topk_idx == targets.unsqueeze(1)).any(dim=1).float().mean().item()

def mrr_at_k(topk_idx: torch.Tensor, targets: torch.Tensor) -> float:
    # Compute reciprocal rank if target appears in top-K
    targets_exp = targets.unsqueeze(1).expand_as(topk_idx)
    matches = (topk_idx == targets_exp).float()
    # positions start at 1
    ranks = torch.arange(1, topk_idx.size(1) + 1, device=topk_idx.device).float().unsqueeze(0)
    rr = (matches / ranks).max(dim=1).values  # highest reciprocal rank per row (0 if not found)
    return rr.mean().item()

# --------------------------
# Training loop (with AMP, logging, early stopping, checkpointing)
# --------------------------
def train_one_epoch(model, loader, optimizer, device, epoch, log_product_metrics=False):
    model.train()
    total_loss = 0.0
    total_event_correct = 0
    total_samples = 0
    # optional running product recall@K on train (can be expensive)
    total_prod_recallk = 0.0
    total_prod_mrrk = 0.0

    for batch in tqdm(loader, desc=f"Training {epoch}", leave=False):
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        y_event = batch["y_event"].to(device)
        y_product = batch["y_product"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits_event, pooled = model.forward_event_logits(x, mask)
            loss_event = criterion_event(logits_event, y_event)
            loss_product = model.product_head.loss(pooled, y_product)
            loss = loss_event + loss_product

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        total_event_correct += (logits_event.argmax(dim=1) == y_event).sum().item()
        total_samples += x.size(0)

        # Optionally compute Recall/MRR@K on train
        if log_product_metrics:
            with torch.no_grad():
                topk_idx, _ = model.product_head.topk(pooled, k=K_EVAL)
                total_prod_recallk += recall_at_k(topk_idx, y_product) * x.size(0)
                total_prod_mrrk += mrr_at_k(topk_idx, y_product) * x.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    acc_event = total_event_correct / max(total_samples, 1)
    if log_product_metrics:
        return avg_loss, acc_event, (total_prod_recallk / total_samples), (total_prod_mrrk / total_samples)
    else:
        return avg_loss, acc_event, None, None

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_event_correct = 0
    total_samples = 0
    total_prod_recallk = 0.0
    total_prod_mrrk = 0.0

    for batch in tqdm(loader, desc="Validation", leave=False):
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        y_event = batch["y_event"].to(device)
        y_product = batch["y_product"].to(device)

        # Filter any degenerate batch (NaN/Inf or out-of-range labels)
        if torch.isnan(x).any() or torch.isinf(x).any():
            valid_idx = ~(torch.isnan(x).any(dim=(1,2)) | torch.isinf(x).any(dim=(1,2)))
            x, mask, y_event, y_product = x[valid_idx], mask[valid_idx], y_event[valid_idx], y_product[valid_idx]
            if x.shape[0] == 0:
                continue
        out_of_range = (y_product < 0) | (y_product >= product_vocab)
        if out_of_range.any():
            valid_idx = ~out_of_range
            x, mask, y_event, y_product = x[valid_idx], mask[valid_idx], y_event[valid_idx], y_product[valid_idx]
            if x.shape[0] == 0:
                continue

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits_event, pooled = model.forward_event_logits(x, mask)
            loss_event = criterion_event(logits_event, y_event)
            loss_product = model.product_head.loss(pooled, y_product)
            loss = loss_event + loss_product

        total_loss += loss.item() * x.size(0)
        total_event_correct += (logits_event.argmax(dim=1) == y_event).sum().item()
        total_samples += x.size(0)

        topk_idx, _ = model.product_head.topk(pooled, k=K_EVAL)
        total_prod_recallk += recall_at_k(topk_idx, y_product) * x.size(0)
        total_prod_mrrk += mrr_at_k(topk_idx, y_product) * x.size(0)

    if total_samples == 0:
        return float("nan"), 0.0, 0.0, 0.0

    avg_loss = total_loss / total_samples
    acc_event = total_event_correct / total_samples
    recallk = total_prod_recallk / total_samples
    mrrk = total_prod_mrrk / total_samples
    return avg_loss, acc_event, recallk, mrrk

# --------------------------
# Run training with early stopping and checkpointing
# --------------------------
best_val_loss = float("inf")
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    # To keep training fast, compute train Recall/MRR only every few epochs
    log_train_rank_metrics = (epoch == 1) or (epoch % 5 == 0)
    train_loss, train_event_acc, train_prod_recall, train_prod_mrr = train_one_epoch(
        model, train_loader, optimizer, DEVICE, epoch, log_product_metrics=log_train_rank_metrics
    )
    val_loss, val_event_acc, val_prod_recall, val_prod_mrr = eval_model(model, val_loader, DEVICE)
    scheduler.step(val_loss)

    # Logging metrics
    msg = [
        f"Epoch {epoch}/{EPOCHS}",
        f"train_loss={train_loss:.4f}",
        f"val_loss={val_loss:.4f}",
        f"train_event_acc={train_event_acc:.4f}",
        f"val_event_acc={val_event_acc:.4f}",
        f"val_prod_recall@{K_EVAL}={val_prod_recall:.4f}",
        f"val_prod_mrr@{K_EVAL}={val_prod_mrr:.4f}",
    ]
    if train_prod_recall is not None:
        msg.insert(3, f"train_prod_recall@{K_EVAL}={train_prod_recall:.4f}")
        msg.insert(4, f"train_prod_mrr@{K_EVAL}={train_prod_mrr:.4f}")
    print(" | ".join(msg))

    # Early stopping and checkpointing
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        os.makedirs("pre-processed", exist_ok=True)
        torch.save(model.state_dict(), "pre-processed/transformer_next_event_product_best.pt")
        print("  -> Saved best model.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

# Save last epoch model
os.makedirs("pre-processed", exist_ok=True)
torch.save(model.state_dict(), "pre-processed/transformer_next_event_product_last.pt")
print("Saved last epoch model.")
print("Training finished. Best val_loss:", best_val_loss)