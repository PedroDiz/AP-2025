# Standard library
import os
import time
import random

# Data handling
import requests
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# PyTorch & vision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import torchvision.transforms as transforms

# Scikit-learn metrics & utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

# Plotting
import matplotlib.pyplot as plt

"""#Step 3: Network and Loss function

Instantiate a pretrained EfficientNet-B0 model with a single output neuron for binary (benign vs. malignant) classification.
"""

model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=1  # output neuron
)

"""Focal Binary Cross‐Entropy (Focal BCE) loss down‐weights easy examples and focuses training on hard, misclassified samples.
"""

class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=0.5, reduction="mean", use_focal=True):
        """
        alpha: weight for the positive (malignant) class (0 < alpha < 1)
               smaller alpha → less emphasis on positives
        gamma: focusing parameter (≥0), smaller → less focus on hard examples
        reduction: 'mean' or 'sum'
        use_focal: if False, falls back to plain BCEWithLogitsLoss
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.use_focal = use_focal

        if self.use_focal:
            # keep one BCE-with-logits loss per example for focal computation
            self.bce = nn.BCEWithLogitsLoss(reduction='none')
        else:
            # if not using focal, just use standard BCEWithLogitsLoss with reduction
            self.bce = nn.BCEWithLogitsLoss(reduction=self.reduction)

    def forward(self, logits, targets):
        # logits: raw model outputs (no sigmoid), shape [batch_size]
        # targets: ground-truth labels (0.0 or 1.0), shape [batch_size]

        if not self.use_focal:
            # simple binary cross-entropy on logits
            return self.bce(logits, targets)

        # 1. Compute per-example BCE loss (no reduction)
        bce_loss = self.bce(logits, targets)  # shape [batch_size]

        # 2. Convert logits to probabilities in [0,1]
        prob = torch.sigmoid(logits)          # shape [batch_size]

        # 3. p_t: model's probability of the true class
        #    for positive: prob; for negative: 1 - prob
        p_t = prob * targets + (1 - prob) * (1 - targets)

        # 4. alpha factor: alpha for positives, (1 - alpha) for negatives
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # 5. focal factor: (1 - p_t)^gamma — focuses on hard (uncertain) examples
        focal_factor = (1 - p_t) ** self.gamma

        # 6. Combine: alpha * focal * bce
        loss = alpha_factor * focal_factor * bce_loss  # shape [batch_size]

        # 7. Reduce to scalar
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # return per-example losses if reduction is None

"""**Training configuration:**  
- **Loss:** Focal Binary Cross‐Entropy.  
- **Optimizer:** AdamW (learning rate = 3e-4, weight decay = 1e-4).  
- **LR Scheduler:** Cosine annealing over 30 epochs (`T_max=30`).  

"""

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn


criterion = FocalBCELoss(gamma=2.0,alpha=0.75)
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=30)

"""Set a fixed random seed to ensure reproducible training and evaluation results across runs.

"""

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""#Step 4: Model train and metrics"""

# --- 2) Instantiate loaders --------------------------------------

#--------APAGAR----------- (mini dataset)
#small = train_df.groupby('label').sample(10, random_state=42)
#train_dataset = SkinCancerDataset(small, PATH, transform=train_transform)
#train_loader  = DataLoader(train_dataset, batch_size=4, shuffle=True)
# train on just small_loader and see if the model can learn to predict both classes
#--------APAGAR-----------



epoch_times = []

# training loader (no filenames)
train_dataset = SkinCancerDataset(train_df, PATH, transform=train_transform, return_filename=False)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=2)

# validation loader (with filenames)
val_dataset = SkinCancerDataset(val_df, PATH, transform=val_transform, return_filename=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# --- 3) Prepare model & optimizer & scheduler ------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)

# (Assume criterion, optimizer, scheduler already defined)
# criterion = FocalBCELoss(...)
# optimizer = AdamW(...)
# scheduler = CosineAnnealingLR(...)

# --- 4) Set up metric arrays ------------------------------------
n_epochs       = 10
train_losses   = np.zeros(n_epochs)
train_bal_accs = np.zeros(n_epochs)

val_losses     = np.zeros(n_epochs)
val_aurocs     = np.zeros(n_epochs)
val_f1s        = np.zeros(n_epochs)
val_bal_accs   = np.zeros(n_epochs)

# --- 5) Training + Eval loop with tracking ----------------------

for epoch in range(n_epochs):
    start = time.time()
    # ---- TRAINING ----
    model.train()
    running_loss = 0.0

    # prepare storage
    train_labels, train_probs, train_logits = [], [], []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze()          # raw scores
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            current = torch.cuda.memory_allocated(device) / 1e9
            peak    = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"GPU memory — current: {current:.2f} GB, peak: {peak:.2f} GB")

        running_loss += loss.item() * images.size(0)

        # store for analysis
        train_logits.extend(logits.detach().cpu().numpy())
        probs = torch.sigmoid(logits)
        train_probs.extend(probs.detach().cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(
            loss=loss.item(),
            lr=optimizer.param_groups[0]['lr'],
            l_min=f"{min(logits.detach().cpu()).item():.2f}",
            l_max=f"{max(logits.detach().cpu()).item():.2f}"
        )

    # epoch statistics
    train_losses[epoch]   = running_loss / len(train_loader.dataset)
    train_bal_accs[epoch] = balanced_accuracy_score(
        train_labels,
        (np.array(train_probs) > 0.5).astype(int)
    )
    scheduler.step()

    print(f"Epoch {epoch+1} logits → min {np.min(train_logits):.3f}, max {np.max(train_logits):.3f}, mean {np.mean(train_logits):.3f}")


    # ---- VALIDATION ----
    model.eval()
    val_logits, val_labels, val_files = [], [], []

    with torch.no_grad():
        vbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]  ", leave=False)
        for images, labels, fnames in vbar:
            images = images.to(device)
            logits = model(images).squeeze()

            val_logits.extend(logits.cpu().numpy())
            val_labels.extend(labels.numpy())
            val_files.extend(fnames)

    y_true = np.array(val_labels)
    y_prob = torch.sigmoid(torch.tensor(val_logits)).numpy()
    y_pred = (y_prob > 0.5).astype(int)

    val_losses[epoch]   = criterion(
        torch.tensor(val_logits),
        torch.tensor(val_labels)
    ).item()
    val_aurocs[epoch]   = roc_auc_score(y_true, y_prob)
    val_bal_accs[epoch] = balanced_accuracy_score(y_true, y_pred)
    val_f1s[epoch]      = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )[2]

    # Print epoch summary + classification report
    print(f"Epoch {epoch+1:2d}: "
          f"Train Loss={train_losses[epoch]:.4f}, "
          f"Train BalAcc={train_bal_accs[epoch]:.4f} | "
          f"Val Loss={val_losses[epoch]:.4f}, "
          f"AUROC={val_aurocs[epoch]:.4f}, "
          f"F1={val_f1s[epoch]:.4f}, "
          f"Val BalAcc={val_bal_accs[epoch]:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Benign","Malignant"]))

    # --- 6) Save predictions to CSV for later review ---------
    df_preds = pd.DataFrame({
        "filename": val_files,
        "true":     y_true,
        "pred":     y_pred,
        "prob":     y_prob
    })
    df_preds.to_csv(f"val_preds_epoch_{epoch+1}.csv", index=False)
    df_preds[df_preds["true"] != df_preds["pred"]] \
        .to_csv(f"wrong_preds_epoch_{epoch+1}.csv", index=False)

    elapsed = time.time() - start
    epoch_times.append(elapsed)
    print(f"Epoch {epoch+1} took {elapsed:.1f}s")

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# y_true and y_prob from the final epoch
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUROC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Chance')
plt.fill_between(fpr, tpr, alpha=0.3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Validation ROC Curve (Final Epoch)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# --- 7) Final plots -------------------------------------------

# Loss curves
plt.figure(figsize=(8,3))
plt.plot(train_losses,   label="Train Loss")
plt.plot(val_losses,     label="Val Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Accuracy & AUROC
plt.figure(figsize=(8,3))
plt.plot(train_bal_accs, label="Train Balanced Acc")
plt.plot(val_bal_accs,   label="Val Balanced Acc")
plt.plot(val_aurocs,     label="Val AUROC")
plt.title("Balanced Acc & AUROC Over Epochs")
plt.xlabel("Epoch")
plt.ylim(0,1)
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(range(1, n_epochs + 1), epoch_times, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Time (seconds)')
plt.title('Time per Epoch')
plt.grid(True)
plt.show()

# 0) Compute and print overall AUROC
auroc = roc_auc_score(y_true, y_prob)
print(f"Overall AUROC = {auroc:.4f}")

# 1) Youden’s J threshold
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
J = tpr - fpr
idx_j = J.argmax()
youden_thresh = thresholds[idx_j]
y_pred_youden = (y_prob >= youden_thresh).astype(int)
youden_balacc = balanced_accuracy_score(y_true, y_pred_youden)
youden_f1     = precision_recall_fscore_support(y_true, y_pred_youden, average='macro')[2]
print(f"Youden’s J best threshold = {youden_thresh:.3f} | "
      f"BalAcc = {youden_balacc:.3f} | F1 = {youden_f1:.3f}")

# 2) F1‐optimal threshold
ths = np.linspace(0, 1, 101)
f1s = [f1_score(y_true, (y_prob >= t).astype(int)) for t in ths]
idx_f1 = int(np.argmax(f1s))
f1_thresh = ths[idx_f1]
y_pred_f1 = (y_prob >= f1_thresh).astype(int)
f1_balacc = balanced_accuracy_score(y_true, y_pred_f1)
f1_f1     = f1s[idx_f1]
print(f"F1‐opt threshold = {f1_thresh:.3f} | "
      f"BalAcc = {f1_balacc:.3f} | F1 = {f1_f1:.3f}")

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# y_true and y_prob from the final epoch
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUROC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Chance')
plt.fill_between(fpr, tpr, alpha=0.3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Validation ROC Curve (Final Epoch)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

def visualize_predictions(csv_file, image_dir, n=5):
    """
    Displays n correctly classified and n misclassified images side by side.

    Args:
        csv_file (str): Path to the CSV with columns ['filename', 'true', 'pred', 'prob'].
        image_dir (str): Directory containing the image files.
        n (int): Number of examples from each group to display.
    """
    # Load predictions
    df = pd.read_csv(csv_file)

    # Split correct and wrong
    correct = df[df['true'] == df['pred']]
    wrong   = df[df['true'] != df['pred']]

    # Sample up to n examples
    n_corr = min(n, len(correct))
    n_wrong = min(n, len(wrong))
    sample_correct = correct.sample(n_corr, random_state=42)
    sample_wrong = wrong.sample(n_wrong, random_state=42)

    # Create grid: 2 rows (correct, wrong), n columns
    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))

    # Plot correct predictions
    for i, row in enumerate(sample_correct.itertuples()):
        img = Image.open(os.path.join(image_dir, row.filename)).convert('RGB')
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"True={row.true}")
        axes[0, i].axis('off')

    # Plot wrong predictions
    for i, row in enumerate(sample_wrong.itertuples()):
        img = Image.open(os.path.join(image_dir, row.filename)).convert('RGB')
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"T={row.true}, P={row.pred}")
        axes[1, i].axis('off')

    # Label rows
    axes[0, 0].set_ylabel('Correct', size=14)
    axes[1, 0].set_ylabel('Wrong', size=14)

    plt.tight_layout()
    plt.show()

visualize_predictions('val_preds_epoch_3.csv', PATH, n=5)

# Get the first batch from train_loader
first_batch = next(iter(train_loader))

# Unpack images and labels
images, labels = first_batch

# Print shapes
print("images.shape:", images.shape)   # e.g. [32, 3, 224, 224]
print("labels.shape:", labels.shape)   # e.g. [32]

# Print the first example in that batch
print("First image tensor:", images[0])
print("First label:", labels[0].item())

plt.plot(epoch_times, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Time (s)")
plt.title("Epoch Duration")
plt.grid(True)
plt.show()

# Load validation predictions
df = pd.read_csv('val_preds_epoch_3.csv')

# Number of actual cancer cases
true_cancer = df[df['true'] == 1]
n_true_cancer = len(true_cancer)

# Among those, how many were correctly predicted as cancer (true positive rate)
tp = len(true_cancer[true_cancer['pred'] == 1])
pct_correct_cancer = tp / n_true_cancer * 100 if n_true_cancer else 0.0

# Number of actual non-cancer cases
true_noncancer = df[df['true'] == 0]
n_true_noncancer = len(true_noncancer)

# Among those, how many were correctly predicted as non-cancer (true negative rate)
tn = len(true_noncancer[true_noncancer['pred'] == 0])
pct_correct_noncancer = tn / n_true_noncancer * 100 if n_true_noncancer else 0.0

print(f'Correct cancer diagnoses: {tp}/{n_true_cancer} = {pct_correct_cancer:.2f}%')
print(f'Correct non-cancer diagnoses: {tn}/{n_true_noncancer} = {pct_correct_noncancer:.2f}%')

# Check train split
counts = np.bincount([lbl for _, lbl in train_loader.dataset])
print("Train labels 0,1:", counts)

# Check val split
counts = np.bincount([lbl for _, lbl, _ in val_loader.dataset])
print("Val   labels 0,1:", counts)

# Choose an example index
index = 6  # change this to test a different image
test_dataset = SkinCancerDataset(test_df, PATH, transform=train_transform)
# Get the image tensor and true label from your dataset
image_tensor, true_label = train_dataset[index]

# Prepare the input batch and move to device
input_tensor = image_tensor.unsqueeze(0).to(device)

# Run inference
model.eval()
with torch.inference_mode():
    logit = model(input_tensor).squeeze()
    prob = torch.sigmoid(logit).item()         # probability of class “1” (malignant)
    pred = 1 if prob > 0.5 else 0              # threshold at 0.5

# Map numeric labels to strings
label_map = {0: "Benign", 1: "Malignant"}

# Print results
print(f"True label:      {label_map[int(true_label.item())]}")
print(f"Predicted label: {label_map[pred]}  (probability = {prob:.4f})")

print("Val split counts:\n", val_df["label"].value_counts())

import importlib.util
import subprocess
import sys

# Check if torchinfo is installed
package_name = 'torchinfo'
spec = importlib.util.find_spec(package_name)

# If not installed, install it
if spec is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

import torchinfo

torchinfo.summary(model=model,
        input_size=(16, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)



"""Visualize samples image and tensor values:

def show_tensor_image(tensor, mean=0.5, std=0.5):

    #Display a single image given a tensor of shape (C, H, W).
    #Assumes the tensor was normalized with transforms.Normalize([mean]*3, [std]*3).

    img = tensor.cpu().clone()          # clone to avoid modifying original
    img = img * std + mean             # unnormalize
    img = img.permute(1, 2, 0).numpy()  # C×H×W -> H×W×C
    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

index = 8111
sample = train_dataset[index][0] #substituir por val para ver
label = train_dataset[index][1]
show_tensor_image(sample)
print(label)
"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title(f"Confusion Matrix — Epoch {epoch+1}")
plt.show()