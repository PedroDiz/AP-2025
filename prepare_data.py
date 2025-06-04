import argparse
import os
import time
import random
import zipfile
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit, GroupKFold


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def fetch_metadata():
    print("Fetching metadata from ISIC API...")
    base_url = "https://api.isic-archive.com/api/v2/images/search/"
    params = {"collections": "66,67,73"}
    all_ids, results = [], []

    while True:
        response = requests.get(base_url, params=params)
        data = response.json()

        for result in data.get("results", []):
            all_ids.append(result["isic_id"])
            results.append(result)

        next_cursor = data.get("next")
        if not next_cursor:
            break
        base_url, params = next_cursor, {}

    print(f"Collected {len(all_ids)} metadata entries.")
    return results

def extract_all_images(zip_dir, extract_dir):
    print(f"Extracting images from ZIP files in: {zip_dir}")
    os.makedirs(extract_dir, exist_ok=True)
    all_filenames = []

    for file in os.listdir(zip_dir):
        if file.lower().endswith(".zip"):
            zip_path = os.path.join(zip_dir, file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for name in zip_ref.namelist():
                    if name.lower().endswith(".jpg"):
                        target_path = os.path.join(extract_dir, os.path.basename(name))
                        if not os.path.exists(target_path):
                            with zip_ref.open(name) as source, open(target_path, "wb") as target:
                                target.write(source.read())
                        all_filenames.append(os.path.basename(name))
    print(f"Extracted {len(all_filenames)} images to '{extract_dir}'.")
    return all_filenames

def build_dataframe(results, image_filenames):
    print("Building dataframe from metadata and filtering by extracted images...")
    valid_images = set(image_filenames)
    rows = []

    for result in results:
        try:
            isic_id = result["isic_id"]
            filename = f"{isic_id}.jpg"
            if filename not in valid_images:
                continue

            clinical = result["metadata"]["clinical"]
            patient = clinical.get("lesion_id", "unknown")
            benign_malignant = clinical.get("benign_malignant")
            diagnosis_1 = clinical.get("diagnosis_1", "")
            label = None

            if benign_malignant:
                label = 1 if benign_malignant.lower() == "malignant" else 0
            elif diagnosis_1 in ["Benign", "Malignant"]:
                label = 1 if diagnosis_1 == "Malignant" else 0
            else:
                continue

            rows.append({"file": filename, "patient": patient, "label": label})
        except:
            continue

    return pd.DataFrame(rows)
"""
def split_and_save(df, output_dir, test_size=0.2, n_splits=5, seed=42):
    print(f"Step 1: Splitting off {int(test_size * 100)}% of patients as test set using GroupShuffleSplit...")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1 — hold out test set
    groups = df["patient"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss.split(df, df["label"], groups=groups))

    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    test_df.to_csv(os.path.join(output_dir, "lesions_test.csv"), index=False)
    print(f"Saved test set: {test_df['patient'].nunique()} patients, {len(test_df)} samples")

    # Step 2 — apply GroupKFold on trainval set
    print(f"\nStep 2: Applying GroupKFold(n_splits={n_splits}) on remaining 80%...")
    gkf = GroupKFold(n_splits=n_splits)
    groups = trainval_df["patient"]

    for fold, (train_idx, val_idx) in enumerate(gkf.split(trainval_df, trainval_df["label"], groups=groups)):
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
        val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

        train_df.to_csv(os.path.join(fold_dir, "lesions_train.csv"), index=False)
        val_df.to_csv(os.path.join(fold_dir, "lesions_val.csv"), index=False)

        print(f"\nFold {fold}:")
        print(f"  Train patients: {train_df['patient'].nunique()}, samples: {len(train_df)}")
        print(f"  Val patients:   {val_df['patient'].nunique()}, samples: {len(val_df)}")
"""


def split_and_save(df, output_dir, train_size=0.7, val_size=0.15, test_size=0.15, seed=42):
    # sanity check
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "train/val/test must sum to 1"

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "lesions_full.csv"), index=False)

    groups = df["patient"]

    # 1) split off train (70%) vs temp (30%)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, test_size=(val_size + test_size), random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, df["label"], groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df  = df.iloc[temp_idx].reset_index(drop=True)

    # 2) split temp into val (15%) vs test (15%)
    rel_val = val_size / (val_size + test_size)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=rel_val, test_size=(1 - rel_val), random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, temp_df["label"], groups=temp_df["patient"]))

    val_df  = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    # 3) save to CSV
    train_df.to_csv(os.path.join(output_dir, "lesions_train.csv"), index=False)
    val_df.to_csv(  os.path.join(output_dir, "lesions_val.csv"),   index=False)
    test_df.to_csv( os.path.join(output_dir, "lesions_test.csv"),  index=False)

    # 4) print stats
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"{name}: {split_df['patient'].nunique()} patients, {len(split_df)} samples")
        d = split_df["label"].value_counts(normalize=True)*100
        print(f"  Benign {d.get(0,0):.1f}% | Malignant {d.get(1,0):.1f}%\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--zip_dir", type=str, required=True, help="Directory containing ZIPs with images")
    parser.add_argument("--output_dir", type=str, default="data_cache", help="Directory to save processed CSVs")
    parser.add_argument("--image_dir", type=str, default="ISIC_IMAGES", help="Directory to extract images to")
    args = parser.parse_args()

    set_seed(args.seed)
    image_filenames = extract_all_images(args.zip_dir, args.image_dir)
    metadata = fetch_metadata()
    df = build_dataframe(metadata, image_filenames)
    split_and_save(df, args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
