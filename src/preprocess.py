import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_metadata(csv_path):
    meta = pd.read_csv(csv_path)

    # 🎯 Simulated skin tone groups
    np.random.seed(42)
    meta['skin_tone_group'] = np.random.choice(
        ['light', 'medium', 'dark'],
        size=len(meta)
    )

    # 🧹 Handle missing values
    meta['age'] = meta['age'].fillna(meta['age'].median())
    meta['sex'] = meta['sex'].fillna('unknown')

    # 🎯 Binary classification (melanoma vs rest)
    meta['is_mel'] = (meta['dx'] == 'mel').astype(int)
    meta['dx_enc'] = meta['is_mel']

    return meta


def add_image_paths(meta, image_dir):
    def get_path(img_id):
        for ext in ['.jpg', '.jpeg', '.png']:
            p = os.path.join(image_dir, img_id + ext)
            if os.path.exists(p):
                return p
        return None

    meta['image_path'] = meta['image_id'].apply(get_path)

    # ❗ Remove rows without images
    meta = meta.dropna(subset=['image_path']).reset_index(drop=True)

    return meta


def split_data(meta):
    train_df, test_df = train_test_split(
        meta,
        test_size=0.2,
        stratify=meta['dx_enc'],
        random_state=42
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def encode_features(train_df, test_df):
    for col in ['sex', 'localization', 'skin_tone_group']:
        le = LabelEncoder()

        combined = pd.concat([train_df[col], test_df[col]], axis=0)
        le.fit(combined)

        train_df[col + '_le'] = le.transform(train_df[col])
        test_df[col + '_le'] = le.transform(test_df[col])

    return train_df, test_df