import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

class HamDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

        self.transform = T.Compose([
    T.Resize((128,128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225])
])

        self.meta_cols = ['age','sex_le','localization_le','skin_tone_group_le']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row['image_path']).convert('RGB')
        img = self.transform(img)

        meta = torch.tensor(row[self.meta_cols].values, dtype=torch.float32)
        label = int(row['dx_enc'])

        return img, meta, label, row['skin_tone_group']


def get_loaders(train_df, test_df, batch_size):
    train_ds = HamDataset(train_df)
    test_ds = HamDataset(test_df)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size)
    )