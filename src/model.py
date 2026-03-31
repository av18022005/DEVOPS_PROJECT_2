import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MultimodalFairNet(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.resnet18(pretrained=True)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()

        self.img_enc = base

        self.meta_enc = nn.Sequential(
            nn.Linear(4,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + 128, 256),
            nn.ReLU(),
            nn.Linear(256,2)
        )

    def forward(self, img, meta):
        img_f = self.img_enc(img)
        meta_f = self.meta_enc(meta)

        x = F.relu(torch.cat([img_f, meta_f], dim=1))
        logits = self.classifier(x)

        return logits, img_f, meta_f, None