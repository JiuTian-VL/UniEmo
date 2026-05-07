from torchvision import models
import torch.nn as nn
import torch
from collections import OrderedDict
from timm.models.layers import trunc_normal_
from clip.model import QuickGELU


class BackBone(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.cnn = models.resnet50(pretrained=True)

        self.backbone = nn.Sequential(*list(self.cnn.children())[:-2])
        self.flaten = nn.Sequential(nn.AvgPool2d(kernel_size=7), nn.Flatten())
        self.fc_1 = nn.Linear(2048, 768)
        self.fc_2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(768, 8)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.flaten(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc = nn.Linear(768, 768)

    def forward(self, x):
        out = self.fc(x)
        return out


class MLP(nn.Module):
    def __init__(self, num_fc_layers=2, need_ReLU=False, need_LN=False, need_Dropout=False):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(1024, 1024))
        if need_LN is True:
            layers.append(nn.LayerNorm(1024))
        if need_ReLU is True:
            layers.append(nn.ReLU())
        if need_Dropout is True:
            layers.append(nn.Dropout(0.5))
        for _ in range(num_fc_layers - 2):
            layers.append(nn.Linear(1024, 1024))
            if need_LN is True:
                layers.append(nn.LayerNorm(1024))
            if need_ReLU is True:
                layers.append(nn.ReLU())
            if need_Dropout is True:
                layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(1024, 768))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out = self.mlp(x)
        return out


class SimpleMLP(nn.Module):
    def __init__(self, need_ReLU=False, need_Dropout=False):
        super(SimpleMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(768, 1024))
        if need_ReLU is True:
            layers.append(nn.ReLU())
        if need_Dropout is True:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(1024, 768))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out = self.mlp(x)
        return out


class emo_classifier(nn.Module):
    def __init__(self, ):
        super(emo_classifier, self).__init__()
        self.fc = nn.Linear(768, 8)

    def forward(self, x):
        x = self.fc(x)
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MultitokenIntegrationTransformer(nn.Module):
    def __init__(self, T, embed_dim=512, layers=1,):
        super().__init__()
        self.T = T
        transformer_heads = embed_dim // 64
        self.positional_embedding = nn.Parameter(torch.empty(1, T, embed_dim))
        trunc_normal_(self.positional_embedding, std=0.02)
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(d_model=embed_dim, n_head=transformer_heads) for _ in range(layers)])

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear,)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        ori_x = x#####16x8x512
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)  
        x = x.type(ori_x.dtype) + ori_x
        
        return x.mean(dim=1, keepdim=False)






