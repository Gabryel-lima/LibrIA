
import torch
from torch import nn
import torchvision
from torchvision import models
import math
import string
import os
import random
import numpy as np
import tensorflow as tf

class CFG:
    # Paths
    TRAIN_PATH = "/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
    
    # Labels (29 classes)
    LABELS = list(string.ascii_uppercase) + ["del", "nothing", "space"]

    # Imagens
    IMG_SIZE = 224  # compatível com VGG16
    IMG_CHANNELS = 3

    # Hiperparâmetros
    NUM_CLASSES = len(LABELS)
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 0.001
    MOMENTUM = 0.9
    DROPOUT = 0.5
    
    # Semente
    SEED = 42

    @staticmethod
    def seed_everything():
        random.seed(CFG.SEED)
        os.environ["PYTHONHASHSEED"] = str(CFG.SEED)
        np.random.seed(CFG.SEED)
        tf.random.set_seed(CFG.SEED)

# Positional Encoding para Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Módulo de memória temporal (Transformer Encoder)
class TransformerMemory(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_ff, dropout=dropout,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (batch, seq_len, feat_dim)
        x = self.pos_enc(x)
        return self.encoder(x.transpose(0,1)).transpose(0,1)  # volta a (batch, seq, d_model)

# Decoder GPT puro (Transformer Decoder causual)
class GPTDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_ff, dropout=dropout,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tgt_ids, memory):
        # tgt_ids: (tgt_seq, batch)
        # memory:  (src_seq, batch, d_model)
        tgt = self.token_emb(tgt_ids) * math.sqrt(self.token_emb.embedding_dim)
        tgt = self.pos_enc(tgt)
        seq_len = tgt.size(0)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=tgt.device)).bool()
        out  = self.decoder(tgt, memory, tgt_mask=mask)
        return self.lm_head(out)

# Classificador estático custom ASLNetVGG (sem última camada)
class ASLNetVGG(nn.Module):
    def __init__(self, num_classes, freeze_vgg=True):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        # Congela convolucionais se desejado
        if freeze_vgg:
            for p in vgg.features.parameters(): 
                p.requires_grad = False

        self.features   = vgg.features
        self.avgpool    = vgg.avgpool
        # Remove saída final, ficamos até o penúltimo FC
        orig_cls = list(vgg.classifier.children())[:-1]
        self.classifier = nn.Sequential(*orig_cls)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)  # → (batch, 4096)

# Extrator estático unificado (VGG + ASLNet)
class StaticFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        # 1) VGG16 (apenas features)
        self.vgg_feats = torchvision.models.vgg16(pretrained=True).features
        for p in self.vgg_feats.parameters(): 
            p.requires_grad = False

        # pooling e flatten para VGG
        self.vgg_pool = nn.AdaptiveAvgPool2d((1,1))

        # 2) ASLNetVGG (sem última camada)
        self.asl_feats = ASLNetVGG(num_classes=CFG.NUM_CLASSES, freeze_vgg=True)

        # Projeção concatenada → feature_dim
        self.proj = nn.Linear(512 + 4096, feature_dim)
        self.act  = nn.ReLU()

    def forward(self, x):
        # x: (B, C, H, W)
        f1 = self.vgg_feats(x)
        f1 = self.vgg_pool(f1)
        f1 = torch.flatten(f1, 1)            # → (B, 512)

        f2 = self.asl_feats(x)               # → (B, 4096)

        f  = torch.cat([f1, f2], dim=1)      # → (B, 4608)
        return self.act(self.proj(f))        # → (B, feature_dim)

# Modelo unificado final
class UnifiedVisionTransformerGPT(nn.Module):
    def __init__(self, vocab_size, feature_dim=512):
        super().__init__()
        self.static_extractor = StaticFeatureExtractor(feature_dim)
        self.memory  = TransformerMemory(d_model=feature_dim)
        self.gpt     = GPTDecoder(vocab_size, d_model=feature_dim)

    def forward(self, images_seq, tgt_ids):
        # images_seq: (B, S, C, H, W)
        B, S, C, H, W = images_seq.shape
        imgs = images_seq.view(B*S, C, H, W)
        feat = self.static_extractor(imgs)    # → (B*S, feature_dim)
        feat_seq = feat.view(B, S, -1)        # → (B, S, feature_dim)

        mem = self.memory(feat_seq)           # → (B, S, feature_dim)
        mem = mem.transpose(0,1)              # → (S, B, feature_dim)

        out = self.gpt(tgt_ids, mem)          # → (T_tgt, B, vocab_size)
        return out  

if __name__ == "__main__":
    # Teste rápido do modelo
    CFG.seed_everything()
    model = UnifiedVisionTransformerGPT(vocab_size=CFG.NUM_CLASSES)
    images_seq = torch.randn(CFG.BATCH_SIZE, 10, CFG.IMG_CHANNELS, CFG.IMG_SIZE, CFG.IMG_SIZE)
    tgt_ids = torch.randint(0, CFG.NUM_CLASSES, (CFG.BATCH_SIZE, 20))
    out = model(images_seq, tgt_ids)