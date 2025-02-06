import os
import string
import torch

__DEVICE__ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
class Config_Img_Classifier:
    # Data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "../data/archive/ASL_Alphabet_Dataset/asl_alphabet_train")
    LABELS = list(string.ascii_uppercase) + ["del", "nothing", "space"]
    IMG_SIZE = 32
    BATCH_SIZE = 64
    
    # Model
    NUM_CLASSES = len(LABELS)
    DROPOUT = 0.5
    
    # Training
    EPOCHS = 10
    LR = 0.001
    MOMENTUM = 0.9
    
    # Paths
    MODEL_DIR = os.path.join(BASE_DIR, "./saved")
    BEST_MODEL = os.path.join(MODEL_DIR, "classification.pth")
    
class Config_Transformer:
    # model parameter setting
    batch_size = 128
    max_len = 256
    d_model = 512
    n_layers = 6
    n_heads = 8
    ffn_hidden = 2048
    drop_prob = 0.1

    # optimizer parameter setting
    init_lr = 1e-5
    factor = 0.9
    adam_eps = 5e-9
    patience = 10
    warmup = 100
    epoch = 1000
    clip = 1.0
    weight_decay = 5e-4
    inf = float('inf')

__all__ = [
    '__DEVICE__',
    'Config_Img_Classifier',
    'Config_Transformer'
]