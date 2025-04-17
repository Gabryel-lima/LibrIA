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
    MODEL_DIR = os.path.join(BASE_DIR, "./")
    BEST_MODEL = os.path.join(MODEL_DIR, "asl_vgg_best_weights.keras")

__all__ = [
    '__DEVICE__',
    'Config_Img_Classifier',
]