import os
import string
import torch
import random
import numpy as np

# Tentar importar TensorFlow (requer suporte AVX)
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except (ImportError, RuntimeError) as e:
    TENSORFLOW_AVAILABLE = False
    print(f"⚠️  TensorFlow não disponível: {type(e).__name__}")
    print("   Motivo provável: CPU não suporta AVX (necessário para TensorFlow)")
    print("   Algumas funcionalidades podem estar limitadas.")
    # Criar um stub do TensorFlow para evitar erros
    class TensorFlowStub:
        class random:
            @staticmethod
            def set_seed(*args, **kwargs):
                pass
    tf = TensorFlowStub()

# Dispositivo (usa CUDA se disponível)
__DEVICE__ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config_Img_Classifier:
    # Diretórios
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "../data/archive/ASL_Alphabet_Dataset/asl_alphabet_train")

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

    # Caminhos
    BEST_MODEL = os.path.join(BASE_DIR, "./saved/classification_vgg16.pth")
    PLOTS_DIR = os.path.join(BASE_DIR, "./training_plots")

    # Semente
    SEED = 42

    @staticmethod
    def seed_everything():
        random.seed(Config_Img_Classifier.SEED)
        os.environ["PYTHONHASHSEED"] = str(Config_Img_Classifier.SEED)
        np.random.seed(Config_Img_Classifier.SEED)
        
        # Apenas usar TensorFlow se disponível
        if TENSORFLOW_AVAILABLE:
            try:
                tf.random.set_seed(Config_Img_Classifier.SEED)
            except Exception as e:
                print(f"⚠️  Não foi possível configurar seed do TensorFlow: {e}")
        
        torch.manual_seed(Config_Img_Classifier.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config_Img_Classifier.SEED)

__all__ = [
    '__DEVICE__',
    'Config_Img_Classifier',
]