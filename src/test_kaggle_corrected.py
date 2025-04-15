"""
ASL Classifier - Script convertido de notebook Kaggle
Rodando 100% local e compatível com execução direta

Autor: Gabryel Lima
Data: 2025-04-14
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # força uso de CPU

import warnings
warnings.filterwarnings("ignore")

import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import string
import random
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Configuração
class CFG:
    batch_size = 64
    img_height = 64
    img_width = 64
    epochs = 10
    num_classes = 29
    img_channels = 3

TRAIN_PATH = "./ASL_Alphabet_Dataset/asl_alphabet_train"
TEST_PATH = "./ASL_Alphabet_Dataset/asl_alphabet_test"
LABELS = list(string.ascii_uppercase) + ["del", "nothing", "space"]

def seed_everything(seed=2023):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
