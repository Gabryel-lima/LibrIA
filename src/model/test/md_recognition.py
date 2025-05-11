import tensorflow as tf
import pandas as pd
import numpy as np
import json
import random
import os
from glob import glob

# utils
def flatten_l_o_l(nested_list):
    """Flatten a list of lists into a single list.

    Args:
        nested_list (list): 
            – A list of lists (or iterables) to be flattened.

    Returns:
        list: A flattened list containing all items from the input list of lists.
    """
    return [item for sublist in nested_list for item in sublist]

def print_ln(symbol="-", line_len=110, newline_before=False, newline_after=False):
    """Print a horizontal line of a specified length and symbol.

    Args:
        symbol (str, optional): 
            – The symbol to use for the horizontal line
        line_len (int, optional): 
            – The length of the horizontal line in characters
        newline_before (bool, optional): 
            – Whether to print a newline character before the line
        newline_after (bool, optional): 
            – Whether to print a newline character after the line
    """
    if newline_before: print();
    print(symbol * line_len)
    if newline_after: print();
        
def read_json_file(file_path):
    """Read a JSON file and parse it into a Python object.

    Args:
        file_path (str): The path to the JSON file to read.

    Returns:
        dict: A dictionary object representing the JSON data.
        
    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If the specified file path does not contain valid JSON data.
    """
    try:
        # Open the file and load the JSON data into a Python object
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        return json_data
    except FileNotFoundError:
        # Raise an error if the file path does not exist
        raise FileNotFoundError(f"File not found: {file_path}")
    except ValueError:
        # Raise an error if the file does not contain valid JSON data
        raise ValueError(f"Invalid JSON data in file: {file_path}")
        
def get_sign_df(pq_path, invert_y=True):
    sign_df = pd.read_parquet(pq_path)
    
    # y value is inverted (Thanks @danielpeshkov)
    if invert_y: sign_df["y"] *= -1 
        
    return sign_df

ROWS_PER_FRAME = 543  # number of landmarks per frame
def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

# Define the path to the root data directory
DATA_DIR         = "/kaggle/input/asl-signs"
EXTEND_TRAIN_DIR = "/kaggle/input/gislr-extended-train-dataframe" 

print("\n... BASIC DATA SETUP STARTING ...\n")
print("\n\n... LOAD TRAIN DATAFRAME FROM CSV FILE ...\n")

LOAD_EXTENDED = True
if LOAD_EXTENDED and os.path.isfile(os.path.join(EXTEND_TRAIN_DIR, "extended_train.csv")):
    train_df = pd.read_csv(os.path.join(EXTEND_TRAIN_DIR, "extended_train.csv"))
else:
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    train_df["path"] = DATA_DIR+"/"+train_df["path"]
print(train_df)

print("\n\n... LOAD SIGN TO PREDICTION INDEX MAP FROM JSON FILE ...\n")
s2p_map = {k.lower():v for k,v in read_json_file(os.path.join(DATA_DIR, "sign_to_prediction_index_map.json")).items()}
p2s_map = {v:k for k,v in read_json_file(os.path.join(DATA_DIR, "sign_to_prediction_index_map.json")).items()}
encoder = lambda x: s2p_map.get(x.lower())
decoder = lambda x: p2s_map.get(x)
print(s2p_map)

DEMO_ROW = 283
print(f"\n\n... DEMO SIGN/EVENT DATAFRAME FOR ROW {DEMO_ROW} - SIGN={train_df.iloc[DEMO_ROW]['sign']} ...\n")
demo_sign_df = get_sign_df(train_df.iloc[DEMO_ROW]["path"])
print(demo_sign_df)


####

train_x    = np.load("./models/data/feature_data.npy").astype(np.float32)
train_y    = np.load("./models/data/feature_labels.npy").astype(np.uint8)
BATCH_SIZE = 64

N_TOTAL = train_x.shape[0]
VAL_PCT = 0.1
N_VAL   = int(N_TOTAL * VAL_PCT)
N_TRAIN = N_TOTAL-N_VAL

random_idxs = random.sample(range(N_TOTAL), N_TOTAL)
train_idxs, val_idxs = np.array(random_idxs[:N_TRAIN]), np.array(random_idxs[N_TRAIN:])

val_x, val_y = train_x[val_idxs], train_y[val_idxs]
train_x, train_y = train_x[train_idxs], train_y[train_idxs]

# Estava comentado
train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))\
                          .shuffle(N_TRAIN)\
                          .batch(BATCH_SIZE, drop_remainder=True)\
                          .prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))\
                          .shuffle(N_VAL)\
                          .batch(BATCH_SIZE, drop_remainder=True)\
                          .prefetch(tf.data.AUTOTUNE)

#print(train_ds, val_ds)

def fc_block(inputs, output_channels, dropout=0.2):
    x = tf.keras.layers.Dense(output_channels)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("gelu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    return x

def get_model(n_labels=250, init_fc=512, n_blocks=2, _dropout_1=0.2, _dropout_2=0.6, flat_frame_len=3258):
    _inputs = tf.keras.layers.Input(shape=(flat_frame_len,))
    x = _inputs
    
    # Define layers
    for i in range(n_blocks):
        x = fc_block(
            x, output_channels=init_fc//(2**i), 
            dropout=_dropout_1 if (1+i)!=n_blocks else _dropout_2
        )
    
    # Define output layer
    _outputs = tf.keras.layers.Dense(n_labels, activation="softmax")(x)
    
    # Build the model
    model = tf.keras.models.Model(inputs=_inputs, outputs=_outputs)
    return model

model = get_model()
model.compile(tf.keras.optimizers.Adam(0.000333), "sparse_categorical_crossentropy", metrics=["acc"])
model.summary()

tf.keras.utils.plot_model(model)

cb_list = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.8, verbose=1)
]
history = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=1, callbacks=cb_list, batch_size=BATCH_SIZE) # epochs 100
model.save("./models/asl_model.keras")

model.evaluate(val_x, val_y)
for x,y in zip(val_x[:10], val_y[:10]):
    print(f"PRED: {decoder(np.argmax(model.predict(tf.expand_dims(x, axis=0), verbose=0), axis=-1)[0]):<20} – GT: {decoder(y)}")

class PrepInputs(tf.keras.layers.Layer):
    def __init__(self, face_idx_range=(0, 468), lh_idx_range=(468, 489), 
                 pose_idx_range=(489, 522), rh_idx_range=(522, 543)):
        super(PrepInputs, self).__init__()
        self.idx_ranges = [face_idx_range, lh_idx_range, pose_idx_range, rh_idx_range]
        self.flat_feat_lens = [3*(_range[1]-_range[0]) for _range in self.idx_ranges]
    
    def call(self, x_in):
        
        # Split the single vector into 4
        xs = [x_in[:, _range[0]:_range[1], :] for _range in self.idx_ranges]
        
        # Reshape based on specific number of keypoints
        xs = [tf.reshape(_x, (-1, flat_feat_len)) for _x, flat_feat_len in zip(xs, self.flat_feat_lens)]
        
        # Drop empty rows - Empty rows are present in 
        #   --> pose, lh, rh
        #   --> so we don't have to for face
        xs[1:] = [
            tf.boolean_mask(_x, tf.reduce_all(tf.logical_not(tf.math.is_nan(_x)), axis=1), axis=0)
            for _x in xs[1:]
        ]
        
        # Get means and stds
        x_means = [tf.math.reduce_mean(_x, axis=0) for _x in xs]
        x_stds  = [tf.math.reduce_std(_x,  axis=0) for _x in xs]
        
        x_out = tf.concat([*x_means, *x_stds], axis=0)
        x_out = tf.where(tf.math.is_finite(x_out), x_out, tf.zeros_like(x_out))
        return tf.expand_dims(x_out, axis=0)
    
PrepInputs()(load_relevant_data_subset(train_df.path[0]))

class TFLiteModel(tf.Module):
    """
    TensorFlow Lite model that takes input tensors and applies:
        – a preprocessing model
        – the ISLR model 
    """

    def __init__(self, islr_model):
        """
        Initializes the TFLiteModel with the specified preprocessing model and ISLR model.
        """
        super(TFLiteModel, self).__init__()

        # Load the feature generation and main models
        self.prep_inputs = PrepInputs()
        self.islr_model   = islr_model
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs):
        """
        Applies the feature generation model and main model to the input tensors.

        Args:
            inputs: Input tensor with shape [batch_size, 543, 3].

        Returns:
            A dictionary with a single key 'outputs' and corresponding output tensor.
        """
        x = self.prep_inputs(tf.cast(inputs, dtype=tf.float32))
        outputs = self.islr_model(x)[0, :]

        # Return a dictionary with the output tensor
        return {'outputs': outputs}

tflite_keras_model = TFLiteModel(islr_model=model)
demo_output = tflite_keras_model(load_relevant_data_subset(train_df.path[0]))["outputs"]
decoder(np.argmax(demo_output.numpy(), axis=-1))

import tensorflow as tf
import zipfile

# ——————————————————————————————————————————————
# 1) Converter o modelo Keras para TFLite  
#    (use aqui a variável `model` que você treinou)
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Habilita o suporte a resource variables no TFLite
converter.experimental_enable_resource_variables = True                     # :contentReference[oaicite:0]{index=0}

# Permite tanto ops nativas do TFLite quanto ops do TF (incluindo READ_VARIABLE)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,   # ops comuns do TFLite
    tf.lite.OpsSet.SELECT_TF_OPS      # ops do TensorFlow que não existem no TFLite puro :contentReference[oaicite:1]{index=1}
]

tflite_model = converter.convert()

# Salvar o .tflite
tflite_path = '/kaggle/working/models/model.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

# Embalar para submissão
with zipfile.ZipFile('submission.zip', 'w') as z:
    z.write(tflite_path, arcname='model.tflite')
# ——————————————————————————————————————————————

# 2) Carregar e preparar o runtime TFLite
#import tflite_runtime.interpreter as tflite

from tensorflow import lite as tflite

interpreter = tflite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()            # ← ESSENCIAL para criar/alinhar as variáveis
prediction_fn = interpreter.get_signature_runner("serving_default")

# (Opcional) para conferir nomes de inputs/outputs:
print(interpreter.get_signature_list())

# 3) Fazer inferência
# supondo que load_relevant_data_subset retorne um np.array com shape correto
output = prediction_fn(inputs=load_relevant_data_subset(train_df.path[0]))
sign = tf.argmax(output["outputs_0"], axis=-1).numpy()

print("PRED :", decoder(sign))
print("GT   :", train_df.sign[0])

def decode_redux_example(serialized_example, n_keyframes=3, n_ax=2, override_frame_idx=None):
    """ Parses a set of features and label from the given `serialized_example`.
        
        It is used as a map function for `dataset.map`

    Args:
        serialized_example (tf.Example): A serialized example containing the
            following features:
                – 'face'
                – 'left_hand'
                – 'pose'
                – 'right_hand'
                – 'lbl'
        
    Returns:
        A decoded tf.data.Dataset object representing the tfrecord dataset
    """
    
    _frame_idx_map  = FRAME_TYPE_IDX_MAP if override_frame_idx is None else override_frame_idx
    
    feature_dict = {
        "lbl": tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0)
    }
    feature_dict.update({k:tf.io.FixedLenFeature(shape=[len(v)*n_keyframes*n_ax], dtype=tf.float32, default_value=[0.0]*len(v)*n_keyframes*n_ax) for k,v in _frame_idx_map.items()})
    
    # Define a parser
    features = tf.io.parse_single_example(serialized_example, features=feature_dict)
    
    # Decode the tf.string
    label = features["lbl"]
    feats = {k:tf.cast(tf.reshape(features[k], (n_keyframes,len(v),n_ax)), tf.float32) for k,v in _frame_idx_map.items()}
    return feats, tf.cast(label, tf.int32)

def flatten_x(x, keys_to_use):
    # sorted --> [f]ace, [l]eft_hand, [p]ose, [r]ight_hand
    return [x[_k] for _k in sorted(keys_to_use)]

# Define a function to flatten the first dimension into individual examples
def flatten_first_dim(x,y):
    return tf.data.Dataset.from_tensor_slices((x, y))

N_KEYFRAMES    = 3
N_HAND_LMS     = 21
N_AX           = 2
BATCH_SIZE     = 128
SHUFFLE_BUFFER = BATCH_SIZE*10
KEYS_TO_USE    = ["face", "left_hand", "pose", "right_hand"]
TFREC_DIR      = "/kaggle/input/gislr-tfrecords-dataset-creation/tfrecords_seqredux"

# Get all created tfrecords
all_tfrec_paths = glob(os.path.join(TFREC_DIR, "*.tfrec"))

# Different val tfrecord means different 'fold'
val_tfrec_paths = random.sample(all_tfrec_paths, 1)
train_tfrec_paths = [x for x in all_tfrec_paths if x not in val_tfrec_paths]

# Interleave normally
# .map(lambda x,y: (flatten_frames(x,y,N_KEYFRAMES, N_HAND_LMS, N_AX)), num_parallel_calls=tf.data.AUTOTUNE)\
train_ds = tf.data.TFRecordDataset(train_tfrec_paths, num_parallel_reads=tf.data.AUTOTUNE)\
                  .map(decode_redux_example, num_parallel_calls=tf.data.AUTOTUNE)\
                  .map(lambda x,y: (*flatten_x(x, KEYS_TO_USE), y), num_parallel_calls=tf.data.AUTOTUNE)\
                  .map(lambda x1,x2,x3,x4,y: ((x1,x2,x3,x4), y))\
                  .shuffle(SHUFFLE_BUFFER)\
                  .batch(BATCH_SIZE, drop_remainder=True)\
                  .prefetch(tf.data.AUTOTUNE)
#                .flat_map(flatten_first_dim)\

val_ds = tf.data.TFRecordDataset(val_tfrec_paths, num_parallel_reads=tf.data.AUTOTUNE)\
                .map(decode_redux_example, num_parallel_calls=tf.data.AUTOTUNE)\
                .map(lambda x,y: (*flatten_x(x, KEYS_TO_USE), y), num_parallel_calls=tf.data.AUTOTUNE)\
                .map(lambda x1,x2,x3,x4,y: ((x1,x2,x3,x4), y))\
                .shuffle(SHUFFLE_BUFFER//5)\
                .batch(BATCH_SIZE, drop_remainder=True)\
                .prefetch(tf.data.AUTOTUNE)
#                .flat_map(flatten_first_dim)\
# .map(lambda x,y: (*flatten_x(x, KEYS_TO_USE), tf.repeat(tf.expand_dims(y, axis=0), N_KEYFRAMES, axis=0)), num_parallel_calls=tf.data.AUTOTUNE)\
print(f"\nDATASETS:\n\tTRAIN --> {train_ds}\n\tVAL   --> {val_ds}")

class PointVectorizer(tf.keras.layers.Layer):
    """
    A custom tf.keras.layers.Layer for computing pairwise angles 
    between connections in a set of hand landmarks.

    Args:
        n_connections (int, optional): 
            – The number of possible connections between pairs of landmarks.
    """
    
    def __init__(self, n_connections=21, **kwargs):
        super(PointVectorizer, self).__init__(**kwargs)
        self.n_connections = n_connections
        self.connections = tf.constant([(i, j) for i in range(n_connections) for j in range(n_connections)], dtype=tf.int32)
    
    def call(self, landmarks):
        """Computes the pairwise angles between connections in a set of hand landmarks.

        Args:
            landmarks: A tensor of shape (batch_size, num_points, point_dimensions) containing the hand landmarks.

        Returns:
            A tensor of shape (batch_size, n_connections * n_connections) containing the pairwise angles between connections.
        """
        
        # Compute the connection vectors
        connections = tf.gather(landmarks, self.connections, axis=1)
        connection_vectors = connections[:, :, 1, :] - connections[:, :, 0, :]

        # Compute the pairwise angles between the connection vectors
        angles = []
        for i in range(self.n_connections):
            for j in range(self.n_connections):
                angle = self._get_angle_between_vectors(connection_vectors[:, i, :], connection_vectors[:, j, :])
                angle = tf.where(tf.math.is_nan(angle), tf.zeros_like(angle), angle)
                angles.append(angle)

        # Return the flattened list of angles
        return tf.concat(angles, axis=-1)

    @staticmethod
    def _get_angle_between_vectors(u, v):
        """Computes the pairwise angles between two sets of vectors.

        Args:
            u, v: Tensors of shape (batch_size, n_frames, num_vectors, vector_dimensions) containing the vectors.

        Returns:
            A tensor of shape (batch_size, n_frames, num_vectors, num_vectors) containing the pairwise angles between the vectors.
        """
        
        # Compute the dot product and norms of the vectors
        dot_product = tf.reduce_sum(u * v, axis=-1)
        norm_u = tf.norm(u, axis=-1)
        norm_v = tf.norm(v, axis=-1)

        # Compute the cosine similarity and angle between the vectors
        cosine_similarity = dot_product / (norm_u * norm_v)
        angle = tf.acos(cosine_similarity)

        return angle

    def get_config(self):
        config = super(PointVectorizer, self).get_config()
        config.update({'n_connections': self.n_connections})
        return config

def residual_block(inputs, output_channels, kernel_size=3, strides=1, dropout=0.0):
    # Save the input tensor to add to the output later
    shortcut = inputs
    
    # Apply a convolution layer with batch normalization and activation
    x = tf.keras.layers.Conv1D(output_channels, kernel_size, strides=strides, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Apply another convolution layer with batch normalization, but no activation
    x = tf.keras.layers.Conv1D(output_channels, kernel_size, strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Adjust the shortcut connection to match the output channels of the second convolution layer
    if shortcut.shape[-1] != output_channels:
        shortcut = tf.keras.layers.Conv1D(output_channels, kernel_size=1, strides=strides, padding='same')(shortcut)
    
    # Add the shortcut connection to the output of the second convolution layer
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    return x




class HandModel(tf.keras.Model):
    def __init__(self, n_landmarks=21, n_ax=2, fc_nodes=(256,64), fc_dropout=(0.025, 0.15)):
        super(HandModel, self).__init__()
        self.n_landmarks = n_landmarks
        self.n_ax = n_ax
        self.fc_nodes = fc_nodes
        self.fc_dropout = fc_dropout
        
        self.reduce_mean = tf.keras.layers.Lambda(
            lambda _x: tf.reduce_sum(_x, axis=1)/tf.cast(tf.math.count_nonzero(_x, axis=1), tf.float32)
        )
        self.pv = PointVectorizer(self.n_landmarks)
        
        # Modelling layers
        self.fc_layers = [tf.keras.layers.Dense(_n, activation="relu", name=f"fc_dense_{i+1}") for i, _n in enumerate(self.fc_nodes)]
        self.fc_dropouts = [tf.keras.layers.Dropout(_d, name=f"fc_dropout_{i+1}") for i, _d in enumerate(self.fc_dropout)]
    
    def call(self, inputs, training=None):
        # shape (1, n_frames, n_landmarks, n_ax)
        
        # for now we just grab the non-zero mean of the input_array
        # so that our output is (1, n_landmarks, n_ax)
        x = self.reduce_mean(inputs)
        x = self.pv(x)
        
        for _fc, _drop in zip(self.fc_layers, self.fc_dropouts):
            x = _fc(x) # residual?
            x = _drop(x, training=training)
        return x

class PoseModel(tf.keras.Model):
    def __init__(self, n_landmarks=33, n_ax=2, fc_nodes=(64,32), fc_dropout=(0.025, 0.15)):
        super(PoseModel, self).__init__()
        self.n_landmarks = n_landmarks
        self.n_ax = n_ax
        self.fc_nodes = fc_nodes
        self.fc_dropout = fc_dropout
        
        self.reduce_mean = tf.keras.layers.Lambda(
            lambda _x: tf.reduce_sum(_x, axis=1)/tf.cast(tf.math.count_nonzero(_x, axis=1), tf.float32)
        )
        self.pv = PointVectorizer(self.n_landmarks)
        
        # Modelling layers
        self.fc_layers = [tf.keras.layers.Dense(_n, activation="relu", name=f"fc_dense_{i+1}") for i, _n in enumerate(self.fc_nodes)]
        self.fc_dropouts = [tf.keras.layers.Dropout(_d, name=f"fc_dropout_{i+1}") for i, _d in enumerate(self.fc_dropout)]
    
    def call(self, inputs, training=None):
        # shape (1, n_frames, n_landmarks, n_ax)
        
        # for now we just grab the non-zero mean of the input_array
        # so that our output is (1, n_landmarks, n_ax)
        x = self.reduce_mean(inputs)
        x = self.pv(x)
        
        for _fc, _drop in zip(self.fc_layers, self.fc_dropouts):
            x = _fc(x)
            x = _drop(x, training=training)
        return x

class FaceModel(tf.keras.Model):
    def __init__(self, n_landmarks=468, n_ax=2, fc_nodes=(128, 32), fc_dropout=(0.1, 0.25)):
        super(FaceModel, self).__init__()
        self.n_landmarks = n_landmarks
        self.n_ax = n_ax
        self.fc_nodes = fc_nodes
        self.fc_dropout = fc_dropout
        
        self.reduce_mean = tf.keras.layers.Lambda(
            lambda _x: tf.reduce_sum(_x, axis=1)/tf.cast(tf.math.count_nonzero(_x, axis=1), tf.float32)
        )
        self.pv = PointVectorizer(self.n_landmarks)
        
        # Modelling layers
        self.fc_layers = [tf.keras.layers.Dense(_n, activation="relu", name=f"fc_dense_{i+1}") for i, _n in enumerate(self.fc_nodes)]
        self.fc_dropouts = [tf.keras.layers.Dropout(_d, name=f"fc_dropout_{i+1}") for i, _d in enumerate(self.fc_dropout)]
    
    def call(self, inputs, training=None):
        # shape (1, n_frames, n_landmarks, n_ax)
        
        # for now we just grab the non-zero mean of the input_array
        # so that our output is (1, n_landmarks, n_ax)
        x = self.reduce_mean(inputs)
        x = self.pv(x)
        print(x.shape)
        for _fc, _drop in zip(self.fc_layers, self.fc_dropouts):
            x = _fc(x) # residual?
            x = _drop(x, training=training)
        
        return x
    
class ISLite(tf.keras.Model):
    def __init__(self, n_total_landmarks=543, 
                 face_idx_range=(0, 468), 
                 lh_idx_range=(468, 489),
                 pose_idx_range=(489, 522),
                 rh_idx_range=(522, 543),
                 n_ax=2, raw_frame_shape=(543,3),
                 n_labels=250, head_dropout=0.2):
        super(ISLite, self).__init__()
        
        self.n_total_landmarks = n_total_landmarks
        self.face_idx_range    = face_idx_range 
        self.n_face_landmarks  = face_idx_range[1]-face_idx_range[0]
        self.lh_idx_range      = lh_idx_range
        self.n_lh_landmarks    = lh_idx_range[1]-lh_idx_range[0]
        self.pose_idx_range    = pose_idx_range
        self.n_pose_landmarks  = pose_idx_range[1]-pose_idx_range[0]
        self.rh_idx_range      = rh_idx_range
        self.n_rh_landmarks    = rh_idx_range[1]-rh_idx_range[0]
        self.n_ax              = n_ax
        self.raw_frame_shape   = raw_frame_shape
        self.n_labels          = n_labels
        self.head_dropout      = head_dropout
        
        self.fix_nans = tf.keras.layers.Lambda(
            lambda _x: tf.where(tf.math.is_nan(_x), tf.zeros_like(_x), _x), name="nan_to_zero",
        )
        
        self.add_batch_dim = tf.keras.layers.Lambda(
            lambda _x: tf.expand_dims(_x, axis=0), "add_batch_dim"
        )
        
        # Submodels
        self.face_model = FaceModel(self.n_face_landmarks, self.n_ax)
        self.lh_model   = HandModel(self.n_lh_landmarks, self.n_ax)
        self.pose_model = PoseModel(self.n_pose_landmarks, self.n_ax)
        self.rh_model   = HandModel(self.n_rh_landmarks, self.n_ax)
        
        # Head info
        self.head_concat  = tf.keras.layers.Concatenate(axis=-1)
        self.head_dropout = tf.keras.layers.Dropout(head_dropout)
        self.head_fc      = tf.keras.layers.Dense(n_labels, activation="softmax")
        
    def call(self, inputs, training=None):
        """
        Forward pass for the ISLite class.

        Args:
          inputs: A tensor of shape (n, 543, 3).

        Returns:
          A tuple of four tensors, each of shape (n, 543, k), where k is the number of landmarks for the corresponding body part.
        """
        
        # Fix nans and add batch dimension
        inputs = self.fix_nans(inputs)
        inputs = self.add_batch_dim(inputs)
        
        # Reduce last dimension to x and y (1, n, 543, 3) --> (1, n, 543, 2)
        # and split into relevant landmarks for each piece
        #  1. (1, n, self.n_face_landmarks, 2)
        #  2. (1, n, self.n_lh_landmarks,   2)
        #  3. (1, n, self.n_pose_landmarks, 2)
        #  4. (1, n, self.n_rh_landmarks,   2)
        face_lmarks = inputs[:, :, self.face_idx_range[0] : self.face_idx_range[1], :self.n_ax]
        lh_lmarks   = inputs[:, :, self.lh_idx_range[0]   : self.lh_idx_range[1],   :self.n_ax]
        pose_lmarks = inputs[:, :, self.pose_idx_range[0] : self.pose_idx_range[1], :self.n_ax]
        rh_lmarks   = inputs[:, :, self.rh_idx_range[0]   : self.rh_idx_range[1],   :self.n_ax]
        
        # Get type wise outputs from our modells
        face_outputs = self.face_model(face_lmarks, training=training)
        lh_outputs   = self.lh_model(lh_lmarks,     training=training)
        pose_outputs = self.pose_model(pose_lmarks, training=training)
        rh_outputs   = self.rh_model(rh_lmarks,     training=training)
        
        head_output  = self.head_concat([face_outputs, lh_outputs, pose_outputs, rh_outputs]) 
        head_output  = self.head_dropout(head_output, training=training)
        head_output  = self.head_fc(head_output)
        
        return head_output
    
demo_arr = load_relevant_data_subset(train_df.path[0])
islr_model = ISLite()
preds = islr_model(demo_arr)
for i in range(4):
    print(preds[i].shape)
islr_model.summary()

def get_model(n_labels=250, init_fc=1024, n_blocks=3, _dropout_1=0.1, _dropout_2=0.5, hand_input_shape=(21, 2), face_input_shape=(468, 2), pose_input_shape=(33,2), n_keyframes=3, n_ax=2, ):
    
    # Define input layers and join hands
    face_inputs = tf.keras.layers.Input(shape=(n_keyframes, *face_input_shape))
    lh_inputs = tf.keras.layers.Input(shape=(n_keyframes, *hand_input_shape))
    pose_inputs = tf.keras.layers.Input(shape=(n_keyframes, *pose_input_shape))
    rh_inputs = tf.keras.layers.Input(shape=(n_keyframes, *hand_input_shape))
    
    hand_x = tf.keras.layers.Maximum()([lh_inputs, rh_inputs])
    x = tf.keras.layers.Concatenate(axis=2)([face_inputs, hand_x, pose_inputs])
    x = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0,2,1,3]))(x)
    x = tf.keras.layers.Reshape((hand_input_shape[0]+face_input_shape[0]+pose_input_shape[0], n_keyframes*n_ax))(x)
    
    # Define residual layers
    for i in range(n_blocks):
        x = residual_block(
            x, output_channels=init_fc//(2**i), 
            dropout=_dropout_1 if (1+i)!=n_blocks else _dropout_2
        )
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Define output layer
    _outputs = tf.keras.layers.Dense(n_labels, activation="softmax")(x)
    
    # Build the model
    model = tf.keras.models.Model(inputs=(face_inputs, lh_inputs, pose_inputs, rh_inputs), outputs=_outputs)
    return model

model = get_model()
model.compile("adam", "sparse_categorical_crossentropy", metrics="acc")
model.summary()

tf.keras.utils.plot_model(model)

class HandModel(tf.keras.Model):
    def __init__(self, n_landmarks=21, n_ax=2, fc_nodes=(128,64,32), fc_dropout=(0.025, 0.05, 0.1)):
        super(HandModel, self).__init__()
        self.n_landmarks = n_landmarks
        self.n_ax = n_ax
        self.fc_nodes = fc_nodes
        self.fc_dropout = fc_dropout
        
        self.reduce_mean = tf.keras.layers.Lambda(
            lambda _x: tf.reduce_sum(_x, axis=1)/tf.cast(tf.math.count_nonzero(_x, axis=1), tf.float32)
        )
        self.reshape_arr = tf.keras.layers.Lambda(
            lambda _x: tf.reshape(_x[0], (_x[1], n_landmarks*n_ax))
        )
        # Modelling layers
        self.fc_layers = [tf.keras.layers.Dense(_n, activation="relu") for i, _n in enumerate(self.fc_nodes)]
        self.fc_dropouts = [tf.keras.layers.Dropout(_d) for i, _d in enumerate(self.fc_dropout)]
    
    def call(self, inputs, training=None):
        # shape (1, n_frames, n_landmarks, n_ax)
        batch_dim = tf.shape(inputs)[0]
        # for now we just grab the non-zero mean of the input_array
        # so that our output is (1, n_landmarks, n_ax)
        x = self.reduce_mean(inputs)
        x = self.reshape_arr([x, batch_dim])
        for _fc, _drop in zip(self.fc_layers, self.fc_dropouts):
            x = _fc(x) # residual?
            x = _drop(x, training=training)
        return x


class ISLite(tf.keras.Model):
    def __init__(self, n_total_landmarks=543, 
                 face_idx_range=(0, 468), 
                 lh_idx_range=(468, 489),
                 pose_idx_range=(489, 522),
                 rh_idx_range=(522, 543),
                 n_ax=2, raw_frame_shape=(543,3),
                 n_labels=250, head_dropout=0.2):
        super(ISLite, self).__init__()
        
        self.n_total_landmarks = n_total_landmarks
        
        self.lh_idx_range      = lh_idx_range
        self.n_lh_landmarks    = lh_idx_range[1]-lh_idx_range[0]
        
        self.rh_idx_range      = rh_idx_range
        self.n_rh_landmarks    = rh_idx_range[1]-rh_idx_range[0]
        
        self.n_ax              = n_ax
        
        self.raw_frame_shape   = raw_frame_shape
        self.n_labels          = n_labels
        self.head_dropout      = head_dropout
        
        self.fix_nans = tf.keras.layers.Lambda(
            lambda _x: tf.where(tf.math.is_nan(_x), tf.zeros_like(_x), _x)
        )
        
        self.add_batch_dim = tf.keras.layers.Lambda(
            lambda _x: tf.expand_dims(_x, axis=0)
        )
        
        # Submodels
        #self.face_model = FaceModel(self.n_face_landmarks, self.n_ax)
        #self.pose_model = PoseModel(self.n_pose_landmarks, self.n_ax)
        self.lh_model   = HandModel(self.n_lh_landmarks, self.n_ax)
        self.rh_model   = HandModel(self.n_rh_landmarks, self.n_ax)
        
        # Head info
        self.head_concat  = tf.keras.layers.Concatenate(axis=-1)
        self.head_dropout = tf.keras.layers.Dropout(head_dropout)
        # self.head_fc      = tf.keras.layers.Dense(n_labels, activation="softmax")
        
    def call(self, inputs, training=None):
        """
        Forward pass for the ISLite class.

        Args:
          inputs: A tensor of shape (n, 543, 3).

        Returns:
          A tuple of four tensors, each of shape (n, 543, k), where k is the number of landmarks for the corresponding body part.
        """
        
        lh_lmarks   = inputs[:, :, self.lh_idx_range[0]   : self.lh_idx_range[1],   :self.n_ax]
        rh_lmarks   = inputs[:, :, self.rh_idx_range[0]   : self.rh_idx_range[1],   :self.n_ax]

        lh_outputs   = self.fix_nans(self.lh_model(lh_lmarks,     training=training))
        rh_outputs   = self.fix_nans(self.rh_model(rh_lmarks,     training=training))
        
        head_output  = self.head_concat([lh_outputs, rh_outputs]) 
        head_output  = self.head_dropout(head_output, training=training)
        
        return head_output
    
def get_model():
    _inputs = tf.keras.layers.Input(shape=(None,543,3), name="inputs")
    x = ISLite()(_inputs)
    _outputs = tf.keras.layers.Dense(250, activation="softmax", name="outputs")(x)
    model = tf.keras.Model(inputs=_inputs, outputs=_outputs)
    model.compile("adam", "sparse_categorical_crossentropy")
    return model
    
islr_model = get_model()
# demo_arr = load_relevant_data_subset(train_df.path[0])
# islr_model = ISLite()
# preds = islr_model(demo_arr)
# islr_model.summary()


# Fix nans and add batch dimension
# inputs = self.fix_nans(inputs)
# inputs = self.add_batch_dim(inputs)

def residual_block(inputs, output_channels, kernel_size=3, strides=1, dropout=0.0):
    # Save the input tensor to add to the output later
    shortcut = inputs
    
    # Apply a convolution layer with batch normalization and activation
    x = tf.keras.layers.Conv1D(output_channels, kernel_size, strides=strides, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Apply another convolution layer with batch normalization, but no activation
    x = tf.keras.layers.Conv1D(output_channels, kernel_size, strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Adjust the shortcut connection to match the output channels of the second convolution layer
    if shortcut.shape[-1] != output_channels:
        shortcut = tf.keras.layers.Conv1D(output_channels, kernel_size=1, strides=strides, padding='same')(shortcut)
    
    # Add the shortcut connection to the output of the second convolution layer
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    return x



def get_fc_model(n_labels=250, init_fc=64, n_blocks=2):
    
    # Define input layer
    _inputs = tf.keras.layers.Input(shape=(MAX_N_FRAMES, 543, 3))
    
    # Access just the hands stuff
    
    lh_x = tf.slice(_inputs, [0, 0, FRAME_TYPE_IDX_MAP["left_hand"][0], 0], [-1, -1, len(FRAME_TYPE_IDX_MAP["left_hand"]), 2])
    rh_x = tf.slice(_inputs, [0, 0, FRAME_TYPE_IDX_MAP["right_hand"][0], 0], [-1, -1, len(FRAME_TYPE_IDX_MAP["right_hand"]), 2])
    x = tf.concat([lh_x, rh_x], axis=2)
    x = tf.transpose(x, perm=[0, 1, 3, 2])
    x = tf.keras.layers.AveragePooling2D()(x)
    x = tf.squeeze(tf.transpose(x, perm=[0, 1, 3, 2]), axis=-1)
    # Define residual layers
    for i in range(n_blocks):
        x = residual_block(
            x, output_channels=init_fc//(2**i), 
            dropout=0 if (1+i)!=n_blocks else 0.25
        )
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Flatten()(x)
    
    # Define output layer
    _outputs = tf.keras.layers.Dense(n_labels, activation="softmax")(x)
    
    # Build the model
    model = tf.keras.models.Model(inputs=_inputs, outputs=_outputs)
    
    return model

model = get_fc_model()
model.compile("adam", "sparse_categorical_crossentropy", metrics=["acc"])
model.summary()

tf.keras.utils.plot_model(model)


