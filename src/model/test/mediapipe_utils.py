import tensorflow as tf

frame_map = {
    "face":  list(range(0, 468)),
    "left_hand":  list(range(468, 489)),
    "pose":  list(range(489, 522)),
    "right_hand": list(range(522, 543)),
}

# -----------------------------------------------------
# Fully connected block
# -----------------------------------------------------
def fc_block(inputs, output_channels, dropout=0.2):
    """
    Bloco totalmente conectado reutilizável:
    - Dense
    - BatchNormalization
    - Ativação GELU
    - Dropout
    """
    x = tf.keras.layers.Dense(output_channels)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("gelu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    return x

# -----------------------------------------------------
# MLP model for flattened landmarks
# -----------------------------------------------------
def get_mlp_model(n_labels=250, init_fc=512, n_blocks=2,
                  dropout_1=0.2, dropout_2=0.6, flat_frame_len=3258):
    """
    Constrói um modelo MLP baseado em landmarks:
    - n_blocks blocos FC com dropout variável
    - Saída softmax com n_labels classes
    """
    inputs = tf.keras.layers.Input(shape=(flat_frame_len,))
    x = inputs
    for i in range(n_blocks):
        drop = dropout_1 if i < n_blocks - 1 else dropout_2
        x = fc_block(x, output_channels=init_fc // (2 ** i), dropout=drop)
    outputs = tf.keras.layers.Dense(n_labels, activation="softmax")(x)
    model = tf.keras.models.Model(inputs, outputs)
    return model

# -----------------------------------------------------
# Preprocessing layer: split landmarks and compute features
# -----------------------------------------------------
class PrepInputs(tf.keras.layers.Layer):
    def __init__(self,
                 face_idx_range=(0, 468),
                 lh_idx_range=(468, 489),
                 pose_idx_range=(489, 522),
                 rh_idx_range=(522, 543)):
        super().__init__()
        self.idx_ranges = [face_idx_range, lh_idx_range, pose_idx_range, rh_idx_range]
        self.flat_feat_lens = [3 * (rng[1] - rng[0]) for rng in self.idx_ranges]

    def call(self, x):
        # x: [batch, 543, 3]
        feats = []
        for rng, feat_len in zip(self.idx_ranges, self.flat_feat_lens):
            slc = x[:, rng[0]:rng[1], :]
            slc = tf.reshape(slc, (-1, feat_len))
            means = tf.reduce_mean(slc, axis=0)
            stds = tf.math.reduce_std(slc, axis=0)
            feat = tf.concat([means, stds], axis=0)
            feat = tf.where(tf.math.is_finite(feat), feat, tf.zeros_like(feat))
            feats.append(feat)
        return tf.expand_dims(tf.concat(feats, axis=0), axis=0)

# -----------------------------------------------------
# TFLite wrapper combining preprocessing and model
# -----------------------------------------------------
class TFLiteModel(tf.Module):
    """
    Wrapper para conversão em TFLite, encapsulando PrepInputs e o modelo Keras.
    """
    def __init__(self, keras_model):
        super().__init__()
        self.prep = PrepInputs()
        self.model = keras_model

    @tf.function(input_signature=[tf.TensorSpec([None, 543, 3], tf.float32)])
    def __call__(self, x):
        x = self.prep(tf.cast(x, tf.float32))
        out = self.model(x)[0]
        return {"outputs": out}

# -----------------------------------------------------
# TFRecord example decoder and flatten utilities
# -----------------------------------------------------
def decode_redux_example(serialized_example, n_keyframes=3, n_ax=2, frame_map=None):
    """
    Decodifica um TFRecord serializado em features e rótulo.

    Args:
        serialized_example: tf.Example serializado.
        frame_map: dict[str, list[int]] mapeando feature names para índices de landmarks.
    """
    if frame_map is None:
        raise ValueError("frame_map must be provided")
    features = {"lbl": tf.io.FixedLenFeature([], tf.int64)}
    for name, idxs in frame_map.items():
        features[name] = tf.io.FixedLenFeature([len(idxs) * n_keyframes * n_ax], tf.float32)
    parsed = tf.io.parse_single_example(serialized_example, features)
    label = tf.cast(parsed["lbl"], tf.int32)
    feats = {name: tf.reshape(parsed[name], (n_keyframes, len(idxs), n_ax))
             for name, idxs in frame_map.items()}
    return feats, label

# Flatten features dict into ordered list
# keys must match names in frame_map

def flatten_x(x_dict, keys):
    return [x_dict[k] for k in sorted(keys)]

# -----------------------------------------------------
# PointVectorizer: compute pairwise angles of landmarks
# -----------------------------------------------------
class PointVectorizer(tf.keras.layers.Layer):
    def __init__(self, n_connections=21):
        super().__init__()
        self.connections = tf.constant([
            (i, j) for i in range(n_connections) for j in range(n_connections)
        ], dtype=tf.int32)

    def call(self, lands):
        # lands: [batch, n_points, dim]
        pairs = tf.gather(lands, self.connections, axis=1)
        vecs = pairs[:, :, 1, :] - pairs[:, :, 0, :]
        dots = tf.reduce_sum(vecs[:, :, None, :] * vecs[:, None, :, :], axis=-1)
        norms = tf.norm(vecs, axis=-1)
        cos = dots / (norms[:, :, None] * norms[:, None, :])
        cos = tf.clip_by_value(cos, -1.0, 1.0)
        angles = tf.acos(cos)
        return tf.reshape(angles, (tf.shape(angles)[0], -1))

# -----------------------------------------------------
# Residual block 1D
# -----------------------------------------------------
def residual_block(inputs, output_channels, kernel_size=3,
                   strides=1, dropout=0.0):
    x = tf.keras.layers.Conv1D(output_channels, kernel_size,
                               strides=strides, padding="same",
                               activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(output_channels, kernel_size,
                               strides=strides, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    shortcut = inputs
    if inputs.shape[-1] != output_channels:
        shortcut = tf.keras.layers.Conv1D(output_channels, 1,
                                          padding="same")(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    return x

# -----------------------------------------------------
# Modular models by part
# -----------------------------------------------------
class HandModel(tf.keras.Model):
    def __init__(self, n_landmarks=21, n_ax=2,
                 fc_nodes=(256, 64), fc_dropout=(0.025, 0.15)):
        super().__init__()
        self.reduce_mean = tf.keras.layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=1)
        )
        self.pv = PointVectorizer(n_landmarks)
        self.fcs = [tf.keras.layers.Dense(n, activation="relu") for n in fc_nodes]
        self.drops = [tf.keras.layers.Dropout(d) for d in fc_dropout]

    def call(self, x, training=False):
        x = self.reduce_mean(x)
        x = self.pv(x)
        for f, d in zip(self.fcs, self.drops):
            x = f(x)
            x = d(x, training=training)
        return x

class PoseModel(HandModel):
    def __init__(self, n_landmarks=33, n_ax=2,
                 fc_nodes=(64, 32), fc_dropout=(0.025, 0.15)):
        super().__init__(n_landmarks, n_ax, fc_nodes, fc_dropout)

class FaceModel(HandModel):
    def __init__(self, n_landmarks=468, n_ax=2,
                 fc_nodes=(128, 32), fc_dropout=(0.1, 0.25)):
        super().__init__(n_landmarks, n_ax, fc_nodes, fc_dropout)

# -----------------------------------------------------
# Unified ISLite model
# -----------------------------------------------------
class ISLite(tf.keras.Model):
    def __init__(self, frame_map, n_ax=2,
                 n_labels=250, head_dropout=0.2):
        super().__init__()
        self.models = {
            name: (
                FaceModel if name=="face" else
                PoseModel if name=="pose" else HandModel
            )(len(idxs), n_ax)
            for name, idxs in frame_map.items()
        }
        self.concat = tf.keras.layers.Concatenate()
        self.drop = tf.keras.layers.Dropout(head_dropout)
        self.fc = tf.keras.layers.Dense(n_labels, activation="softmax")

    def call(self, x, training=False):
        parts = []
        for name, model in self.models.items():
            rng = frame_map[name]
            feat = x[:, :, rng[0]:rng[1], :2]
            parts.append(model(feat, training=training))
        h = self.concat(parts)
        h = self.drop(h, training=training)
        return self.fc(h)

# -----------------------------------------------------
# Build unified model from frame_map
# -----------------------------------------------------
def get_unified_model(frame_map, n_labels=250):
    """
    Cria um modelo ISLite configurado com frame_map.
    """
    inp = tf.keras.layers.Input(
        shape=(None, sum(len(v) for v in frame_map.values()), 3)
    )
    out = ISLite(frame_map, n_labels=n_labels)(inp)
    return tf.keras.models.Model(inp, out)
