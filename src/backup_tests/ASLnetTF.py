"""A ideia é a gente unir a arquitetura ao notebook do Kaggle, que usa o VGG-16. 
A gente vai passar essa arquitetura do torch para receber RGB em vez da escala de cinza. 
E vamos aumentar a capacidade da rede, porque a gente vai rodar ela pelo próprio servidor do Kaggle."""

# Implementação em TensorFlow “baixo nível” (tf.Module + tf.nn)
# Está arquitetura será levada para o a gpu P-100 da Google Colab

import tensorflow as tf

class ASLNetTF(tf.Module):
    def __init__(self, img_size, num_classes, dropout_rate, name=None):
        super().__init__(name=name)
        glorot = tf.initializers.GlorotUniform()
        zeros  = tf.zeros_initializer()

        # Conv1
        self.w1 = tf.Variable(glorot([3, 3, 1, 32]), name='w1')
        self.b1 = tf.Variable(zeros([32]), name='b1')
        self.gamma1 = tf.Variable(tf.ones([32]), name='gamma1')
        self.beta1  = tf.Variable(tf.zeros([32]), name='beta1')
        self.m1_mean = tf.Variable(tf.zeros([32]), trainable=False)
        self.m1_var  = tf.Variable(tf.ones([32]), trainable=False)

        # Conv2
        self.w2 = tf.Variable(glorot([3, 3, 32, 64]), name='w2')
        self.b2 = tf.Variable(zeros([64]), name='b2')
        self.gamma2 = tf.Variable(tf.ones([64]), name='gamma2')
        self.beta2  = tf.Variable(tf.zeros([64]), name='beta2')
        self.m2_mean = tf.Variable(tf.zeros([64]), trainable=False)
        self.m2_var  = tf.Variable(tf.ones([64]), trainable=False)

        # Conv3
        self.w3 = tf.Variable(glorot([3, 3, 64, 128]), name='w3')
        self.b3 = tf.Variable(zeros([128]), name='b3')
        self.gamma3 = tf.Variable(tf.ones([128]), name='gamma3')
        self.beta3  = tf.Variable(tf.zeros([128]), name='beta3')
        self.m3_mean = tf.Variable(tf.zeros([128]), trainable=False)
        self.m3_var  = tf.Variable(tf.ones([128]), trainable=False)

        # Fully connected
        flat_dim = (img_size // 8) * (img_size // 8) * 128
        self.w4 = tf.Variable(glorot([flat_dim, 512]), name='w4')
        self.b4 = tf.Variable(zeros([512]), name='b4')
        self.w5 = tf.Variable(glorot([512, num_classes]), name='w5')
        self.b5 = tf.Variable(zeros([num_classes]), name='b5')

        self.dropout_rate = dropout_rate

    @tf.function
    def __call__(self, x, training=False):
        # Conv1 -> BN -> ReLU -> Pool
        x = tf.nn.conv2d(x, self.w1, strides=1, padding='SAME') + self.b1
        x = tf.nn.batch_normalization(x, self.m1_mean, self.m1_var, self.beta1, self.gamma1, 1e-5)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')

        # Conv2 -> BN -> ReLU -> Pool
        x = tf.nn.conv2d(x, self.w2, strides=1, padding='SAME') + self.b2
        x = tf.nn.batch_normalization(x, self.m2_mean, self.m2_var, self.beta2, self.gamma2, 1e-5)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')

        # Conv3 -> BN -> ReLU -> Pool
        x = tf.nn.conv2d(x, self.w3, strides=1, padding='SAME') + self.b3
        x = tf.nn.batch_normalization(x, self.m3_mean, self.m3_var, self.beta3, self.gamma3, 1e-5)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')

        # Flatten
        x = tf.reshape(x, [tf.shape(x)[0], -1])

        # Dense 512 -> ReLU -> Dropout
        x = tf.matmul(x, self.w4) + self.b4
        x = tf.nn.relu(x)
        if training:
            x = tf.nn.dropout(x, rate=self.dropout_rate)

        # Saída
        logits = tf.matmul(x, self.w5) + self.b5
        return logits

# Exemplo de instância:
model_tf = ASLNetTF(
    img_size=224,
    num_classes=29,
    dropout_rate=0.5
)
