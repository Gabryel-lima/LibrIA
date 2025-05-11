"""A ideia é a gente unir a arquitetura ao notebook do Kaggle, que usa o VGG-16. 
A gente vai passar essa arquitetura do torch para receber RGB em vez da escala de cinza. 
E vamos aumentar a capacidade da rede, porque a gente vai rodar ela pelo próprio servidor do Kaggle."""

# Implementação em TensorFlow “baixo nível” (tf.Module + tf.nn)
# Está arquitetura será levada para o a gpu P-100 da Google Colab

import tensorflow as tf

class ASLNetTF(tf.Module):
    def __init__(self, img_size, num_classes, dropout_rate=0.5, name=None):
        super().__init__(name=name)
        init = tf.initializers.GlorotUniform()
        zeros = tf.zeros_initializer()

        # Variáveis do backbone VGG16 (poderiam ser carregadas via TF Hub,
        # mas aqui consideramos apenas o classificador)

        flat_dim = (img_size//32)*(img_size//32)*512  # 7×7×512 = 25088

        # Classificador: 25088→4096→2048→1024→num_classes
        self.w1 = tf.Variable(init([flat_dim, 4096]), name='w1')
        self.b1 = tf.Variable(zeros([4096]),      name='b1')
        self.w2 = tf.Variable(init([4096,    2048]), name='w2')
        self.b2 = tf.Variable(zeros([2048]),      name='b2')
        self.w3 = tf.Variable(init([2048,    1024]), name='w3')
        self.b3 = tf.Variable(zeros([1024]),      name='b3')
        self.w4 = tf.Variable(init([1024, num_classes]), name='w4')
        self.b4 = tf.Variable(zeros([num_classes]),      name='b4')

        self.dropout_rate = dropout_rate

    @tf.function
    def __call__(self, x, training=False):
        # x deve vir de VGG16 sem topo: shape=(B,7,7,512)
        x = tf.reshape(x, [tf.shape(x)[0], -1])  # (B,25088)
        x = tf.matmul(x, self.w1) + self.b1
        x = tf.nn.relu(x)
        if training: x = tf.nn.dropout(x, rate=self.dropout_rate)

        x = tf.matmul(x, self.w2) + self.b2
        x = tf.nn.relu(x)
        if training: x = tf.nn.dropout(x, rate=self.dropout_rate)

        x = tf.matmul(x, self.w3) + self.b3
        x = tf.nn.relu(x)
        if training: x = tf.nn.dropout(x, rate=self.dropout_rate)

        logits = tf.matmul(x, self.w4) + self.b4
        return logits

# Instância (após extrair features via VGG16):
model_tf = ASLNetTF(
    img_size=224, 
    num_classes=29, 
    dropout_rate=0.5
)
