import numpy as np

# Definindo a função de ativação (degrau)
def step_function(x):
    return 1 if x >= 0 else 0

# Definindo o Perceptron
class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # Inicializando pesos e bias com zeros
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Treinamento
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = step_function(linear_output)
                
                # Atualização dos pesos
                update = self.learning_rate * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = [step_function(i) for i in linear_output]
        return np.array(y_pred)
