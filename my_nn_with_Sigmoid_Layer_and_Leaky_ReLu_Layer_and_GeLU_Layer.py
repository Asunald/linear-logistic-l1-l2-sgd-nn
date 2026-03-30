import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_mnist_from_csv(train_csv_path, test_csv_path, val_split=0.1):
    train_data = pd.read_csv(train_csv_path)

    test_data = pd.read_csv(test_csv_path)

    y_train_full = train_data.iloc[:, 0].to_numpy(np.int64)
    X_train_full = train_data.iloc[:, 1:].to_numpy(np.float32) / 255.0

    y_test = test_data.iloc[:, 0].to_numpy(np.int64)
    X_test = test_data.iloc[:, 1:].to_numpy(np.float32) / 255.0

    n_val = int(len(X_train_full) * val_split)

    np.random.seed(42)
    val_indices = np.random.choice(len(X_train_full), n_val, replace=False)

    train_mask = np.ones(len(X_train_full), dtype=bool)
    train_mask[val_indices] = False

    X_val = X_train_full[val_indices]
    y_val = y_train_full[val_indices]

    X_train = X_train_full[train_mask]
    y_train = y_train_full[train_mask]

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Validation labels shape: {y_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


class Layer:
    def __init__(self):
        pass

    def forward(self, input):
        return input

    def backward(self, grad_output):
        return grad_output


class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, grad_output):
        relu_grad = self.input > 0
        return grad_output * relu_grad


class SigmoidLayer(Layer):
    def forward(self, input):
        self.input = input
        x = np.clip(input, -500, 500)
        self.output = 1.0 / (1.0 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        return grad_output * (self.output * (1.0 - self.output))


class LeakyReLULayer(Layer):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, input):
        self.input = input
        return np.where(input > 0, input, self.alpha * input)

    def backward(self, grad_output):
        grad = np.where(self.input > 0, 1.0, self.alpha)
        return grad_output * grad


class GeLULayer(Layer):
    def forward(self, input):
        self.input = input
        c = np.sqrt(2.0 / np.pi)
        self._inner = c * (input + 0.044715 * (input ** 3))
        self._tanh = np.tanh(self._inner)
        output = 0.5 * input * (1.0 + self._tanh)
        return output

    def backward(self, grad_output):
        c = np.sqrt(2.0 / np.pi)
        x = self.input
        tanh_s = self._tanh
        ds_dx = c * (1.0 + 3.0 * 0.044715 * (x ** 2))
        dy_dx = 0.5 * (1.0 + tanh_s) + 0.5 * x * (1.0 - tanh_s * tanh_s) * ds_dx
        return grad_output * dy_dx


class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        self.learning_rate = learning_rate

        self.weights = np.random.randn(input_units, output_units) * np.sqrt(
            2.0 / input_units
        )
        self.biases = np.zeros(output_units)

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, grad_output):
        grad_weights = np.dot(self.input.T, grad_output)

        grad_biases = np.sum(grad_output, axis=0)

        grad_input = np.dot(grad_output, self.weights.T)

        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input


def softmax_crossentropy_with_logits(logits, labels):
    batch_size = logits.shape[0]
    one_hot_labels = np.zeros_like(logits)
    one_hot_labels[np.arange(batch_size), labels] = 1

    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    loss = -np.sum(one_hot_labels * np.log(softmax_probs + 1e-9)) / batch_size

    grad = (softmax_probs - one_hot_labels) / batch_size

    return loss, grad


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def forward(network, X):
    activations = []
    input = X

    for layer in network:
        input = layer.forward(input)
        activations.append(input)

    return activations


def predict(network, X):
    logits = forward(network, X)[-1]
    probs = softmax(logits)
    return np.argmax(probs, axis=-1)


def train(network, X, y):
    activations = forward(network, X)
    logits = activations[-1]

    loss, grad_logits = softmax_crossentropy_with_logits(logits, y)

    grad_output = grad_logits
    for i in range(len(network))[::-1]:
        layer = network[i]
        grad_output = layer.backward(grad_output)

    return loss


def train_mnist_network(
    X_train, y_train, X_val, y_val, num_epochs=200
):
    network = [
        Dense(784, 64),
        SigmoidLayer(),
        Dense(64, 32),
        LeakyReLULayer(alpha=0.01),
        Dense(32, 16),
        GeLULayer(),
        Dense(16, 10),
    ]

    print("Network architecture:")
    for i, layer in enumerate(network):
        if isinstance(layer, Dense):
            print(
                f"Layer {i}: Dense ({layer.weights.shape[0]} -> {layer.weights.shape[1]})"
            )
        else:
            print(f"Layer {i}: {layer.__class__.__name__}")

    print(
        f"\nTraining on {len(X_train)} examples using Gradient Descent (full batch)"
    )

    for epoch in range(num_epochs):
        loss = train(network, X_train, y_train)

        train_predictions = predict(network, X_train)
        train_accuracy = np.mean(train_predictions == y_train)

        val_predictions = predict(network, X_val)
        val_accuracy = np.mean(val_predictions == y_val)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_from_csv(
        "./mnist_train.csv", "./mnist_test.csv", val_split=0.1
    )
    train_mnist_network(X_train, y_train, X_val, y_val)
