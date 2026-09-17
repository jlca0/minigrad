import numpy as np
from functools import partial
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import autograd as ag

def lse(
        logit: ag.Tensor,
        Y: ag.Tensor,
        K: int
        )-> ag.Tensor:
    """
    Computes the least squares loss.
    Args:
        logit (ag.Tensor): The predicted logits of shape (K, N).
        Y (ag.Tensor): The true labels of shape (K, N).
    Returns:
        ag.Tensor: The computed least squares loss.
    """
    N = logit.data.shape[1]
    ones_N = ag.Tensor(np.ones((N, 1)), label='ones_N')
    ones_K = ag.Tensor(np.ones((1, K)), label='ones_K')
    return (ones_K @ ((Y + (-logit)) * (Y + (-logit))) @ ones_N) * (1/N)

def ce(
        logit: ag.Tensor, 
        Y: ag.Tensor, 
        K: int
        )-> ag.Tensor:
    """
    Computes the cross-entropy loss.
    Args:
        logit (ag.Tensor): The predicted logits of shape (K, N).
        Y (ag.Tensor): The true labels of shape (K, N).
    Returns:
        ag.Tensor: The computed cross-entropy loss.
    """
    N = logit.data.shape[1]
    ones_K = ag.Tensor(np.ones((1, K)), label='ones_K')
    ones_N = ag.Tensor(np.ones((N, 1)), label='ones_N')
    return (ones_K @ (Y * logit.softmax().log())) @ ones_N * (-1 / N)

class NN:
    """
    A class representing a neural network with one hidden layer.
    Attributes:
        M (int): Number of input feature.
        K (int): Number of output classes. 
        A (ag.Tensor): Derived features weights.
        a0 (ag.Tensor): Derived features bias.
        B (ag.Tensor): Hidden layer weights.
        b0 (ag.Tensor): Hidden layer bias.
    """
    def __init__(
            self, 
            M: int, 
            K: int,
            type: str = 'Regression'
            ) -> None:
        """
        Initializes the neural network with one hidden layer.
        Args:
            M (int): Number of input features.
            K (int): Number of output classes.
            type (str): Use case for the neural network, e. g., classification or regression.
        Raises:
            ValueError: type must be either Regression or Classification.
        """
        self.M = M
        self.K = K

        # He initialization of weights and biases.
        self.A = ag.Tensor(np.random.randn(M, M) * np.sqrt(2.0 / M), label='A')
        self.a0 = ag.Tensor(np.zeros((M, 1)), label='a0')
        self.B = ag.Tensor(np.random.randn(K, M) * np.sqrt(2.0 / M), label='B')
        self.b0 = ag.Tensor(np.zeros((K, 1)), label='b0')

        if type == 'Regression':
            self._loss = partial(lse, K = self.K)
        elif type == 'Classification':
            self._loss = partial(ce, K = self.K)
        else:
            raise ValueError(f'{type} is not a valid type. Try Regression or Classification instead.')

    def train(
            self,
            X: ag.Tensor,
            Y: ag.Tensor, 
            epochs: int = 100,
            learning_rate: float = 0.01,
            verbose: bool = False
            ) -> None:
        """
        Trains a simple neural network with one hidden layer using backpropagation.
        Args:
            X (ag.Tensor): Input data of shape (M, N).
            Y (ag.Tensor): One-hot encoded labels of shape (K, N).
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for gradient descent.
        """  
        params = [self.A, self.a0, self.B, self.b0]

        for epoch in range(epochs):
            # forward pass
            logit = self.B @ (self.A @ X + self.a0).sigmoid() + self.b0
            l = self._loss(logit, Y)
            
            # backward pass
            l.backward()

            # parameter update
            for param in params:
                param.data -= learning_rate * param.grad
                param.grad = np.zeros_like(param.data)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {l.data.squeeze():.4f}")

    def predict(
            self, 
            X: ag.Tensor
            ) -> np.ndarray:
        """
        Predicts the expected class by the model.
        Args:
            X (ag.Tensor): Input data of shape (M, N).
        Returns:
            np.ndarray: Predicted class labels of shape (N,).
        """
        logit = self.B @ (self.A @ X + self.a0).sigmoid() + self.b0

        return np.argmax(logit.data, axis=0)

if __name__ == "__main__":
    M = 64  # 8x8 pixels
    K = 10  # Digits from 0 to 9

    def to_one_hot(y, num_classes=10):
        """
        Converts a vector of labels into one-hot encoded format.
        Args:
            y (np.ndarray): Array of labels of shape (N,).
            num_classes (int): Number of classes for one-hot encoding.
        Returns:
            np.ndarray: One-hot encoded array of shape (num_classes, N)."""
        return np.eye(num_classes)[y].T

    # Loading the digits dataset.
    digits = load_digits()
    X_raw = digits.data
    y_raw = digits.target

    X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

    X_train = ag.Tensor(X_train_raw.T / 16.0, label='X')
    X_test = ag.Tensor(X_test_raw.T / 16.0, label='X_test')
    Y_train = ag.Tensor(to_one_hot(y_train_raw), label='Y')

    # Train.
    model = NN(M, K, 'Regression')
    model.train(X_train, Y_train, epochs=500, learning_rate=0.02)
    y_pred = model.predict(X_test)

    # Compare.
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)
    print(f"Accuracy: {np.trace(matrix) / np.sum(matrix):.4f}")
