import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from matplotlib.colors import ListedColormap


def _sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


class MyOwnLinearRegressionWithL1L2AndSGD:
    def __init__(
        self,
        learning_rate=0.01,
        n_epochs=500,
        batch_size=32,
        l1_lambda=0.0,
        l2_lambda=0.0,
        random_state=42,
        verbose_every=50,
    ):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.random_state = random_state
        self.verbose_every = verbose_every

        self.weights = None
        self.bias = None

    def _regularization_loss(self):
        w = self.weights
        l1 = self.l1_lambda * np.sum(np.abs(w))
        l2 = self.l2_lambda * np.sum(w * w)
        return l1 + l2

    def _regularization_grad(self):
        w = self.weights
        l1_grad = self.l1_lambda * np.sign(w)
        l2_grad = 2.0 * self.l2_lambda * w
        return l1_grad + l2_grad

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        indices = np.arange(n_samples)

        for epoch in range(1, self.n_epochs + 1):
            rng.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                X_b = X[batch_idx]
                y_b = y[batch_idx]

                y_pred = X_b @ self.weights + self.bias
                error = y_pred - y_b
                m = X_b.shape[0]

                grad_w = (2.0 / m) * (X_b.T @ error)
                grad_b = (2.0 / m) * np.sum(error)

                grad_w += self._regularization_grad()

                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            if self.verbose_every and epoch % self.verbose_every == 0:
                y_all_pred = X @ self.weights + self.bias
                mse = mean_squared_error(y, y_all_pred)
                loss = mse + self._regularization_loss()
                print(f"[Linear][Epoch {epoch}] loss={loss:.6f} mse={mse:.6f}")

    def predict(self, X):
        return X @ self.weights + self.bias


class LogisticRegressionWithL1L2AndSGD:
    def __init__(
        self,
        learning_rate=0.1,
        n_epochs=300,
        batch_size=32,
        l1_lambda=0.0,
        l2_lambda=0.0,
        random_state=42,
        verbose_every=50,
    ):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.random_state = random_state
        self.verbose_every = verbose_every

        self.weights = None
        self.bias = None

    def _regularization_loss(self):
        w = self.weights
        l1 = self.l1_lambda * np.sum(np.abs(w))
        l2 = self.l2_lambda * np.sum(w * w)
        return l1 + l2

    def _regularization_grad(self):
        w = self.weights
        l1_grad = self.l1_lambda * np.sign(w)
        l2_grad = 2.0 * self.l2_lambda * w
        return l1_grad + l2_grad

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        indices = np.arange(n_samples)
        eps = 1e-9

        for epoch in range(1, self.n_epochs + 1):
            rng.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                X_b = X[batch_idx]
                y_b = y[batch_idx]
                m = X_b.shape[0]

                logits = X_b @ self.weights + self.bias
                y_pred = _sigmoid(logits)

                # gradient of average BCE loss
                error = y_pred - y_b  # shape: (m,)
                grad_w = (1.0 / m) * (X_b.T @ error)
                grad_b = (1.0 / m) * np.sum(error)

                grad_w += self._regularization_grad()  # weights only

                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            if self.verbose_every and epoch % self.verbose_every == 0:
                logits_all = X @ self.weights + self.bias
                y_pred_all = _sigmoid(logits_all)
                bce = -np.mean(y * np.log(y_pred_all + eps) + (1 - y) * np.log(1 - y_pred_all + eps))
                loss = bce + self._regularization_loss()
                print(f"[Logistic][Epoch {epoch}] loss={loss:.6f} bce={bce:.6f}")

    def predict_proba(self, X):
        logits = X @ self.weights + self.bias
        return _sigmoid(logits)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)


def run_my_own_linear_regression():
    dataset = pd.read_csv("Salary_Data.csv")
    X = dataset.iloc[:, :-1].values.astype(float)
    y = dataset.iloc[:, -1].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 / 3, random_state=0
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MyOwnLinearRegressionWithL1L2AndSGD(
        learning_rate=0.01,
        n_epochs=800,
        batch_size=32,
        l1_lambda=0.001,
        l2_lambda=0.001,
        verbose_every=100,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print("Linear Regression MSE:", mse)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ax = axes[0]
    ax.scatter(X_train, y_train, color="red")
    if X_train.shape[1] == 1:
        X_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
        X_line_scaled = scaler.transform(X_line)
        y_line = model.predict(X_line_scaled)
        ax.plot(X_line, y_line, color="blue")
    ax.set_title("Salary vs Experience (Training set)")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")

    ax = axes[1]
    ax.scatter(X_test, y_test, color="red")
    if X_test.shape[1] == 1:
        X_line = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
        X_line_scaled = scaler.transform(X_line)
        y_line = model.predict(X_line_scaled)
        ax.plot(X_line, y_line, color="blue")
    ax.set_title("Salary vs Experience (Test set)")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")

    plt.tight_layout()
    plt.show()


def run_my_own_logistic_regression():
    data = pd.read_csv("Social_Network_Ads.csv")
    X_raw = data[["Age", "EstimatedSalary"]].values.astype(float)
    y = data["Purchased"].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=0
    )

    X_train_raw, X_test_raw, _, _ = train_test_split(
        X_raw, y, test_size=0.25, random_state=0
    )

    model = LogisticRegressionWithL1L2AndSGD(
        learning_rate=0.1,
        n_epochs=400,
        batch_size=64,
        l1_lambda=0.001,
        l2_lambda=0.001,
        verbose_every=100,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    cmap = ListedColormap(("red", "green"))
    colors = ["red", "green"]

    ax = axes[0]
    X_set, y_set = X_test, y_test

    x_min, x_max = X_set[:, 0].min() - 1, X_set[:, 0].max() + 1
    y_min, y_max = X_set[:, 1].min() - 1, X_set[:, 1].max() + 1

    X1, X2 = np.meshgrid(
        np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[X1.ravel(), X2.ravel()]
    Z = model.predict(grid).reshape(X1.shape)

    ax.contourf(X1, X2, Z, alpha=0.75, cmap=cmap)
    for i, label in enumerate(np.unique(y_set)):
        ax.scatter(
            X_set[y_set == label, 0],
            X_set[y_set == label, 1],
            color=colors[i],
            label=label,
        )
    ax.set_title("Scaled Features")
    ax.set_xlabel("Age (Standardized)")
    ax.set_ylabel("Estimated Salary (Standardized)")
    ax.legend()

    ax = axes[1]
    X_set, y_set = X_test_raw, y_test

    x_min, x_max = X_set[:, 0].min() - 5, X_set[:, 0].max() + 5
    y_min, y_max = X_set[:, 1].min() - 10000, X_set[:, 1].max() + 10000

    X1, X2 = np.meshgrid(
        np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300)
    )
    grid_raw = np.c_[X1.ravel(), X2.ravel()]
    grid_scaled = scaler.transform(grid_raw)
    Z = model.predict(grid_scaled).reshape(X1.shape)

    ax.contourf(X1, X2, Z, alpha=0.75, cmap=cmap)
    for i, label in enumerate(np.unique(y_set)):
        ax.scatter(
            X_set[y_set == label, 0],
            X_set[y_set == label, 1],
            color=colors[i],
            label=label,
        )

    ax.set_title("Original Features")
    ax.set_xlabel("Age")
    ax.set_ylabel("Estimated Salary")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_my_own_linear_regression()
    run_my_own_logistic_regression()

