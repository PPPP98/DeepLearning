from .function import sigmoid, softmax, cross_entropy_error
from .gradient import numerical_gradient
import numpy as np
from matplotlib import pyplot as plt


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        """
        추론 함수 - 입력 신호를 받아서 출력 신호를 반환하는 함수
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        """
        손실 함수 - 추론 결과와 정답 레이블 간의 오차를 구하는 함수
        """
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        """
        정확도 계산 함수
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """˝˝
        기울기 계산 함수 - 손실 함수의 기울기를 구하는 함수
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads

    def train(self, data, epochs=10, learning_rate=0.1, batch_size=100, epoch=None):
        """
        학습 함수 - 미니배치 학습을 수행하는 함수
        """
        if epoch is not None:
            epochs = epoch

        (x_train, t_train), (x_test, t_test) = data
        train_size = x_train.shape[0]
        train_loss = []
        train_acc = []
        test_acc = []

        for epoch_idx in range(epochs):
            # 미니배치 추출
            for i in range(0, train_size, batch_size):
                batch_mask = np.random.choice(train_size, batch_size)
                x_batch = x_train[batch_mask]
                t_batch = t_train[batch_mask]

                # 기울기 계산
                grads = self.numerical_gradient(x_batch, t_batch)

                # 매개변수 갱신
                for key in self.params:
                    self.params[key] -= learning_rate * grads[key]
            # 학습 경과 기록
            loss = self.loss(x_train, t_train)
            train_loss.append(loss)
            current_train_acc = self.accuracy(x_train, t_train)
            current_test_acc = self.accuracy(x_test, t_test)
            train_acc.append(current_train_acc)
            test_acc.append(current_test_acc)
            print(
                f"epoch {epoch_idx + 1}/{epochs} - "
                f"loss: {loss:.4f} - "
                f"train acc: {current_train_acc:.4f} - "
                f"test acc: {current_test_acc:.4f}"
            )

        # 학습 경과 시각화
        plt.plot(train_loss, label="train loss")
        plt.plot(train_acc, label="train acc")
        plt.plot(test_acc, label="test acc")
        plt.xlabel("epoch")
        plt.legend()
        plt.show()
