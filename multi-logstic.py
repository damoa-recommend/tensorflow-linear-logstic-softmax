import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

# 2개의 feature를 가진 샘플(sample) 정의
X = np.array([[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]])
y = np.array([0, 0, 0, 1, 1, 1])

# 레이어
model = Sequential()

# 모델구축
# 학습 데이터의 feature(차원) 2개
model.add(Dense(1, input_dim=2, activation='linear'))

# 비용(손실)함수: MSE(mean squared error)
# 최적화 함수: SGD(stochastic gradient descent), lr(learning rate)은 학습률을 의미
model.compile(optimizer=optimizers.SGD(lr=0.00001), loss='mse', metrics=['mse'])

# 학습
model.fit(x, y, epochs=100)

# model layer 출력
model.summary()

# 예측
predict = print(model.predict(X))
print(predict)
