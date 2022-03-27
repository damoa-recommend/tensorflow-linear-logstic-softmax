import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

# 공부 시간
# 1개의 feature를 가진 샘플(sample) 정의
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9] )
# 성적
y = np.array([11, 22, 33, 44, 53, 66, 77, 87, 95] )

# 레이어
model = Sequential()

# 모델구축
model.add(Dense(1, input_dim=1, activation='linear'))

# 비용(손실)함수: MSE(mean squared error)
# 최적화 함수: SGD(stochastic gradient descent), lr(learning rate)은 학습률을 의미
model.compile(optimizer=optimizers.SGD(lr=0.01), loss='mse', metrics=['mse'])

# 학습
model.fit(x, y, epochs=100)

# model layer 출력
model.summary()

# 예측
predict = model.predict([9.5])
print(predict)


# plt.plot(x, model.predict(x), 'b', x, y, 'k.')
# plt.grid()
# plt.show()
