import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

# 1차, 2차, 3차시험 점수
# 3개의 feature를 가진 샘플(sample) 정의
x = np.array([
  [73., 80., 75.],
  [93., 88., 93.],
  [89., 91., 90.],
  [96., 98., 100.],
  [73., 66., 70.] 
])
# 성적
y = np.array([
  [152.],
  [185.],
  [180.],
  [196.],
  [142.]
])

# 레이어
model = Sequential()

# 모델구축
# 학습 데이터의 feature(차원) 3개 
model.add(Dense(1, input_dim=3, activation='linear'))

# 비용(손실)함수: MSE(mean squared error)
# 최적화 함수: SGD(stochastic gradient descent), lr(learning rate)은 학습률을 의미
model.compile(optimizer=optimizers.SGD(lr=0.00001), loss='mse', metrics=['mse'])

# 학습
model.fit(x, y, epochs=100)

# model layer 출력
model.summary()

# 예측
predict = model.predict(np.array([[72., 93., 90.]]))
print(predict)
