import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

# xor data set
input = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
output = np.array([[0.], [1.], [1.], [0.]])

# layer 설계
model = Sequential()
model.add(Dense(16, input_dim=2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# 손실함수, 최적화 함수 설정
sgd = optimizers.SGD(lr=0.01, decay=0, momentum=0.99, nesterov=True)
model.compile(optimizer=sgd, loss='mse', metrics=['mae', 'mse'])

model.fit(input, output, epochs = 1500)

predict = model.predict(input)
print(predict)