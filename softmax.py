import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.datasets import load_iris

# 데이터 로드
iris = load_iris()

# input, output 분리
X = iris['data']

# 원 핫 인코딩을 수행한다.
# 출력 라벨은 3개가 있다. => ['setosa' 'versicolor' 'virginica']
# 원 핫 인코딩을 수행하면 0(setosa): [0, 0, 1], 1(versicolor): [0, 1, 0], 2(virginica): [1, 0, 0] 이런식으로 나온다.
Y = to_categorical(iris['target']) 
print(X)
print(Y)

model = Sequential()
model.add(Dense(3, input_dim=4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, Y, epochs=200, batch_size=1)

predict = model.predict(np.array([[5.6, 2.7, 4.2, 1.3]]))

# [[1.5873306e-04 8.1704116e-01 1.8280007e-01]] 
# 원 핫 인코딩이 된 라벨 데이터와 매핑된 확률이 나온다. 여기서 가장 가까운 값은 2이므로 해당 데이터는 최종 라벨이 2가 된다.
# 2에 해당하는 라벨값을 출력하면 된다.
print(predict)