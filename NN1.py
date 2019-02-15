import numpy as np
import pandas as pd
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Russian language

# мой путь к файлам
os.chdir("C:/Users/User/Desktop/CSC 2019 весна/other/cyber-punk")

# делаем новые факторы, а именно квадраты текущих факторов
def quadr_factors(data):
    target = data[64].copy()
    for i in range(64):
        col = i + 64
        data[col] = data[i]**2
    data[128] = target
    return data

# на данный момент у нас 128 факторов
n = 128
test_size = 400

# считываем данные, объединяем, размешиваем и делим на тест и обучение
def read_data():
    allFiles = ['0.csv', '1.csv', '2.csv', '3.csv']
    list = []
    for file in allFiles:
        df = pd.read_csv(file,index_col=None, header=None)
        df = quadr_factors(df)
        read_matrix = np.asmatrix(df)
        list.append(read_matrix)
    data = np.concatenate(list)
    values = np.asmatrix(data)
    np.random.shuffle(values)
    x_train = values[test_size:,:n]
    y_train = values[test_size:,n]
    x_test = values[:test_size,:n]
    y_test = values[:test_size,n]
    print('x_train ', x_train.shape)
    print('y_train ', y_train.shape)
    print('x_test ', x_test.shape)
    print('y_test ', y_test.shape)
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = read_data()

# делаем стандартизацию
for i in range(n):
    mean_ = np.mean(list(np.asarray(x_train[:, i])))
    std_ = np.std(list(np.asarray(x_train[:, i])))
    x_train[:, i] -= mean_
    x_train[:, i] /= std_
    x_test[:, i] -= mean_
    x_test[:, i] /= std_

# изменяем формат таргета на нужный для нейронной сети   
y_train = keras.utils.to_categorical(y_train, num_classes=4)
y_test = keras.utils.to_categorical(y_test, num_classes=4)

# также делим еще на валидацию, чтобы проверять переобучение/недообучение от итераций
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]


model = Sequential()
model.add(Dense(40, activation='relu', input_dim=n))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='nadam' , metrics=['accuracy'])
model.fit(partial_x_train, partial_y_train, epochs=50, batch_size=100, validation_data=(x_val, y_val))

score = model.evaluate(x_test, y_test, batch_size=test_size)
print(score)

last_score_list = [0.9645, 0.9675, 0.9575, 0.9650, 0.9500, 0.9600, 0.9550, 0.9500, 
                   0.9550, 0.9450, 0.9725, 0.9500, 0.9525, 0.9650, 0.9525, 0.9525, 0.9625]
print(np.mean(last_score_list))


"""
Что мне дало улучшить предыдущий алгоритм:
    1) Добавить квадраты факторов
    2) Добавить стандартизацию
    3) Немного изменил кол-во нейронов внутри слоя
    4) Добавил всем слоям dropout, чтобы было меньше переобучения
    5) Изменил оптимизатор на "nadam"
"""