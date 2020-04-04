```python
%matplotlib inline                

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```


```python
### 1. 讀入 MNSIT 數據集
```


```python
from tensorflow.keras.datasets import fashion_mnist  #讀入Fashion 版的 MNIST 
```


```python
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```


```python
### 2. 欣賞數據集內容
```


```python
n=1234  #想看第1234筆資料長怎樣
```


```python
x_train[n]   #看x的第1234筆資料
```


```python
y_train[n]   #看y的第1234筆資料
```


```python
plt.imshow(x_train[n], cmap='Greys')
```


```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```


```python
plt.imshow(x_train[n],cmap='Reds')    #把資料以圖形呈現，紅色
```


```python
n = 1234
print('這是', class_names[y_train[n]])
plt.imshow(x_train[n], cmap='Greys');
```


```python
pick = np.random.choice(60000, 5, replace=False)

for i in range(5):
    n = pick[i]
    ax = plt.subplot(151+i)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(class_names[y_train[n]], fontsize=10)
    plt.imshow(x_train[n], cmap='Greys')  
```


```python
x_train = x_train.reshape(60000,784)/255
x_test = x_test.reshape(10000,784)/255
```


```python
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```


```python
### 3.打造神經網路
```


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
```


```python
model=Sequential()
```


```python
model.add(Dense(99,input_dim=784,activation='relu'))
model.add(Dense(99,activation='relu'))
model.add(Dense(99,activation='relu'))
```


```python
model.add(Dense(10,activation='softmax'))
```


```python
### 4. 組裝我們的神經網路
```


```python
model.compile(loss='mse',optimizer=SGD(lr=0.08),metrics=['accuracy'])
```


```python
model.summary()
```


```python
model.fit(x_train,y_train,batch_size=120,epochs=50)
```


```python
### 6. 訓練成果
```


```python
result = model.predict_classes(x_test)
```


```python
n = 1234

print('神經網路預測是:', class_names[result[n]])
plt.imshow(x_test[n].reshape(28,28), cmap='Greys')
```


```python

```


```python

```
