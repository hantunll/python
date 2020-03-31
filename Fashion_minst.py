#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


import tensorflow as tf  # 讀入Tensorflow


# In[ ]:


from tensorflow.keras.datasets import fashion_mnist  #讀入Fashion 版的 MNIST 


# In[ ]:


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# In[ ]:


len(x_train)


# In[ ]:


len(x_test)


# In[ ]:


n=1234  #想看第1234筆資料長怎樣


# In[ ]:


x_train[n]   #看x的第1234筆資料


# In[ ]:


y_train[n]   #看y的第1234筆資料


# In[ ]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[ ]:


plt.imshow(x_train[n],cmap='Reds')    #把資料以圖形呈現，紅色


# In[ ]:


n = 1234
print('這是', class_names[y_train[n]])
plt.imshow(x_train[n], cmap='Greys');


# In[ ]:


pick = np.random.choice(60000, 5, replace=False)

for i in range(5):
    n = pick[i]
    ax = plt.subplot(151+i)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(class_names[y_train[n]], fontsize=10)
    plt.imshow(x_train[n], cmap='Greys')  


# In[ ]:


x_train = x_train.reshape(60000,784)/255
x_test = x_test.reshape(10000,784)/255


# In[ ]:


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


# In[ ]:


model=Sequential()


# In[ ]:


model.add(Dense(99,input_dim=784,activation='relu'))
model.add(Dense(99,activation='relu'))
model.add(Dense(99,activation='relu'))


# In[ ]:


model.add(Dense(10,activation='softmax'))


# In[ ]:


model.compile(loss='mse',optimizer=SGD(lr=0.015),metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(x_train,y_train,batch_size=100,epochs=25)


# In[ ]:


result = model.predict_classes(x_test)


# In[ ]:


n = 1234

print('神經網路預測是:', class_names[result[n]])
plt.imshow(x_test[n].reshape(28,28), cmap='Greys')


# In[ ]:





# In[ ]:




