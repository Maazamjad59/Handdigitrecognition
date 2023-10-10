import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist=tf.keras.datasets.mnist
import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
    
(X_train,y_train),(X_test,y_test)=mnist.load_data()

#Normalizing_data
X_train=tf.keras.utils.normalize(X_train,axis=1)
X_test=tf.keras.utils.normalize(X_test,axis=1)

'''from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128,activation='relu'))
model.add(tf.keras.layers.Dense(units=128,activation='relu'))
model.add(tf.keras.layers.Dense(units='10',activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.fit(X_train,y_train,epochs=10)

model.save('handdigitrec')'''
'''
model=tf.keras.models.load_model('handdigitrec')

loss,accuracy=model.evaluate(X_test,y_test)
print(f'The loss is: {loss}')
print(f'The accuracy is: {accuracy}')'''

model=tf.keras.models.load_model('handdigitrec')
image_number=1
while os.path.isfile(f'digits/digit{image_number}.png'):
    try:
        img=cv2.imread(f'digits/digit{image_number}.png')[:,:,0]
        img=np.invert(np.array([img]))
        prediction=model.predict(img)
        print(f'This image is probably is: {np.argmax(prediction)}')
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print('Error')
    finally:
        image_number+=1
    