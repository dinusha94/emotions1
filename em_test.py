import cv2
import numpy as np
from keras.models import load_model

model = load_model('emotion.h5')
print("Model loaded")

im = cv2.imread('1.jpg',0)
im = cv2.resize(im,(48,48))
im = im.astype('float32')
im /= 255
im= np.expand_dims(im, axis=4)
im= np.expand_dims(im, axis=0)

#print(im.shape)
    
prediction = model.predict(im)
print(np.argmax(prediction))