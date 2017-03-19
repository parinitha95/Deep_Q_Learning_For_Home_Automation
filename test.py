from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from PIL import Image
from scipy import misc
from skimage import color
from keras.preprocessing.image import ImageDataGenerator
import csv
import numpy as np
import random

repD=np.zeros(4)
repD=repD.reshape(1,4)
st=0
def read(state):
    path="/home/vignesh/Desktop/Projects/Home_automation/images/state_"+state+".jpg"
    im=misc.imread(path);
    im=misc.imresize(im,(80,80,3))
    im=np.transpose(im,(2,0,1))
    im = np.reshape(im,(1,3,80,80))
    return im

def makemove(st):
    st = st + 1
    #urllib.urlretrieve("http://192.168.0.9:8000/state"+str(st)+".jpg","/home/vignesh/Desktop/Projects/Home_automation/state"+str(st)+".jpg")
    return st

def getreward(state,action):
    if(int(state)==0 and action==0):
        return 100
    elif(int(state)==0 and action==1):
        return -100 
    elif(int(state)==1 and action==0):
        return 100
    elif(int(state)==1 and action==1):
        return 100 
 


model = Sequential()

model.add(Convolution2D(32,8,8, border_mode='valid',input_shape=(3, 80, 80))) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 4, 4, border_mode='valid')) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Convolution2D(64, 3, 3, border_mode='valid')) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Convolution2D(64, 3, 3, border_mode='valid')) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, init='normal'))
model.add(Activation('relu'))

model.add(Dense(256, init='normal'))
model.add(Activation('relu'))

model.add(Dense(2, init='normal'))
model.add(Activation('linear'))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.load_weights("/home/projectweights.h5")
print "Model Loaded"


#def Q(state,actions):
#	r=np.zeros(2)
#	r[0]=getreward(state,actions[0])
#        r[1]=getreward(state,actions[1])
#	return r
for i in range(13):
	print np.argmax(model.predict(read(str(i))))
#print model.predict(read('1'))
