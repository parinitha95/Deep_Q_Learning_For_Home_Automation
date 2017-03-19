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
print "Model Loaded"

#def Q(state,actions):
#	r=np.zeros(2)
#	r[0]=getreward(state,actions[0])
#        r[1]=getreward(state,actions[1])
#	return r

terminated=0
i=0
s="0"
r=0
gamma=0.9
while(1):
    if(i==0):
        fetchSignal=raw_input("Can I fetch the photo?") 
    	#makemove(st)
    	st = st+1
    	st=st-1
    #print model.predict(read(s)) 
    if(i<=10): 
        if(random.uniform(0.0,1.0)<0.35):
            action=random.randint(0,1)
        else:
            action=np.argmax(model.predict(read(str(s))))
    else:
        action=np.argmax(model.predict(read(str(s))))
    print "Initial State: %s" %s
    print "Action taken is: "
    if(action==0):
    	print "LIGHTS_OFF"
    else:
    	print "LIGHTS_ON"	 	  
    r = int(raw_input("Enter the reward: "))
    #print "Action taken: %d" %action  
    fetchSignal=raw_input("Can I fetch the photo?") 
    #r=getreward(str(s),action)
    #snew=makemove(str(s),action)
    #snew=makemove(st)
    snew = st+1
    st=st+1 
    print snew
    #print "Reward : %d" %r
    if(i==0):
        repD=np.array([[int(s),action,r,int(snew)]])
    else:
        repD=np.concatenate((repD,np.array([[int(s),action,r,int(snew)]])))
    #print ("State %s action %d reward %d new state %s" %(s,action,r,snew))
    #print repD  
    index=random.randint(0,np.shape(repD)[0]-1)
    #print "training"
    #print index
    #print ("S	tate %d action %d reward %d new state %d" %(repD[index][0],repD[index][1],repD[index][2],repD[index][3]))
    print "Random set %d %d %d %d" %(repD[index][0],repD[index][1],repD[index][2],repD[index][3])
    target=model.predict(read(str(repD[index][0])))
    print "initial target:"
    print target
    if(repD[index][3]==1):
        target[0][repD[index][1]]=repD[index][2]
    else:
        target[0][repD[index][1]]=repD[index][2]+gamma*max(model.predict(read(str(repD[index][3])))[0])
    #loss=(tt-getreward(repD[index][0],repD[index][1]))**2
    s=str(snew)
    print target
    i=i+1
    model.fit(read(str(repD[index][3])),target,nb_epoch=1)
    if(i==50):
        print model.predict(read('0'))
        print model.predict(read('1'))
        model.save_weights("/home/projectweights.h5",overwrite=False)
        break 
