from skimage import io
from keras.layers import Dense, Conv2D, Flatten,AveragePooling2D,MaxPooling2D, Dropout
import numpy as np
from os import listdir
from keras.models import Sequential
from keras.optimizers import Adam,Nadam,SGD

#%%

direc=r'E:\Project\Captcha\data'
filenames=listdir(direc)
images=[]
labels=[]

for name in filenames:   
    labels.append(name[:5])
    temp=io.imread(name)
    temp1=temp[:,:,3]
    images.append(temp1)
    

id=0
letdic={}
for i in range(ord('a'),ord('z')+1):
    letdic[chr(i)]=id
    id+=1
for i in range(10):
    letdic[str(i)]=id
    id+=1
    

def labeltosevhot(arr):
    sevhot=[]
    for lab in arr:
        tempar=[]
        for c in lab:
            tempp=[0 for x in range(36)]
            tempp[letdic[c]]=1
            tempar+=tempp     
        
        sevhot.append(tempar)         
            
    return sevhot

def sevhottolab(hotlab):
    st=''
    for i in range(5):
        let=hotlab[:36]
        hotlab=hotlab[36:]
        
        ind=let.index(1)
        ch=[chara for chara,enc in letdic.items() if enc==ind][0]   
        st+=ch
        
    return st 
        

labels=labels[6800:]        
  
      
   
"""
def showsomeim(n):
    io.imshow(images[n])
    print(sevhottolab(labelsenc[n]))
"""

images=images[6800:]



direc=r'E:\Project\Captcha\data10k'
filenames=[]
filenames=listdir(direc)

for name in filenames:
    namepath=direc+'\\'+name
    labels.append(name[:5])
    temp=io.imread(namepath)
    temp1=temp[:,:,3]
    images.append(temp1)

labelsenc=labeltosevhot(labels)


xtrain=np.reshape(images,(len(images),50,200,1))
ytrain=np.reshape(labelsenc,(len(labels),180))
xtrain=xtrain.astype('float')
xtrain/=255
del images,labels,labelsenc
#%%


images=[]
labels=[]

direc=r'E:\Project\Captcha\data5k'
filenames=[]
filenames=listdir(direc)

for name in filenames:
    namepath=direc+'\\'+name
    labels.append(name[:5])
    temp=io.imread(namepath)
    temp1=temp[:,:,3]
    images.append(temp1)

labelsenc=labeltosevhot(labels)


xtest=np.reshape(images,(len(images),50,200,1))
ytest=np.reshape(labelsenc,(len(labels),180))
xtest=xtest.astype('float')
xtest/=255
del images,labels,labelsenc

#%%

from keras.models import load_model

model=load_model(r'E:\Project\Captcha\captchamodelverygood.h5')

#%%
model.fit(xtrain,ytrain,validation_data=(xtest,ytest),batch_size=64,shuffle=True,epochs=50)
#%%
import random

n=random.randint(0,4800)

io.imshow(np.reshape(xtest[n],(50,200)))
var=sevhottolab(np.ndarray.tolist(ytest[n]))
print(sevhottolab(np.ndarray.tolist(ytest[n])))

#%%
from math import exp

def sevhottolab2(hotlab):
    st=''
    for i in range(5):
        let=hotlab[:36]
        hotlab=hotlab[36:]
        esum=sum([exp(x-max(let)) for x in let])
        newar=[exp(x-max(let))/esum for x in let]
                    
        #print(max(newar))
        
        ind=newar.index(max(newar))
        ch=[chara for chara,enc in letdic.items() if enc==ind][0]   
        st+=ch
        del let,newar,esum
    return st 


testpreds=[]

for ii in range(len(xtest)):
    ar=np.ndarray.tolist(model.predict(np.reshape(xtest[ii],(1,50,200,1))))
    testpreds.append([sevhottolab(np.ndarray.tolist(ytest[ii])),sevhottolab2(ar[0])])
    print(ii)


#%%

ara=[1 if x[0]==x[1] else 0 for x in testpreds]

ar2=[1 if x[0]==x[1] else 0 for x in trainpreds]


onlywrong=[  [x,y] for x,y  in zip(testpreds,ara) if y==0 ]