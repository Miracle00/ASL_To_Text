import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%

# Tahmin edilen harf indis cinsinden dönüyor o indisin hangi harfe karşılık geldiğini bulmak için oluşturuldu
arrays=["a","b","c","d","e","f","g","h","i","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y"]
cap = cv2.VideoCapture(0)

#Train ve test datamızı aldık.
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')

labels = train['label'].values

#%%
unique_val = np.array(labels)
np.unique(unique_val)

plt.figure(figsize = (18,8))
sns.countplot(x =labels)
#%%
# preprocessing yapmak için kategorik label kolonunu sildik.
train.drop('label', axis = 1, inplace = True)

images = train.values

images = np.array([np.reshape(i, (28, 28)) for i in images])
# Datayı 27455, 784 matrisine çeviriyoruz
images = np.array([i.flatten() for i in images])

# Label ı 27455, 24 matrisine çeviriyoruz
from sklearn.preprocessing import LabelBinarizer
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)

#%%

plt.imshow(images[0].reshape(28,28))

#%%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)

# Keras kütüphanesini implemente ediyoruz
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
batch_size = 128
epochs = 10
num_classes=24

# Datayı ölçeklendiyoruz
x_train = x_train / 255
x_test = x_test / 255


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Modelimizi yapılandırıyoruz
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))
model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=0.5,  
        zoom_range = 0.5,
        width_shift_range=0.5,  
        height_shift_range=0.5,  
        horizontal_flip=False,  
        vertical_flip=False)  

datagen.fit(x_train)
#%%
# Modelimizi eğitiyoruz
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)

#%%
# İndisleri kategorize ediyoruz
test_labels = test['label']

test.drop('label', axis = 1, inplace = True)

test_images = test.values
test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])

test_labels = label_binrizer.fit_transform(test_labels)

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
# Imageları test ediyoruz
y_pred = model.predict(test_images)

from sklearn.metrics import accuracy_score

accuracy_score(test_labels, y_pred.round())


#%%
while(1):
        
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    
    roi=frame[50:300, 50:300]        
    # Frame oluşturup resim karesi alıyoruz ve grayscale ediyoruz
    cv2.rectangle(frame,(50,50),(300,300),(0,255,0),0)    
    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    
    # resmimizi resize ediyoruz
    hsv=cv2.resize(hsv,(28,28))
    
    hsv=np.expand_dims(np.array(hsv),axis=2)
    
    hsv=np.expand_dims(np.array(hsv),axis=0)
    
    # Resize edilen resmi modelime tahmin için gönderiyoruz.
    y_predicated = model.predict(hsv)
    
    # Tahmin edilen sonucun icinde 1.0 olan kolon u buluyoruz.
    sonuc=y_predicated.flatten()
    a=0
    b=0
    for x in sonuc:    
        if x==1.0:
            b=a
            break
        a=a+1
        
    font = cv2.FONT_HERSHEY_SIMPLEX  
    # Belirlenen indisi arrayde bulup ekrana yazıyoruz
    cv2.putText(frame,arrays[b],(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    
    cv2.imshow('frame',frame)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
 
cv2.destroyAllWindows()
cap.release() 
    




