# Real-Time-Sign-Language-Recognition-Using-Machine-Learning
ABSTRACT
Sign Language Recognition (SLR) systems have emerged as a breakthrough in facilitating communication between individuals with speech and hearing impairments and those without.The proposed method involves filtering the hand gesture and applying a classifier to predict the corresponding ASL fingerspelling class. The experimental results demonstrate a high accuracy of 95.7% for the recognition of the 26 letters of the alphabet. This approach provides a promising avenue for improving communication between deaf-mute individuals and the general population, leveraging the power of neural networks and image processing techniques.
STEPS OF BUILDING THE PROJECT

STEP 1 - Preparing and Cleaning Data Set

To prepare and clean the dataset, First we have read two CSV files containing sign language images and their corresponding labels, and we have stored them as Pandas Data Frames. Then we have extracted the labels from the Data Frame and stored them separately in the label’s variable.
To prepare and clean the dataset, First we have read two CSV files(MNIST DATA SET TAKEN FROM KAGGLE) containing sign language images and their corresponding labels, and we have stored them as Pandas Data Frames. Then we have extracted the labels from the Data Frame and stored them separately in the label’s variable.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
test= pd.read_csv(r"C:\Users\user1\Downloads\archive\sign_mnist_test.csv")
train = pd.read_csv(r"C:\Users\user1\Downloads\archive\sign_mnist_train.csv")
train.head()
labels = train['label'].values
unique_val = np.array(labels)
np.unique(unique_val)
plt.figure(figsize = (18,8))
sns.countplot(x = labels)
train.drop('label', axis = 1,inplace = True)
images = train.values
images = np.array([np.reshape(i,(28,28))for i in images])
images = np.array([i.flatten() for i in images])
from sklearn.preprocessing import LabelBinarizer
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)
labels
index = 2
print(labels[index])
plt.imshow(images[index].reshape(28,28))
import cv2
import numpy as np
for i in range(0,10):
    rand = np.random.randint(0, len(images))
    input_im = images[rand]
    sample = input_im.reshape(28,28).astype(np.uint8)
    sample = cv2.resize(sample, None, fx=10, fy=10, interpolation = cv2.INTER_CUBIC)
    cv2.imshow("sample image",sample)
    cv2.waitKey(0)
cv2.destroyAllWindows()
```
STEP 2 - SPLITTING THE DATA

We have split the preprocessed images and labels arrays into training and testing subsets.
30% of the examples will be used for testing and 70% for training.

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
batch_size = 128
num_classes = 24
epochs = 10
```
STEP 3 - SCALING AND RESHAPING
```python
x_train = x_train / 255
x_test = x_test / 255
x_train  = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test  = x_test.reshape(x_test.shape[0], 28, 28, 1)
plt.imshow(x_train[0].reshape(28,28))
```
STEP 4 - BUILDING A CNN MODEL
```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras import backend as K 
from tensorflow.keras.optimizers import Adam
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28,1))) 
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense (128, activation = 'relu'))
model.add(Dropout (0.20))
model.add(Dense (num_classes, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy',
              optimizer= Adam(),
              metrics=['accuracy'])
```
STEP 5 - TRAINING CNN MODEL
```python
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)
```
STEP 6 - SAVING CNN MODEL
```python
model.save("sign_mnist_cnn_50_Epochs.h5")
print("Model Saved")
```
STEP 7 - ACCURACY
```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'])
plt.show()
```

STEP 8 - TESTING THE MODEL
```python
test_labels = test['label']
test.drop('label', axis = 1, inplace = True)
test_images = test.values
test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])
test_labels = label_binrizer.fit_transform(test_labels)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
test_images.shape
y_pred = model.predict(test_images)
```

STEP 9 - TESTING ACCURACY
```python
from sklearn.metrics import accuracy_score
accuracy_score(test_labels, y_pred.round())
```
STEP 10 - LABELLING 
```python
def getLetter(result): 
    classLabels = { 0:'A',
                    1:'B',
                    2:'C',
                    3:'D',
                    4:'E',
                    5:'F',
                    6:'G',
                    7:'H',
                    8:'I',
                    9:'K',
                    10:'L',
                    11:'M',
                    12:'N',
                    13:'0', 
                    14:'P',
                    15:'Q',
                    16:'R',
                    17:'S',
                    18:'T',
                    19:'U',
                    20:'V',
                    21:'W',
                    22:'X',
                    23:'Y'}
    try:
        res = int(result)
        return classLabels[res]
    except:
      return "Error"
```

STEP 11 - PREDCTING ON LIVE WEBCAM
```python
cap = cv2.VideoCapture(0)
while True:
    
    ret, frame = cap.read()
    roi = frame[100:400, 320:620] 
    cv2.imshow('roi', roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28, 28), interpolation = cv2.INTER_AREA)
    cv2.imshow('roi sacled and gray', roi)
    copy = frame.copy() 
    cv2.rectangle(copy, (320, 100), (620, 400), (255,0,0), 5)
    roi = roi.reshape(1,28,28,1)
    result = str(np.argmax(model.predict(roi, 1, verbose = 0),axis=1)[0])
    cv2.putText(copy, getLetter(result), (300 , 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('frame', copy)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
cap.release()
cv2.destroyAllWindows()
```
TOOLS REQUIRED 

Python 3.10, Tensorflow(present version),Jupyter notebook and required libraries