

pip install tensorflow  


# In[2]:


pip install streamlit


# In[3]:


pip install matplot


# In[4]:


pip install numpy 


# In[5]:


pip  install pandas


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import tensorflow as tf 

from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras import utils


# In[7]:


data_train_path= r"C:\Users\mrpra\Desktop\Fruits_Vegetables\train"
data_test_path= r"C:\Users\mrpra\Desktop\Fruits_Vegetables\test"
data_val_path= r"C:\Users\mrpra\Desktop\Fruits_Vegetables\validation"


# In[8]:


img_width = 180
img_height = 180


# In[9]:


data_train = tf.keras.utils.image_dataset_from_directory(
    data_train_path,
    shuffle=True,
    image_size=(img_width,img_height),
    batch_size=32,
    validation_split=False)


# In[10]:


data_train.class_names


# In[11]:


data_cat = data_train.class_names


# In[12]:


data_val = tf.keras.utils.image_dataset_from_directory(data_val_path,
                                                       image_size=(img_height,img_width),
                                                       batch_size=32,
                                                       shuffle=False,
                                                       validation_split=False)


# In[13]:


data_test = tf.keras.utils.image_dataset_from_directory(
data_test_path,
    image_size=(img_height,img_width),
    batch_size=32,
    shuffle=False,
    validation_split=False)


# In[14]:


plt.figure(figsize=(10,10))
for image, labels in data_train.take(1):
   for i in range(9):
       plt.subplot(3,3,i+1)
       plt.imshow(image[i].numpy().astype('uint8'))
       plt.title(data_cat[labels[i]])
       plt.axis('off')


# In[15]:


from tensorflow.keras.models import Sequential


# In[16]:


data_train


# In[17]:


model = Sequential([
    layers.Rescaling(1./225),
    layers.Conv2D(16,3,padding='same' , activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,padding='same' , activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,padding='same' , activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(128),
    layers.Dense(len(data_cat))
])


# In[18]:


model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


# In[19]:


epochs_size = 25
history = model.fit(data_train, validation_data=data_val,epochs=epochs_size)


# In[20]:


epochs_range = range(epochs_size)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range,history.history['accuracy'],label = 'Training Accuracy')
plt.plot(epochs_range,history.history['val_accuracy'],label = 'Validation Accuracy')
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range,history.history['loss'],label = 'Training loss')
plt.plot(epochs_range,history.history['val_loss'],label = 'Validation loss')
plt.title('loss')


# In[21]:


import tensorflow as tf
img_width = 180
img_height = 180

image = r"C:\Users\mrpra\Desktop\Fruits_Vegetables\Banana.jpg"
image =  tf.keras.utils.load_img(image , target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image)
img_bat = tf.expand_dims(img_arr,0)


# In[22]:


predict = model.predict(img_bat)


# ***down statement didn't understand***

# In[23]:


score = model.predict(img_bat)
score = tf.nn.softmax(score).numpy()


# In[24]:


print('Veg/Fruit in image is{} with accuracy of {:0.2f}'.format(data_cat[np.argmax(score)],np.max(score)*100))


# In[25]:


model.save('Image_classify.keras')


# In[ ]:




