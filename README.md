# Fruit-and-Vegetable-Image-Classifier
This repository contains a deep learning model built using TensorFlow and Keras for classifying images of fruits and vegetables. The classifier is trained on a dataset of labeled images and can identify the type of fruit or vegetable in an input image with a confidence score.

How it works :-
1. Data Preprocessing:-The dataset is organized into three directories: training, validation, and testing, each containing subfolders for different categories (fruits/vegetables).
The images are loaded and preprocessed using image_dataset_from_directory, which generates batches of image data along with their respective labels. Each image is resized to 180x180 pixels.

2. Model Architecture:-The model is a Convolutional Neural Network (CNN) built using the Sequential API from TensorFlow/Keras.
Key layers in the model:
Rescaling Layer: Normalizes the pixel values to a range of [0, 1] by rescaling them.
Convolutional Layers: Three convolutional layers with increasing filters (16, 32, 64) and ReLU activation functions are used to capture spatial features from the images.
MaxPooling Layers: Applied after each convolutional layer to reduce the spatial dimensions.
Flatten Layer: Converts the 2D outputs from the convolutional layers into a 1D vector.
Dropout Layer: Introduced to prevent overfitting by randomly dropping 20% of the neurons during training.
Dense Layers: Fully connected layers to perform the final classification. The last dense layer outputs predictions for each class (one class per type of fruit or vegetable).

3. Training:-The model is compiled using the Adam optimizer and Sparse Categorical Crossentropy as the loss function since this is a multi-class classification problem.
The model is trained for 25 epochs on the training data, while the validation data is used to monitor performance. The metrics tracked are accuracy and loss for both training and validation sets.

4. Visualization:-After training, accuracy and loss for both training and validation sets are plotted to visualize the model's performance over the epochs.

5. Model Prediction:-A test image (e.g., a banana) is loaded and preprocessed, and the model predicts which fruit or vegetable it is. The model outputs a confidence score for each class, and the class with the highest score is selected as the predicted category.

6. Model Saving:-The trained model is saved as a .keras file using model.save('Image_classify.keras') for future use.

This repository can be used to train and deploy an image classifier for fruits and vegetables, making it suitable for applications such as grocery store automation, farming, or educational tools.
