# Image_Similarity_model

This model is trained using Convolutional Autoencoders and K-Nearest Neighbours in such a way that it can retrieve the similar images of the query image.

### Steps :
First, the **get_features** function takes a directory path as input and loads the images from that directory. It resizes the images to a target size of 256x256 pixels, converts them to arrays, and adds them to a list. The list of image arrays is then returned.

The **create_autoencoder_model** function builds an autoencoder model using the TensorFlow Keras library. It takes input parameters for image dimensions, number of color channels, and the size of the latent image. It defines the encoder and decoder parts of the autoencoder using convolutional and transpose convolutional layers, respectively. The model is then returned.

**Architecture of Convolutional Autoencoder :**

![image](https://github.com/DynamVraj/Image_Similarity_model/assets/99869914/bea4cc9f-f109-42c8-bd53-b9a159c4f749)

The **train_model** function trains the autoencoder model. It takes training and validation data, the number of epochs, learning rate, and batch size as input. It creates the autoencoder model using the create_autoencoder_model function and compiles it with the Adam optimizer and mean squared error loss. It then fits the model to the training data and validates it on the validation data. The training history is recorded and plotted, and the trained model is saved to a file.

After training the model, the **generate_features** function is called to create feature vectors for indexing. It loads the trained autoencoder model, extracts the encoder part of the model, and applies it to the training data to obtain the latent features. The indices and features are then stored in a dictionary and saved to a pickle file.

Next, the **get_most_similar_images** function calculates the Euclidean distance between a given source vector and a set of destination vectors. It returns the indices of the destination vectors with the least Euclidean distance to the source vector.

Finally, the **get_similar_images** function performs image similarity search. It loads the trained autoencoder model, extracts the encoder part, and applies it to the validation data to obtain the latent features. It then iterates over the test sample indices and finds the most similar images using the get_most_similar_images function. The search image and the resultant similar images are displayed using matplotlib.

### Output :

![image](https://github.com/DynamVraj/Image_Similarity_model/assets/99869914/3f134e5e-39be-4a95-b396-b5e56b41a8cf)
![image](https://github.com/DynamVraj/Image_Similarity_model/assets/99869914/ce3a0a0b-a5a7-4479-90e8-fb8f4784ede4)
![image](https://github.com/DynamVraj/Image_Similarity_model/assets/99869914/64af7f65-7490-47ee-ab21-826ff1240b9a)
