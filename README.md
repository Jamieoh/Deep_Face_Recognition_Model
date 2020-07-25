# MNIST_Dataset_Analysis_using_Keras
The following repo contains the Python code for an assignment which was based on conducting specific analysis using Keras on the MNIST Dataset.

# 1. (Part 1): Program a network in Keras with a triplet loss function.

## Steps

- Choose an image from our training or test data with the given label.

- Choose a triplet (anchor, positive, negative) of images such that anchor and positive have the same label and anchor and negative have different labels.

- Generate an un-ending stream (ie a generator) of triplets for training or test.

- This loss function just takes the mean of y_pred. Because of the way we wire the network (see complete_model below), y_pred is the output of the triplet loss, so minimising it is what we want to do.

- The triplet loss is ||A - P|| - ||A - N|| + alpha, where ||.|| is the Euclidean norm. Notice that this is not a loss function in the format expected by Keras, ie f(y_true, y_pred).

- A tiny model similar to the network we used for MNIST classification. We assume the architecture should be good for MNIST embedding. Its input is an image and output is an embedding, not a classification, so the final layer is not a softmax. We don't compile or add a loss since this will become a component in the complete model below.

- This part of the model is quite tricky. Rather than a Sequential model, we declare a Model and say which are its inputs and outputs, and we declare how the outputs are calculated from the inputs. In particular, there are no layers in this model, *other than* the layers in the embedding model discussed above.

- A further complication is that our triplet loss can't be calculated as a function of y_true and y_predicted as usual. Instead we calculate the triplet loss as an extra Lambda layer. Then the Model's loss is set to be equal to the triplet loss via the identity function.

# 2. (Part 2): Train the network on the MNIST data

# 3. (Part 3): Report the value of the loss function over time.

- Used a plot to model loss over time acorss 5 epochs.

-  added an extra 28 pixels to allow for images whose bottom-left is at the top or right border.

# 4. (Part 4): Program a recognition function: given a new image, it should recognize the image as a digit (and which digit) or report (e.g. if the image is a letter, instead of a digit) that it is unknown.

- Reload the datasets to make own database for our digit recognition. 

- Convert to encoding for the digits.

# 5 (Part 5): Test the network with (unseen) images of letters (not numbers), to demonstrate that it works in this case.

- Load unseen data and encode

# 6. (Part 6): Find any pre-trained deep convolutional neural network face model on the internet which uses an embedding, a Siamese network, or a similar approach which allows you to carry out the final tasks as follows.

- For the following, i had the files vgg_face_weights.h5 and the six images ('joly1.jpg','s1.jpg','joly2.jpg','joly3.jpg','s3.jpg','s2.jpg') stored on my google drive. To run these just store them in the directory and remove the "gdrive/My Drive/". The links for these files are the following:

- vgg_face_weights.h5: https://drive.google.com/uc?id=1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo&export=download
- images.zip: https://drive.google.com/file/d/1p8QlRSm4f7muWCSKM8R-KlsNMfCZMLf1/view?usp=sharing

# 7 (Paer 7): Used the output model (possibly after removing some final layers) to make face clusters, i.e. ran a clustering algorithm so that multiple pictures of the same person are in the same cluster. This can run on any face dataset, whether open/public. For each image in the dataset we clustered, we pre-computed its embedding, and then worked with the embeddings. We didn't run all the images through the network at every step of the clustering algorithm.

 - Please see the ipynb file for our VGG Face model. It is based off of the following pdf with help from the GITHUB below.
- http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf
- https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

## Model overview
- Our model has 22 layers and 37 deep units.
- We used the Sequential Keras model, the Sequential model is a linear stack of layers
- Defined the input shape. It is 224x224x3 sized images, where the number three is the RGB color of the input image
- We then loaded the weights, defined the depiction and preprocessed the image.
- We then compared the vector representations to see if the distance between the Cosine and Euclidiean distance is low we can say the images are of the same person.
- We clustered the images that were similar or of the same person with each other using the DBSCAN (Density-based spatial clustering of applications with noise) algorithm from scikit-learn.
- Calculate the number of unique faces we have. Result = 2.
- Finally, we plotted the clusters by looping throught the two face ID's.
