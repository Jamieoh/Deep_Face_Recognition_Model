# MNIST Dataset Analysis using Keras

Completed as a group assignment during the MSc Computer Science - Data Analytics with Kevin Derrane.

The following repo contains the Python code for building a Deep Face Recognition Model that aimed to classify facial images from the MNIST Dataset. Analysis conducted using Keras.

## Objectives:

- Part 1: Program a network in Keras with a triplet loss function.
- Part 2: Train the network on the MNIST data
- Part 3: Report the value of the loss function over time.
- Part 4: Program a recognition function: given a new image, it should recognize the image as a digit (and which digit) or report (e.g. if the image is a letter, instead of a - - Part 5: Test the network with (unseen) images of letters (not numbers), to demonstrate that it works in this case.
- Part 6: Find any pre-trained deep convolutional neural network face model on the internet which uses an embedding, a Siamese network, or a similar approach which allows you to carry out the final tasks as follows.
- Part 7: Used the output model (possibly after removing some final layers) to make face clusters, i.e. ran a clustering algorithm so that multiple pictures of the same person are in the same cluster. This can run on any face dataset, whether open/public. For each image in the dataset we clustered, we pre-computed its embedding, and then worked with the embeddings. We didn't run all the images through the network at every step of the clustering algorithm.

 - Please see the ipynb file for our VGG Face model. 
 
## Model overview
- Our model has 22 layers and 37 deep units.
- We used the Sequential Keras model, which is a linear stack of layers
- Firstly, weefined the input shape. The images were 224x224x3 sized images, the number three is the RGB color of the input image.
- Secondly, we loaded the weights, defined the depiction and preprocessed the image.
- Thirdly, we compared the vector representations to see if the distance between the Cosine and Euclidiean distance is low so we can interpret that the images are of the same person.
- Fourthly, we clustered the images that were similar or of the same person with each other using the DBSCAN (Density-based spatial clustering of applications with noise) algorithm from scikit-learn.
- We then prooceeded to identify the number of unique faces we have. Result = 2.
- Finally, we plotted the clusters by looping throught the two face ID's.
