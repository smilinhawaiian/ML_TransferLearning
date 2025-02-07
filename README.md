# ML_TransferLearning
_Sharice Mayer, Cyrus Rice, Lane Barton, Shehab Ibrahim, David Djernaes, Marina Neumann_ 

Project to evaluate Transfer Learning Potential between MNIST/NIST/ImageNet data sets 


## Transfer Learning using a Multilayer Neural Network
### ML Spring 2020 Project Proposal


#### Tasks:
- Evaluate transfer learning as it applies to training a network to classify handwritten 
characters to digits, and digits to characters. (Transfer classification from one domain 
to a similar domain)
- Evaluate any patterns the pre-trained network may identify within an image set, if trained 
to classify handwritten characters or digits, and tested on a set of images. 
(Transfer classification from one domain to another)
- Evaluate the outcomes of transfer learning in our trained neural networks when processed 
both in a supervised and unsupervised learning setting for both the character to digit domains, 
as well as from the characters to the image domains.


#### Algorithms:
For this project, we have decided to implement a fully-connected multilayer neural network 
with the following characteristics:
- Feed forward 
- Back propagation
- Stochastic gradient descent
- Input layer: 784 units
- Minimum 1 hidden layer with n(64) hidden units 
- Possible Additional Test - 2 hidden layers with m(196) and n(64) units each respectively
- Output layer - k classifications(26 for chars, 10 for digits, test both for images)

The second portion of our program will be particularly written for our unsupervised learning 
setting with unlabeled target data:
- Using k-means to cluster and test the data for image/character classification. 
- Using PCA to project the data into a smaller subspace to generate a predictive model.

We plan to implement our code for our project using python. 


#### Datasets:
We will be using: 
- MNIST Database of handwriten digits 
  http://yann.lecun.com/exdb/mnist/
- NIST Special Database 19 - handprinted forms and characters dataset
  https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format/home
- ImageNet Image Database
  http://image-net.org/download-images


#### Hypothesis: 
We think that the Neural Network will learn to classify handwritten digits faster using 
a neural network already trained to classify handwritten characters than it would from scratch. 
We also think that the reverse should be true - a MLNN trained to classify handwritten digits 
will learn to classify handwritten characters faster than from scratch. We also think that we 
may see some interesting classification when we apply the trained network to a new domain - 
classifying images. 

It's possible that we may not see the "class" of classification right away by looking at the 
images in each class, but we do hope to see some patterns forming in terms of the network's 
classification. For example, we may perhaps see that the images recognized arches in photos 
and grouped them, or perhaps saw similar building structures and grouped them together, or 
perhaps similar people or animals may be grouped together. We aren't entirely sure what 
results we may get, but hope that we see some interesting or unexpected classification happening. 
Given time, we may also use K-means to classify each of the three datasets and compare 
the results with our Neural Network results. 

We also hypothesize that it is possible that the training from digits to characters, or 
characters to digits could possibly take longer(than it would from scratch), 
due to over-fitting or under-fitting of the features belonging to each dataset, 
or due to a network structure that is not quite compatible/tuned for the task. 

In addition, we believe that Multilayer Neural Network training is powerful enough 
in the supervised learning setting, but also that K-means clustering and PCA are effective 
at generating predictive models. We are curious to see how well the outcomes of the 
two methods compare to each other, particularly how transfer learning might be more 
effective in one learning setting more than another. 





