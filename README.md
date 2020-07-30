# Handwritten_Digit_Recognizer-DL

ALL the notebooks are implemented in GOOGLE COLAB(free cloud service and  requires no setup to use specifically for ML projects.)

This is a getting started on computer vision [challenge on kaggle](https://www.kaggle.com/c/digit-recognizer/notebooks?sortBy=voteCount&group=everyone&pageSize=20&competitionId=3004)

The dataset is available at [Kaggle](https://www.kaggle.com/c/digit-recognizer/data)(both train and test csv files)

The prediction on test data after traning the model of keras CNNis in the file(cnn_mnist_datagen (1).csv)            

## Task-Correctly identify digits from a dataset of tens of thousands of handwritten images.(MNIST datset)

> We will use various machine learning models to see how accurate they are in recogninzing the wriiten digit.

* The models explored are using keras CNN(including  setting the optimizer and annealer and  Data augmentation) is implemented in notebook CNN_keras.ipynb

* Neural Network using SGD,MOMENTUM,L2 REGULIZER in the training process(Using_optimizers.ipynb)

* Using Random forest classification , Principal component analysis (PCA) + k-nearest neighbours (kNN) for the MNIST classification problem(PCA,_forest_.ipynb)

* RNNs and LSTM implemented in pytorch framework(PyTorch_RNNs_and_LSTMs.ipynb)

* Using most commonly applied  techniques of deep neural networks - VGG,Inception,Xception((VGG,Inception,Xception).ipynb) and resnet(resnet.ipynb)

## Overall outline of Keras CNN:(build it with keras API (Tensorflow backend) which is very intuitive)  
* Data preparation
  1 Load data
  2 Check for null and missing values
  3 Normalization
  4 Reshape
  5 Label encoding
  6 Split training and valdiation set
* CNN
  1 Define the model
  2 Set the optimizer and annealer
  3 Data augmentation
* Evaluate the model
   Training and validation curves
   Confusion matrix
* Prediction on test data

**CNN** - The modelling steps are:          
* Set the CNN model - my CNN architechture is In -> ((Conv2D->relu)*2 -> MaxPool2D -> Dropout)*2 -> Flatten -> Dense -> Dropout -> Out
* Set the optimizer and annealer - RMSprop (with default values), it is a very effective optimizer with loss function "categorical_crossentropy" and Set a learning rate annealer by using ReduceLROnPlateau function
* Data augmentation by using function ImageDataGenerator and create a very robust model which can improve accuracy.

> Without data augmentation i obtained an accuracy of 98.114%
> With data augmentation i achieved 99.67% of accuracy
 
## Outline of simple Fully Connected Neural Network using Stochastic Gradient Descent,Momentum and L2-regulizer in training:

* Data exploration
  * import the dataset
  * visualization of the data
  * count of examples per digit
  * normalization
  * one hot encoding
* Describe the model
  * activation functions
  * creating weights,dropout, predict,metrics,cross - validation
* Training
  * sgd (stochastic gradient descent)
  * momentum
  * l2 regulizer
 
* The accuracy score is  0.98.                       

**We'll use the ReLu (Rectifier Linear Function), this function is similar to the functioning of the biological neuron, and allow us to run the model more fast and the Softmax function allow us predict the model, because it normalize the data in one hot encoding.**           

**We will use a Neural Network with 3 layer, 800-300 Hidden Units with Softmax Classification, Cross-Entropy as Loss Function and weight decay.**            

###### TRAINING             
1.SGD (Stochastic Gradient Descent)- is a optimizer used for fit the neural network, this technique is based by Gradient Descent.                                                  2.MOMENTUM - SGD in training have a lot of oscillations,momentum term is used for soft the oscillations and accelerates of the convergence.                                  
3.L2 REGULIZER - Technique which is used for penalize the higher weights.

**The third model is Comparing random forest, PCA and kNN for analysis of the MNIST dataset by examining number of components in PCA vs the variance feature and number of estimators vs the accuracy using sklearn functions - RandomForestClassifier, KNeighborsClassifier,PCA.**

**The accuracy score in PCA+KNN is 0.97 and slightly worse score (0.96) for random forest.**

# PyTorch RNNs and LSTM

## Overview
 * 1 Layer RNNs
 * Multiple Neurons RNN
 * Vanilla RNN for MNIST Classification
 * Multilayer RNNs
 * Tanh Activation Function
 * Multilayer RNN for MNIST
 * LSTMs and Vanishing Gradient Problem
 * Bidirectional LSTMs
 * LSTM for MNIST Classification 
 
###### Accuracy score results:
* RNN Architecture for MNIST Classification is 0.96
* Multilayer RNN-0.972
* LSTM (Long Short Term Memory RNNs): 0.991 :fire:

**Resnet keras(Using TensorFlow backend) - Resnets (Residual Networks) are used when we need to train very deep neural networks.To avoid Vanishing Gradient problem,we use residual network.**                     

**Building our own Resnet from scratch including data augmentation and callbacks.**                                 
**Accuracy score - 0.994 :fire: :fire:**

In all of the models implemented, it's AMAZING how important hyperparameters are. Try changing the learning_rate to 0.01 and see what happens. Also try changing the batch_size to 20 instead of 64. Try adding weight_decay to the optimizer functions. The accuracy of the model will be improved , but by altering some of these hyperparameters can change this "sweet spot" we found instantly.

## References:
[Yassine Ghouzam blog](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6), 
[Andrada Olteanu-Master in kaggle](https://www.kaggle.com/andradaolteanu/pytorch-rnns-and-lstms-explained-acc-0-99), 
[Guide to RNN](https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9), 
[Guide to LSTM](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21), 
[Jean Carlo Codogno-github](https://www.kaggle.com/jcodogno/neural-network-using-sgd-98-9), 
[Kaggle notebook on PCA,forest](https://www.kaggle.com/sflender/comparing-random-forest-pca-and-knn), 
[rishabhdhiman-github](https://www.kaggle.com/rishabhdhiman/resnet-keras-lb-0-997)

## Support 

If you like this repo and find it useful, please consider (★) starring it (on top right of the page) so that it can reach a broader audience.


