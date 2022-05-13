


List of Topics:


## 1. Intro to ML : regression, k-Means, PCA
- Unsupervised(clustering, PCA/dimensionality reduction) VS supervised(regression,classifications) learning

-  ***Supervised***
-  Linear regression/multiple linear regsression - hypothesis, cost , weights and bias
-  Non-linear regression 

-  Logistic regression  - hypothesis, cost , weights and bias
-  Non-linear logistic regression (half-moon dataset, MNIST Dataset)

-  ***Unsupervised***
-  k-Means - centroids, initalization
-  PCA

### Steps in ML
- data cleaning
- data exploration
- converting to neural network(using torch tensors)
- select hyperparameters - lr, (I,H,O), epochs, batch sizes, momentum
- select algorithm
- calculate loss function
- make a forward pass
- backpropogation
- updates the parameters
- 
## Concepts
- Derivations - backpropogation(chain rule) , forwardpass
- Standardization/ Uniformity of a NN
- Loss Functions(mean squred error, binary cross entropy,categorical cross entropy)
- Activation Functions(sigmoid(not centered around zero), tanh, ReLu(dead neurons), Leaky ReLU, ELU, Softplus ) - vanishing gradient problem
- Gradient Descent(SGD, MBGD, BGD, Adam) - momentum, variable and adaptive learning rates
- Hyper Parameter Tuning

## Pytorch
- Important Functions (tensor, nn/modules, optim, autograd, distributions, utils, torchvision)
- Basic Pytorch Tensor Operations (setting seed, declaring tensors,creating random tensor, plotting tensors, using gpus, using autograd, using optim)
- Autograd
- Backpropogation
- Gradient descent 
- Training Pipeline
- Linear Regression
- Logistic Regression
- Dataset and Dataloader
- Dataset Transforms
- Softmax and Crossentropy
- Activation Functions
- Feed Forward Net
- CNN
- Transfer Learning
- Tensorboard
- Save & Load Models



## 2. Feed-Foward Neural Networks

## 3. Regularization, Bias, Variance

## 4. Convolutional Neural Networks

- CNN artchitecture - layers, kernals, filters, channels
- convolutional arithematic - output size, output image
- Layers - convolutional layer, pooling layer, upsampling, downsampling 
- Transposed Convolutions
- Transfer Learning

Steps in Creating a CNN (LeNet5)
- Downalod the dataset 
- Split the Data - get the train validation and test - indices and data and labels
- Create Standardization function
- Apply Standardizarion
- Create tensorDatasets - TensorDataset to be able to operate on the dataset without having to load it all in memory

- Making the ML Functions :
    - Training Function
    - Validation Function
    - Evaluate Function

## 5. Recurrent Networks and LSTMs 




One-hot encoding

One-hot encoding
Some of these data types are clearly categorical. In order to use categorical features in ML models we would have to one-hot encode them. This is easy enough to do using the pd.get_dummies() function. For more information, here is the documentation.