# -*- coding: utf-8 -*-
import torch
print(torch.__version__)

"""# Autograd

PyTorch uses a technique called automatic differentiation. It records all the operations that we are performing (computational graph) and replays it backward to compute gradients

The autograd package provides automatic differentiation for all operations on Tensors
"""

# requires_grad = True -> tracks all operations on the tensor.
torch.manual_seed(42)

x = torch.randn(3, requires_grad=True)
print(x, '\n')

y = x + 2  #Since y was created as a result of an operation, it has a grad_fn attribute.


print(y)

#print(y.grad_fn)

#Further operation
z = y * y * 3
print(z, '\n')

h = z.mean()
print(h)

"""To  compute the gradients we can call ***.backward()*** and have all the gradients computed automatically.

The gradient for this tensor will be accumulated into .grad attribute.
It is the partial derivate of the function w.r.t. the tensor.
"""

# copute dh/dx
h.backward() #if you run this cell twice will receive and error, since we will be trying to backward through the graph a second time

#display the values of the gradient
print(x.grad)

"""### Exercise

Try to compute this gradient manually and copare the result with x.grad

"""

## WRITE YOU CODE HERE
2*(x+2)

"""### Note:
* During the forward pass, PyTorch computes the forward operations and builds a computation graph.
* During the backward pass (backpropagation), PyTorch uses this graph to compute gradients by applying the chain rule.
* By default, PyTorch only retains gradients for leaf nodes in the computation graph (input tensors or parameters with requires_grad=True). Intermediate tensors, unless specified otherwise, have their gradients discarded after the backward pass to save memory.

`retain_grad()` method is used to indicate that gradients for a particular tensor should be retained during backpropagation, even if the default behavior is to discard them

#### Illustration
"""

x = torch.tensor([2.0], requires_grad=True)
y = x * 3
z = y**2

# Retain gradients for the intermediate tensor 'y' i.e the derivative w.r.t  y
y.retain_grad()

# Perform more operations
w = z.sum()

# Perform backward pass
w.backward() # dw/dx=dw/dz * dz/dy * dy/dx

# Access gradients
print(x.grad)  # Gradient for x
print(y.grad)  # Gradient for y (retained) = dz/dy=2*y

"""## Zeros gradient

backward() accumulates the gradient for this tensor into .grad attribute. We need to be careful during optimization !!! Use .zero_() to empty the gradients before a new optimization step so that the parameter will be updated correctly. Otherwise, the gradient would be a combination of the old gradient, which we have already used to update our model parameters, and the newly-computed gradient. It would therefore point in some other direction than the intended direction towards the minimum
"""

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    # just a dummy example
    model_output = (weights*3).sum()
    model_output.backward()

    print(weights.grad)

# print(weights)
# print(model_output)

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    # just a dummy example
    model_output = (weights*3).sum()
    model_output.backward()

    print(weights.grad)

    # this is important! It affects the final weights & output
    weights.grad.zero_()

# print(weights)
# print(model_output)

"""## Backpropagation

Note:
Three options to stop/prevent pytorch for tracking the gradient for some operation.

for example during the training when we want to update the weights, this operation should not be part of the gradient computation
- option 1: `requires_grad_(False)`
- option 2: `detach()`, this will create a new tensor that doesn't require the gradient
- option 3: is to wrap it into a with statement: `with torch.no_grad():`

#### Illustration
"""

x = torch.tensor(1.0)
y = torch.tensor(2.0)

# This is the parameter we want to optimize -> requires_grad=True
w = torch.tensor(1.0, requires_grad=True)

# forward pass to compute loss
y_predicted = w * x

loss = (y_predicted - y)**2
print(loss)

# backward pass to compute gradient dLoss/dw
loss.backward()
print(w.grad)


# update weights, this operation should not be part of the computational graph
with torch.no_grad():
    w -= 0.01 * w.grad
# don't forget to zero the gradients
w.grad.zero_()

"""## Pytorch optim module

The Optim module in PyTorch has pre-written codes for most of the optimizers that are used while building a neural network. We just have to import them and then they can be used to build models.
"""

# importing the optim module
from torch import optim

# sgd
## SGD = optim.SGD(model.parameters(), lr=learning_rate)

# adam
## adam = optim.Adam(model.parameters(), lr=learning_rate)



# During training:
# optimizer.step(): to update the weights
# optimizer.zero_grad(): to zero the gradients

"""## Torch nn Module

It provides an easy and modular way to build and train simple or complex neural networks using Torch:




*   Simple layers: nn.Linear
*   Convolutional layers: nn.Conv1D, nn.Conv2D, ...
*   Pooling layers: nn.MaxPool1d, nn.MaxPool2d, ....
*   Criterion: nn.MSELoss, nn.CrossEntropyLoss
*   Activation functions:  nn.ReLU, nn.Sigmoid, ...
nn.RNN, nn.LSTM
*   ....







"""

import torch.nn as nn

"""# Linear regression with pytorch"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

"""### Prepare data"""

#Generate a regression dataset using sklearn
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

# cast to float Tensor
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

print(n_samples, n_features)

plt.scatter(X,y)

"""### Model"""

#Linear model f = wx + b
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

model

for name, w in model.named_parameters():
  print(name)
  print(w)

"""### Loss and optimizer"""

learning_rate = 0.01

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

"""### Training"""

num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass and loss
    y_predicted = model(X)

    ##compute the loss between your prediction and the true y
    loss = criterion(y_predicted,y)

    # Backward pass
    loss.backward()

    ## update parameters
    optimizer.step()

    # zero grad before new step
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# Plot
predicted = model(X).detach().numpy()

plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()

"""# Exercise: Implement Logistic regression using PyTorch"""

## FILL THE MISSING CODE ##

# Step 1: Prepare the dataset (We are going to use the breast cancer dataset- a binary classification problem)

bc = datasets.load_breast_cancer()
X,y = bc.data, bc.target

#Get the number of features and the number of samples
samples, features = X.shape

#Scale X to have 0 mean and unit variance

X_centered = (X - X.mean(axis=0))/np.std(X, axis=0)

#Split the data into train and test sets (X_train, y_train, X_test, y_test)
def train_test_split(X,y):
  '''
  this function takes as input the sample X and the corresponding features y
  and output the training and test set
  '''
  np.random.seed(0)

  train_size = 0.8
  n = int(len(X)*train_size)
  indices = np.arange(len(X))
  np.random.shuffle(indices)
  train_idx = indices[: n]
  test_idx = indices[n:]
  X_train, y_train = X[train_idx], y[train_idx]
  X_test, y_test = X[test_idx], y[test_idx]

  return X_train, y_train, X_test, y_test

X_train, y_train, X_test,y_test= train_test_split(X_centered,y)


# Convert X_train, y_train, X_test, y_test into tensor
X_train1 = torch.from_numpy(X_train.astype("float32"))
y_train1 = torch.from_numpy(y_train.astype("float32"))
X_test1 = torch.from_numpy(X_test.astype("float32"))
y_test1 = torch.from_numpy(y_test.astype("float32"))

# Reshape y_train and y_test if needed

y_train1 = y_train1.reshape(-1,1)
y_test1 = y_test1.reshape(-1,1)

X_train.shape

#Step 2: Design the model

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features,1)

    def forward(self, x):
        # use torch.sigmoid to get the prbabilities values
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegression(features)

#Step 3: Loss and optimizer
num_epochs = 200
learning_rate = 0.01
criterion = nn.BCELoss() #Check online how to define the binary cross entropy using nn
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)#use SGD

# Step 4: Training loop

for epoch in range(num_epochs):

    # Forward pass and loss
    y_pred = model(X_train1)
    loss = criterion(y_pred,y_train1)

    # Backward pass
    loss.backward()

    # update
    optimizer.step()

    # zero grad before new step
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')


with torch.no_grad():
    y_predicted = model(X_test1)
    y_predicted_cls = [0 if p <=0.5 else 1 for p in y_predicted] #use threshold=0.5 to define the classes
    accuracy =( y_predicted_cls==y_test1.numpy().flatten()).sum() / y_test1.shape[0]
    print(f'accuracy: {accuracy.item():.4f}')

"""# Pytorch Dataset

PyTorch provides two data primitives: `torch.utils.data.DataLoader` and `torch.utils.data.Dataset` that allow us to use pre-loaded datasets as well as our own data. Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.

PyTorch domain libraries provide a number of pre-loaded datasets (such as FashionMNIST) that subclass torch.utils.data.Dataset and implement functions specific to the particular data.

Dataset are accessible through  
* TorchVision: for images dataset
* TorchText: for text datasets
* TorchAudio: for audios dataset


Here is an example of how to load the Fashion-MNIST

`root` is the path where the train/test data is stored,

`train` specifies training or test dataset,

`download=True` downloads the data from the internet if it’s not available at root.

`transform` and target_transform specify the feature and label transformations (convert data to tensor, normalize data, ...(see documentation))



"""

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=16, shuffle=False)
first_batch = next(iter(train_dataloader)) # retrieves the first bacth of the data from the train_dataloader
image, label = first_batch
print(image.shape)
print(label)

"""# Creating a Custom Dataset for your files

A custom Dataset class must implement three functions: `__init__`, `__len__`, and `__getitem__`.
"""

#To connect colab to drive
from google.colab import drive
drive.mount("/content/drive")

from torch.utils.data import Dataset
import pandas as pd

class WineDataset(Dataset):

    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = pd.read_csv('/content/drive/MyDrive/AMMI 2023-24/Tutorials/Data/wine.csv')

        xy = xy.to_numpy()
        self.n_samples = xy.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, [0]]) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples



# create dataset
dataset = WineDataset()

# get first sample and unpack
features, labels = dataset[0]
print(features, labels)

print(features.shape)

"""# Using pytorch dataset transform on custom dataset"""

class WineDataset1(Dataset):

    def __init__(self, transform=None):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = pd.read_csv('/content/drive/MyDrive/AMMI 2023-24/Tutorials/Data/wine.csv')

        xy = xy.to_numpy()
        self.n_samples = xy.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = xy[:, 1:] # size [n_samples, n_features]
        self.y_data = xy[:, [0]] # size [n_samples, 1]
        self.transform = transform

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        sample =  self.x_data[index], self.y_data[index]
        if self.transform:
          sample = self.transform(sample)
        return sample

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

class ToTensor:
  def __call__(self, sample):
    inputs, target = sample
    return torch.from_numpy(inputs), torch.from_numpy(target)

# create dataset
dataset = WineDataset1(transform=ToTensor())

# get first sample and unpack
first_data = dataset[0]
features, labels = first_data
print(features, labels)

print(features.shape)

"""## Create a Dataloader for our custum dataset"""

# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
train_loader = DataLoader(dataset=dataset,
                          batch_size=4,
                          shuffle=True)

# convert to an iterator and look at one  sample
data = next(iter(train_loader))
features, labels = data
print(features, labels)
print(features.shape)

"""## Let's see how to train with the dataloader"""

import math

# Dummy Training loop

num_epochs = 2
num_samples = len(dataset)
n_iterations = math.ceil(num_samples/4)  #number of pass in one epoch

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):

      #forward pass, backward, update ...

        # Run your training process
        if (i+1) % 5 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')

"""# Saving and loading models"""

torch.save(model.state_dict(), 'model_weights.pth')

model.load_state_dict(torch.load('model_weights.pth'))

"""# Extra reading

https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html

https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
"""
