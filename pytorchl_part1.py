# -*- coding: utf-8 -*-


"""![picture](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/09/pytorch.png)

# Definition


PyTorch is a Python-based library used to build neural networks.

It provides two main features:

* An n-dimensional Tensor object

* Automatic differentiation for building and training neural networks


## Tensors

At its core, PyTorch is a library for processing tensors. Tensors are a specialized data structure similar to NumPy's ndarrays, except that tensors can run on GPUs (to accelerate their numeric computations). This is a major advantage of using tensors. A tensor can be a number, vector, matrix, or any n-dimensional array.

PyTorch supports multiple types of tensors, including:

1. FloatTensor: 32-bit float
2. DoubleTensor: 64-bit float
3. HalfTensor: 16-bit float
4. IntTensor: 32-bit int
5. LongTensor: 64-bit int

# Setup
"""

# !conda install pytorch torchvision -c pytorch
# # or with GPU
# ! conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

#https://pytorch.org/

import torch
print(torch.__version__)

"""# Tensor creation

### Scalar
"""

t1 = torch.tensor(4)
print(t1)

"""### 1D tensor (vector)"""

t2 = torch.tensor([1., 2, 3, 4], dtype=torch.float32)
print(t2)

"""### 2D tensor (matrix)"""

t3 = torch.tensor([[5., 6],
                   [7, 8],
                   [9, 10]])

print(t3)

"""### 3D tensor"""

t4 = torch.tensor([
    [[11, 12, 13],
     [13, 14, 15]],
    [[15, 16, 17],
     [17, 18, 19.]]])
print(t4)

"""### Tensor filled with zeros"""

a = torch.zeros(3,3)
a

"""### Tensor filled with ones"""

a = torch.ones(3,3)
print(a)
b = torch.full((3,3),9)
print("\n\n",b)

"""### Tensor filled with random values"""

# setting the random seed for pytorch
torch.manual_seed(42)

## torch.randn returns a tensor filled with random numbers from a normal distribution with mean `0` and variance `1`
torch.randn(3,3)

# torch.randint returns a tensor filled with random integers generated uniformly
torch.randint(0, 10, (3,3))

## torch.rand returns a tensor filled with random numbers from a uniform distribution on the interval `[0, 1)`
torch.rand(3,3)

"""# Attributes of a tensor"""

a = torch.rand(3,4)
print(a, '\n')

print(f"Shape of tensor: {a.shape}")
# print(f"Shape of tensor: {a.size()}")
print(f"Data type of tensor: {a.dtype}")
print(f"Number of dimenssion: {a.ndim}")
print(f"Device tensor is stored on: {a.device}")

"""# Torch tensors and numpy conversion

### From tensor to numpy
"""

## Tensor to numpy array
a = torch.rand(3,3)
print(a, '\n')

## convert a into numpy array b
b = a.numpy()
print(b, '\n')

## update the tensor a by adding a value to it
b +=2
print(a, '\n')

"""Note: if the tensor a is on the cpu and not the gpu, then both a and b will share/point to the same memory

### From numpy to tensor
"""

# torch.set_printoptions(precision=8)
import numpy as np

a = np.random.rand(3,3)
print(a, '\n')

## convert array a into tensor b
#.clone is used to make a copy
b = torch.from_numpy(a).clone()
print(b, '\n')
a += 2
print(b, '\n')
print(a)

"""### Move tensor to a specific  GPU/CPU device"""

import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
a = torch.ones(2, device=device)

#Move to GPU if available
if torch.cuda.is_available():
  device = torch.device('cuda')
  x = torch.ones(5, device=device)
  y = torch.ones(5).to(device)
  z = x+y

#z.cpu().numpy()

"""Note: if we want to convert the z below deirectly to numpy (z.numpy()), this will return an error, because numpy can only handle cpu tensors; So we wil have to bring first z to cpu first and later on do the convertion"""

z.to("cpu").numpy()
z.cpu().numpy()

"""# Operations on tensors

## Standard numpy-like indexing and slicing
"""

a =  torch.randint(0, 10, (4,3))
print(a, '\n')

print(a[0,0])
print(f"First row: {a[0]}")
print(f"First column: {a[:, 0]}")
print(f"Last column: {a[:, -1]}")
a[:,1] = 0
print(a)

"""### Exercise 1

Consider the array x below and reply to the following questions:
1. Convert x to a tensor
2. Get the attributes of the obtained tensor
3. Get the first element in the second dim of the second block (9)
4. select the third row in each block
"""

import numpy as np

x = np.array([[[1, 2], [3, 4], [5, 6], [7, 8]],
                      [[9, 10], [11, 12], [13, 14], [15, 16]],
                      [[17, 18], [19, 20], [21, 22], [23, 24]]])
print(x)

## WRITE YOUR CODE HERE ##

x = np.array([[[1, 2], [3, 4], [5, 6], [7, 8]],
                      [[9, 10], [11, 12], [13, 14], [15, 16]],
                      [[17, 18], [19, 20], [21, 22], [23, 24]]])

x = torch.from_numpy(x.astype("float32"))

print("\n\n Shape:", x.shape)
print(f"\n\n Data type of tensor: {x.dtype}")
print(f"\n\n Number of dimenssion: {x.ndim}")
print(f"\n\n Device tensor is stored on: {x.device}")

print("\n\n First element in the second dim of the second block",x[1,0,0])

print("\n\n Third row of each block \n", x[:,2,:])

"""## Arithmetic operations

### Addition, substraction,  division

For addition, substraction and division we can either use torch.add, torch.sub, torch.div or +,  - and /
"""

torch.manual_seed(42)
a = torch.randn(3,3)
b = torch.randn(3,3)
print(a, '\n')
print(b)

## compute sum of a and b
print(a + b) #torch.add(a,b)

## compute a -b
a - b #torch.sub(a,b)

##compute a / b
a / b #torch.div(a, b)

xy=torch.tensor([[1,2,3],[4,2,1],[4,6,5]])
print(xy,"\n\n")
torch.sum(xy, dim=1)

"""### Multiplication

#### Elementwise multiplication
"""

##
a * b

"""#### matrix multiplication"""

##using torch.mm
torch.mm(a, b)

##using torch.matmul
torch.matmul(a,b)

##using @
a @ b

"""## Reshape, Transpose, concatenate, flatten  and squeeze tensors

### Reshaping
"""

torch.manual_seed(42)

a = torch.randn(3,4)
print(a.shape)

b = a.reshape(-1, 1) #a.view(-1,1)
print(b.shape)

c = a.reshape(6,2)
print(c.shape)

"""### Transpose"""

a_t = a.T #torch.t(a)
print(a_t.shape)

"""### Concatenating"""

print(a)
print(b)

b = torch.randn(1, 4)
print(b.shape)
concat = torch.cat((a, b),dim=0)
print(concat.shape)

"""### Flattening"""

flat_vect = a.flatten()
print(flat_vect)
flat_vect.shape

"""### Squeeze a tensor


To compress tensors along their singleton dimensions we can use the .squeeze() method and use the .unsqueeze() method to do the opposite.
"""

x = torch.randn(1, 10)
y = x.squeeze()
torch.unsqueeze(y,1).shape

"""## In-place operations"""

x = torch.rand(2,2)
y = torch.rand(2,2)
print(y, '\n')
y.add_(x)
x.sub_(y)

print(y, '\n')
print(x)

"""## Get the value of a tensor"""

x = torch.rand(5,3)
print(x)
print(x[1,1])
print(x[1,1].item())

"""# Exercise 2:

* Create two random tensors of shape (4, 3) and send them both to the GPU. Set torch.manual_seed(1234) when creating the tensors.
* Perform a matrix multiplication on these two tensors;
* Convert the result to a Numpy array.
"""

torch.manual_seed(1267)
p= torch.rand(4,3).cuda()
q= torch.rand(4,3).cuda()

(p @ q.T).cpu().numpy()

## WRITE YOUR CODE HERE ##
torch.manual_seed(1234)

x = torch.rand(4,3).cuda()
y = torch.rand(4,3).cuda().reshape(3,4)

results = x @ y

print("Matrix multiplication: \n",results )

print("\n\n Matrix multiplication: \n",results.cpu().numpy() )



"""# Exercise 3:

Create a function that takes in a square matrix A and returns a 2D tensor consisting of a flattened A with the index of each element appended to this tensor in the row dimension, e.g.,



 \begin{equation*}
 A=
 \begin{bmatrix}
  2 & 3 \\
  4 & -2
\end{bmatrix}, \ \
output = \begin{bmatrix}
  0 & 2 \\
  1 & 3 \\
  2 & 4 \\
  3 & -2
\end{bmatrix}
\end{equation*}

"""

def function_concat(A):
  """
  This function takes in a square matrix A and returns a 2D tensor
  consisting of a flattened A with the index of each element
  appended to this tensor in the row dimension.

  Args:
    A: torch.Tensor

  Returns:
    output: torch.Tensor
  """
  if A.shape[0]==A.shape[1]:

    # TODO flatten A
    A_flatten = A.flatten().reshape(-1,1)
    # TODO create the idx tensor to be concatenated to A
    idx_tensor = torch.tensor(np.arange(len(A_flatten))).reshape(-1,1)
    # TODO concatenate the two tensors
    output = torch.cat((idx_tensor,A_flatten),dim=1)
  else:
    return "A not a square matrix"

  return output

## TEST YOUR CODE HERE ##
A = torch.tensor([[2,3,3],[4,-2,5],[4,2,4]])

function_concat(A)

"""# Exercise 4:

Write a function to check device-compatiblity with computations. Fill the missing part in the code below.

Note: You have to set the device when creating each tensor.
"""

import time

torch.full((2,2),2).dtype

def cpu_gpu(dim, device):
  """
  Function to check device-compatiblity with computations

  Args:
    dim: Integer
    device: String ("cpu" or "cuda") or torch.device('cpu'), torch.device('cuda')

  Returns: the execution time

  """

  # TODO: create 2D tensor filled with uniform random numbers in [0,1), dim x dim
  x = torch.rand(dim,dim).to(device)
  # TODO: create 2D tensor filled with uniform random numbers in [0,1), dim x dim
  y = torch.rand(dim,dim).to(device)
  # TODO: create 2D tensor filled with the scalar value 2, dim x dim

  #z =2*(torch.ones((dim,dim))).to(device)

  z =torch.full((dim,dim),float(2)).to(device)

  start = time.time()
  # elementwise multiplication of x and y
  a = x * y
  # matrix multiplication of x and z
  b = x @ z #.astype("float342"))
  return time.time()-start

##  TEST YOUR CODE HERE ##

dim=10000
print(f"Execution time when using the cpu: {cpu_gpu(dim, device='cpu')}")

print(f"Execution time when using the gpu: {cpu_gpu(dim, device='cuda')}")

"""#### Comment the results

# Extra reading

https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
"""
