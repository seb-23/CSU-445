#!/usr/bin/env python
# coding: utf-8

# # A4 Classification of Hand-Drawn Digits
# 
# In this assignment, you will define a new class named `NeuralNetworkClassifier` that extends the `NeuralNetwork` class provided here and is the solution to Assignment A2.  You will use `NeuralNetworkClassifier` to train a classifier of hand-drawn digits.
# 
# You will also define the function `confusion_matrix`.

# ## `NeuralNetwork` class

# In[1]:


import matplotlib.pyplot as plt


# The following code cell will write its contents to `optimizers.py` so the `import optimizers` statement in the code cell after it will work correctly.

# In[2]:


get_ipython().run_cell_magic('writefile', 'optimizers.py', "import numpy as np\n\n######################################################################\n## class Optimizers()\n######################################################################\n\nclass Optimizers():\n\n    def __init__(self, all_weights):\n        '''all_weights is a vector of all of a neural networks weights concatenated into a one-dimensional vector'''\n        \n        self.all_weights = all_weights\n\n        # The following initializations are only used by adam.\n        # Only initializing m, v, beta1t and beta2t here allows multiple calls to adam to handle training\n        # with multiple subsets (batches) of training data.\n        self.mt = np.zeros_like(all_weights)\n        self.vt = np.zeros_like(all_weights)\n        self.beta1 = 0.9\n        self.beta2 = 0.999\n        self.beta1t = 1\n        self.beta2t = 1\n\n        \n    def sgd(self, error_f, gradient_f, fargs=[], n_epochs=100, learning_rate=0.001, verbose=True, error_convert_f=None):\n        '''\nerror_f: function that requires X and T as arguments (given in fargs) and returns mean squared error.\ngradient_f: function that requires X and T as arguments (in fargs) and returns gradient of mean squared error\n            with respect to each weight.\nerror_convert_f: function that converts the standardized error from error_f to original T units.\n        '''\n\n        error_trace = []\n        epochs_per_print = n_epochs // 10\n\n        for epoch in range(n_epochs):\n\n            error = error_f(*fargs)\n            grad = gradient_f(*fargs)\n\n            # Update all weights using -= to modify their values in-place.\n            self.all_weights -= learning_rate * grad\n\n            if error_convert_f:\n                error = error_convert_f(error)\n            error_trace.append(error)\n\n            if verbose and ((epoch + 1) % max(1, epochs_per_print) == 0):\n                print(f'sgd: Epoch {epoch+1:d} Error={error:.5f}')\n\n        return error_trace\n\n    def adam(self, error_f, gradient_f, fargs=[], n_epochs=100, learning_rate=0.001, verbose=True, error_convert_f=None):\n        '''\nerror_f: function that requires X and T as arguments (given in fargs) and returns mean squared error.\ngradient_f: function that requires X and T as arguments (in fargs) and returns gradient of mean squared error\n            with respect to each weight.\nerror_convert_f: function that converts the standardized error from error_f to original T units.\n        '''\n\n        alpha = learning_rate  # learning rate called alpha in original paper on adam\n        epsilon = 1e-8\n        error_trace = []\n        epochs_per_print = n_epochs // 10\n\n        for epoch in range(n_epochs):\n\n            error = error_f(*fargs)\n            grad = gradient_f(*fargs)\n\n            self.mt[:] = self.beta1 * self.mt + (1 - self.beta1) * grad\n            self.vt[:] = self.beta2 * self.vt + (1 - self.beta2) * grad * grad\n            self.beta1t *= self.beta1\n            self.beta2t *= self.beta2\n\n            m_hat = self.mt / (1 - self.beta1t)\n            v_hat = self.vt / (1 - self.beta2t)\n\n            # Update all weights using -= to modify their values in-place.\n            self.all_weights -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)\n    \n            if error_convert_f:\n                error = error_convert_f(error)\n            error_trace.append(error)\n\n            if verbose and ((epoch + 1) % max(1, epochs_per_print) == 0):\n                print(f'Adam: Epoch {epoch+1:d} Error={error:.5f}')\n\n        return error_trace\n\nif __name__ == '__main__':\n\n    import matplotlib.pyplot as plt\n    plt.ion()\n\n    def parabola(wmin):\n        return ((w - wmin) ** 2)[0]\n\n    def parabola_gradient(wmin):\n        return 2 * (w - wmin)\n\n    w = np.array([0.0])\n    optimizer = Optimizers(w)\n\n    wmin = 5\n    optimizer.sgd(parabola, parabola_gradient, [wmin],\n                  n_epochs=500, learning_rate=0.1)\n\n    print(f'sgd: Minimum of parabola is at {wmin}. Value found is {w}')\n\n    w = np.array([0.0])\n    optimizer = Optimizers(w)\n    optimizer.adam(parabola, parabola_gradient, [wmin],\n                   n_epochs=500, learning_rate=0.1)\n    \n    print(f'adam: Minimum of parabola is at {wmin}. Value found is {w}')")


# In[3]:


import numpy as np
import optimizers
import sys  # for sys.float_info.epsilon

######################################################################
## class NeuralNetwork()
######################################################################

class NeuralNetwork():


    def __init__(self, n_inputs, n_hiddens_per_layer, n_outputs, activation_function='tanh'):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation_function = activation_function

        # Set self.n_hiddens_per_layer to [] if argument is 0, [], or [0]
        if n_hiddens_per_layer == 0 or n_hiddens_per_layer == [] or n_hiddens_per_layer == [0]:
            self.n_hiddens_per_layer = []
        else:
            self.n_hiddens_per_layer = n_hiddens_per_layer

        # Initialize weights, by first building list of all weight matrix shapes.
        n_in = n_inputs
        shapes = []
        for nh in self.n_hiddens_per_layer:
            shapes.append((n_in + 1, nh))
            n_in = nh
        shapes.append((n_in + 1, n_outputs))

        # self.all_weights:  vector of all weights
        # self.Ws: list of weight matrices by layer
        self.all_weights, self.Ws = self.make_weights_and_views(shapes)

        # Define arrays to hold gradient values.
        # One array for each W array with same shape.
        self.all_gradients, self.dE_dWs = self.make_weights_and_views(shapes)

        self.trained = False
        self.total_epochs = 0
        self.error_trace = []
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None


    def make_weights_and_views(self, shapes):
        # vector of all weights built by horizontally stacking flatenned matrices
        # for each layer initialized with uniformly-distributed values.
        all_weights = np.hstack([np.random.uniform(size=shape).flat / np.sqrt(shape[0])
                                 for shape in shapes])
        # Build list of views by reshaping corresponding elements from vector of all weights
        # into correct shape for each layer.
        views = []
        start = 0
        for shape in shapes:
            size =shape[0] * shape[1]
            views.append(all_weights[start:start + size].reshape(shape))
            start += size
        return all_weights, views


    # Return string that shows how the constructor was called
    def __repr__(self):
        return f'{type(self).__name__}({self.n_inputs}, {self.n_hiddens_per_layer}, {self.n_outputs}, \'{self.activation_function}\')'


    # Return string that is more informative to the user about the state of this neural network.
    def __str__(self):
        result = self.__repr__()
        if len(self.error_trace) > 0:
            return self.__repr__() + f' trained for {len(self.error_trace)} epochs, final training error {self.error_trace[-1]:.4f}'


    def train(self, X, T, n_epochs, learning_rate, method='sgd', verbose=True):
        '''
train: 
  X: n_samples x n_inputs matrix of input samples, one per row
  T: n_samples x n_outputs matrix of target output values, one sample per row
  n_epochs: number of passes to take through all samples updating weights each pass
  learning_rate: factor controlling the step size of each update
  method: is either 'sgd' or 'adam'
        '''

        # Setup standardization parameters
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xstds[self.Xstds == 0] = 1  # So we don't divide by zero when standardizing
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
            
        # Standardize X and T
        X = (X - self.Xmeans) / self.Xstds
        T = (T - self.Tmeans) / self.Tstds

        # Instantiate Optimizers object by giving it vector of all weights
        optimizer = optimizers.Optimizers(self.all_weights)

        # Define function to convert value from error_f into error in original T units, 
        # but only if the network has a single output. Multiplying by self.Tstds for 
        # multiple outputs does not correctly unstandardize the error.
        if len(self.Tstds) == 1:
            error_convert_f = lambda err: (np.sqrt(err) * self.Tstds)[0] # to scalar
        else:
            error_convert_f = lambda err: np.sqrt(err)[0] # to scalar
            

        if method == 'sgd':

            error_trace = optimizer.sgd(self.error_f, self.gradient_f,
                                        fargs=[X, T], n_epochs=n_epochs,
                                        learning_rate=learning_rate,
                                        verbose=True,
                                        error_convert_f=error_convert_f)

        elif method == 'adam':

            error_trace = optimizer.adam(self.error_f, self.gradient_f,
                                         fargs=[X, T], n_epochs=n_epochs,
                                         learning_rate=learning_rate,
                                         verbose=True,
                                         error_convert_f=error_convert_f)

        else:
            raise Exception("method must be 'sgd' or 'adam'")
        
        self.error_trace = error_trace

        # Return neural network object to allow applying other methods after training.
        #  Example:    Y = nnet.train(X, T, 100, 0.01).use(X)
        return self

    def relu(self, s):
        s[s < 0] = 0
        return s

    def grad_relu(self, s):
        return (s > 0).astype(int)
    
    def forward_pass(self, X):
        '''X assumed already standardized. Output returned as standardized.'''
        self.Ys = [X]
        for W in self.Ws[:-1]:
            if self.activation_function == 'relu':
                self.Ys.append(self.relu(self.Ys[-1] @ W[1:, :] + W[0:1, :]))
            else:
                self.Ys.append(np.tanh(self.Ys[-1] @ W[1:, :] + W[0:1, :]))
        last_W = self.Ws[-1]
        self.Ys.append(self.Ys[-1] @ last_W[1:, :] + last_W[0:1, :])
        return self.Ys

    
    # Function to be minimized by optimizer method, mean squared error
    def error_f(self, X, T):
        Ys = self.forward_pass(X)
        mean_sq_error = np.mean((T - Ys[-1]) ** 2)
        return mean_sq_error

    # Gradient of function to be minimized for use by optimizer method
    def gradient_f(self, X, T):
        '''Assumes forward_pass just called with layer outputs in self.Ys.'''
        error = T - self.Ys[-1]
        n_samples = X.shape[0]
        n_outputs = T.shape[1]
        delta = - error / (n_samples * n_outputs)
        n_layers = len(self.n_hiddens_per_layer) + 1
        # Step backwards through the layers to back-propagate the error (delta)
        for layeri in range(n_layers - 1, -1, -1):
            # gradient of all but bias weights
            self.dE_dWs[layeri][1:, :] = self.Ys[layeri].T @ delta
            # gradient of just the bias weights
            self.dE_dWs[layeri][0:1, :] = np.sum(delta, 0)
            # Back-propagate this layer's delta to previous layer
            if self.activation_function == 'relu':
                delta = delta @ self.Ws[layeri][1:, :].T * self.grad_relu(self.Ys[layeri])
            else:
                delta = delta @ self.Ws[layeri][1:, :].T * (1 - self.Ys[layeri] ** 2)
        return self.all_gradients

    def use(self, X):
        '''X assumed to not be standardized'''
        # Standardize X
        X = (X - self.Xmeans) / self.Xstds
        Ys = self.forward_pass(X)
        Y = Ys[-1]
        # Unstandardize output Y before returning it
        return Y * self.Tstds + self.Tmeans


# In[4]:


X = np.arange(100).reshape((-1, 1))
T = (X - 20) ** 3 / 300000

hiddens = [10]
nnet = NeuralNetwork(X.shape[1], hiddens, T.shape[1])
nnet.train(X, T, 250, 0.01, method='adam')

plt.subplot(1, 2, 1)
plt.plot(nnet.error_trace)

plt.subplot(1, 2, 2)
plt.plot(T, label='T')
plt.plot(nnet.use(X), label='Y')
plt.legend()


# ## Your `NeuralNetworkClassifier` class

# Complete the following definition of `NeuralNetworkClassifier` as discussed in class. You will need to override the functions
# 
# * `train`
# * `error_f`
# * `gradient_f`
# * `use`
# 
# and define the following new functions
# 
# * `makeIndicatorVars`
# * `softmax`

# In[5]:


class NeuralNetworkClassifier(NeuralNetwork):
    
    def train(self, X, T, n_epochs, learning_rate, method='sgd', verbose=True):

        self.classes = np.unique(T)

        # Setup standardization parameters
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xstds[self.Xstds == 0] = 1 
            
        X = (X - self.Xmeans) / self.Xstds
        T_indicator_vars = self.makeIndicatorVars(T)

        # Instantiate Optimizers object by giving it vector of all weights
        optimizer = optimizers.Optimizers(self.all_weights)

        # Define function to convert value from error_f into error in original T units, 
        # but only if the network has a single output. Multiplying by self.Tstds for 
        # multiple outputs does not correctly unstandardize the error.
        # if len(self.Tstds) == 1:
        if self.Tstds:
            error_convert_f = lambda err: (np.exp(-err) * self.Tstds)[0] # to scalar
        else:
            error_convert_f = lambda err: np.exp(-err) # to scalar

        if method == 'sgd':

            error_trace = optimizer.sgd(self.error_f, self.gradient_f,
                                        fargs=[X, T_indicator_vars], n_epochs=n_epochs,
                                        learning_rate=learning_rate,
                                        verbose=True,
                                        error_convert_f=error_convert_f)

        elif method == 'adam':

            error_trace = optimizer.adam(self.error_f, self.gradient_f,
                                         fargs=[X, T_indicator_vars], n_epochs=n_epochs,
                                         learning_rate=learning_rate,
                                         verbose=True,
                                         error_convert_f=error_convert_f)

        else:
            raise Exception("method must be 'sgd' or 'adam'")
        
        self.error_trace = error_trace

        # Return neural network object to allow applying other methods after training.
        #  Example:    Y = nnet.train(X, T, 100, 0.01).use(X)
        return self
    
    # Function to be minimized by optimizer method, mean squared error
    def error_f(self, X, T):
        Ys = self.forward_pass(X)
        Y = self.softmax(Ys[-1])
        return - np.mean(T * np.log(Y))

    # Gradient of function to be minimized for use by optimizer method
    def gradient_f(self, X, T):
        
        Yz = self.softmax(self.Ys[-1])
        
        error = T - Yz
        n_samples = X.shape[0]
        n_outputs = T.shape[1]
        delta = - error / (n_samples * n_outputs)
        n_layers = len(self.n_hiddens_per_layer) + 1
        # Step backwards through the layers to back-propagate the error (delta)
        for layeri in range(n_layers - 1, -1, -1):
            # gradient of all but bias weights
            self.dE_dWs[layeri][1:, :] = self.Ys[layeri].T @ delta
            # gradient of just the bias weights
            self.dE_dWs[layeri][0:1, :] = np.sum(delta, 0)
            # Back-propagate this layer's delta to previous layer
            if self.activation_function == 'relu':
                delta = delta @ self.Ws[layeri][1:, :].T * self.grad_relu(self.Ys[layeri])
            else:
                delta = delta @ self.Ws[layeri][1:, :].T * (1 - self.Ys[layeri] ** 2)
        return self.all_gradients
    
    def softmax(self, X):
        '''LECTURE: 3/08'''
        fs = np.exp(X)  # N x K
        denom = np.sum(fs, axis=1).reshape((-1, 1))
        gs = fs / denom
        return gs
    
    def makeIndicatorVars(self, T):
        # Make sure T is two-dimensional. Should be nSamples x 1.
        '''LECTURE: 3/08'''
        if T.ndim == 1:
            T = T.reshape((-1, 1))    
        return (T == np.unique(T)).astype(int)
    
    def use(self, X):
        X = (X - self.Xmeans) / self.Xstds
        Ys = self.forward_pass(X)
        Y = self.softmax(Ys[-1])
        classes = self.classes[np.argmax(Y, axis=1)].reshape(-1, 1)
        return classes, Y


# Here is a simple test of your new class.  For inputs from 0 to 100, classify values less than or equal to 25 as Class Label 25, greater than 25 and less than or equal to 75 as Class Label 75, and greater than 75 as Class Label 100. 

# In[6]:


X = np.arange(100).reshape((-1, 1))
T = X.copy()
T[T <= 25] = 25
T[np.logical_and(25 < T, T <= 75)] = 75
T[T > 75] = 100

plt.plot(X, T, 'o-')
plt.xlabel('X')
plt.ylabel('Class');


# In[7]:


# X = np.arange(20).reshape(20, 1)
# X = np.hstack((X, X[::-1, :]))
# T = np.array(['ends', 'mid'])[(np.abs(X[:, 0:1] - X[:, 1:2]) < 6).astype(int)]

# np.random.seed(42)

# nnet = NeuralNetworkClassifier(X.shape[1], [10, 10], len(np.unique(T)), activation_function='relu')
# nnet.train(X, T, 500, 0.001, method='adam', verbose=False)
hiddens = [10]
nnet = NeuralNetworkClassifier(X.shape[1], hiddens, len(np.unique(T)))
nnet.train(X, T, 200, 0.01, method='adam', verbose=True)

plt.subplot(1, 2, 1)
plt.plot(nnet.error_trace)
plt.xlabel('Epoch')
plt.ylabel('Likelihood')

plt.subplot(1, 2, 2)
plt.plot(T + 5, 'o-', label='T + 5')  # to see, when predicted overlap T very closely
plt.plot(nnet.use(X)[0], 'o-', label='Y')
plt.legend()


# ## Now for the Hand-Drawn Digits
# 
# We will use a bunch (50,000) images of hand drawn digits from [this deeplearning.net site](http://deeplearning.net/tutorial/gettingstarted.html).  Download `mnist.pkl.gz`. 
# 
# deeplearning.net goes down a lot.  If you can't download it from there you can try getting it from [here](https://gitlab.cs.washington.edu/colinxs/neural_nets/blob/master/mnist.pkl.gz).
# 
# This pickle file includes data already partitioned into training, validation, and test sets.  To read it into python, use the following steps

# In[8]:


import pickle
import gzip
import time

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

Xtrain = train_set[0]
Ttrain = train_set[1].reshape(-1, 1)

Xval = valid_set[0]
Tval = valid_set[1].reshape(-1, 1)

Xtest = test_set[0]
Ttest = test_set[1].reshape(-1, 1)




with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set1, valid_set1, test_set1 = pickle.load(f, encoding='latin1')
    
Xtrain1 = train_set1[0]
Ttrain1 = train_set1[1].reshape(-1, 1)

Xval1 = valid_set1[0]
Tval1 = valid_set1[1].reshape(-1, 1)

Xtest1 = test_set1[0]
Ttest1 = test_set1[1].reshape(-1, 1)





with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set2, valid_set2, test_set2 = pickle.load(f, encoding='latin1')
    
Xtrain2 = train_set2[0]
Ttrain2 = train_set2[1].reshape(-1, 1)

Xval2 = valid_set2[0]
Tval2 = valid_set2[1].reshape(-1, 1)

Xtest2 = test_set2[0]
Ttest2 = test_set2[1].reshape(-1, 1)





with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set3, valid_set3, test_set3 = pickle.load(f, encoding='latin1')
    
Xtrain3 = train_set3[0]
Ttrain3 = train_set3[1].reshape(-1, 1)

Xval3 = valid_set3[0]
Tval3 = valid_set3[1].reshape(-1, 1)

Xtest3 = test_set3[0]
Ttest3 = test_set3[1].reshape(-1, 1)





with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set4, valid_set4, test_set4 = pickle.load(f, encoding='latin1')
    
Xtrain4 = train_set4[0]
Ttrain4 = train_set4[1].reshape(-1, 1)

Xval4 = valid_set4[0]
Tval4 = valid_set4[1].reshape(-1, 1)

Xtest4 = test_set4[0]
Ttest4 = test_set4[1].reshape(-1, 1)





with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set5, valid_set5, test_set5 = pickle.load(f, encoding='latin1')

Xtrain5 = train_set5[0]
Ttrain5 = train_set5[1].reshape(-1, 1)

Xval5 = valid_set5[0]
Tval5 = valid_set5[1].reshape(-1, 1)

Xtest5 = test_set5[0]
Ttest5 = test_set5[1].reshape(-1, 1)





with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set6, valid_set6, test_set6 = pickle.load(f, encoding='latin1')

Xtrain6 = train_set6[0]
Ttrain6 = train_set6[1].reshape(-1, 1)

Xval6 = valid_set6[0]
Tval6 = valid_set6[1].reshape(-1, 1)

Xtest6 = test_set6[0]
Ttest6 = test_set6[1].reshape(-1, 1)

print(Xtrain.shape, Ttrain.shape,  Xval.shape, Tval.shape,  Xtest.shape, Ttest.shape)


# In[9]:


Ttrain[:10]


# Those must be the digits.  What the heck is in those 784 columns in the input matrices?

# In[10]:


plt.plot(Xtrain[0, :]);


# Well, values between 0 and 1.  That doesn't help much.  These are actually intensity values for 784 pixels in an image.
# 
# How can we rearrange these values into an image to be displayed?  We must first figure out how many columns and rows the image would have.  Perhaps the image is a square image, with equal numbers of rows and columns.

# In[11]:


import math
math.sqrt(784)


# Ah, cool.

# In[12]:


28 * 28


# Ok Let's reshape it and look at the numbers.

# In[13]:


image0 = Xtrain[0, :]
image0 = image0.reshape(28, 28)
image0


# Not that helpful.  Ok, let's use `matplotlib` to make an image display.

# In[14]:


plt.imshow(image0);


# Humm.  Try a grayscale color map.

# In[15]:


plt.imshow(image0, cmap='gray');


# With a little more work, we can make it look like a pencil drawing.

# In[16]:


plt.imshow(-image0, cmap='gray')  # notice the negative sign
plt.axis('off');


# Looks like a 5.  What class label is associated with this image?

# In[17]:


Ttrain[0]


# Okay.  Makes sense.  Let's look at the first 100 images and their labels, as plot titles.

# In[18]:


plt.figure(figsize=(20, 20))
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow(-Xtrain[i, :].reshape(28, 28), cmap='gray')
    plt.title(Ttrain[i, 0])
    plt.axis('off');


# Okay.  We are ready to try to classify, right?
# 
# First we should check the proportions of each digit in the given data partitions.

# In[19]:


classes = np.arange(10)
(Ttrain == classes).shape


# In[20]:


(Ttrain == classes).sum(axis=0)


# In[21]:


(Ttrain == classes).sum(axis=0) / Ttrain.shape[0]


# In[22]:


['Ttrain', *(Ttrain == classes).sum(axis=0) / Ttrain.shape[0]]


# In[23]:


import pandas

def rmse(Y, T):
    return np.sqrt(np.mean((T - Y)**2))

result = []
result.append(['Train', *(Ttrain == classes).sum(axis=0) / Ttrain.shape[0]])
result.append(['Tval', *(Tval == classes).sum(axis=0) / Tval.shape[0]])
result.append(['Ttest', *(Ttest == classes).sum(axis=0) / Ttest.shape[0]])
pandas.DataFrame(result)


# All very close to 0.1. Super.

# Time for our first experiment.  Let's train a small neural net with 5 hidden units in one layer for a small number of epochs using Adam.

# In[24]:


n_epochs = 100
learning_rate = 0.01

np.random.seed(142)

nnet = NeuralNetworkClassifier(Xtrain.shape[1], [5], len(classes))
nnet.train(Xtrain, Ttrain, n_epochs, learning_rate, method='adam', verbose=True)


# In[25]:


print(nnet)  # uses the __str__ method


# In[ ]:


plt.plot(nnet.error_trace)

'''
All you can really change is the learning_rate & hidden layers.
This is because n_epochs must remain at 500.
Lowest RMSE Val is the best one.
'''

# start1 = time.time()
# n_epochs1 = 500  # 100 small number of epochs
# learning_rate1 = 0.01
# hiddens1 = [10]

# np.random.seed(67)

# nnet1 = NeuralNetworkClassifier(Xtrain1.shape[1], hiddens1, len(classes))
# nnet1.train(Xtrain1, Ttrain1, n_epochs1, learning_rate1, method='sgd', verbose=True)

# plt.plot(nnet1.error_trace)

# elapsed1 = (time.time() - start1) 

# Y_classes1, Y_probs1 = nnet1.use(Xtrain1)
# Y_classes2, Y_probs2 = nnet1.use(Xval1)
# Y_classes3, Y_probs3 = nnet1.use(Xtest1)

# result1 = []
# result1.append([hiddens1,
#                 100 * np.mean(Y_classes1 == Ttrain1),
#                 100 * np.mean(Y_classes2 == Tval1),
#                 100 * np.mean(Y_classes3 == Ttest1),
#                 elapsed1])

# # load these into a dataframe and give it some column titles
# df1 = pandas.DataFrame(result1, columns=('Hidden Layers','Train','Validate','Test','Time'))
# print("First Network\n", df1)












# start2 = time.time()
# n_epochs2 = 500  
# learning_rate2 = 0.01
# hiddens2 = [15]

# np.random.seed(99)

# nnet2 = NeuralNetworkClassifier(Xtrain2.shape[1], hiddens2, len(classes))
# nnet2.train(Xtrain2, Ttrain2, n_epochs2, learning_rate2, method='sgd', verbose=True)

# plt.plot(nnet2.error_trace)

# elapsed2 = (time.time() - start2) 

# Y_classes4, Y_probs4 = nnet2.use(Xtrain2)
# Y_classes5, Y_probs5 = nnet2.use(Xval2)
# Y_classes6, Y_probs6 = nnet2.use(Xtest2)

# result2 = []
# result2.append([hiddens2,
#                 100 * np.mean(Y_classes4 == Ttrain2),
#                 100 * np.mean(Y_classes5 == Tval2),
#                 100 * np.mean(Y_classes6 == Ttest2),
#                 elapsed2])

# # load these into a dataframe and give it some column titles
# df2 = pandas.DataFrame(result2, columns=('Hidden Layers','Train','Validate','Test','Time'))
# print("Second Network\n", df2)










# start3 = time.time()
# n_epochs3 = 500  
# learning_rate3 = 0.01
# hiddens3 = [25]

# np.random.seed(33)

# nnet3 = NeuralNetworkClassifier(Xtrain3.shape[1], hiddens3, len(classes))
# nnet3.train(Xtrain3, Ttrain3, n_epochs3, learning_rate3, method='sgd', verbose=True)

# plt.plot(nnet3.error_trace)

# elapsed3 = (time.time() - start3) 

# Y_classes7, Y_probs7 = nnet3.use(Xtrain3)
# Y_classes8, Y_probs8 = nnet3.use(Xval3)
# Y_classes9, Y_probs9 = nnet3.use(Xtest3)

# result3 = []
# result3.append([hiddens3,
#                 100 * np.mean(Y_classes7 == Ttrain3),
#                 100 * np.mean(Y_classes8 == Tval3),
#                 100 * np.mean(Y_classes9 == Ttest3),
#                 elapsed3])

# # load these into a dataframe and give it some column titles
# df3 = pandas.DataFrame(result3, columns=('Hidden Layers','Train','Validate','Test','Time'))
# print("Third Network\n", df3)






# start4 = time.time()
# n_epochs4 = 500 
# learning_rate4 = 0.01
# hiddens4 = [10]

# np.random.seed(4)

# nnet4 = NeuralNetworkClassifier(Xtrain4.shape[1], hiddens4, len(classes))
# nnet4.train(Xtrain4, Ttrain4, n_epochs4, learning_rate4, method='adam', verbose=True)

# plt.plot(nnet4.error_trace)

# elapsed4 = (time.time() - start4)

# Y_classes10, Y_probs10 = nnet4.use(Xtrain4)
# Y_classes11, Y_probs11 = nnet4.use(Xval4)
# Y_classes12, Y_probs12 = nnet4.use(Xtest4)

# result4 = []
# result4.append([hiddens4,
#                 100 * np.mean(Y_classes10 == Ttrain4),
#                 100 * np.mean(Y_classes11 == Tval4),
#                 100 * np.mean(Y_classes12 == Ttest4),
#                 elapsed4])

# # load these into a dataframe and give it some column titles
# df4 = pandas.DataFrame(result4, columns=('Hidden Layers','Train','Validate','Test','Time'))
# print("Fourth Network\n", df4)




# start5 = time.time()
# n_epochs5 = 500  
# learning_rate5 = 0.01
# hiddens5 = [20,25]

# np.random.seed(22)

# nnet5 = NeuralNetworkClassifier(Xtrain5.shape[1], hiddens5, len(classes))
# nnet5.train(Xtrain5, Ttrain5, n_epochs5, learning_rate5, method='adam', verbose=True)

# plt.plot(nnet5.error_trace)

# elapsed5 = (time.time() - start5) 

# Y_classes13, Y_probs13 = nnet5.use(Xtrain5)
# Y_classes14, Y_probs14 = nnet5.use(Xval5)
# Y_classes15, Y_probs15 = nnet5.use(Xtest5)

# result5 = []
# result5.append([hiddens5,
#                 100 * np.mean(Y_classes13 == Ttrain5),
#                 100 * np.mean(Y_classes14 == Tval5),
#                 100 * np.mean(Y_classes15 == Ttest5),
#                 elapsed5])
# # load these into a dataframe and give it some column titles
# df5 = pandas.DataFrame(result5, columns=('Hidden Layers','Train','Validate','Test','Time'))
# print("Fifth Network\n", df5)







start6 = time.time()
n_epochs6 = 500  
learning_rate6 = 0.01
hiddens6 = [30]

np.random.seed(22)

nnet6 = NeuralNetworkClassifier(Xtrain6.shape[1], hiddens6, len(classes))
nnet6.train(Xtrain6, Ttrain6, n_epochs6, learning_rate6, method='adam', verbose=True)

plt.plot(nnet6.error_trace)

elapsed6 = (time.time() - start6) 

Y_classes16, Y_probs13 = nnet6.use(Xtrain6)
Y_classes17, Y_probs14 = nnet6.use(Xval6)
Y_classes18, Y_probs15 = nnet6.use(Xtest6)

result6 = []
result6.append([hiddens6,
                100 * np.mean(Y_classes16 == Ttrain6),
                100 * np.mean(Y_classes17 == Tval6),
                100 * np.mean(Y_classes18 == Ttest6),
                elapsed6])
# load these into a dataframe and give it some column titles
df6 = pandas.DataFrame(result6, columns=('Hidden Layers','Train','Validate','Test','Time'))
print("Sixth Network\n", df6)






df6


# Now it is time for you to run some longer experiments.  You must write the code to do the following steps:
# 
# 1. For each of at least five different hidden layer structures
# 
#     1. Train a network for 500 epochs.
#     1. Collect percent of samples correctly classified in the given train, validate, and test partitions.
# 
# 2. Create a `pandas.DataFrame` with these results and with column headings `('Hidden Layers', 'Train', 'Validate', 'Test', 'Time')` where `'Time'` is the number of seconds required to train each network.
# 
# 3. Retrain a network using the best hidden layer structure, judged by the percent correct on the validation set.
# 4. Use this network to find several images in the test set for which the network's probability of the correct class is the closest to zero, meaning images for which your network does the worst.  Draw these images and discuss why your network might not be doing well for those images.
# 
# 
# 
# ------------------------------------------------------------------------------------------------------------------
# 
# 
# 
# ANSWER TO #4:
# 
# The network doesn't do well for these particular images because the dataset is very large. The larger a dataset is, the longer it takes to run if the hyperparameters in the network remain the same. Hence, because this particular dataset has 50,000 hand drawn images to learn from, even running a network with a small number of hidden layers can have us waiting on the computation for several minutes. This is part of the reason why those particular images did not perform so well, it is due to the fact that the dataset is just too large for the time I have to run more expirements. Another reason why the network does not perform well on those particular images is due to the fact that certain numbers look very similar to other numbers. For example, 2 looks like a more accomplished 7; 3 looks like an unfinished 8 and a deformed 5. While 9 gets confused  by 7. There is certainly more confusion when making predictions on those particular images which look similar to other images. Also, what could be improved upon are the data transformation techniques. So if we create more data variables from the already established data we can then use that data transformation in our machine learning to extract a better prediction.

# ## `confusion_matrix`
# 
# Now, write a function named `confusion_matrix` that returns a confusion matrix for any classification problem, returned as a `pandas.DataFrame` as shown in Lecture Notes 12.  It must require two arguments, the predicted classes for each sample and the true classes for each sample.  Here is an example.

# In[ ]:


def confusion_matrix(Y_classes, Ttest):
    class_names = np.unique(Ttest)
    table = []
    for true_class in class_names:
        row = []
        for predicted_class in class_names:
            row.append(100 * np.mean(Y_classes[Ttest == true_class] == predicted_class))
            # row.append(f'{100 * np.mean(Y_classes[Ttest == true_class] == predicted_class):0.1f}')
        table.append(row)
    
    
    df = pandas.DataFrame(table, index=class_names, columns=class_names)
    
    return df

'''
Reasons why these are the worst images:
1. The network isn't large enough because we can't run a large b/c it would take too long
2. Certain numbers look like others so the algorithm/s get confused.  For example 2 looks like 5. 3 looks like 8.
3. This network is not doing so well for these images because... LECTURE 03/17/21
'''

# Y_clazzez1, Y_probz1 = nnet1.use(Xtrain1)
# best1 = confusion_matrix(Y_clazzez1, Ttrain1)

# Y_clazzez2, Y_probz2 = nnet2.use(Xtrain2)
# best2 = confusion_matrix(Y_clazzez2, Ttrain2)

# Y_clazzez3, Y_probz3 = nnet3.use(Xtrain3)
# best3 = confusion_matrix(Y_clazzez3, Ttrain3)

# Y_clazzez4, Y_probz4 = nnet4.use(Xtrain4)
# best4 = confusion_matrix(Y_clazzez4, Ttrain4)

# Y_clazzez5, Y_probz5 = nnet5.use(Xtrain5)
# best5 = confusion_matrix(Y_clazzez5, Ttrain5)

Y_clazzez6, Y_probz6 = nnet6.use(Xtrain6)
best6 = confusion_matrix(Y_clazzez6, Ttrain6)

# image number 3
worst_image = Xtrain6[7, :]
worst_image = worst_image.reshape(28, 28)
plt.subplot(1, 3, 1)
plt.imshow(-worst_image, cmap='gray')  # notice the negative sign
plt.axis('off')

# image number 2
worst_image1 = Xtrain6[5, :]
worst_image1 = worst_image1.reshape(28, 28)
plt.subplot(1, 3, 2)
plt.imshow(-worst_image1, cmap='gray')  # notice the negative sign
plt.axis('off')

# image number 2
worst_image2 = Xtrain6[4, :]
worst_image2 = worst_image2.reshape(28, 28)
plt.subplot(1, 3, 3)
plt.imshow(-worst_image2, cmap='gray')  # notice the negative sign
plt.axis('off')

best6


# ## Grading and Check-In
# 
# You will receive 70 points for correct code, and 30 points for other results and your discussions.  As before, you can test your code against the grading script yourself by downloading [A4grader.zip](https://www.cs.colostate.edu/~cs445/notebooks/A4grader.zip) and extracting `A4grader.py` parallel to this notebook.  We recommend keeping this notebook and the grader script in a dedicated folder with *just those two files.* Run the code in the in the following cell to see an example grading run.  Submit assignments **through Canvas** following the pattern of the previous assignments. *Do not send your file to the instructor/TA via email or any other medium!*

# In[ ]:


get_ipython().run_line_magic('run', '-i A4grader.py')


# ## Extra Credit
# Earn 5 extra credit point on this assignment by doing the following.
# 
# 1. Combine the train, validate, and test partitions loaded from the MNIST data file into two matrices, `X` and `T`. 
# 2. Using `adam` , `relu` and just one value of `learning_rate` and `n_epochs`, compare several hidden layer architectures. Do so by applying our `generate_k_fold_cross_validation_sets` function as defined in Lecture Notes 10 which forms stratified partitioning, for use in classification problems, to your `X` and `T` matrices using `n_fold` of 3.
# 3. Show results and discuss which architectures you find works the best, and how you determined this.

# In[ ]:




