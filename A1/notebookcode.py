#!/usr/bin/env python
# coding: utf-8

# # A1.1 Linear Regression with SGD

# * A1.1: *Added preliminary grading script in last cells of notebook.*

# In this assignment, you will implement three functions `train`, `use`, and `rmse` and apply them to some weather data.
# Here are the specifications for these functions, which you must satisfy.

# `model = train(X, T, learning_rate, n_epochs, verbose)`
# * `X`: is an $N$ x $D$ matrix of input data samples, one per row. $N$ is the number of samples and $D$ is the number of variable values in
# each sample.
# * `T`: is an $N$ x $K$ matrix of desired target values for each sample.  $K$ is the number of output values you want to predict for each sample.
# * `learning_rate`: is a scalar that controls the step size of each update to the weight values.
# * `n_epochs`: is the number of epochs, or passes, through all $N$ samples, to take while updating the weight values.
# * `verbose`: is True or False (default value) to control whether or not occasional text is printed to show the training progress.
# * `model`: is the returned value, which must be a dictionary with the keys `'w'`, `'Xmeans'`, `'Xstds'`, `'Tmeans'` and `'Tstds'`.
# 
# `Y = use(X, model)`
# * `X`: is an $N$ x $D$ matrix of input data samples, one per row, for which you want to predict the target values.
# * `model`: is the dictionary returned by `train`.
# * `Y`: is the returned $N$ x $K$ matrix of predicted values, one for each sample in `X`.
# 
# `result = rmse(Y, T)`
# * `Y`: is an $N$ x $K$ matrix of predictions produced by `use`.
# * `T`: is the $N$ x $K$ matrix of target values.
# * `result`: is a scalar calculated as the square root of the mean of the squared differences between each sample (row) in `Y` and `T`.

# To get you started, here are the standard imports we need.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas


# ## 60 points: 40 for train, 10 for use, 10 for rmse

# Now here is a start at defining the `train`, `use`, and `rmse`
# functions.  Fill in the correct code wherever you see `. . .` with
# one or more lines of code.

# In[2]:


def train(X, T, learning_rate, n_epochs, verbose=False):

    # Calculate means and standard deviations of each column in X and T
    meanX = np.mean(X, axis = 0)
    meanT = np.mean(T, axis = 0)
    devX = np.std(X, axis = 0)
    devT = np.std(T, axis = 0)
    # print(meanX)
    # print(meanT)
    # print(devX)
    # print(devT)
    
    
    # Use the means and standard deviations to standardize X and T
    stdX = (X - meanX) / devX
    stdT = (T - meanT) / devT
    T = stdT
    # print(T)
    
    # Insert the column of constant 1's as a new initial column in X
    X = np.insert(stdX, 0, 1, 1)
    
    # Initialize weights to be a numpy array of the correct shape and all zeros values.
    weight = np.zeros((X.shape[1], T.shape[1]))
    n_samples = X.shape[0]

    for epoch in range(n_epochs):
        sqerror_sum = 0

        for n in range(n_samples):

            # Use current weight values to predict output for sample n, then
            # calculate the error, and
            # update the weight values.
            Y = X[n : n + 1, :] @ weight
            error = T[n : n + 1, :] - Y
            weight = weight + learning_rate * X[n : n + 1, :].T * error
            # Add the squared error to sqerror_sum
            sqerror_sum = sqerror_sum + error**2
            
        if verbose and (n_epochs < 11 or (epoch + 1) % (n_epochs // 10) == 0):
            rmse = np.sqrt(sqerror_sum / n_samples)
            rmse = rmse[0, 0]  # because rmse is 1x1 matrix
            print(f'Epoch {epoch + 1} RMSE {rmse:.2f}')

    return {'w': weight, 'Xmeans': meanX, 'Xstds': devX,
            'Tmeans': meanT, 'Tstds': devT}


# In[3]:


def use(X, model):
    # Standardize X using Xmeans and Xstds in model
    X = (X - model['Xmeans']) / model['Xstds']
    X = np.insert(X, 0, 1, axis = 1)
    # Predict output values using weights in model
    weights = model["w"]
    stdY = X @ weights
    # Unstandardize the predicted output values using Tmeans and Tstds in model
    meanT = model["Tmeans"]
    devT = model["Tstds"]
    Y = (stdY * devT) + meanT
    # Return the unstandardized output values
    return Y


# In[4]:


def rmse(A, B):
    meanAB = np.mean((A - B)**2, axis = 0)
    return np.sqrt(meanAB)


# Here is a simple example use of your functions to help you debug them.  Your functions must produce the same results.

# In[5]:


X = np.arange(0, 100).reshape(-1, 1)  # make X a 100 x 1 matrix
T = 0.5 + 0.3 * X + 0.005 * (X - 50) ** 2
plt.plot(X, T, '.')
plt.xlabel('X')
plt.ylabel('T');


# In[6]:


model = train(X, T, 0.01, 50, verbose=True)
model


# In[7]:


Y = use(X, model)
plt.plot(T, '.', label='T')
plt.plot(Y, '.', label='Y')
plt.legend()


# In[8]:


plt.plot(Y[:, 0], T[:, 0], 'o')
plt.xlabel('Predicted')
plt.ylabel('Actual')
a = max(min(Y[:, 0]), min(T[:, 0]))
b = min(max(Y[:, 0]), max(T[:, 0]))
plt.plot([a, b], [a, b], 'r', linewidth=3)


# ## Weather Data

# Now that your functions are working, we can apply them to some real data. We will use data
# from  [CSU's CoAgMet Station Daily Data Access](http://coagmet.colostate.edu/cgi-bin/dailydata_form.pl).
# 
# You can get the data file [here](http://www.cs.colostate.edu/~cs445/notebooks/A1_data.txt)

# ## 5 points:
# 
# Read in the data into variable `df` using `pandas.read_csv` like we did in lecture notes.
# Missing values in this dataset are indicated by the string `'***'`.

# In[9]:


df = pandas.read_csv('A1_data.txt', delim_whitespace=True, na_values="***")
df


# ## 5 points:
# 
# Check for missing values by showing the number of NA values, as shown in lecture notes.

# In[10]:


df.isna().sum()


# ## 5 points:
# 
# If there are missing values, remove either samples or features that contain missing values. Prove that you
# were successful by counting the number of missing values now, which should be zero.

# In[11]:


df = df.dropna()
df.isna().sum()


# Your job is now to create a linear model that predicts the next day's average temperature (tave) from the previous day's values. A discription of all features can be found [here](https://coagmet.colostate.edu/rawdata_docs.php). To start, consider just focusing on these features: 
# 1. tave: average temperature
# 2. tmax: maximum temperature
# 3. tmin: minimum temperature
# 4. vp: vapor pressure
# 5. rhmax: maximum relative humidity
# 6. rhmin: minimum relative humidity
# 7. pp: precipitation
# 8. gust: wind gust speed
# 
# First, modify the datafile to add a new column: 'next tave' -- here's a hint on your X and T vectors names:

# In[12]:


Xnames = ['tave', 'tmax', 'tmin', 'vp', 'rhmax', 'rhmin', 'pp', 'gust']
Tnames = ['next tave']


# ## 5 points:
# 
# Now select those eight columns from `df` and convert the result to a `numpy` array.  (Easier than it sounds.)
# Then assign `X` to be all columns and all but the last row.  Assign `T` to be just the first column (tave) and all but the first sample.  So now the first row (sample) in `X` is associated with the first row (sample) in `T` which tave for the following day.

# In[16]:


df = pandas.read_csv('A1_data.txt', delim_whitespace=True, na_values="***")
df["next tave"] = ""
df.to_csv("A1_data.csv", index=False)
data = df[Xnames].to_numpy()
X = data[:-1, :]
T = data[1:, 0:1]
T.shape, X.shape


# ## 15 points:
# 
# Use the function `train` to train a model for the `X`
# and `T` data.  Run it several times with different `learning_rate`
# and `n_epochs` values to produce decreasing errors. Use the `use`
# function and plots of `T` versus predicted `Y` values to show how
# well the model is working.  Type your observations of the plot and of the value of `rmse` to discuss how well the model succeeds.

# In[17]:


m1 = train(X, T, 0.001, 10, verbose=True)
Y1 = use(X, m1)
m2 = train(X, T, 0.001, 100, verbose=True)
Y2 = use(X, m2)
m3 = train(X, T, 0.0001, 10, verbose=True)
Y3 = use(X, m3)
m4 = train(X, T, 0.0001, 100, verbose=True)
Y4 = use(X, m4)
m5 = train(X, T, 0.01, 100, verbose=True)
Y5 = use(X, m5)

plt.plot(Y1[:365,:], '.', label='Y1')
plt.plot(T[:365,:], '.', label='T')
plt.legend()
plt.show()
plt.plot(Y2[:365,:], '.', label='Y2')
plt.plot(T[:365,:], '.', label='T')
plt.legend()
plt.show()
plt.plot(Y3[:365,:], '.', label='Y3')
plt.plot(T[:365,:], '.', label='T')
plt.legend()
plt.show()
plt.plot(Y4[:365,:], '.', label='Y4')
plt.plot(T[:365,:], '.', label='T')
plt.legend()
plt.show()
plt.plot(Y5[:365,:], '.', label='Y5')
plt.plot(T[:365,:], '.', label='T')
plt.legend()
plt.show()
# Observations: 
# We trained 4 sets of data, with different sets of learning rate and number of epochs, 
# The RMSE and the plot shows that the data predicts best at learning rate of 0.001 and number 
# of epochs at n = 100. This gives us the least number of RMSE and the predicted dots give us
# the best fit to the actual data. The model also gives a decent prediction when the learning
# rate is around 0.001(or slightly above), and deviates from the real value more as it gets lower. 


# ## 5 points:
# 
# Print the weight values in the resulting model along with their corresponding variable names (in `Xnames`). Use the relative magnitude
# of the weight values to discuss which input variables are most significant in predicting the changes in the tave values.

# In[18]:


row_num = len(Xnames)
col_num = range(row_num)
model_list = [m1, m2, m3, m4, m5]
index = 1
for model in model_list: 
    print("Model", index, ": ")
    index += 1
    for col in col_num:
        var_name = Xnames[col]
        print(Xnames[col], ": ", model['w'][col])
    print()

    
# Observations: 
# Maximum relative humidity, vp have the highest relative weights in the five models we selected, and therefore should have the most impact on the predicted temperature of the next day. 


# ## Grading and Check-in
# 
# You must name your notebook as `Lastname-A1.ipynb` with `Lastname` being your last name, and then save this notebook and check it in at the A1 assignment link in our Canvas web page.

# ## Extra Credit: 1 point

# A typical problem when predicting the next value in a time series is
# that the best solution may be to predict the previous value.  The
# predicted value will look a lot like the input tave value shifted on
# time step later.
# 
# To do better, try predicting the change in tave from one day to the next. `T` can be assigned as

# In[ ]:


T = data[1:, 0:1] -  data[:-1, 0:1]


# Now repeat the training experiments to pick good `learning_rate` and
# `n_epochs`.  Use predicted values to produce next day tave values by
# adding the predicted values to the previous day's tave.  Use `rmse`
# to determine if this way of predicting next tave is better than
# directly predicting tave.

# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# # A1.1 Linear Regression with SGD

# * A1.1: *Added preliminary grading script in last cells of notebook.*

# In this assignment, you will implement three functions `train`, `use`, and `rmse` and apply them to some weather data.
# Here are the specifications for these functions, which you must satisfy.

# `model = train(X, T, learning_rate, n_epochs, verbose)`
# * `X`: is an $N$ x $D$ matrix of input data samples, one per row. $N$ is the number of samples and $D$ is the number of variable values in
# each sample.
# * `T`: is an $N$ x $K$ matrix of desired target values for each sample.  $K$ is the number of output values you want to predict for each sample.
# * `learning_rate`: is a scalar that controls the step size of each update to the weight values.
# * `n_epochs`: is the number of epochs, or passes, through all $N$ samples, to take while updating the weight values.
# * `verbose`: is True or False (default value) to control whether or not occasional text is printed to show the training progress.
# * `model`: is the returned value, which must be a dictionary with the keys `'w'`, `'Xmeans'`, `'Xstds'`, `'Tmeans'` and `'Tstds'`.
# 
# `Y = use(X, model)`
# * `X`: is an $N$ x $D$ matrix of input data samples, one per row, for which you want to predict the target values.
# * `model`: is the dictionary returned by `train`.
# * `Y`: is the returned $N$ x $K$ matrix of predicted values, one for each sample in `X`.
# 
# `result = rmse(Y, T)`
# * `Y`: is an $N$ x $K$ matrix of predictions produced by `use`.
# * `T`: is the $N$ x $K$ matrix of target values.
# * `result`: is a scalar calculated as the square root of the mean of the squared differences between each sample (row) in `Y` and `T`.

# To get you started, here are the standard imports we need.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## 60 points: 40 for train, 10 for use, 10 for rmse

# Now here is a start at defining the `train`, `use`, and `rmse`
# functions.  Fill in the correct code wherever you see `. . .` with
# one or more lines of code.

# In[25]:


def train(X, T, learning_rate, n_epochs, verbose=False):
    # Calculate means and standard deviations of each column in X and T

    Xmeans = np.mean(X, axis = 0)
    Xstds = np.std(X, axis = 0)

    Tmeans = np.mean(T, axis = 0)
    Tstds = np.std(T, axis = 0)
   
    # Use the means and standard deviations to standardize X and T
    X = (X - Xmeans)/Xstds
    #np.mean(X, axis=0), np.std(X,axis=0)

    T = (T - Tmeans) / Tstds
    #np.mean(T, axis=0), np.std(T,axis=0)

    # Insert the column of constant 1's as a new initial column in X

    X1 = np.insert(X,0,1,axis=1)
    n_samples, n_inputs = X1.shape
   
    # Initialize weights to be a numpy array of the correct shape and all zeros values.
    rows, cols = T.shape
    w = np.zeros((n_inputs,cols))

    for epoch in range(n_epochs):
        sqerror_sum = 0

        for n in range(n_samples):

            # Use current weight values to predict output for sample n, then
            # calculate the error, and
            # update the weight values.
            y = X1[n:n + 1, :] @ w
            error = T[n:n + 1, :] - y
            w += learning_rate * X1[n:n + 1, :].T * error
           
           
            # Add the squared error to sqerror_sum
            sqerror_sum += error ** 2
           
        if verbose and (n_epochs < 11 or (epoch + 1) % (n_epochs // 10) == 0):
            rmse = np.sqrt(sqerror_sum / n_samples)
            rmse = rmse[0, 0]  # because rmse is 1x1 matrix
            print(f'Epoch {epoch + 1} RMSE {rmse:.2f}')

    return {'w': w, 'Xmeans': Xmeans, 'Xstds': Xstds,
            'Tmeans': Tmeans, 'Tstds': Tstds}


# In[26]:


def use(X, model):
    # Standardize X using Xmeans and Xstds in model
    X = (X - model['Xmeans'])/model['Xstds']
    #np.mean(X, axis=0), np.std(X,axis=0)

    # Predict output values using weights in model
    X1 = np.insert(X,0,1,axis=1)
    predict = X1 @ model['w']
   
    # Unstandardize the predicted output values using Tmeans and Tstds in model
    Y = predict * model['Tstds'] + model['Tmeans']
   
    # Return the unstandardized output values
    return Y


# In[4]:


def rmse(A, B):
    # Y = Predictions  :  T = Targets
    return np.sqrt(np.mean((A-B)**2))


# Here is a simple example use of your functions to help you debug them.  Your functions must produce the same results.

# In[5]:


X = np.arange(0, 100).reshape(-1, 1)  # make X a 100 x 1 matrix
T = 0.5 + 0.3 * X + 0.005 * (X - 50) ** 2
plt.plot(X, T, '.')
plt.xlabel('X')
plt.ylabel('T');


# In[6]:


model = train(X, T, 0.01, 50, verbose=True)
model


# In[7]:


Y = use(X, model)
plt.plot(T, '.', label='T')
plt.plot(Y, '.', label='Y')
plt.legend()


# In[8]:


plt.plot(Y[:, 0], T[:, 0], 'o')
plt.xlabel('Predicted')
plt.ylabel('Actual')
a = max(min(Y[:, 0]), min(T[:, 0]))
b = min(max(Y[:, 0]), max(T[:, 0]))
plt.plot([a, b], [a, b], 'r', linewidth=3)


# ## Weather Data

# Now that your functions are working, we can apply them to some real data. We will use data
# from  [CSU's CoAgMet Station Daily Data Access](http://coagmet.colostate.edu/cgi-bin/dailydata_form.pl).
# 
# You can get the data file [here](http://www.cs.colostate.edu/~cs445/notebooks/A1_data.txt)

# ## 5 points:
# 
# Read in the data into variable `df` using `pandas.read_csv` like we did in lecture notes.
# Missing values in this dataset are indicated by the string `'***'`.

# In[9]:


df = pd.read_csv('A1_data.txt', delim_whitespace=True, na_values="***")
df


# ## 5 points:
# 
# Check for missing values by showing the number of NA values, as shown in lecture notes.

# In[10]:


df.isna().sum()


# ## 5 points:
# 
# If there are missing values, remove either samples or features that contain missing values. Prove that you
# were successful by counting the number of missing values now, which should be zero.

# In[11]:


# Due to the way my code is written this has to come before I can remove the missing values
Xnames = ['tave', 'tmax', 'tmin', 'vp', 'rhmax', 'rhmin', 'pp', 'gust']
Tnames = ['next tave']


# Your job is now to create a linear model that predicts the next day's average temperature (tave) from the previous day's values. A discription of all features can be found [here](https://coagmet.colostate.edu/rawdata_docs.php). To start, consider just focusing on these features: 
# 1. tave: average temperature
# 2. tmax: maximum temperature
# 3. tmin: minimum temperature
# 4. vp: vapor pressure
# 5. rhmax: maximum relative humidity
# 6. rhmin: minimum relative humidity
# 7. pp: precipitation
# 8. gust: wind gust speed
# 
# First, modify the datafile to add a new column: 'next tave' -- here's a hint on your X and T vectors names:

# In[12]:


df = df[Xnames]
df = df.dropna()
df.isna().sum()


# ## 5 points:
# 
# Now select those eight columns from `df` and convert the result to a `numpy` array.  (Easier than it sounds.)
# Then assign `X` to be all columns and all but the last row.  Assign `T` to be just the first column (tave) and all but the first sample.  So now the first row (sample) in `X` is associated with the first row (sample) in `T` which tave for the following day.

# In[13]:


T_df = df[['tave']][1:]
T = T_df.to_numpy()
X = df.to_numpy()
X = X[:-1]
T.shape, X.shape


# ## 15 points:
# 
# Use the function `train` to train a model for the `X`
# and `T` data.  Run it several times with different `learning_rate`
# and `n_epochs` values to produce decreasing errors. Use the `use`
# function and plots of `T` versus predicted `Y` values to show how
# well the model is working.  Type your observations of the plot and of the value of `rmse` to discuss how well the model succeeds.

# In[81]:


print('Model:1')

model1 = train(X, T, .1, 50, verbose=True)
Y1 = use(X, model1)
rmse1 = rmse(Y1,T)
print('rmse 1:', rmse1)

plt.plot(Y1[:365,:], '.', label='Y1')
plt.plot(T[:365,:], '.', label='T')
plt.legend()
plt.show()

'''
The plot fits very good but we can probably do better because
the learning_rate is large. With a learning_rate of .1 the 
lowest rmse that can be reached is 6.33. It doesn't matter how 
large the epoch gets either, it will remain at a constant rmse of
6.33. A large concentration of the predictions do line up with the 
target values, but we can do better than this. Meaning we can 
get a lower rmse than 6.33.
'''

print('\nModel:2')

model2 = train(X, T, 0.01, 10, verbose=True)
Y2 = use(X, model2)
rmse2 = rmse(Y2,T)
print('rmse 2:', rmse2)

plt.plot(Y2[:365,:], '.', label='Y2')
plt.plot(T[:365,:], '.', label='T')
plt.legend()
plt.show()

'''
The plot is at an extremely great fit. There is a high
concentration of predicted values in the target value area.
The epoch is at 10 because it doesn't need to get much 
bigger to produce the same constant of rmse 3.43. At an 
epoch of 10, rmse is at a constant of 3.43 and the same 
is true when epoch is at 100 & 1000 & so on. This leads me to 
believe there is likely a lower constant than 3.43 that we 
can manipulate the plot to be. With rmse being 3.43.
'''

print('\nModel:3')

model3 = train(X, T, 0.001, 50, verbose=True)
Y3 = use(X, model3)
rmse3 = rmse(Y3,T)
print('rmse 3:', rmse3)

plt.plot(Y3[:365,:], '.', label='Y3')
plt.plot(T[:365,:], '.', label='T')
plt.legend()
plt.show()

'''
From my investigation with the learning rate & epoch
manipulation, I found that the lowest rmse can be is 3.32. So 
because this is the lowest rmse value produced from all the 
models provided, it makes it the most successful model of them 
all. Again, the epoch does not need to be much higher than 50 
because the lowest rmse can be according to this model is 3.32
and an epoch of 50 already produces this.
'''

print('\nModel:4')

model4 = train(X, T, 0.00001, 50, verbose=True)
Y4 = use(X, model4)
rmse4 = rmse(Y4,T)
print('rmse 4:', rmse4)

plt.plot(Y4[:365,:], '.', label='Y4')
plt.plot(T[:365,:], '.', label='T')
plt.legend()
plt.show()

'''
This model is not very successful in having a large 
concentration of predicted values fall in the target area.
The reason for this is that the rmse is at 5.74. This is a
good illustration to compare model 3 with, so that you can 
see with your own eyes why the learning rate and epoch matter
when calculating this.
'''

print('\nModel:5')

model5 = train(X, T, 0.0000001, 100, verbose=True)
Y5 = use(X, model5)
rmse5 = rmse(Y5,T)
print('rmse 5:', rmse5)

'''
This model has the worst fit out of all the ones provided.
the rmse is at a 9.87, which is the highest out of all the
models which is why it is the worst model of all. I displayed
this model to show what the model should not look like. Do 
not use this model for any practical use.
'''

plt.plot(Y5[:365,:], '.', label='Y5')
plt.plot(T[:365,:], '.', label='T')
plt.legend()
plt.show()

models = [model1,model2,model3,model4,model5]


# ## 5 points:
# 
# Print the weight values in the resulting model along with their corresponding variable names (in `Xnames`). Use the relative magnitude
# of the weight values to discuss which input variables are most significant in predicting the changes in the tave values.

# In[82]:


num = 0
for _model in models:
    weights = pd.DataFrame(_model['w'][1:], index=Xnames, columns=['weights'])
    print(weights)


# ## Grading and Check-in
# 
# Your notebook will be partially graded automatically.  You can test this grading process yourself by downloading [A1grader.zip](https://www.cs.colostate.edu/~cs445/notebooks/A1grader.zip) and extract `A1grader.py` parallel to this notebook.  Run the code in the in the following cell to see an example grading run.  If your functions are defined correctly, you should see a score of 60/60.  The remaining 40 points are based on testing other data and your discussion.

# In[ ]:


get_ipython().run_line_magic('run', '-i A1grader.py')


# A different but similar grading script will be used to grade yout checked-in notebook.  It will include different tests.
# 
# You must name your notebook as `Lastname-A1.ipynb` with `Lastname` being your last name, and then save this notebook and check it in at the A1 assignment link in our Canvas web page.

# ## Extra Credit: 1 point

# A typical problem when predicting the next value in a time series is
# that the best solution may be to predict the previous value.  The
# predicted value will look a lot like the input tave value shifted on
# time step later.
# 
# To do better, try predicting the change in tave from one day to the next. `T` can be assigned as

# In[ ]:


T = data[1:, 0:1] -  data[:-1, 0:1]


# Now repeat the training experiments to pick good `learning_rate` and
# `n_epochs`.  Use predicted values to produce next day tave values by
# adding the predicted values to the previous day's tave.  Use `rmse`
# to determine if this way of predicting next tave is better than
# directly predicting tave.
