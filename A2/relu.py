import numpy as np
import copy

def relu(s):
	'''S is n_samples x n_units'''
	Y = s.copy()
	Y[Y < 0] = 0
	return Y

def grad_relu(s):
	'''S is n_samples x n_units'''
	dY = s.copy()
	return (dY > 0).astype(int)


S = 5 - np.arange(5*2).reshape(5,2)
#S = np.array([-5.5,-4.5,-3.5,-2.5,-1.5,-0.5,0,0.5,1.5,2.5,3.5])
print(S)

print(relu(S))


S = grad_relu(S)

print(S)

