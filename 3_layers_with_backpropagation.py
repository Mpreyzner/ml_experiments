import numpy as np

#collect data
x = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])

y = np.array([[0], [1],[1],[0]])

print x

print y

#build model

num_epochs = 60000

syn0 = 2 * np.random.random((3,4)) -1
syn1 = 2 * np.random.random((4,1)) -1


print syn0
print syn1	

#train mode

def nonlin(x, deriv=False):
	if (deriv == True):
			return x  * (1-x)

	return 1/(1+np.exp(-x))

for j in xrange(num_epochs):
	#feed forward through layers 0,1,2
	l0 = x
	l1 = nonlin(np.dot(l0,syn0))
	l2 = nonlin(np.dot(l1, syn1))


	l2_error = y - l2

	#in what direction s the target value
	l2_delta = l2_error * nonlin(l2, deriv=True)

	#how much l2 contributes to l2 error
	l1_error = l2_delta.dot(syn1.T)
	l1_delta = l1_error * nonlin(l1, deriv=True)

	syn1 += l1.T.dot(l2_delta)
	syn0 += l0.T.dot(l1_delta)

