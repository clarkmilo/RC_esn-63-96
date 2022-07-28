
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.linalg as linalg
import scipy.sparse as sparse
import optuna as opt

# numpy.linalg is also an option for even fewer dependencies
#define lorenz 63
'''
Hyperparameters as given in Nogueira's paper:

leaking_rate = 0.2212
reservoir_size = 828
sparsity = .5770
spectral_radius = 1.3498
a_regularizer = 1.2848 * 10 ** -5
input_vector = 3
'''
def lorenz63(t, y):
      
      dx = 10.0 * (y[1] - y[0])
      dy = y[0] * (28.0 - y[2]) - y[1]
      dz = y[0] * y[1] - (8/3) * y[2]
      
      return [dx, dy, dz]
#generate some solutions for training
#hollow time vector
trainLen = 4000
testLen = 2000
initLen = 100
dt = .00500000
Training_time_vector = np.arange(0, (trainLen+testLen)*dt, dt)

data = integrate.solve_ivp(lorenz63, (0, (trainLen+testLen)*dt), [17.67715816276679, 12.931379185960404, 43.91404334248268] ,t_eval = Training_time_vector,  method='RK23')

# plot training data
plt.figure(10).clear()
for i in data.y:
    
    plt.plot(data.t[:1000],(i[:1000]))
plt.title('sample plots of training data')
plt.savefig('ftp')
# generate the ESN reservoir
inSize = outSize = 3
resSize = 828
a =  0.2212 # leaking rate
#np.random.seed(42)
Win = (np.random.rand(resSize,inSize) - 0.5) * 2
#W = (np.random.rand(resSize,resSize) - 0.5) * 1 
#establish desired sparsity

Sparsity = .5770
"""
deletion_matrix = np.random.rand(resSize, resSize)
deletions_id = []
for a,i  in enumerate(deletion_matrix):
    for b,j in enumerate(i):
        if j > Sparsity:
            deletions_id.append([a,b])

    for i in deletions_id:
        W[i[0]][i[1]] = 0
"""
W = sparse.rand(resSize,resSize,density=Sparsity).todense() 


# normalizing and setting spectral radius (correct, slow):
print('Computing spectral radius...')
rhoW = np.max(np.abs(linalg.eigvals(W)))
print('done.', rhoW)
spectral_r = 1.3498
W = (W / rhoW) * .3498
#print(W)
#print(Win)
lno = np.zeros((resSize,1))
print(np.dot(W,lno))
# allocated memory for the design (collected states) matrix
X = np.zeros((resSize,trainLen-initLen))
# set the corresponding target matrix directly
Yt = data.y[0:,initLen:trainLen] 
dataa = np.transpose(data.y)
# run the reservoir with the data and collect X
x = np.zeros((resSize,1))
for t in range(trainLen):
    u = dataa[t] / 15
    
    x = (a)*x +  (1-a)*np.tanh( np.dot( Win, np.vstack((u)) ) + np.dot( W, x ) )
    #print(x)
    if t >= initLen:
        X[:,t-initLen] = np.hstack((x))[:,0]
   
# train the output by ridge regression
reg = 1.2848 * 10 ** -5  # regularization coefficient
# direct equations from texts:
#X_T = X.T
#Wout = np.dot( np.dot(Yt,X_T), linalg.inv( np.dot(X,X_T) + \
#    reg*np.eye(1+inSize+resSize) ) )
# using scipy.linalg.solve:
#print(X)
for i in X:
    for a,j in enumerate(i):
        if a%2 == 0:
            j=j**2

Wout = linalg.solve( np.dot(X,X.T) + reg*np.eye(resSize), 
    np.dot(X,Yt.T) ).T

# run the trained ESN in a generative mode. no need to initialize here, 
# because x is initialized with training data and we continue from there.
Y = np.zeros((outSize,testLen))
u = dataa[trainLen+1]
for t in range(testLen):
    
    x = (a)*x + (1-a)*np.tanh( np.dot( Win, np.vstack((u)) ) + np.dot( W, x ) )
    for g,i in enumerate(x):
        if g%2==0:
            i=i**2
    y = np.dot( Wout, np.vstack((x)) )
    Y[:,t] = np.transpose(y)
    # generative mode:
    u = y
    ## this would be a predictive mode:
    #u = data[trainLen+t+1] 
'''
# compute MSE for the first errorLen time steps
errorLen = 500
#print(dataa)
#print(np.transpose(Y))
mse = sum( np.square( dataa[trainLen+1:trainLen+errorLen+1] - np.transpose(Y)[0:errorLen] ) ) / errorLen
print('MSE = ' + str( mse ))
'''
y_news = np.transpose(Y.T)
# plot some signals
plt.figure(1).clear()
plt.plot( data.y[1][trainLen+1:trainLen+800+1], 'g' )
plt.plot( y_news[1][0:800] * 15, 'b' )
plt.title('Target and generated signals $y(n)$ starting at $n=0$')
plt.legend(['Target signal', 'Free-running predicted signal'])

plt.figure(2).clear()
plt.plot( X[0:20,0:200].T )
plt.title(r'Some reservoir activations $\mathbf{x}(n)$')

#plt.figure(3).clear()
#plt.bar( np.arange(1+inSize+resSize), Wout[0].T )
#plt.title(r'Output weights $\mathbf{W}^{out}$')

plt.show()
#plt.savefig('fml')

