import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
import pandas as pd
import scipy.integrate as sciint
import matplotlib.pyplot as plt
import optuna as opt
def lorenz63(t, y):
      
      dx = 10.0 * (y[1] - y[0])
      dy = y[0] * (28.0 - y[2]) - y[1]
      dz = y[0] * y[1] - (8/3) * y[2]
      
      return [dx, dy, dz]
sols_vector = np.arange(0, 80, .01)
data = sciint.solve_ivp(lorenz63, (0, 80), [17.67715816276679, 12.931379185960404, 43.91404334248268] ,t_eval = sols_vector,  method='RK23').y
# global variables
#This will change the initial condition used. Currently it starts from the first# value
shift_k = 0

approx_res_size = 828


model_params = {'dt': 0.01,
                'total_steps': 8000,
                'N': 3,
                'd': 22}


#Paramaters of our reservoir, what we'll have to optimiz/Tune
res_params = {'radius':0.1,
             'degree': 477,
             'sigma': .5,
             'train_length': 6000,
             'N': 1000,
             'num_inputs': model_params['N'],
             'predict_length': 2000,
             'a_regular': 1.2848*10**-5
              }

res_params = {'radius':0.09894719502645223,
             'degree': 427,
             'sigma': 1.1912158964838269,
             'train_length': 6000,
             'N': 1113,
             'num_inputs': model_params['N'],
             'predict_length': 2000,
             'a_regular': 1.5235677226131174*10**-5
              }

# The ESN functions for training
def generate_reservoir(size,radius,degree):
    sparsity = degree/float(size);
    A = sparse.rand(size,size,density=sparsity).todense()
    vals = np.linalg.eigvals(A)
    e = np.max(np.abs(vals))
    A = (A/e) * radius
    return A

def reservoir_layer(A, Win, input, res_params):
    states = np.zeros((res_params['N'],res_params['train_length']))
    for i in range(res_params['train_length']-1):
        states[:,i+1] = np.tanh(np.dot(A,states[:,i]) + np.dot(Win,input[:,i]))
    return states


def train_reservoir(res_params, data):
    A = generate_reservoir(res_params['N'], res_params['radius'], res_params['degree'])
    q = int(res_params['N']/res_params['num_inputs'])
    Win = np.zeros((res_params['N'],res_params['num_inputs']))
    for i in range(res_params['num_inputs']):
        np.random.seed(seed=i)
        Win[i*q: (i+1)*q,i] = res_params['sigma'] * (-1 + 2 * np.random.rand(1,q)[0])
        
    states = reservoir_layer(A, Win, data, res_params)
    Wout = train(res_params, states, data)
    x = states[:,-1]
    return x, Wout, A, Win

def train(res_params,states,data):
    a_regular = res_params['a_regular']
    idenmat = a_regular * sparse.identity(res_params['N'])
    states2 = states.copy()
    for j in range(2,np.shape(states2)[0]-2):
        if (np.mod(j,2)==0):
            states2[j,:] = (states[j-1,:]*states[j-2,:]).copy()
    U = np.dot(states2,states2.transpose()) + idenmat
    Uinv = np.linalg.inv(U)
    Wout = np.dot(Uinv,np.dot(states2,data.transpose()))
    return Wout.transpose()

def predict(A, Win, res_params, x, Wout):
    output = np.zeros((res_params['num_inputs'],res_params['predict_length']))
    for i in range(res_params['predict_length']):
        x_aug = x.copy()
        for j in range(2,np.shape(x_aug)[0]-2):
            if (np.mod(j,2)==0):
                x_aug[j] = (x[j-1]*x[j-2]).copy()
        out = np.squeeze(np.asarray(np.dot(Wout,x_aug)))
        output[:,i] = out
        x1 = np.tanh(np.dot(A,x) + np.dot(Win,out))
        x = np.squeeze(np.asarray(x1))
    return output, x



# Train reservoir
x,Wout,A,Win = train_reservoir(res_params,data[:,shift_k:shift_k+res_params['train_length']])

# Prediction

output, _ = predict(A, Win,res_params,x,Wout)
np.save('Expansion_2step_back'+'R_size_train_'+str(res_params['train_length'])+'_Rd_'+str(res_params['radius'])+'_Shift_'+str(shift_k)+'.npy',output)

fig, axs = plt.subplots(3, 2, figsize = [17,15] )
axs[0,0].plot(sols_vector[6000:],data[0][6000:],'r', label = 'Actual lorenz value')
axs[0,0].plot(sols_vector[6000:],output[0],'b', label = 'RC x prediction')
axs[0,0].set_title('X')
axs[0,0].legend()
axs[0,1].plot(sols_vector[6000:],np.abs(data[0][6000:] - output[0]),'r',label = 'x-Error (actual- predicted)')
axs[0,1].set_title('X Error')
axs[0,1].legend()
axs[1,0].plot(sols_vector[6000:],data[1][6000:],'r', label = 'Actual lorenz y-value')
axs[1,0].plot(sols_vector[6000:],output[1],'b', label = 'RC y-prediction')
axs[1,0].legend()
axs[1,0].set_title('Y')
axs[1,1].plot(sols_vector[6000:],np.abs(data[1][6000:] - output[1]),'r',label = 'y-Error (actual- predicted)')
axs[1,1].set_title('Y Error')
axs[1,1].legend()
axs[2,0].plot(sols_vector[6000:],data[2][6000:],'r', label = 'Actual lorenz z-value')
axs[2,0].plot(sols_vector[6000:],output[2],'b', label = 'RC z-prediction')
axs[2,0].legend()
axs[0,0].set_title('Z')
axs[2,1].plot(sols_vector[6000:],np.abs(data[2][6000:] - output[2]),'r',label = 'z-Error (actual- predicted)')
axs[2,1].set_title('Z Error')
axs[2,1].legend()
plt.show
plt.savefig('fml')



####ERROR

# compute MSE for the first errorLen time steps
errorLen = 500
#print(dataa)
#print(np.transpose(Y))
trainLen = res_params['train_length']
mse = sum( np.square( np.transpose(data)[trainLen+1:trainLen+errorLen+1] - np.transpose(output)[0:errorLen] ) ) / errorLen
print('MSE = ' + str( mse ))