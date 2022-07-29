import numpy as np
from BM import *
import scipy.sparse as sparse
from scipy.sparse import linalg
import pandas as pd
import scipy.integrate as sciint
import matplotlib.pyplot as plt
import optuna 
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
             'N': int(np.floor(approx_res_size/model_params['N']) * model_params['N']),
             'num_inputs': model_params['N'],
             'predict_length': 2000,
             'a_regular': 1.2848*10**-5
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





def objective(trial):
    modpams = {'dt': 0.01,
                'total_steps': 8000,
                'N': 3,
                'd': 22}
    respams = {'radius': trial.suggest_float('x',.05,0.2),
             'degree': trial.suggest_int('degree',400,550),
             'sigma': trial.suggest_float('sigma',.1,1.5),
             'train_length': 6000,
             'N': trial.suggest_int('N',700,1300),
             'num_inputs': modpams['N'],
             'predict_length': 2000,
             'a_regular': trial.suggest_float('a_regular',.8*10**-5,1.6*10**-5)
              }
    x,Wout,A,Win = train_reservoir(respams,data[:,0:0+respams['train_length']])

    #Selected trial can be uncommented
    #####FOR BFLOAT 16:#####
    #A = np.float32(A)
    #Win = np.float32(Win)
    #A = f32tob16(A)
    #Win = f32tob16(Win)

    ######FOR FLOAT 16#######
    #A = np.float16(A)
    #Win = np.float16(Win)

    #For neither just leave them commented 
    output, _ = predict(A, Win,respams,x,Wout)
    # compute MSE for the first errorLen time steps
    errorLen = 500

    trainLen = res_params['train_length']
    mse = sum( np.square( np.transpose(data)[trainLen+1:trainLen+errorLen+1] - np.transpose(output)[0:errorLen] ) ) / errorLen
    print('MSE = ' + str( mse ))
    
    
    return np.mean(mse)


study = optuna.create_study()
study.optimize(objective, n_trials=20)

#





