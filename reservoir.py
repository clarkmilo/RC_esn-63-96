import sklearn as skl
import numpy as np
import scipy as sci
import matplotlib as plt
#echo state network###
#first,  define lorenz eqs
#funtion takes input t(time) and y(position) where y is a 3 component vector
#We'll use the parameters given in the paper for now
# all else (for this function) is same from Gauthier paper
def lorenz63(t, y):
      
      dx = 10.0 * (y[1] - y[0])
      dy = y[0] * (28.0 - y[2]) - y[1]
      dy = y[0] * y[1] - (8/3) * y[2]
      
      return [dy0, dy1, dy2]
# let's integrate to find a solution vector
y_otpeq = solve_ivp(lorenz63, (0, maxtime), [17.67715816276679, 12.931379185960404, 43.91404334248268] , t_eval=t_eval, method='RK23')


#define training time (method from gauthier)
#start time
train_npts = 21
starts = 5
train_start = 4
train_finish = 24
steps = (train_finish-train_start)/ train_npts
training_vector = np.linspace(train_start,train_finish,train_npts)
dt = .025

