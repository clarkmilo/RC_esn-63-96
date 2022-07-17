#import sklearn as skl (not necessary for now)
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

#echo state network###
#first,  define lorenz eqs
#funtion takes input t(time) and y(position) where y is a 3 component vector
#We'll use the parameters given in the paper for now
# all else (for this function) is same from Gauthier paper
def lorenz63(t, y):
      
      dx = 10.0 * (y[1] - y[0])
      dy = y[0] * (28.0 - y[2]) - y[1]
      dz = y[0] * y[1] - (8/3) * y[2]
      
      return [dx, dy, dz]



#define  time (method from gauthier)
#start time
time_start = 0
train_start = 0
train_finish = 60
total_finish = 80
train_npts = total_finish / .05
dt = .050000000000
sols_vector = np.arange(time_start,total_finish,dt)
Training_time_vector = np.arange(train_start, train_finish, dt)

# find solutions (method from gauthier)
yyy = sci.integrate.solve_ivp(lorenz63, (time_start, total_finish), [17.67715816276679, 12.931379185960404, 43.91404334248268] ,t_eval = sols_vector,  method='RK23')


#we've now got a vector of time steps in yyy.t, and our xyz values in yyy.y[]
####
#define W_in
#####
#Define reservoir  
#####

#create training vector

yyy_new = yyy.y[0:,0:400]
yyy_new1 = yyy.y[0:,1:401]

yyy_act_new = yyy_new
yyy_t = yyy.t[200:600]
def res(in_size, res_size, leak_rate, sradius, sparseness):
      #input weight matrix 
      W_in = ((np.random.rand(res_size, in_size)) / 3) 
      #reservoir weight matrix 
      A_w = (np.random.rand(res_size,res_size)-0.5)
      ##establish desired connection density
      deletion_matrix = np.random.rand(res_size, res_size)
      deletions_id = []
      for a,i  in enumerate(deletion_matrix):
            for b,j in enumerate(i):
                  if j < sparseness:
                        deletions_id.append([a,b])

      for i in deletions_id:
            A_w[i[0]][i[1]] = 0


      ##compute spectral radius
      sr1 = np.max(np.abs(np.linalg.eigvals(A_w)))
      ###resize to desired spectral radius
      A_w = (A_w * (1 / sr1)) * sradius
      
      return W_in, A_w

def train(in_weights, res_weights, inputs, outputs, leakrate, alpha):
      #compute reservoir states
      reservoir_states =np.zeros(((len(inputs[0])+1),len(res_weights)) )
      
      reference_ins = np.transpose(inputs)
      for j in range(len(reservoir_states)-1):
            
            reservoir_states[j+1] = (leakrate)*reservoir_states[j] + (1-leakrate)*(np.tanh(np.matmul(res_weights,reservoir_states[j]) +  np.matmul(in_weights,reference_ins[j])))
            
      reservoir_states = reservoir_states[:-1]
      
      #linear transformation must be applied to reservoir states, not entirely sure why, discussed in Noguiera's paper
      #wew'll just use the simplest one outlined, can try others later.
      for a,i in enumerate(reservoir_states,start = 0):
            if a % 2  == 0:
                  i = i ** 2
      Id = np.identity(len(reservoir_states[0]))
      print(reservoir_states)
      reservoir_states = np.transpose(reservoir_states)
      
      #print(inputs)
      Wout = np.matmul(np.matmul(inputs,np.transpose(reservoir_states)),(np.matmul(reservoir_states,np.transpose(reservoir_states))+ alpha*Id))
      #Wout = Ridge(alpha).fit(inputs, reservoir_states)
      return Wout, reservoir_states

###let's try it...
###HYPER PARAMETERS (as given in Nogueira's paper)
leaking_rate = 0.0
reservoir_size = 828
sparsity = .5770
spectral_radius = .13498
a_regularizer = 1.2848 * 10 ** -5
input_vector = 3

###first, intitalize revervoir and inputs

trial_w_in, trial_reservoir = res(input_vector,reservoir_size,leaking_rate,spectral_radius,sparsity)
##Train w_out
q,l = train(trial_w_in, trial_reservoir, yyy_act_new, yyy_new1, leaking_rate, a_regularizer)
#print(q)

## W_out is trained, plot learned values and real solutions side by side

newy= np.zeros((400,3))
newy_res_states = np.zeros((400,reservoir_size))


newy_res_states[0] = np.transpose(l)[-1]
newy[0] = [yyy_new[0][-1],yyy_new[1][-1],yyy_new[2][-1]]
#print(newy)
#print(newy_res_states)

for i in range(len(newy)-1):
      newy_res_states[i+1] = (1-leaking_rate)*newy_res_states[i] + leaking_rate*(np.tanh(np.matmul(trial_reservoir,newy_res_states[i]) +  np.matmul(trial_w_in,newy[i])))
      if (i+1) % 2 ==0:
            newy_res_states[i+1] == newy_res_states[i+1] ** 2
      newy[i+1] = np.matmul(q,newy_res_states[i])
#read values into array
plotter_trainstage = np.zeros(1200)
plotter_teststage = np.zeros(400)
for i in range(400):
      
      plotter_trainstage[i] = yyy_act_new[1][i]
for i in range(400):
      plotter_teststage[i] = newy[i][1]
times_test = yyy.t[400:800]   
plt.plot(yyy.t,yyy.y[1],'r')
plt.plot(times_test,plotter_teststage,'b')




plt.savefig('fml')
