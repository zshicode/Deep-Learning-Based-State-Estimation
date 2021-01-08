import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import lstm
import transformer
import time
import torch

random_state = np.random.RandomState(0)
torch.manual_seed(0)
T = 0.01
step = 200

'''
Step 1: Provide a de facto system
'''

A= np.array([[1,T, 0.5*T*T],
 [ 0, 1,T],
 [ 0, 0, 1]])
B= [0, 0, 0]
C= [1,0,0]
D= [0]
Q= 0.01*np.eye(3)
R= 0.005*np.eye(1)
m0= [ 0,0,0.1]
P0= 0.1*np.eye(3)

kft = KalmanFilter(
    A,C,Q,R,B,D,m0,P0,
    random_state=random_state
)# model should be
state, observation = kft.sample(
    n_timesteps=step,
    initial_state=m0
)# provide data
#filtered_state_estimatet, f_covt = kft.filter(observation)
#smoothed_state_estimatet, s_covt = kft.smooth(observation)

'''
Step 2: Initialize our model
'''

# specify parameters
transition_matrix = A
transition_offset = B
observation_matrix = C
observation_offset = D
transition_covariance = 0.02*np.eye(3)
observation_covariance = np.eye(1)
initial_state_mean =[0,0,1]
initial_state_covariance = 5*np.eye(3)

# sample from model

kf = KalmanFilter(
    transition_matrix, observation_matrix, transition_covariance,
    observation_covariance, transition_offset, observation_offset,initial_state_mean,initial_state_covariance,
    random_state=random_state,
    em_vars=[
      #'transition_matrices', 'observation_matrices',
      'transition_covariance','observation_covariance',
      #'transition_offsets', 'observation_offsets',
      'initial_state_mean', 'initial_state_covariance'
      ]
)
data = kf.sample(n_timesteps=step,initial_state=initial_state_mean)[1]
filtered_state_estimater, nf_cov = kf.filter(observation)
smoothed_state_estimater, ns_cov = kf.smooth(observation)

'''
Step 3: Learn good values for parameters named in `em_vars` using the EM algorithm
'''

def compute_tr(a):
    size = a.shape[0]
    return (np.trace(a)/size)*np.eye(size)

def test(data,method='TL',n_iteration=10):
    t_start = time.process_time()
    if method == 'TL':
        print('----transformer+lstm----')
        data,loss_list = transformer.train(data,step)
        data,loss_list = lstm.train(data)
        labelfilter = 'TL-KF'
        labelsmooth = 'TL-KS'
    elif method == 'L':
        print('----lstm----')
        data,loss_list = lstm.train(data)
        labelfilter = 'LSTM-KF'
        labelsmooth = 'LSTM-KS'
    elif method == 'T':
        print('----transformer----')
        data,loss_list = transformer.train(data,step)
        labelfilter = 'Transformer-KF'
        labelsmooth = 'Transformer-KS'
    else:
        print('----EM----')
        labelfilter = 'EM-KF'
        labelsmooth = 'EM-KS'
    
    t_train = time.process_time()
    kfem = kf.em(X=data, n_iter=n_iteration)
    t_em = time.process_time()
    print('train-time/sec',t_train-t_start)
    print('em-time/sec',t_em-t_train)
    Qem = compute_tr(kfem.transition_covariance)
    Rem = compute_tr(kfem.observation_covariance)
    P0em = compute_tr(kfem.initial_state_covariance)
    m0em = [0,0,np.abs(kfem.initial_state_mean[2])]
    print('Q=',Qem)
    print('R=',Rem)
    print('m0=',m0em)
    print('P0=',P0em)
    kfem = KalmanFilter(
        A,C,Qem,Rem,B,D,m0em,P0em,
        random_state=random_state
    )
    #obsem = kfem.sample(n_timesteps=step,initial_state=m0)[1]
    filtered_state_estimates, f_cov = kfem.filter(observation)
    smoothed_state_estimates, s_cov = kfem.smooth(observation)
    return filtered_state_estimates, f_cov, smoothed_state_estimates, s_cov,labelfilter,labelsmooth


# draw estimates
filtered_state_estimates, f_cov, smoothed_state_estimates, s_cov, labelfilter,labelsmooth = test(data[:,0],n_iteration=10)
#print('emkf=',filtered_state_estimates[:,0].tolist())
#print('emks=',smoothed_state_estimates[:,0].tolist())
filtered_delta_estimater = filtered_state_estimater[:,0] - state[:,0]
smoothed_delta_estimater = smoothed_state_estimater[:,0] - state[:,0]
filtered_delta_estimates = filtered_state_estimates[:,0] - state[:,0]
smoothed_delta_estimates = smoothed_state_estimates[:,0] - state[:,0]
'''
filtered_delta_estimates_lstm = filtered_state_estimates_lstm[:,0] - state[:,0]
smoothed_delta_estimates_lstm = smoothed_state_estimates_lstm[:,0] - state[:,0]
filtered_delta_estimates_tranf = filtered_state_estimates_tranf[:,0] - state[:,0]
smoothed_delta_estimates_tranf = smoothed_state_estimates_tranf[:,0] - state[:,0]
'''
#smoothed_delta_estimates[step-1] = smoothed_delta_estimates[step-3]
# lines_true = plt.plot(state[:,0],state[:,1] ,color='c',label='true')
#lines_model = plt.plot(state, color='m')
msefr = np.linalg.norm(filtered_delta_estimater)**2/step
msesr = np.linalg.norm(smoothed_delta_estimater)**2/step
msefs = np.linalg.norm(filtered_delta_estimates)**2/step
msess = np.linalg.norm(smoothed_delta_estimates)**2/step
print('----MSE----')
print('KF',msefr)
print('KS',msesr)
print(labelfilter,msefs)
print(labelsmooth,msess)

#draw
taxis = np.linspace(0,step*T,step)
plt.figure()
lines_filter = plt.scatter(taxis,state[:,0], color='c',label='True')
lines_filter = plt.plot(taxis,filtered_state_estimater[:,0], 'r',label='KF')
lines_smoother = plt.plot(taxis,smoothed_state_estimater[:,0], 'r--',label='KS')
lines_filt = plt.plot(taxis,filtered_state_estimates[:,0], 'b',label=labelfilter)
lines_smooth = plt.plot(taxis,smoothed_state_estimates[:,0], 'b--',label=labelsmooth)
plt.xlim(0,step*T)
plt.xlabel('Time/s')
plt.ylabel('x/m')
plt.legend()
plt.grid()
plt.figure()
dlines_filter = plt.plot(taxis,filtered_delta_estimater, 'r',label='KF')
dlines_smoother = plt.plot(taxis,smoothed_delta_estimater, 'r--',label='KS')
dlines_filt = plt.plot(taxis,filtered_delta_estimates, 'b',label=labelfilter)
dlines_smooth = plt.plot(taxis,smoothed_delta_estimates, 'b--',label=labelsmooth)
#plt.plot(observation[:,0] - state[:,0],color='c')
plt.xlim(0,step*T)
plt.xlabel('Time/s')
plt.ylabel('Error/m')
plt.legend()
plt.grid()
plt.show()
#lines_filt1 = plt.plot(filtered_state_estimates1, color='b')
#lines_smooth1 = plt.plotsmoothed_delta_estimates1, color='k')
