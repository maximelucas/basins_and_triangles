import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numpy import linalg as LA
from parfor import parfor
plt.rc('text', usetex=True)

# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

def kuramoto(N,T,K1,K2,sigma,Q,D):
    # Kuramoto oscillators on a ring w/ higher-order interactions
    # N: number of oscillators, 83
    # T: simulation time, 100
    # K1: pairwise coupling range, 2
    # K2: triadic coupling range, 2
    # sigma: relative triadic coupling strength, 1
    # Q: winding number of the initial condition, 0
    # D: perturbation strength, 1e-10

    sync = np.linspace(0, 2*np.pi*Q, N+1)
    sync = sync[:-1]
    if Q == 0:
        sync = sync + np.pi

    perturbe = 2 * np.pi * (np.random.rand(N) - 0.5)
    perturbe = perturbe - np.mean(perturbe)

    def rhs(theta, t, N, K1, K2, sigma):
        dtheta_dt = np.zeros(N)

        # pairwise coupling
        for ii in range(N):
            for jj in range(-K1,K1+1):
                dtheta_dt[ii] +=  np.sin(theta[(ii+jj)%N] - theta[ii])/K1

        # triadic coupling
        idx = list(range(-K2,0)) + list(range(1,K2+1))
        for ii in range(N):
            for jj in idx:
                for kk in idx:
                    if jj != kk:
                        dtheta_dt[ii] +=  sigma*np.sin(theta[(ii+kk)%N] + theta[(ii+jj)%N] - 2*theta[ii])/(K2*(2.0*K2-1))

        return dtheta_dt

    time = np.linspace(0, T, 1001)
    #initial = sync + D * perturbe
    initial = 2 * np.pi * np.random.rand(N)
    x = odeint(rhs, initial, time, args=(N, K1, K2, sigma))

    return x, time

def winding_number(xa, N, t):
    q = 0
    Delta = np.zeros(N)
    for ii in range(N):
        delta = xa[t, (ii+1)%N] - xa[t, ii]

        if delta > np.pi:
            delta = delta - 2*np.pi
        elif delta < -np.pi:
            delta = delta + 2*np.pi

        q = q + delta
        Delta[ii] = delta

    w_no = round(q / (2*np.pi))
    #print(LA.norm(Delta-np.mean(Delta)))
    Is_twisted_state = LA.norm(Delta-np.mean(Delta))<1e-1

    return w_no, Is_twisted_state

N=83
T=[400,300,200,100,80]
K1=2
K2=2
#sigma=[0,1,2,3,4]
sigma=[0.5,1.5,2.5,3.5,4.5]
Q=5
D=1e-5

if __name__ == '__main__':
    @parfor(range(500000))
    def fun(i):
        idx = np.floor(i/1e5).astype(int)
        x, time = kuramoto(N,T[idx],K1,K2,sigma[idx],Q,D)
        xa = x % (2 * np.pi)

        # winding number of the initial state
        #q_0, Is_twisted_state_0 = winding_number(xa, N, 0)
        # winding number of the final state
        q, Is_twisted_state = winding_number(xa, N, -1)
        #print([q,Is_twisted_state])

        if ~Is_twisted_state:
            q = 99
        
        return q
    #print(fun)
    w = np.array(fun).astype(int)
    w = w.reshape(1, w.shape[0])
    with open("winding_numbers.txt", "ab") as f:
        np.savetxt(f, w, fmt='%i', delimiter=',')