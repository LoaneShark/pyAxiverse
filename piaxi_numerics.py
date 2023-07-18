import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
#import multiprocess as mp
import time
from datetime import timedelta
#import pathos
#import dill

params = {}
signstr = {1: '+', -1: '-', 0: '±'}

## Set parameters of model for use in numerical integration
def set_params(params_in: dict, sample_delta=True, sample_theta=True, t_max=10, t_min=0, t_N=500, 
               k_max=200, k_min=1, k_N=200):
    # Define global variables
    global params
    global t
    global t_span
    global t_num
    global t_step
    global k_values
    global k_span
    global k_num
    global k_step
    global e
    global F
    global p_t
    global eps
    global L3
    global L4
    global l1
    global l2
    global l3
    global l4
    global A_0
    global Adot_0
    global A_pm
    global m
    global p
    global amps
    global d
    global Th
    
    # t domain
    t_span = [t_min, t_max]  # Time range
    t_num  = t_N             # Number of timesteps
    t, t_step = np.linspace(t_span[0], t_span[1], t_num, retstep=True)
    t_sens = params_in['t_sens']
    # k domain
    k_span = [k_min, k_max] # Momentum range
    k_num  = k_N            # Number of k-modes
    k_values, k_step = np.linspace(k_span[0], k_span[1], k_num, retstep=True)
    # resonance condition threshold
    res_con = params_in['res_con']

    # Set constants of model (otherwise default)
    e = params_in['e'] if 'e' in params_in else 0.3        #
    F = params_in['F'] if 'F' in params_in else 1e20       # eV
    p_t = params_in['p_t'] if 'p_t' in params_in else 0.4  # eV / cm^3
    
    ## Tuneable constants
    # millicharge, vary to enable/disable charged species (10e-15 = OFF, 10e-25 = ON)
    #eps  = 1e-25   # (unitless)
    eps  = params_in['eps'] #
    
    # dQCD sector parameters
    qm    = params_in['qm']    if 'qm'    in params_in else np.full((3, ), None)
    qc    = params_in['qc']    if 'qc'    in params_in else np.full((6, ), None)
    dqm   = params_in['dqm']   if 'dqm'   in params_in else np.full((6, ), None)
    eps_c = params_in['eps_c'] if 'eps_c' in params_in else np.full((9, ), None)
    xi    = params_in['xi']    if 'xi'    in params_in else np.full((9, ), None)
    
    # Coupling constants
    L3 = params_in['L3'] # eV
    L4 = params_in['L4'] # eV
    l1 = params_in['l1'] #
    l2 = params_in['l2'] #
    l3 = params_in['l3'] #
    l4 = params_in['l4'] #
    
    # Initial Conditions
    A_0    = params_in['A_0']       # initial implitude
    Adot_0 = params_in['Adot_0']    # initial rate of change
    A_pm   = params_in['A_pm']      # specify A± case (+1 or -1)
    A_sens = params_in['A_sens']    # amplitude sensitivity/scale for classification of resonance strength

    # masses for real, complex, and charged species in range (10e-8, 10e-4) [eV]
    m   = params_in['m']
    N_r = m_r.count() if 'm_r' in params_in else 1                           # number of real species
    m_r = params_in['m_r'] if 'm_r' in params_in else np.full((N_r, ), m[0]) # (neutral) real species
    N_n = m_n.count() if 'm_n' in params_in else 1                           # number of neutral species
    m_n = params_in['m_n'] if 'm_n' in params_in else np.full((N_n, ), m[1]) # neutral (complex) species
    N_c = m_c.count() if 'm_c' in params_in else 1                           # number of charged species
    m_c = params_in['m_c'] if 'm_c' in params_in else np.full((N_c, ), m[2]) # charged (complex) species

    # local DM densities for each species [eV/cm^3]
    p    = params_in['p'] if 'p' in params_in else np.array([np.full(p_t/3,N_r), np.full(p_t/3,N_n), np.full(p_t/3,N_c)], dtype=object) # default to equal distribution
    p_r  = params_in['p_r'] if 'p_r' in params_in else np.full((N_r, ), None)   # (neutral) real species
    p_n  = params_in['p_n'] if 'p_n' in params_in else np.full((N_n, ), None)   # neutral (complex) species
    p_c  = params_in['p_c'] if 'p_c' in params_in else np.full((N_c, ), None)   # charged (complex) species
    # initial amplitudes for each species
    amps = params_in['amps'] if 'amps' in params_in else [np.sqrt(2 * p[i]) / m[i] for i in range(len(m))] # default to equal distribution

    # local phases for each species in (0, 2pi)
    d    = params_in['d'] if 'd' in params_in else np.array([np.zeros(float(N_r)), np.zeros(float(N_n)), np.zeros(float(N_c))], dtype=object)

    # global phase for neutral complex species in (0, 2pi)
    Th   = params_in['Th'] if 'Th' in params_in else np.array([np.zeros(float(N_r)), np.zeros(float(N_n)), np.zeros(float(N_c))], dtype=object)
    
    # Sample phases from normal distribution, sampled within range (0, 2pi)
    mu_d  = params_in['mu_d'] if 'mu_d' in params_in else np.pi           # local phase mean
    sig_d = params_in['sig_d'] if 'sig_d' in params_in else np.pi / 3     # local phase standard deviation
    mu_Th  = params_in['mu_Th'] if 'mu_Th' in params_in else np.pi        # global phase mean
    sig_Th = params_in['sig_Th'] if 'sig_Th' in params_in else np.pi / 3  # global phase standard deviation
    if sample_delta and 'd' not in params_in:
            d = [np.mod(np.random.normal(mu_d, sig_d, len(d_i)), 2*np.pi) for d_i in d]
    if sample_theta and 'Th' not in params_in:
            Th = [np.mod(np.random.normal(mu_Th, sig_Th, len(Th_i)), 2*np.pi) for Th_i in Th]
            
    # Store for local use, and then return
    params = {'e': e, 'F': F, 'p_t': p_t, 'eps': eps, 'L3': L3, 'L4': L4, 'l1': l1, 'l2': l2, 'l3': l3, 'l4': l4,
              'A_0': A_0, 'Adot_0': Adot_0, 'A_pm': A_pm, 'amps': amps, 'd': d, 'Th': Th,
              'qm': qm, 'qc': qc, 'dqm': dqm, 'eps_c': eps_c, 'xi': xi, 'N_r': N_r, 'N_n': N_n, 'N_c': N_c,
              'm': m, 'm_r': m_r, 'm_n': m_n, 'm_c': m_c, 'p': p, 'p_r': p_r, 'p_n': p_n, 'p_c': p_c,
              'mu_d': mu_d, 'sig_d': sig_d, 'mu_Th': mu_Th, 'sig_Th': sig_Th, 'k_span': k_span, 'k_num': k_num,
              't_span': t_span, 't_num': t_num, 'A_sens': A_sens, 't_sens': t_sens, 'res_con': res_con}
    
    return params

'''
def system(t, y, k):
    #l3 = params['l3']
    P = lambda t: 4*l3/(L3**2) * eps**2 * (np.abs(amps[2])**2 * np.cos(m[2]*t + d[2])**2) + \
                  4*l4/(L4**2) * eps**2 * (np.abs(amps[1])**2 * np.cos(m[1]*t + d[1])**2 + \
                                           np.abs(amps[0])**2 * np.cos(m[0]*t + d[0])**2 + \
                                           2*np.abs(amps[0])*np.abs(amps[1]) * np.cos(m[0]*t + d[0])*np.cos(m[1]*t + d[1]) * np.cos(Th[1]))

    B = lambda t: (-1)*8*l3/(L3**2) * eps**2 * (np.abs(amps[2])**2 * m[2] * np.sin(m[2]*t + d[2])**2 * np.cos(m[2]*t + d[2])**2) + \
                  (-1)*8*l4/(L4**2) * eps**2 * (np.abs(amps[0])**2 * m[0] * np.sin(m[0]*t + d[0])**2 * np.cos(m[0]*t + d[0])**2 + \
                                                np.abs(amps[1])**2 * m[1] * np.sin(m[1]*t + d[1])**2 * np.cos(m[1]*t + d[1])**2 + \
                                                np.abs(amps[1])*np.abs(amps[0]) * np.cos(Th[1]) * \
                                                    (m[0] * np.sin(m[0]*t + d[0])**2 * np.cos(m[1]*t + d[1])**2 + \
                                                     m[1] * np.sin(m[1]*t + d[1])**2 * np.cos(m[0]*t + d[0])**2))

    C = lambda t, pm: (-1) * pm * (2*l1 / F * eps**2) * np.abs(amps[0]) * m[0] * np.sin(m[0]*t + d[0])

    D = lambda t: 2*l2 * eps**2 * e**2 * np.abs(amps[2])**2 * np.cos(m[2]*t + d[2])**2

    Alpha = lambda t, k: (k**2 + C(t, A_pm)*k + D(t)) / (1 + P(t))

    Beta = lambda t: B(t) / (1 + P(t))
    
    dy0dt = y[1]
    dy1dt = -1/(1 + P(t)) * (B(t)*y[1] + (C(t, A_pm)*k + D(t))*y[0]) - k**2*y[0]
    return [dy0dt, dy1dt]
'''

# Solve the system over all desired k_values. Specify whether multiprocessing should be used.
def solve_system(system_in, parallelize=False, jupyter=False, num_cores=4):
    if jupyter: import multiprocess as mp
    else: import multiprocessing as mp
    # Start timing
    start_time = time.monotonic()
    
    # Solve the differential equation for each k, in parallel
    if parallelize:
        with mp.Pool(num_cores) as p:
            #solutions = np.array(p.map(solve_subsystem, k_values))
            solutions = p.starmap(solve_subsystem, [(system_in, params, k_i) for k_i in k_values])
    else:
        solutions = np.zeros((len(k_values), 2, len(t)))
        for i, k in enumerate(k_values): 
            solutions[i] = solve_subsystem(system_in, params, k) # Store the solution
    
    # `solutions` contains the solutions for A(t) for each k.
    # e.g. `solutions[i]` is the solution for `k_values[i]`.
    
    # Finish timing
    end_time = time.monotonic()
    
    time_elapsed = timedelta(seconds=end_time - start_time) 
    timestr = str(time_elapsed) + (' on %d cores' % num_cores if parallelize else '')
    
    # update parameters with performance statistics
    params['time_elapsed'] = time_elapsed
    params['num_cores'] = num_cores if parallelize else 1
    params['parallel'] = parallelize
    params['jupyter'] = jupyter
    
    return solutions, params, time_elapsed, timestr
    

# Solve the differential equation for a singular given k
def solve_subsystem(system_in, params, k):
    # Initial conditions
    y0 = [params['A_0'], params['Adot_0']]
    # Time domain
    t_span = params['t_span']
    t = np.linspace(t_span[0], t_span[1], params['t_num'])

    sol = solve_ivp(system_in, t_span, y0, args=(k, params), dense_output=True)
    
    # Evaluate the solution at the times in the array
    y = sol.sol(t)
    return y
    
# Helper function to print parameters of model
def get_text_params(case='full'):
    if case == 'simple':
        textstr1 = '\n'.join((
                r'$A_\pm(t = 0)=%.2f$' % (A_0, ),
                r'$\dot{A}_\pm (t = 0)=%.2f$' % (Adot_0, ),
                r'$\pm=%s$' % (signstr[A_pm], ),
                '\n',
                r'$\varepsilon=%.0e$' % (eps, ),
                '\n',
                r'$\lambda_1=%d$' % (l1, ),
                r'$\lambda_2=%d$' % (l2, ),
                r'$\lambda_3=%d$' % (l3, ),
                r'$\lambda_4=%d$' % (l4, ),
                r'$\Lambda_3=%.0e$' % (L3, ),
                r'$\Lambda_4=%.0e$' % (L4, ),
                '\n',
                r'$\Delta k=%d$' % (k_step, ),
                r'$k \in \left[%d, %d\right]$' % (k_span[0], k_span[1]),

        ))

        textstr2 = '\n'.join((
                r'$m_{(0)}=%.0e eV$' % (m[0], ),
                r'$m_{(\pi)}=%.0e eV$' % (m[1], ),
                r'$m_{(\pm)}=%.0e eV$' % (m[2], ),
                '\n',
                r'$\pi_{(0)}=%.2f$' % (amps[0], ),
                r'$\pi_{(\pi)}=%.2f$' % (amps[1], ),
                r'$\pi_{(\pm)}=%.2f$' % (amps[2], ),
                '\n',
                r'$\delta_{(0)}=%.2f \pi$' % (d[0]/np.pi, ),
                r'$\delta_{(\pi)}=%.2f \pi$' % (d[1]/np.pi, ),
                r'$\delta_{(\pm)}=%.2f \pi$' % (d[2]/np.pi, ),
                '\n',
                r'$\Theta_{(\pi)}=%.2f \pi$' % (Th[1]/np.pi, ),
                '\n',
                r'$\Delta t=%f$' % (t_step, ),
                r'$t \in \left[%d, %d\right]$' % (t_span[0], t_span[1]),
        ))
    else:
        textstr1 = '\n'.join((
                r'$A_\pm(t = 0)=%.2f$' % (A_0, ),
                r'$\dot{A}_\pm (t = 0)=%.2f$' % (Adot_0, ),
                r'$\pm=%s$' % (signstr[A_pm], ),
                '\n',
                r'$\varepsilon=%.0e$' % (eps, ),
                '\n',
                r'$\lambda_1=%d$' % (l1, ),
                r'$\lambda_2=%d$' % (l2, ),
                r'$\lambda_3=%d$' % (l3, ),
                r'$\lambda_4=%d$' % (l4, ),
                r'$\Lambda_3=%.0e$' % (L3, ),
                r'$\Lambda_4=%.0e$' % (L4, ),
                '\n',
                r'$\Delta k=%d$' % (k_step, ),
                r'$k \in \left[%d, %d\right]$' % (k_span[0], k_span[1]),
                '\n',
                r'$\Delta t=%f$' % (t_step, ),
                r'$t \in \left[%d, %d\right]$' % (t_span[0], t_span[1]),

        ))
        m0_mask = np.ma.getmask(m[0])
        m1_mask = np.ma.getmask(m[1])
        m2_mask = np.ma.getmask(m[2])
        textstr2 = '\n'.join((
                ' '.join([r'$m_{(0),%d}=%.0e$'   % (i+1, m[0][i], ) for i in range(len(m[0])) if not m0_mask[i]]),
                ' '.join([r'$m_{(\pi),%d}=%.0e$' % (i+1, m[1][i], ) for i in range(len(m[1])) if not m1_mask[i]]),
                ' '.join([r'$m_{(\pm),%d}=%.0e$' % (i+1, m[2][i], ) for i in range(len(m[2])) if not m2_mask[i]]),
                '\n',
                ' '.join([r'$\pi_{(0),%d}=%.2f$'   % (i+1, amps[0][i], ) for i in range(len(amps[0])) if not m0_mask[i]]),
                ' '.join([r'$\pi_{(\pi),%d}=%.2f$' % (i+1, amps[1][i], ) for i in range(len(amps[1])) if not m1_mask[i]]),
                ' '.join([r'$\pi_{(\pm),%d}=%.2f$' % (i+1, amps[2][i], ) for i in range(len(amps[2])) if not m2_mask[i]]),
                '\n',
                ' '.join([r'$\delta_{(0),%d}=%.2f \pi$'   % (i+1, d[0][i]/np.pi, ) for i in range(len(d[0])) if not m0_mask[i]]),
                ' '.join([r'$\delta_{(\pi),%d}=%.2f \pi$' % (i+1, d[1][i]/np.pi, ) for i in range(len(d[1])) if not m1_mask[i]]),
                ' '.join([r'$\delta_{(\pm),%d}=%.2f \pi$' % (i+1, d[2][i]/np.pi, ) for i in range(len(d[2])) if not m2_mask[i]]),
                '\n',
                #' '.join([r'$\Theta_{(0),%d}=%.2f \pi$' % (i+1, Th[0][i]/np.pi, ) for i in range(len(Th[0])) if not m0_mask[i]]),
                ' '.join([r'$\Theta_{(\pi),%d}=%.2f \pi$' % (i+1, Th[1][i]/np.pi, ) for i in range(len(Th[1])) if not m1_mask[i]]),
                ' '.join([r'$\Theta_{(\pm),%d}=%.2f \pi$' % (i+1, Th[2][i]/np.pi, ) for i in range(len(Th[2])) if not m2_mask[i]]),
        ))
        
    return textstr1, textstr2, 