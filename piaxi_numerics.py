import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from datetime import timedelta
from piaxi_utils import signstr, Alpha, Beta

def set_param_space(params_in):
    global params
    global t, t_span, t_num, t_step
    global k_values, k_span, k_num, k_step
    global e
    global eps
    global F
    global c, h, G
    global L3, L4
    global l1, l2, l3, l4
    global A_0, Adot_0, A_pm
    global m, m_u, m_0, m_q, k_0
    global p, p_t, p_0, amps
    global d, Th
    global qm, qc, dqm, eps_c, xi
    
    params  = params_in
    
    k_span  = params['k_span']
    k_num   = params['k_num']
    t_span  = params['t_span']
    t_num   = params['t_num']
    
    eps     = params['eps']
    F       = params['F']
    c       = params['c']
    h       = params['h']
    G       = params['G']
    e       = params['e']
    L3      = params['L3']
    L4      = params['L4']
    l1      = params['l1']
    l2      = params['l2']
    l3      = params['l3']
    l4      = params['l4']
    A_0     = params['A_0']
    Adot_0  = params['Adot_0']
    m       = params['m']
    m_u     = params['m_u']
    m_0     = params['m_0']
    m_q     = params['m_q']
    k_0     = params['k_0']
    qm      = params['qm']
    dqm     = params['dqm']
    qc      = params['qc']
    eps_c   = params['eps_c']
    xi      = params['xi']
    d       = params['d']
    Th      = params['Th']
    p       = params['p']
    amps    = params['amps']
    p_t     = params['p_t']
    
    t, t_step        = np.linspace(t_span[0], t_span[1], t_num, retstep=True)
    k_values, k_step = np.linspace(k_span[0], k_span[1], k_num, retstep=True)
    
    return params

def get_param_space():
    return params

# Solve the system over all desired k_values. Specify whether multiprocessing should be used.
def solve_system(system_in, parallelize=False, jupyter=False, num_cores=4):
    if jupyter: import multiprocess as mp
    else: import multiprocessing as mp
    # Start timing
    start_time = time.time()
    
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
    end_time = time.time()
    
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

## Pi-axion Species Mass Definitions
def define_masses(qc, qm, F, e, eps, eps_c, xi):
    m_r = np.array([0., 0., 0., 0., 0.], dtype=float)                   # real neutral
    m_n = np.array([0., 0., 0., 0., 0., 0.], dtype=float)               # complex neutral
    m_c = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float)   # charged
    # Pi-axion Species Labels
    s_l = np.array([np.full_like(m_r, '', dtype=str), np.full_like(m_n, '', dtype=str), np.full_like(m_c, '', dtype=str)], dtype=object)

    ## Real Neutral Masses
    # pi_3
    m_r[0]    = np.sqrt(((qc[0] + qc[1])*qm[0])*F) if 0. not in [qc[0], qc[1]] else 0.
    s_l[0][0] = '$\pi_{3}$'
    # pi_8
    m_r[1]    = np.sqrt(((qc[0] + qc[1])*qm[0] + qc[2]*qm[1])*F) if 0. not in [qc[0], qc[1], qc[2]] else 0.
    s_l[0][1] = '$\pi_{8}$'
    # pi_29
    m_r[2]    = np.sqrt(((qc[3]*qm[1]) + (qc[4]*qm[2]))*F) if 0. not in [qc[3], qc[4]] else 0.
    s_l[0][2] = '$\pi_{29}$'
    # pi_34
    m_r[3]    = np.sqrt(((qc[3]*qm[1]) + ((qc[4] + qc[5])*qm[2]))*F) if 0. not in [qc[3], qc[4], qc[5]] else 0.
    s_l[0][3] = '$\pi_{34}$'
    # pi_35
    m_r[4]    = np.sqrt(((qc[0]+qc[1])*qm[0] + (qc[2]+qc[3])*qm[1] + (qc[4]+qc[5])*qm[2])*F) if 0. not in [qc[0], qc[1], qc[2], qc[3], qc[4], qc[5]] else 0.
    s_l[0][4] = '$\pi_{35}$'

    ## Complex Neutral Masses
    # pi_6  +/- i*pi_7
    m_n[0]    = np.sqrt((qc[1]*qm[0] + qc[2]*qm[1]) * F) if 0. not in [qc[1], qc[2]] else 0.
    s_l[1][0] = '$\pi_6 \pm i\pi_7$'
    # pi_9  +/- i*pi_10
    m_n[1]    = np.sqrt((qc[0]*qm[0] + qc[3]*qm[1]) * F) if 0. not in [qc[0], qc[3]] else 0.
    s_l[1][1] = '$\pi_9 \pm i\pi_{10}$'
    # pi_17 +/- i*pi_18
    m_n[2]    = np.sqrt((qc[1]*qm[0] + qc[4]*qm[2]) * F) if 0. not in [qc[1], qc[4]] else 0.
    s_l[1][2] = '$\pi_{17} \pm i\pi_{18}$'
    # pi_19 +/- i*pi_20
    m_n[3]    = np.sqrt((qc[2]*qm[1] + qc[4]*qm[2]) * F) if 0. not in [qc[2], qc[4]] else 0.
    s_l[1][3] = '$\pi_{19} \pm i\pi_{20}$'
    # pi_21 +/- i*pi_22
    m_n[4]    = np.sqrt((qc[0]*qm[0] + qc[5]*qm[2]) * F) if 0. not in [qc[0], qc[5]] else 0.
    s_l[1][4] = '$\pi_{21} \pm i\pi_{22}$'
    # pi_30 +/- i*pi_31
    m_n[5]    = np.sqrt((qc[3]*qm[1] + qc[5]*qm[2]) * F) if 0. not in [qc[3], qc[5]] else 0.
    s_l[1][5] = '$\pi_{30} \pm i\pi_{31}$'

    ## Charged Masses
    # pi_1  +/- i*pi_2
    m_c[0]    = np.sqrt((qc[0]*qm[0] + qc[1]*qm[0])*F + 2*xi[0]*(e*eps*F)**2) if 0. not in [qc[0], qc[1]] and abs(eps) < 0.1 else 0.
    s_l[2][0] = '$\pi_1 \pm i\pi_2$'
    # pi_4  +/- i*pi_5
    m_c[1]    = np.sqrt((qc[0]*qm[0] + qc[2]*qm[1])*F + 2*xi[1]*(e*eps*F)**2) if 0. not in [qc[0], qc[2]] and abs(eps) < 0.1 else 0.
    s_l[2][1] = '$\pi_4 \pm i\pi_5$'
    # pi_15 +/- i*pi_16
    m_c[2]    = np.sqrt((qc[0]*qm[0] + qc[4]*qm[2])*F + 2*xi[2]*(e*eps*F)**2) if 0. not in [qc[0], qc[4]] and abs(eps) < 0.1 else 0.
    s_l[2][2] = '$\pi_{15} \pm i\pi_{16}$'
    # pi_11 +/- i*pi_12
    m_c[3]    = np.sqrt((qc[1]*qm[0] + qc[3]*qm[2])*F + 2*xi[3]*(e*eps*F)**2) if 0. not in [qc[1], qc[3]] and abs(eps) < 0.1 else 0.
    s_l[2][3] = '$\pi_{11} \pm i\pi_{12}$'
    # pi_23 +/- i*pi_24
    m_c[4]    = np.sqrt((qc[1]*qm[0] + qc[5]*qm[2])*F + 2*xi[4]*(e*eps*F)**2) if 0. not in [qc[1], qc[5]] and abs(eps) < 0.1 else 0.
    s_l[2][4] = '$\pi_{23} \pm i\pi_{24}$'
    # pi_13 +/- i*pi_14
    m_c[5]    = np.sqrt((qc[2]*qm[1] + qc[3]*qm[1])*F + 2*xi[5]*(e*eps*F)**2) if 0. not in [qc[2], qc[3]] and abs(eps) < 0.1 else 0.
    s_l[2][5] = '$\pi_{13} \pm i\pi_{14}$'
    # pi_25 +/- i*pi_26
    m_c[6]    = np.sqrt((qc[2]*qm[1] + qc[5]*qm[2])*F + 2*xi[6]*(e*eps*F)**2) if 0. not in [qc[2], qc[5]] and abs(eps) < 0.1 else 0.
    s_l[2][6] = '$\pi_{25} \pm i\pi_{26}$'
    # pi_27 +/- i*pi_28
    m_c[7]    = np.sqrt((qc[3]*qm[1] + qc[4]*qm[2])*F + 2*xi[7]*(e*eps*F)**2) if 0. not in [qc[3], qc[4]] and abs(eps) < 0.1 else 0.
    s_l[2][7] = '$\pi_{27} \pm i\pi_{28}$'
    # pi_32 +/- i*pi_33
    m_c[8]    = np.sqrt((qc[4]*qm[2] + qc[5]*qm[2])*F + 2*xi[8]*(e*eps*F)**2) if 0. not in [qc[4], qc[5]] and abs(eps) < 0.1 else 0.
    s_l[2][8] = '$\pi_{32} \pm i\pi_{33}$'

    # Mask zero-valued / disabled species in arrays
    m_r = np.ma.masked_where(m_r == 0.0, m_r, copy=True)
    m_n = np.ma.masked_where(m_n == 0.0, m_n, copy=True)
    m_c = np.ma.masked_where(m_c == 0.0, m_c, copy=True)
    r_mask = np.ma.getmask(m_r)
    n_mask = np.ma.getmask(m_n)
    c_mask = np.ma.getmask(m_c)
    masks = np.array([r_mask, n_mask, c_mask], dtype=object)
    N_r = m_r.count()
    N_n = m_n.count()
    N_c = m_c.count()
    counts = (N_r, N_n, N_c)
    
    return m_r, m_n, m_c, counts, masks


def floquet_exponent(p=Beta, q=Alpha, T=2*np.pi, y0_in=None, yp0_in = None, k_modes=[]):
    """
    Calculate the Floquet exponents for a system with multiple k-modes.
    
    Parameters:
    - p: Function representing the coefficient of the differential equations.
    - q: Function representing the coefficient of the differential equations, dependent on k.
    - T: Period of the periodic coefficients.
    - y0, yp0: Initial conditions for the differential equation.
    - k_modes: List of k values for each mode.
    
    Returns:
    - Floquet exponents for each mode.
    """
    global params, k_values
    k_modes = k_modes if len(k_modes) > 0 else k_values
    num_modes = len(k_modes)
    y0 = y0_in if y0_in is not None else params['A_0'] if 'A_0' in params else 0
    yp0 = yp0_in if yp0_in is not None else params['Adot_0'] if 'Adot_0' in params else 0.1
    
    # Convert the second-order differential equations to a system of first-order equations
    def floq_sys(t, Y):
        dydt = np.zeros(2 * num_modes)
        for i, k in enumerate(k_modes):
            dydt[2*i] = Y[2*i+1]
            dydt[2*i+1] = -p(t) * Y[2*i+1] - q(t, k) * Y[2*i]
        return dydt
    
    # Solve the system over one period using the initial conditions
    sol = solve_ivp(floq_sys, [0, T], np.concatenate([y0, yp0]), t_eval=[T])
    
    # Calculate the monodromy matrix
    M = np.zeros((2 * num_modes, 2 * num_modes))
    for i in range(num_modes):
        M[2*i, 0] = sol.y[2*i, -1]
        M[2*i+1, 1] = sol.y[2*i+1, -1]
    
    # Compute the eigenvalues of the monodromy matrix
    eigenvalues = np.linalg.eigvals(M)
    
    # Calculate the Floquet exponents
    floquet_exponents = np.log(eigenvalues) / T
    
    return floquet_exponents

# Example usage
'''
p_func = lambda t: np.cos(t)
q_func = lambda t, k: np.sin(t) + k
T = 2 * np.pi
y0 = [1, 1]
yp0 = [0, 0]
k_modes = [1, 2]

exponents = floquet_exponent(p_func, q_func, T, k_modes)
print(exponents)
'''

# Helper function to print parameters of model (deprecated)
def get_text_params_old(case='full', units_in={}):
    units = {}
    for key in ['c', 'h', 'G', 'm', 'k', 'p', 'amp', 'Lambda', 'lambda', 'F', 't', 'Theta', 'delta', 'eps']:
        if key not in units_in or units_in[key] == 1:
            units[key] = ''
        elif key in ['Lambda','F'] and units_in[key] == 'm_u':
            units[key] = '\quad[%s]' % ('eV')
        elif key in ['k'] and units_in[key] == 'eV':
            units[key] = '\quad[%s]' % ('m_u')
        else:
            units[key] = '\quad[%s]' % (units_in[key])
    
    if case == 'simple':
        textstr1 = '\n'.join((
                r'$A_\pm(t = 0)=%.2f$' % (A_0, ),
                r'$\dot{A}_\pm (t = 0)=%.2f$' % (Adot_0, ),
                r'$\pm=%s$' % (signstr[A_pm], ),
                '\n',
                r'$\varepsilon=%.0e%s$' % (eps, ),
                r'$F_{\pi}=%.0e%s$' % (F, units['F']),
                '\n',
                r'$\lambda_1=%d%s$' % (l1, units['lambda']),
                r'$\lambda_2=%d%s$' % (l2, units['lambda']),
                r'$\lambda_3=%d%s$' % (l3, units['lambda']),
                r'$\lambda_4=%d%s$' % (l4, units['lambda']),
                r'$\Lambda_3=%.0e%s$' % (L3, units['Lambda']),
                r'$\Lambda_4=%.0e%s$' % (L4, units['Lambda']),
                '\n',
                r'$\Delta k=%.2f%s$' % (k_step, units['k']),
                r'$k \in \left[%d, %d\\right]$' % (k_span[0], k_span[1]),

        ))

        textstr2 = '\n'.join((
                r'$m_{(0)}=%.0e%s$' % (m[0], units['m']),
                r'$m_{(\pi)}=%.0e%s$' % (m[1], units['m']),
                r'$m_{(\pm)}=%.0e%s$' % (m[2], units['m']),
                '\n',
                r'$\pi_{(0)}=%.2f%s$' % (amps[0], units['amp']),
                r'$\pi_{(\pi)}=%.2f%s$' % (amps[1], units['amp']),
                r'$\pi_{(\pm)}=%.2f%s$' % (amps[2], units['amp']),
                '\n',
                r'$\delta_{(0)}=%.2f \pi$' % (d[0]/np.pi, ),
                r'$\delta_{(\pi)}=%.2f \pi$' % (d[1]/np.pi, ),
                r'$\delta_{(\pm)}=%.2f \pi$' % (d[2]/np.pi, ),
                '\n',
                r'$\Theta_{(\pi)}=%.2f \pi$' % (Th[1]/np.pi, ),
                '\n',
                r'$\Delta t=%f%s$' % (t_step, units['t']),
                r'$t \in \left[%d, %d\\right]$' % (t_span[0], t_span[1]),
        ))
    else:
        textstr1 = '\n'.join((
                r'$A_\pm(t = 0)=%.2f$' % (A_0, ),
                r'$\dot{A}_\pm (t = 0)=%.2f$' % (Adot_0, ),
                r'$\pm=%s$' % (signstr[A_pm], ),
                '\n',
                r'$\varepsilon=%.0e$' % (eps, ),
                r'$F_{\pi}=%.0e%s$' % (F, units['F']),
                r'$c = h = G = 1$' if all([units[key] == '' for key in ['c', 'h', 'G']]) else '\n'.join([r'$%s=%.2e%s$' % (key, val, units[key]) for key, val in zip(['c', 'h', 'G'], [c, h, G])]),
                '\n',
                r'$\lambda_1=%d%s$' % (l1, units['lambda']),
                r'$\lambda_2=%d%s$' % (l2, units['lambda']),
                r'$\lambda_3=%d%s$' % (l3, units['lambda']),
                r'$\lambda_4=%d%s$' % (l4, units['lambda']),
                r'$\Lambda_3=%.0e%s$' % (L3, units['Lambda']) if L3 > 0 else r'$[\Lambda_3=%s]$' % ('N/A'),
                r'$\Lambda_4=%.0e%s$' % (L4, units['Lambda']) if L4 > 0 else r'$[\Lambda_4=%s]$' % ('N/A'),
                '\n',
                r'$\Delta k=%.2f%s$' % (k_step, units['k']),
                r'$k \in [%d, %d]$' % (k_span[0], k_span[1]),
                '\n',
                r'$\Delta t=%f%s$' % (t_step, units['t']),
                r'$t \in [%d, %d]$' % (t_span[0], t_span[1]),

        ))
        m0_mask = np.ma.getmask(m[0]) if np.ma.getmask(m[0]) else np.full_like(m[0], False)
        m1_mask = np.ma.getmask(m[1]) if np.ma.getmask(m[1]) else np.full_like(m[1], False)
        m2_mask = np.ma.getmask(m[2]) if np.ma.getmask(m[2]) else np.full_like(m[2], False)
        textstr2 = '\n'.join((
                r'$m_{q, dQCD} = [%s]\quad%.0e eV$' % (', '.join('%d' % q for q in qm / m_q), m_q),
                '' if units_in['m'] == 'eV' else r'$m_{u} = %.2e\quad[eV]$' % (m_u, ),
                r'$m%s$' % units['m'],
                ' '.join([r'$m_{(0),%d}=%.2e$'   % (i+1, m[0][i]*m_0, ) for i in range(len(m[0])) if not m0_mask[i]]),
                ' '.join([r'$m_{(\pi),%d}=%.2e$' % (i+1, m[1][i]*m_0, ) for i in range(len(m[1])) if not m1_mask[i]]),
                ' '.join([r'$m_{(\pm),%d}=%.2e$' % (i+1, m[2][i]*m_0, ) for i in range(len(m[2])) if not m2_mask[i]]),
                '\n',
                r'$\pi%s$' % units['amp'],
                ' '.join([r'$\pi_{(0),%d}=%.2e$'   % (i+1, amps[0][i], ) for i in range(len(amps[0])) if not m0_mask[i]]),
                ' '.join([r'$\pi_{(\pi),%d}=%.2e$' % (i+1, amps[1][i], ) for i in range(len(amps[1])) if not m1_mask[i]]),
                ' '.join([r'$\pi_{(\pm),%d}=%.2e$' % (i+1, amps[2][i], ) for i in range(len(amps[2])) if not m2_mask[i]]),
                '\n',
                r'$\rho\quad[eV/cm^3]$',
                ' '.join([r'$\rho_{(0),%d}=%.2e$'   % (i+1, p[0][i]/p_0, ) for i in range(len(p[0])) if not m0_mask[i]]),
                ' '.join([r'$\rho_{(\pi),%d}=%.2e$' % (i+1, p[1][i]/p_0, ) for i in range(len(p[1])) if not m1_mask[i]]),
                ' '.join([r'$\rho_{(\pm),%d}=%.2e$' % (i+1, p[2][i]/p_0, ) for i in range(len(p[2])) if not m2_mask[i]]),
                '\n',
                ' '.join([r'$\delta_{(0),%d}=%.2f \pi$'   % (i+1, d[0][i]/np.pi, ) for i in range(len(d[0])) if not m0_mask[i]]),
                ' '.join([r'$\delta_{(\pi),%d}=%.2f \pi$' % (i+1, d[1][i]/np.pi, ) for i in range(len(d[1])) if not m1_mask[i]]),
                ' '.join([r'$\delta_{(\pm),%d}=%.2f \pi$' % (i+1, d[2][i]/np.pi, ) for i in range(len(d[2])) if not m2_mask[i]]),
                '\n',
                #' '.join([r'$\Theta_{(0),%d}=%.2f \pi$' % (i+1, Th[0][i]/np.pi, ) for i in range(len(Th[0])) if not m0_mask[i]]),
                ' '.join([r'$\Theta_{(\pi),%d}=%.2f \pi$' % (i+1, Th[1][i]/np.pi, ) for i in range(len(Th[1])) if not m1_mask[i]]),
                ' '.join([r'$\Theta_{(\pm),%d}=%.2f \pi$' % (i+1, Th[2][i]/np.pi, ) for i in range(len(Th[2])) if not m2_mask[i]]),
        ))
        
    return textstr1, textstr2

## Set parameters of model for use in numerical integration (deprecated)
def set_params_old(params_in: dict, sample_delta=True, sample_theta=True, t_max=10, t_min=0, t_N=500, 
               k_max=200, k_min=1, k_N=200):
    # Define global variables
    global params
    global t, t_span, t_num, t_step
    global k_values, k_span, k_num, k_step
    global e
    global eps
    global F
    global c, h, G
    global L3, L4
    global l1, l2, l3, l4
    global A_0, Adot_0, A_pm
    global m, m_u, m_0, m_q
    global p, p_t, p_0, amps
    global d, Th
    global qm, qc, dqm, eps_c, xi
    
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
    seed = params_in['seed']

    # Set constants of model (otherwise default)
    e = params_in['e']     if 'e' in params_in else 0.3        #
    F = params_in['F']     if 'F' in params_in else 1e20       # eV
    p_t = params_in['p_t'] if 'p_t' in params_in else 0.4  # eV / cm^3
    
    # Fundamental constants
    c = params_in['c']
    h = params_in['h']
    G = params_in['G']
    
    ## Tuneable constants
    # millicharge, vary to enable/disable charged species (10e-15 = OFF, 10e-25 = ON)
    #eps  = 1e-25   # (unitless)
    eps  = params_in['eps']
    
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
    m_q = params_in['m_scale']                                            # dark quark mass scale
    m_u = params_in['m_u']      if 'm_u' in params_in else min(m[0])      # unit mass value in [eV]
    m_0 = params_in['m_0']      if 'm_0' in params_in else min(m[0])      # mass unitful rescaling parameter
    N_r = len(params_in['m_r']) if 'm_r' in params_in else 1              # number of real species
    m_r = params_in['m_r']      if 'm_r' in params_in else np.full((N_r, ), m[0]) # (neutral) real species
    N_n = len(params_in['m_n']) if 'm_n' in params_in else 1              # number of neutral species
    m_n = params_in['m_n']      if 'm_n' in params_in else np.full((N_n, ), m[1]) # neutral (complex) species
    N_c = len(params_in['m_c']) if 'm_c' in params_in else 1              # number of charged species
    m_c = params_in['m_c']      if 'm_c' in params_in else np.full((N_c, ), m[2]) # charged (complex) species
    m   = params_in['m']        if 'm'   in params_in else np.array([m_r, m_n, m_c], dtype=object)

    # local DM densities for each species [eV/cm^3]
    p_r  = params_in['p_r'] if 'p_r' in params_in else np.full((N_r, ), None)   # (neutral) real species
    p_n  = params_in['p_n'] if 'p_n' in params_in else np.full((N_n, ), None)   # neutral (complex) species
    p_c  = params_in['p_c'] if 'p_c' in params_in else np.full((N_c, ), None)   # charged (complex) species
    #p    = params_in['p']   if 'p'   in params_in else np.array([np.full(p_t/3,N_r), np.full(p_t/3,N_n), np.full(p_t/3,N_c)], dtype=object) # default to equal distribution
    p    = params_in['p']   if 'p'   in params_in else np.array([p_r, p_n, p_c], dtype=object)
    p_0  = params_in['p_unit'] if 'p_unit' in params_in else 1.
    
    # initial amplitudes for each species
    amp_r = params_in['amp_r'] if 'amp_r' in params_in else np.array([np.sqrt(2 * p_r[i]) / m_r[i] for i in range(N_r)], dtype=object)
    amp_n = params_in['amp_n'] if 'amp_n' in params_in else np.array([np.sqrt(2 * p_n[i]) / m_n[i] for i in range(N_n)], dtype=object)
    amp_c = params_in['amp_c'] if 'amp_c' in params_in else np.array([np.sqrt(2 * p_c[i]) / m_c[i] for i in range(N_c)], dtype=object)
    #amps = params_in['amps'] if 'amps' in params_in else [np.sqrt(2 * p[i]) / m[i] for i in range(len(m))] # default to equal distribution
    amps = params_in['amps']   if 'amps'  in params_in else np.array([amp_r, amp_n, amp_c], dtype=object)

    # local phases for each species in (0, 2pi)
    d_r    = params_in['d_r'] if 'd_r' in params_in else np.array([np.zeros(int(N_r))])
    d_n    = params_in['d_n'] if 'd_n' in params_in else np.array([np.zeros(int(N_n))])
    d_c    = params_in['d_c'] if 'd_c' in params_in else np.array([np.zeros(int(N_c))])
    #d    = params_in['d'] if 'd' in params_in else np.array([np.zeros(float(N_r)), np.zeros(float(N_n)), np.zeros(float(N_c))], dtype=object)
    d      = params_in['d']   if 'd' in params_in else np.array([d_r, d_n, d_c], dtype=object)

    # global phase for neutral complex species in (0, 2pi)
    Th_r    = params_in['Th_r'] if 'Th_r' in params_in else np.array([np.zeros(int(N_r))])
    Th_n    = params_in['Th_n'] if 'Th_n' in params_in else np.array([np.zeros(int(N_n))])
    Th_c    = params_in['Th_c'] if 'Th_c' in params_in else np.array([np.zeros(int(N_c))])
    #Th    = params_in['Th'] if 'Th' in params_in else np.array([np.zeros(float(N_r)), np.zeros(float(N_n)), np.zeros(float(N_c))], dtype=object)
    Th      = params_in['Th']   if 'Th' in params_in else np.array([Th_r, Th_n, Th_c], dtype=object)
    
    # Sample phases from normal distribution, sampled within range (0, 2pi)
    mu_d   = params_in['mu_d']   if 'mu_d' in params_in else np.pi           # local phase mean
    sig_d  = params_in['sig_d']  if 'sig_d' in params_in else np.pi / 3      # local phase standard deviation
    mu_Th  = params_in['mu_Th']  if 'mu_Th' in params_in else np.pi          # global phase mean
    sig_Th = params_in['sig_Th'] if 'sig_Th' in params_in else np.pi / 3     # global phase standard deviation
    force_resample = False
    if sample_delta and all([pkey not in params_in for pkey in ['d_r', 'd_n', 'd_c']]) and force_resample:
            d = [np.mod(np.random.normal(mu_d, sig_d, len(d_i)), 2*np.pi) for d_i in d]
    if sample_theta and all([Tkey not in params_in for Tkey in ['Th_r', 'Th_n', 'Th_c']]) and force_resample:
            Th = [np.mod(np.random.normal(mu_Th, sig_Th, len(Th_i)), 2*np.pi) for Th_i in Th]
            
    # rescaling and unit configuration
    unitful_m      = params_in['unitful_m']
    rescale_m      = params_in['rescale_m']
    unitful_k      = params_in['unitful_k']
    rescale_k      = params_in['rescale_k']
    unitful_amps   = params_in['unitful_amps']
    rescale_amps   = params_in['rescale_amps']
    rescale_consts = params_in['rescale_consts']
            
    # Store for local use, and then return
    params = {'e': e, 'F': F, 'p_t': p_t, 'eps': eps, 'L3': L3, 'L4': L4, 'l1': l1, 'l2': l2, 'l3': l3, 'l4': l4,
              'A_0': A_0, 'Adot_0': Adot_0, 'A_pm': A_pm, 'amps': amps, 'd': d, 'Th': Th, 'h': h, 'c': c, 'G': G,
              'qm': qm, 'qc': qc, 'dqm': dqm, 'eps_c': eps_c, 'xi': xi, 'N_r': N_r, 'N_n': N_n, 'N_c': N_c, 'p_0': p_0,
              'm': m, 'm_r': m_r, 'm_n': m_n, 'm_c': m_c, 'p': p, 'p_r': p_r, 'p_n': p_n, 'p_c': p_c, 'm_0': m_0,
              'mu_d': mu_d, 'sig_d': sig_d, 'mu_Th': mu_Th, 'sig_Th': sig_Th, 'k_span': k_span, 'k_num': k_num,
              't_span': t_span, 't_num': t_num, 'A_sens': A_sens, 't_sens': t_sens, 'res_con': res_con, 'm_u': m_u,
              'unitful_m': unitful_m, 'rescale_m': rescale_m, 'unitful_amps': unitful_amps, 'rescale_amps': rescale_amps, 
              'rescale_k': rescale_k, 'rescale_consts': rescale_consts, 'seed': seed}
    
    return params