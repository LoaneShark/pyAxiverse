import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import logsumexp
import pprint
import datetime
import sys
from piaxi_utils import signstr, Alpha, Beta

def set_param_space(params_in):
    global params
    global t, t_span, t_num, t_step, t_0
    global k_values, k_span, k_num, k_step
    global e, bg
    global eps
    global F
    global L3, L4
    global l1, l2, l3, l4
    global A_0, Adot_0, A_pm
    global m, m_u, m_0, m_q, k_0
    global N_r, N_n, N_c
    global p, p_t, p_0, amps
    global d, Th
    global qm, qc, dqm, eps_c, xi
    global use_mass_units, use_natural_units, unitful_m
    
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
    N_r     = params['N_r']
    N_n     = params['N_n']
    N_c     = params['N_c']
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
    t_0     = params['t_0']
    bg      = params['em_bg']
    
    t, t_step        = np.linspace(t_span[0], t_span[1], t_num, retstep=True)
    k_values, k_step = np.linspace(k_span[0], k_span[1], k_num, retstep=True)
    
    return params

def get_param_space():
    return params

k_count_max = 0  # TODO: Make sure this max counter works
# Solve the system over all desired k_values. Specify whether multiprocessing should be used.
def solve_piaxi_system(system_in, params, k_values, parallelize=False, jupyter=None, num_cores=4, verbosity=0, show_progress_bar=False, method='RK45', write_to_params=True):
    global k_count_max
    # Determine the environment
    is_jupyter = jupyter if jupyter is not None else 'ipykernel' in sys.modules
    if is_jupyter and verbosity >= 0:
            print('Jupyter?       ', is_jupyter)
    elif verbosity >= 6:
            print('Jupyter?       ', is_jupyter)
    if verbosity >= 3:
        print('Parallel?      ', parallelize, ' (N = %d)' % num_cores if parallelize and verbosity >= 8 else '')
        print('Integrating using %s' % method)

    if is_jupyter:
        import multiprocess as mp
        from IPython.display import clear_output
    else:
        import pathos.multiprocessing as mp
    
    # Progress bar display subroutine (WIP)
    if show_progress_bar:
        if is_jupyter:
            from tqdm import notebook as tqdm
        else:
            import tqdm

        def update_progress(count, maxval, maxcount=0):
            global k_count_max
            k_count_max = max(maxcount, k_count_max)
            progress = max(k_count_max, count) / maxval
            bar_length = 50
            block = int(round(bar_length * progress))
            clear_output(wait=True) if is_jupyter else None
            text = "Progress: [{0}] {1:.2f}%".format("#" * block + "-" * (bar_length - block), progress * 100)
            print(text, end="\r")
            sys.stdout.flush()
        
        # Worker function for pretty printing a progress bar
        progress_val = lambda i, n=len(k_values), i_max=k_count_max: update_progress(i + 1, n, maxcount=i_max)

    # Start timing
    start_time = datetime.datetime.now()
    if verbosity > 5: 
        print('start time: ', start_time)

    if verbosity > 8:
        print('system in:  ', system_in)
        print('params in:  ')
        pprint.pp(params)
        print('-----------------------------------------------------')

    # Initialize photon apmlitudes via bunch-davies initial conditions
    A_scale    = params['A_0']
    Adot_scale = params['Adot_0']
    k_N  = len(k_values)
    y0_k = init_photons(k_N, A_scale=A_scale, Adot_scale=Adot_scale)

    # Solve the differential equation for each k, in parallel
    if parallelize:
        with mp.Pool(num_cores) as pool:
            #solutions = np.array(p.map(solve_subsystem, k_values))
            pool_params = [(system_in, params, np.float64(y0), np.float64(k), verbosity, method) for k, y0 in zip(k_values, y0_k)]
            pool_inputs = tqdm.tqdm(pool_params, total=k_N) if show_progress_bar else pool_params
            solutions = pool.starmap(solve_subsystem, pool_inputs)
    else:
        t_span = params['t_span']
        t = np.linspace(t_span[0], t_span[1], params['t_num'])
        solutions = np.zeros((k_N, 2, len(t)))
        k_count_max = 0
        for i, k in enumerate(k_values):
            if verbosity > 7 and show_progress_bar:
                #print('i = %d,   k[i] = %d' % (i, k))
                progress_val(i)
            solutions[i] = solve_subsystem(system_in, params, np.float64(y0_k[i]), np.float64(k), verbosity=0, method=method) # Store the solution
    
    # `solutions` contains the solutions for A(t) for each k.
    # e.g. `solutions[i]` is the solution for `k_values[i]`.
    
    # Finish timing
    end_time = datetime.datetime.now()
    
    time_elapsed = end_time - start_time
    timestr = str(time_elapsed) + (' elapsed on %d cores' % num_cores if parallelize else '')
    
    # update parameters with performance statistics
    if write_to_params:
        params['time_elapsed'] = time_elapsed
        params['num_cores']    = num_cores if parallelize else 1
        params['parallel']     = parallelize
        params['jupyter']      = jupyter
        params['mem_per_core'] = params['mem_per_core'] if parallelize and 'mem_per_core' in params and params['mem_per_core'] is not None and (len(params['mem_per_core']) > 1 or (len(params['mem_per_core']) == 1 and int(params['mem_per_core'][0]) > 0)) else None

    if verbosity >= 0:
        print(timestr)
    
    return solutions, params, time_elapsed

# Solve the differential equation for a singular given k
def solve_subsystem(system_in, params, y0_in, k, verbosity=0, method='RK45'):
    # Initial conditions
    y0 = y0_in

    # Debug print statements
    if verbosity > 8:
        print('  k=%.2f  ' % k)
    if verbosity > 9:
        print('y0:   ', y0)
    
    # Time domain
    t_span = params['t_span']
    t = np.linspace(t_span[0], t_span[1], params['t_num'])

    sol = solve_ivp(system_in, t_span, y0, args=(k, params), dense_output=True, method=method)
    
    # Evaluate the solution at the times in the array
    y = sol.sol(t)
    return y

def piaxi_system(t, y, k, params, P, B, C, D, A_pm, bg, k0, c, h, G, use_logsumexp=False):
    # System of differential equations to be solved (bg = photon background)
    '''
    if verbosity >= 9:
            print('\n'.join(['t = %s    k = %.1f' % (t,k), \
                             '  P(t): %.2e' % (bg + P(t)), \
                             '  B(t): %.2e' % B(t), \
                             '  C(t): %.2e' % C(t, A_pm), \
                             '  D(t): %.2e' % D(t)]))
    '''
    disable_B = params['disable_B']
    disable_C = params['disable_C']
    disable_D = params['disable_D']
    if use_logsumexp: # WIP
        # Handle edge cases to sidestep log[0] errors when certain coefficients are turned off
        if disable_B:
            logBeta    = -np.inf
            Beta_sign  = 1
        else:
            logBeta    = np.log(np.abs(B(t))) - np.log(np.abs(bg + P(t)))
            Beta_sign  = np.sign(B(t))*np.sign(bg + P(t))
        if disable_C:
            logCterm   = -np.inf
            Cterm_sign = 1
        else:
            logCterm   = np.log(np.abs(C(t, A_pm))) - np.log(np.abs(bg + P(t))) + np.log(k)
            Cterm_sign = np.sign(C(t, A_pm))*np.sign(bg + P(t))
        if disable_D:
            logDterm   = -np.inf
            Dterm_sign = 1
        else:
            logDterm   = np.log(np.abs(D(t))) - np.log(np.abs(bg + P(t)))
            Dterm_sign = np.sign(D(t))*np.sign(bg + P(t))
        logAlpha, Alpha_sign = logsumexp(a=[logCterm, logDterm, np.log(k**2)],
                                         b=[Cterm_sign, Dterm_sign, 1], return_sign=True)
        
        # Numerically calculate A''[t] in log-space
        logdy1dt, dy1dt_sign = logsumexp(a=[logBeta + np.log(np.abs(y[1])), logAlpha + np.log(np.abs(y[0]))],
                                         b=[Beta_sign * np.sign(y[1]), Alpha_sign * np.sign(y[0])], return_sign=True)
        dy1dt = -dy1dt_sign*np.exp(logdy1dt)
    else:
        # Numerically calculate A''[t]
        dy1dt = -1./(bg + P(t)) * (B(t)*y[1] + (C(t, A_pm)*(k*np.float64(k0)) + D(t))*y[0]) - (k*np.float64(k0))**2*y[0]
    
    # Return A'[t], A''[t]
    dy0dt = y[1]
    return [dy0dt, dy1dt]

def init_photons(k_N, A_scale=1.0, Adot_scale=1.0):

    A_0 = np.fromfunction(lambda k: 1./np.sqrt(2.*(k+1)), (k_N,), dtype=np.float64) * A_scale
    Adot_0 = np.fromfunction(lambda k: np.sqrt((k+1)/2.), (k_N,), dtype=np.float64) * Adot_scale

    return np.array([A_0, Adot_0], dtype=np.float64).T

# TODO: Verify / implement this
def floquet_exponent(p=Beta, q=Alpha, T=2*np.pi, y0_in=None, yp0_in=None, k_modes=[]):
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
    y0  = y0_in  if y0_in  is not None else params['A_0']    if 'A_0'    in params else 0
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
