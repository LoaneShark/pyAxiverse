from math import log
from multiprocessing import context
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import logsumexp
import pprint
import datetime
import sys
import warnings
from pyaxi_utils import signstr, Alpha, Beta, get_kvals, get_times

# (UNUSED / WIP)
def import_multiprocessing(is_jupyter=False, show_progress_bar=True):
    if is_jupyter:
        import multiprocess as mp
        from IPython.display import clear_output
    else:
        import pathos.multiprocessing as mp
    
    # Progress bar display subroutine
    if show_progress_bar:
        if is_jupyter:
            from tqdm import notebook as tqdm
        else:
            import tqdm
    return mp, tqdm

# Solve the system over all desired k_values. Specify whether multiprocessing should be used.
def solve_piaxi_system(system_in, params, k_values, parallelize=False, jupyter=None, num_cores=4, verbosity=0, show_progress_bar=False, method='RK45', write_to_params=True):
    k_count_max = 0 # TODO: Make sure this max counter works and/or is still needed
    # Determine the environment
    is_jupyter = jupyter if jupyter is not None else 'ipykernel' in sys.modules

    # Import libraries if needed
    if not('mp' in sys.modules):
        if is_jupyter:
            import multiprocess as mp
            from IPython.display import clear_output
        else:
            import pathos.multiprocessing as mp
        
        # Progress bar display subroutine
        if show_progress_bar:
            if is_jupyter:
                from tqdm import notebook as tqdm
            else:
                import tqdm
    
    if is_jupyter and verbosity >= 0:
            print('Jupyter?       ', is_jupyter)
    elif verbosity >= 6:
            print('Jupyter?       ', is_jupyter)
    if verbosity >= 3:
        print('Parallel?      ', parallelize, ' (N = %d)' % num_cores if parallelize and verbosity >= 8 else '')
        print('Integrating using %s' % method)

        # Worker function for pretty printing a progress bar
        def update_progress(count, maxval, maxcount=0):
            k_count_max = max(maxcount, k_count_max)
            progress = max(k_count_max, count) / maxval
            bar_length = 50
            block = int(round(bar_length * progress))
            clear_output(wait=True) if is_jupyter else None
            text = "Progress: [{0}] {1:.2f}%".format("#" * block + "-" * (bar_length - block), progress * 100)
            print(text, end="\r")
            sys.stdout.flush()
        
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
    y0_k = init_photons(k_values, A_scale=A_scale, Adot_scale=Adot_scale)

    # Establish cutoff values for resonance and infinities
    res_con = params['res_con']
    inf_con = params['inf_con']

    # Solve the differential equation for each k, in parallel
    if parallelize:
        pool = mp.Pool(num_cores)
        pool_params = [(system_in, params, np.float64(y0), np.float64(k), verbosity, method, res_con, inf_con) for k, y0 in zip(k_values, y0_k)]
        pool_inputs = tqdm.tqdm(pool_params, total=k_N) if show_progress_bar else pool_params
        solutions = pool.starmap(solve_subsystem, pool_inputs)
        pool.close()
        pool.join()
        
    else:
        t_span = params['t_span']
        t = np.linspace(t_span[0], t_span[1], params['t_num'])
        solutions = np.zeros((k_N, 2, len(t)))
        k_count_max = 0
        for i, k in enumerate(k_values):
            if verbosity > 7 and show_progress_bar:
                #print('i = %d,   k[i] = %d' % (i, k))
                progress_val(i)
            solutions[i] = solve_subsystem(system_in, params, np.float64(y0_k[i]), np.float64(k), verbosity=0, \
                                           method=method, resonance_limit=res_con, precision_limit=inf_con) # Store the solution
    
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
def solve_subsystem(system_in, params, y0_in, k, verbosity=0, method='RK45', resonance_limit=1e6, precision_limit=1e100):
    # Initial conditions
    y0 = y0_in

    # Debug print statements
    if verbosity > 8:
        print('\n  k=%.2f %s' % (k, '   | y0:   %.2e' % y0 if verbosity > 9 else ''))
    
    # Time domain
    t_span = params['t_span']
    t = np.linspace(t_span[0], t_span[1], params['t_num'])

    # Define milestone events during integration
    def hit_resonance_limit(t, y, k, params, limit=resonance_limit):
        return limit - y[0]
    def hit_precision_limit(t, y, k, params, limit=precision_limit):
        return limit - y[0]
    
    # Integration will end when we hit a terminal event
    hit_precision_limit.terminal = True
    # (DISABLED) We only care about when we first reach resonance, not dipping back below it
    # hit_resonance_limit.direction = +1

    extra_events = [hit_resonance_limit] if resonance_limit is not None else []
    solve_events = [hit_precision_limit] + extra_events

    sol = solve_ivp(system_in, t_span, y0, args=(k, params), dense_output=True, method=method, events=solve_events)

    # Store the termination reason in params
    status = sol.status
    params['int_status'] = status
    if status != 0:
        if status == 1 and verbosity >= 8:
            print('A termination event was detected')
        if status == -1 and verbosity >= 7:
            print('Integration step failed')
    
    # TODO: Extract important timestamps
    t_events = sol.t_events
    y_events = sol.y_events
    
    # Evaluate the solution at the times in the array
    #t = sol.t
    y = sol.sol(t)
    return y

def piaxi_system(t, y, k, params, P, B, C, D, A_pm, bg, k0, c, h, G, use_logsumexp=False):
    # System of differential equations to be solved (bg = photon background)
    '''
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
            logBeta    = 0
            Beta_sign  = 0
        else:
            logBeta    = np.log(np.abs(B(t))) - np.log(np.abs(bg + P(t)))
            Beta_sign  = np.sign(B(t))*np.sign(bg + P(t))
        if disable_C:
            logCterm   = 0
            Cterm_sign = 0
        else:
            logCterm   = np.log(np.abs(C(t, A_pm))) - np.log(np.abs(bg + P(t))) + np.log(k)
            Cterm_sign = np.sign(C(t, A_pm))*np.sign(bg + P(t))
        if disable_D:
            logDterm   = 0
            Dterm_sign = 0
        else:
            logDterm   = np.log(np.abs(D(t))) - np.log(np.abs(bg + P(t)))
            Dterm_sign = np.sign(D(t))*np.sign(bg + P(t))
        logAlpha, Alpha_sign = logsumexp(a=[logCterm, logDterm, np.log(k**2)],
                                         b=[Cterm_sign, Dterm_sign, 1], return_sign=True)
        
        # Numerically calculate A''[t] in log-space
        logdy1dt, dy1dt_sign = logsumexp(a=[logBeta + np.log(np.abs(y[1])), logAlpha + np.log(np.abs(y[0]))],
                                         b=[Beta_sign * np.sign(y[1]), Alpha_sign * np.sign(y[0])], return_sign=True)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                dy1dt = -dy1dt_sign*np.exp(np.float64(logdy1dt))
            except RuntimeWarning as e:
                print('RuntimeWarning: %s \n   -->   when logdy1dt= %.1f' % (e, logdy1dt))
                if logdy1dt >= 100:
                    dy1dt = -dy1dt_sign*np.inf
                elif logdy1dt <= -100:
                    dy1dt = 0
                else:
                    dy1dt = np.nan
    else:
        # Numerically calculate A''[t]
        dy1dt = -1./(bg + P(t)) * (B(t)*y[1] + (C(t, A_pm)*(k*np.float64(k0)) + D(t))*y[0]) - (k*np.float64(k0))**2*y[0]
    
    # Return A'[t], A''[t]
    dy0dt = y[1]
    return [dy0dt, dy1dt]

def init_photons(k_values, A_scale=1.0, Adot_scale=1.0):
    A_0 = np.array([1./np.sqrt(2.*(k_val)) for k_val in k_values], dtype=np.float64) * A_scale
    Adot_0 = np.array([np.sqrt((k_val)/2.) for k_val in k_values], dtype=np.float64) * Adot_scale

    return np.array([A_0, Adot_0], dtype=np.float64).T

# TODO: Verify / implement this
def floquet_exponent(params_in, p=Beta, q=Alpha, T=2*np.pi, y0_in=None, yp0_in=None, k_modes=[]):
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
    k_modes = k_modes if len(k_modes) > 0 else get_kvals(params_in, None)
    num_modes = len(k_modes)
    y0  = y0_in  if y0_in  is not None else params_in['A_0']    if 'A_0'    in params_in else 0
    yp0 = yp0_in if yp0_in is not None else params_in['Adot_0'] if 'Adot_0' in params_in else 0.1
    
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