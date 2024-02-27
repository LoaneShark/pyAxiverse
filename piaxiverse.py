import numpy as np
import pandas as pd
import argparse
import datetime
from piaxi_utils import *
from piaxi_utils import version, default_output_directory, k_to_Hz, Hz_to_k
from piaxi_numerics import solve_piaxi_system, piaxi_system

## Parameters of model
manual_set = False       # Toggle override mass definitions
unitful_masses = True    # Toggle whether masses are defined in units of [eV] vs. units of mass-ratio [m_unit] (Currently always True)
unitful_k = False        # Toggle whether k values are defined unitfully [eV] vs. units of mass-ratio [m_unit] (Default: False)

# main loop
def main(args):
    scan_args = [args.scan_mass, args.scan_Lambda, args.scan_Lambda3, args.scan_Lambda3, args.scan_F, args.scan_epsilon]
    if any([scan_arg is not None for scan_arg in scan_args]) or args.num_samples > 1:
        run_multiple_cases(args)
    else:
        run_single_case(args)

    return None

get_values = lambda xmin, xmax, N: np.geomspace(10**np.float64(xmin), 10**np.float64(xmax), N)
def run_multiple_cases(args):

    mass_values = [args.m_scale]
    mass_N      = args.scan_mass_N
    L3_values   = [args.L3]
    L4_values   = [args.L4]
    F_values    = [args.F]
    fit_F       = args.fit_F
    F_N         = args.scan_F_N
    Lambda_N    = args.scan_Lambda_N
    Lambda3_N   = args.scan_Lambda3_N if args.scan_Lambda3_N is not None else args.scan_Lambda
    Lambda4_N   = args.scan_Lambda4_N if args.scan_Lambda4_N is not None else args.scan_Lambda
    eps_values  = [args.eps]
    eps_N       = args.scan_epsilon_N

    scan_mass     = np.float64(args.scan_mass)    if args.scan_mass    is not None else None
    scan_F        = np.float64(args.scan_F)       if args.scan_F       is not None else None
    scan_Lambda   = np.float64(args.scan_Lambda)  if args.scan_Lambda  is not None else None
    scan_Lambda3  = np.float64(args.scan_Lambda3) if args.scan_Lambda3 is not None else None
    scan_Lambda4  = np.float64(args.scan_Lambda4) if args.scan_Lambda4 is not None else None
    scan_eps      = np.float64(args.scan_epsilon) if args.scan_epsilon is not None else None
    N_samples     = args.num_samples if args.num_samples is not None else 1

    if scan_mass is not None:
        m_min, m_max = scan_mass
        mass_values = get_values(m_min, m_max, mass_N)

    if scan_eps is not None:
        eps_min, eps_max = scan_eps
        eps_values = get_values(eps_min, eps_max, eps_N)
    
    if scan_F is not None:
        F_min, F_max = scan_F
        F_values = get_values(F_min, F_max, F_N)
        # TODO: Allow this to scan about inferred value from epsilon if said argument is enabled?
        fit_F = False

    if args.verbosity >= 9:
        print('scanning args in: ')
        print('  \n'.join([str(inparam) for inparam in [scan_Lambda, scan_Lambda3, scan_Lambda4, Lambda_N, Lambda3_N, Lambda4_N]]))
    if scan_Lambda is not None:
        L_min, L_max = scan_Lambda
        L3_values = get_values(L_min, L_max, Lambda3_N)
        L4_values = get_values(L_min, L_max, Lambda4_N)
    if scan_Lambda3 is not None:
        L3_min, L3_max = scan_Lambda3
        L3_values = get_values(L3_min, L3_max, Lambda3_N)
    if scan_Lambda4 is not None:
        L4_min, L4_max = scan_Lambda4
        L4_values = get_values(L4_min, L4_max, Lambda4_N)

    i = 0
    for Ni in range(N_samples):
        for mass in mass_values:
            for eps in eps_values:
                for F in F_values:
                    for L3 in L3_values:
                        for L4 in L4_values:
                            if args.verbosity >= 7:
                                print('Running %s case: F=%.1e, m=%.1e, eps=%.1e, L3=%.1e, L4=%.1e' % ('first' if i < 1 else 'next', F, mass, eps, L3, L4))
                            run_single_case(args, Fpi_in=F, L3_in=L3, L4_in=L4, m_scale_in=mass, eps_in=eps, fit_F_in=fit_F)
                            i += 1
    return None

def run_single_case(args, Fpi_in=None, L3_in=None, L4_in=None, m_scale_in=None, eps_in=None, fit_F_in=None):
    ## INPUT PARAMETERS
    verbosity         = args.verbosity            # Set debug print statement verbosity level (0 = Standard, -1 = Off)
    use_mass_units    = args.use_mass_units       # Toggle whether calculations / results are given in units of pi-axion mass (True) or eV (False)
    use_natural_units = args.use_natural_units    # Toggle whether calculations / results are given in c = h = G = 1 (True) or SI units (False)   || NOTE: full SI/phsyical unit support is still WIP!!
    save_output_files = args.save_output_files    # Toggle whether or not the results from this notebook run are written to a data directory
    int_method        = args.int_method           # Scipy.integrate.solve_ivp() integration method

    config_name       = args.config_name          # Descriptive name for the given parameter case. Output files will be saved in a directory with this name.
    seed              = args.seed                 # rng_seed, integer value (None for random)
    num_cores         = args.num_cores            # Number of parallel threads available
    mem_per_core      = args.mem_per_core         # Number of parallel threads available
    data_path         = args.data_path            # Path to directory where output files will be saved
    skip_existing     = args.skip_existing

    # Initialize rng
    rng, rng_seed = get_rng(seed=seed, verbosity=verbosity)

    # Establish file paths
    storage_path = data_path
    output_dir   = os.path.join(storage_path, version, config_name,)

    ## CONSTANTS OF MODEL
    # Unitful fundamental constants
    c_raw = c = np.float64(2.998e10)    # Speed of light in a vacuum [cm/s]
    h_raw = h = np.float64(4.136e-15)   # Planck's constant [eV/Hz]
    G_raw = G = np.float64(1.0693e-19)  # Newtonian constant [cm^5 /(eV s^4)]
    unitful_c = unitful_h = unitful_G = not(use_natural_units)

    # values to use in calculations in order to ensure correct units
    c_u = c if unitful_c else 1.
    h_u = h if unitful_h else 1.
    G_u = G if unitful_G else 1.

    # Tuneable constants
    e    = 0.3         # dimensionless electron charge
    F    = Fpi_in if Fpi_in is not None else args.F      # pi-axion decay constant (GeV) >= 10^11
    p_t  = args.rho    # total local DM density (GeV/cm^3)
    ## --> TODO: Could/Should we support spatially dependent distributions?
    eps  = eps_in if eps_in is not None else args.eps    # millicharge, vary to enable/disable charged species (<= 10e-25 ~ ON, >= 10e-15 ~ OFF)
    L3   = L3_in  if L3_in  is not None else args.L3     # Coupling constant Lambda_3
    L4   = L4_in  if L4_in  is not None else args.L4     # Coupling constant Lambda_4
    l1 = l2 = l3 = l4 = 1
    
    # Unit scaling:
    m_scale = args.m_scale             # dark quark mass scale (eV) <= 10-20
    ## Handle optional fitting of F_pi as a function of fundamental mass scale]
    fit_F = fit_F_in if fit_F_in is not None else args.fit_F
    if fit_F:
        F = fit_Fpi(eps, m_scale, verbosity=verbosity)
    dimensionful_p = not(use_natural_units)
    #p_unit = 1.906e-12
    p_unit = (c_raw*h_raw)**3 if not(dimensionful_p) else (c_u*h_u)**3   # convert densities from units of [1/cm^3] to [eV^3]
    GeV  = 1e9     # GeV -> eV
    #GeV = 1
    F   *= GeV
    p_t *= GeV
    p_t *= p_unit  # 1/cm^3 -> (eV/hc)^3
    L3  *= GeV
    L4  *= GeV

    ## Dark SM Parameters
    sample_qmass = False # TODO
    sample_qcons = args.dqm_c is None

    # SM quark masses for all 3 generations
    qm = m_scale*np.array([1., 2., 40.]) if not sample_qmass else m_scale*np.array([0., 0., 0.]) # TODO

    # dSM quark scaling constants (up, down, strange, charm, bottom, top) sampled from uniform distribution [0.7, 1.3]
    #qc = np.array(args.dqm_c) if not sample_qcons else rng.uniform(0.7, 1.3, (6,))
    #qc_arr_in = args.dqm_c.split('_')
    #print(qc_arr_in)
    #qc = np.array([qc_val if qc_val is not None else rng.uniform(0.7, 1.3) if sample_qcons else 0. for qc_val in qc_arr_in], dtype=float)
    qc = np.array([qc_val if qc_val is not None else rng.uniform(0.7, 1.3) if sample_qcons else 0. for qc_val in args.dqm_c], dtype=float).reshape((6,))

    # Dark quark masses (up, down, strange, charm, bottom, top)
    dqm = np.array([qm[0]*qc[0], qm[0]*qc[1], qm[1]*qc[2], qm[1]*qc[3], qm[2]*qc[4], qm[2]*qc[5]])

    # Scaling parameters
    xi     = np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])  # Charged species scaling paramters
    eps_c  = np.array([+1., +1., +1., -1., -1., -1., -1., +1., -1.])  # Millicharge sign

    # Time domain
    t_span = [0, args.t]   # Units of 1/m_u
    t_N = args.tN          # Number of timesteps
    t_sens = 0.1           # sensitivity for calculating time-averaged values

    # k-space domain
    use_k_eq_0 = False     # Toggle whether or not k = 0 is included in the numerics (div. by 0 error possible if on)
    k_min = 0 if use_k_eq_0 else 1
    k_max = args.k if args.k > 0 else args.kN  # default to a k-mode granularity of 1
    k_span = [k_min, k_max]  # TODO: replace with the appropriate values
    #k_res = args.k/args.kN if args.k > 0 else 1.         # k-mode granularity
    k_res = args.k_res                                    # k-mode granularity
    k_N = int((1./k_res)*max((k_max - k_min), 0) + 1)     # Number of k-modes
    #k_N = args.kN

    # Initial Conditions
    A_0    = args.A_0
    Adot_0 = args.Adot_0
    A_pm   = +1       # specify A± case (+1 or -1)
    A_sens = 1.0      # sensitivity for classification of resonance conditions
    em_bg  = 1.0      # photon background (Default 1)

    # Toggle whether mass-energy values should be computed in units of eV (False) or pi-axion mass (True)
    # (by default, k is defined in units of [m_u] whereas m is defined in units of [eV], so their scaling logic is inverted)
    unitful_amps   = unitful_m = True
    rescale_m      = use_mass_units if unitful_masses else not(use_mass_units)
    rescale_k      = not(rescale_m) if unitful_masses else rescale_m
    rescale_amps   = use_mass_units if unitful_amps   else not(use_mass_units)

    # Toggle whether or not local and global phases should be sampled
    sample_delta = args.sample_delta
    sample_theta = args.sample_theta

    # Define pi-axiverse mass species
    m_r, m_n, m_c, counts, masks = define_mass_species(qm=qm, qc=qc, F=F, e=e, eps=eps, eps_c=eps_c, xi=xi)
    N_r, N_n, N_c = counts

    # Populate masses for real, complex, and charged species (given in units of eV)
    m, m_u = init_masses(m_r, m_n, m_c, natural_units=use_natural_units, c=c, verbosity=verbosity)
    # Handle unit rescaling logic
    m_unit = m_u      # value of m_u in [eV]

    m0     = m0_f(m_u, c_u, rescale_m, unitful_m)      # Desired units  (m --> m_u)
    m0_raw = m0_f(m_u, c_raw, rescale_m, unitful_m)    # Physical units (eV/c^2)
    k0     = k0_f(m_u, c_u, rescale_k, unitful_m)      # Desired units  (k --> m_u)
    k0_raw = k0_f(m_u, c_raw, rescale_k, unitful_m)    # Physical units (eV/c)
    t0     = t0_f(m_u, h_u, rescale_m, unitful_m)      # Desired units  (t --> 1/m_u)
    t0_raw = t0_f(m_u, h_raw, rescale_m, unitful_m)    # Physical units (s)

    if verbosity >= 6:
        print('masks: ', masks)
        print('counts: ', counts)

    ## Populate pi-axion dark matter energy densities for all species
    p = init_densities(masks, p_t=p_t, normalized_subdens=True)

    ## Populate (initial) pi-axion dark matter mass-amplitudes for each species, optional units of [eV/c]
    amps = init_amplitudes(m, p, m_unit=m_u, h=h, c=c, mass_units=use_mass_units, natural_units=use_natural_units, unitful_amps=unitful_masses, rescale_amps=rescale_amps, verbosity=verbosity)

    # Populate and sample local and global phases from normal distribution  for each species, between 0 and 2pi
    d, Th = init_phases(masks, rng, sample_delta, sample_theta, verbosity=verbosity, sample_dist='uniform')

    # For performance gains, omit fully non-existant species from the numerics
    m    = trim_masked_arrays(m)
    p    = trim_masked_arrays(p)
    amps = trim_masked_arrays(amps)
    Th   = trim_masked_arrays(Th)
    d    = trim_masked_arrays(d)

    ## Define the system of ODEs
    # TODO: Verify P(t), B(t), C(t), D(t) are all of the correct form, double check all signs and factors of 2
    # NOTE: Using cosine definitons for amplitudes
    #       i = {0,1,2} correspond to {pi_0, pi, pi_±} respectively

    # Rescale all eV unit constants to unit mass
    rescale_consts = rescale_m if unitful_masses else not(rescale_m)
    L3_sc = abs(L3) if not rescale_consts else L3 / m_unit
    L4_sc = abs(L4) if not rescale_consts else L4 / m_unit
    F_sc  = abs(F)  if not rescale_consts else  F / m_unit

    # Characteristic timescales (minimum binsize to capture full oscillations) by species
    T_min, T_r, T_n, T_c = get_timescales(m, m0, m_u=1, verbosity=verbosity)

    # Get units and pretty print parameters and system configuration to console
    units = get_units(unitful_m, rescale_m, unitful_k, rescale_k, unitful_amps, rescale_amps, rescale_consts, dimensionful_p, unitful_c, unitful_h, unitful_G, use_mass_units, verbosity=verbosity)
    print_params(units, m=m, p=p, amps=amps, Th=Th, d=d, m_q=m_scale, m_0=m0, m_u=m_u, natural_units=use_natural_units, verbosity=verbosity)

    # Shorthand helper function for oscillatory time-dependent terms
    # (time is assumed to be defined in units of [1/m_u] always)
    phi = lambda t, s, i, m=m, d=d, t0=t0: (m[s][i])*(t*t0) + d[s][i]
    #phi = lambda t, s, i, m=m, d=d, M=(1./m_unit if unitful_masses and not rescale_m else 1.): (m[s][i]*M)*t + d[s][i]

    # Define coefficient functions to clean up differential equation representation
    P = lambda t, l3=l3, L3=L3_sc, l4=l4, L4=L4_sc, eps=eps, amps=amps, m=m, M=m0, d=d, Th=Th, c=c_u, h=h_u, G=G_u, phi=phi, np=np: \
            2*l3/(L3**2) * eps**2 * (np.sum([amps[2][i]*amps[2][j] * np.cos(phi(t,2,i))*np.cos(phi(t,2,j)) * np.cos(Th[2][i]-Th[2][j]) \
                                                for i in range(len(m[2])) for j in range(len(m[2]))], axis=0)) + \
            2*l4/(L4**2) * eps**2 * (np.sum([amps[1][i]*amps[1][j] * np.cos(phi(t,1,i))*np.cos(phi(t,1,j)) * np.cos(Th[1][i]-Th[1][j]) \
                                                for i in range(len(m[1])) for j in range(len(m[1]))], axis=0) + \
                                     np.sum([amps[0][i]*amps[0][j] * np.cos(phi(t,0,i))*np.cos(phi(t,0,j)) \
                                                for i in range(len(m[0])) for j in range(len(m[0]))], axis=0) + \
                                   2*np.sum([amps[0][i]*amps[1][j] * np.cos(phi(t,0,i))*np.cos(phi(t,1,j)) * np.cos(Th[1][j]) \
                                                for i in range(len(m[0])) for j in range(len(m[1]))], axis=0))

    B = lambda t, l3=l3, L3=L3_sc, l4=l4, L4=L4_sc, eps=eps, amps=amps, m=m, M=m0, d=d, Th=Th, c=c_u, h=h_u, G=G_u, phi=phi, np=np: \
                (-1)*2*l3/(L3**2) * eps**2 * (np.sum([amps[2][i]*amps[2][j] * np.cos(Th[2][i]-Th[2][j])  * \
                                                    ((m[2][i]*M) * np.sin(phi(t,2,i)) * np.cos(phi(t,2,j)) + \
                                                     (m[2][j]*M) * np.cos(phi(t,2,i)) * np.sin(phi(t,2,j))) \
                                                    for i in range(len(m[2])) for j in range(len(m[2]))], axis=0)) + \
                (-1)*2*l4/(L4**2) * eps**2 * (np.sum([amps[0][i]*amps[0][j] * ((m[0][i]*M) * np.sin(phi(t,0,i)) * np.cos(phi(t,0,j)) + \
                                                                               (m[0][j]*M) * np.cos(phi(t,0,i)) * np.sin(phi(t,0,j))) \
                                                    for i in range(len(m[0])) for j in range(len(m[0]))], axis=0) + \
                                                    np.sum([amps[1][i]*amps[1][j] * np.cos(Th[1][i]-Th[1][j])  * \
                                                            ((m[1][i]*M) * np.sin(phi(t,1,i)) * np.cos(phi(t,1,j)) + \
                                                             (m[1][j]*M) * np.cos(phi(t,1,i)) * np.sin(phi(t,1,j))) \
                                                            for i in range(len(m[1])) for j in range(len(m[1]))], axis=0) + \
                                                    np.sum([np.abs(amps[0][i]*amps[1][j]) * np.cos(Th[1][j]) * \
                                                            ((m[0][i]*M) * np.sin(phi(t,0,i)) * np.cos(phi(t,1,j)) + \
                                                             (m[1][j]*M) * np.cos(phi(t,0,i)) * np.sin(phi(t,1,j))) \
                                                            for i in range(len(m[0])) for j in range(len(m[1]))], axis=0))

    C = lambda t, pm, l1=l1, F=F_sc, eps=eps, amps=amps, m=m, M=m0, d=d, c=c_u, h=h_u, G=G_u, phi=phi, np=np: \
                (-1) * pm * (2*l1 / F) * eps**2 * np.sum([amps[0][i] * (m[0][i]*M) * np.sin(phi(t,0,i)) \
                                                        for i in range(len(m[0]))], axis=0)

    D = lambda t, l2=l2, e=e, eps=eps, amps=amps, m=m, M=m0, d=d, Th=Th, c=c_u, h=h_u, G=G_u, phi=phi, np=np: \
                l2 * eps**2 * e**2 * np.sum([amps[2][i]*amps[2][j] * np.cos(phi(t,2,i))*np.cos(phi(t,2,j)) * np.cos(Th[2][i]-Th[2][j]) \
                                            for i in range(len(m[2])) for j in range(len(m[2]))], axis=0)

    # TODO: Add support for supplying custom functions?
    disable_P = not(args.P)
    disable_B = not(args.B)
    disable_C = not(args.C)
    disable_D = not(args.D)
    override_coefficients = True if any([disable_P, disable_B, disable_C, disable_D]) else False
    if override_coefficients:
        if disable_P:
            P = P_off
        if disable_B:
            B = B_off
        if disable_C:
            C = C_off
        if disable_D:
            D = D_off
        
    # Prepare the numerical integration
    k_values, k_step = np.linspace(k_span[0], k_span[1], k_N, retstep=True)
    #k_values = np.linspace(1./100, 10, 100)

    # Initialize an array to store the solutions
    t, t_step = np.linspace(t_span[0], t_span[1], t_N, retstep=True)  # Array of times at which to evaluate, t > 0
    #t = t[1:]

    # Classification sensitivity threshold
    res_con = 1000
    #res_con = max(100,1./A_sens)

    # Collect all simulation configuration parameters
    parameters = {'e': e, 'F': F, 'p_t': p_t, 'eps': eps, 'L3': L3, 'L4': L4, 'l1': l1, 'l2': l2, 'l3': l3, 'l4': l4, 'res_con': res_con,
                'A_0': A_0, 'Adot_0': Adot_0, 'A_pm': A_pm, 'A_sens': A_sens, 'k_step': k_step,
                't_sens': t_sens, 't_step': t_step, 'T_n': T_n, 'T_r': T_r, 'T_c': T_c, 'T_u': T_min,
                'qm': qm, 'qc': qc, 'dqm': dqm, 'eps_c': eps_c, 'xi': xi, 'm_0': m0, 'm_u': m_unit, 'm_scale': m_scale, 'p_unit': p_unit,
                'm_r': m[0], 'm_n': m[1], 'm_c': m[2], 'p_r': p[0], 'p_n': p[1], 'p_c': p[2], 'Th_r': Th[0], 'Th_n': Th[1], 'Th_c': Th[2],
                'amp_r': amps[0], 'amp_n': amps[1], 'amp_c': amps[2], 'd_r': d[0], 'd_n': d[1], 'd_c': d[2], 'k_0': k0,
                'unitful_m': unitful_masses, 'rescale_m': rescale_m, 'unitful_amps': unitful_amps, 'rescale_amps': rescale_amps, 
                'unitful_k': unitful_k, 'rescale_k': rescale_k, 'rescale_consts': rescale_consts, 'h': h, 'c': c, 'G': G, 'seed': rng_seed, 
                'dimensionful_p': dimensionful_p, 'use_natural_units': use_natural_units, 'use_mass_units': use_mass_units, 'em_bg': em_bg,
                'int_method': int_method, 'disable_P': disable_P, 'disable_B': disable_B, 'disable_C': disable_C, 'disable_D': disable_D}

    # Create unique hash for input parameters (to compare identical runs)
    phash = get_parameter_space_hash(parameters, verbosity=verbosity)

    # Save system performance related input parameters (not to be hashed because these don't affect the final state, only performance time)
    parameters['num_cores']    = num_cores
    parameters['mem_per_core'] = mem_per_core

    if skip_existing and if_output_exists(output_dir, phash):
        if verbosity >= 1:
            print('SKIP: output file already exists for this parameter configuration')
    else:
        # Solve the system, in parallel for each k-mode
        os.environ['NUMEXPR_MAX_THREADS'] = '%d' % (max(int(num_cores), 1))
        is_parallel = (num_cores > 1)
        show_progress = (verbosity >= 0)

        # Initialize parameters
        params = init_params(parameters, t_min=t_span[0], t_max=t_span[1], t_N=t_N, k_min=k_span[0], k_max=k_span[1], k_N=k_N)
        # Define system of equations to solve
        local_system = lambda t, y, k, params: piaxi_system(t, y, k, params, P=P, B=B, C=C, D=D, A_pm=A_pm, bg=em_bg, k0=k0, c=c_u, h=h_u, G=G_u)

        # Solve the equations of motion
        solutions, params, time_elapsed = solve_piaxi_system(local_system, params, k_values, parallelize=is_parallel, num_cores=num_cores, verbosity=verbosity, show_progress_bar=show_progress, method=int_method)

        # Generate plots and optionally show them
        make_plots = args.make_plots
        show_plots = args.show_plots
        result_plots = {}

        # Plot results (Amplitudes)
        k_peak, k_mean = get_peak_k_modes(solutions, k_values, write_to_params=True)
        tot_res = params['res_class']
        if make_plots:
            if verbosity > 0:
                print('max (peak) k mode: ' + str(k_peak))
                print('max (mean) k mode: ' + str(k_mean))

            # Plot the solution
            plt = make_amplitudes_plot(params, units, solutions)
            result_plots['amps'] = plt.gcf()
            if show_plots:
                plt.show()
        
        if make_plots:
            times = t

            scale_n = True
            n_k_local = n_k
            class_method = 'binned'
            
            plt, params, t_res, n_res = make_occupation_num_plots(params, units, solutions, scale_n=scale_n, write_to_params=True, numf_in=n_k_local, class_method=class_method)
            n_tot = sum_n_p(n_k_local, params, solutions, k_values, times)
            result_plots['nums'] = plt.gcf()
            if show_plots:
                plt.show()

            print('n_tot in range [%.2e, %.2e]' % (min(n_tot), max(n_tot)))
            if 'res' in tot_res and verbosity > 2:
                print('resonance classification begins at t = %.2f, n = %.2e' % (t_res, n_res))

        if make_plots:
            # Plot results (Oscillating coefficient values)
            plt = make_coefficients_plot(params, units, P, B, C, D, A_pm, k0)
            result_plots['coeffs'] = plt.gcf()
            if show_plots:
                plt.show()

            if verbosity == 2:
                print_coefficient_ranges(params, P, B, C, D)
            elif verbosity > 2:
                print_coefficient_ranges(params, P, B, C, D, print_all=True)

            if verbosity > 5:
                print('params:\n', params, '\n')
            if verbosity > 2:
                print('params[\'class\']:\n', params['res_class'])
                if verbosity > 6:
                    print('params[\'k_class_arr\']:\n', params['k_class_arr'])
                    print('k_ratio:\n', k_ratio(np.mean, t_sens, A_sens))

        # E^2 = p^2c^2 + m^2c^4
        # Assuming k, m are given in units of eV/c and eV/c^2 respectively
        #k_to_Hz = lambda ki, mi=0, m_0=m0, e=e: 1/h * np.sqrt((ki*k0*e)**2 + ((mi*m_0 * e))**2)
        k_to_Hz_local = lambda ki, k0=k0_raw, h=h_raw, c=c_raw: k_to_Hz(ki, k0, h, c)
        #Hz_to_k = lambda fi, mi=0, m_0=m0, e=e: 1/(e*k0) * np.sqrt((h * fi)**2 - ((mi*m_0 * e))**2)
        Hz_to_k_local = lambda fi, k0=k0_raw, h=h_raw, c=c_raw: Hz_to_k(fi, k0, h, c)

        # Plot k-mode power spectrum (TODO: Verify power spectrum calculation)
        if make_plots:
            plt = make_resonance_spectrum(params, solutions, units, k_to_Hz_local, Hz_to_k_local, numf_in=n_k_local, class_method=class_method)
            result_plots['resonance'] = plt.gcf()
            if show_plots:
                plt.show()

        # Known observable frequency range and bandwidth classification
        res_freq, res_freq_class = get_frequency_class(k_peak, k_to_Hz_local, tot_res, verbosity=verbosity)
        res_band_min, res_band_max, res_band_class = get_resonance_band(k_values, params['k_class_arr'], k_to_Hz_local, verbosity=verbosity)

        params['res_freq'] = res_freq
        params['res_band'] = [res_band_min, res_band_max]
        params['res_freq_class'] = res_freq_class
        params['res_band_class'] = res_band_class

        if make_plots:
            plt = plot_ALP_survey(params, verbosity=verbosity)

            result_plots['alp'] = plt.gcf()
            if show_plots:
                plt.show()
        
        # Optionally save results of this run to data directory
        save_input_params = True
        save_integrations = True
        save_output_plots = make_plots

        if save_output_files:
            output_name  = '_'.join([config_name, phash])
            save_results(output_dir, output_name, params, solutions, result_plots, verbosity=verbosity, save_format='pdf',
                        save_params=save_input_params, save_results=save_integrations, save_plots=save_output_plots,
                        save_coefficients=True, P=P, B=B, C=C, D=D, plot_types=['amps', 'nums', 'coeffs', 'resonance', 'alp'])

        if verbosity > 1:
            print('Done!')

trim_masked_arrays = lambda arr: np.array([np.array(arr_i, dtype=float) if len(arr_i) > 0 else np.array([], dtype=float) for arr_i in arr], dtype=object)

# Functions to rescale mass, k-modes, and time to desired units
m0_f = lambda m_in, c_in, rescale_m, unitful_m: 1./c_in**2 if not rescale_m else (1./(m_in*c_in**2) if unitful_m else (m_in/c_in**2))
k0_f = lambda m_in, c_in, rescale_k, unitful_m: m_in/c_in if rescale_k else 1./c_in
t0_f = lambda m_in, h_in, rescale_m, unitful_m: h_in/m_in if unitful_m else h_in

## Pi-axion Species Mass Definitions
def define_mass_species(qc, qm, F, e, eps, eps_c, xi):
    m_r = np.array([0., 0., 0., 0., 0.], dtype=float)                   # real neutral
    m_n = np.array([0., 0., 0., 0., 0., 0.], dtype=float)               # complex neutral
    m_c = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float)   # charged
    # Pi-axion Species Labels
    s_l = np.array([np.full_like(m_r, '', dtype=str), np.full_like(m_n, '', dtype=str), np.full_like(m_c, '', dtype=str)], dtype=object)
    
    # Millicharge upper bound (charged species do not survive past this)
    eps_bound = 1e-10

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
    m_c[0]    = np.sqrt((qc[0]*qm[0] + qc[1]*qm[0])*F + 2*xi[0]*(e*eps*eps_c[0]*F)**2) if 0. not in [qc[0], qc[1]] and abs(eps) <= eps_bound else 0.
    s_l[2][0] = '$\pi_1 \pm i\pi_2$'
    # pi_4  +/- i*pi_5
    m_c[1]    = np.sqrt((qc[0]*qm[0] + qc[2]*qm[1])*F + 2*xi[1]*(e*eps*eps_c[1]*F)**2) if 0. not in [qc[0], qc[2]] and abs(eps) <= eps_bound else 0.
    s_l[2][1] = '$\pi_4 \pm i\pi_5$'
    # pi_15 +/- i*pi_16
    m_c[2]    = np.sqrt((qc[0]*qm[0] + qc[4]*qm[2])*F + 2*xi[2]*(e*eps*eps_c[2]*F)**2) if 0. not in [qc[0], qc[4]] and abs(eps) <= eps_bound else 0.
    s_l[2][2] = '$\pi_{15} \pm i\pi_{16}$'
    # pi_11 +/- i*pi_12
    m_c[3]    = np.sqrt((qc[1]*qm[0] + qc[3]*qm[2])*F + 2*xi[3]*(e*eps*eps_c[3]*F)**2) if 0. not in [qc[1], qc[3]] and abs(eps) <= eps_bound else 0.
    s_l[2][3] = '$\pi_{11} \pm i\pi_{12}$'
    # pi_23 +/- i*pi_24
    m_c[4]    = np.sqrt((qc[1]*qm[0] + qc[5]*qm[2])*F + 2*xi[4]*(e*eps*eps_c[4]*F)**2) if 0. not in [qc[1], qc[5]] and abs(eps) <= eps_bound else 0.
    s_l[2][4] = '$\pi_{23} \pm i\pi_{24}$'
    # pi_13 +/- i*pi_14
    m_c[5]    = np.sqrt((qc[2]*qm[1] + qc[3]*qm[1])*F + 2*xi[5]*(e*eps*eps_c[5]*F)**2) if 0. not in [qc[2], qc[3]] and abs(eps) <= eps_bound else 0.
    s_l[2][5] = '$\pi_{13} \pm i\pi_{14}$'
    # pi_25 +/- i*pi_26
    m_c[6]    = np.sqrt((qc[2]*qm[1] + qc[5]*qm[2])*F + 2*xi[6]*(e*eps*eps_c[6]*F)**2) if 0. not in [qc[2], qc[5]] and abs(eps) <= eps_bound else 0.
    s_l[2][6] = '$\pi_{25} \pm i\pi_{26}$'
    # pi_27 +/- i*pi_28
    m_c[7]    = np.sqrt((qc[3]*qm[1] + qc[4]*qm[2])*F + 2*xi[7]*(e*eps*eps_c[7]*F)**2) if 0. not in [qc[3], qc[4]] and abs(eps) <= eps_bound else 0.
    s_l[2][7] = '$\pi_{27} \pm i\pi_{28}$'
    # pi_32 +/- i*pi_33
    m_c[8]    = np.sqrt((qc[4]*qm[2] + qc[5]*qm[2])*F + 2*xi[8]*(e*eps*eps_c[8]*F)**2) if 0. not in [qc[4], qc[5]] and abs(eps) <= eps_bound else 0.
    s_l[2][8] = '$\pi_{32} \pm i\pi_{33}$'

    # Mask zero-valued / disabled species in arrays
    m_r = np.ma.masked_where(m_r == 0.0, m_r, copy=True)
    m_n = np.ma.masked_where(m_n == 0.0, m_n, copy=True)
    m_c = np.ma.masked_where(m_c == 0.0, m_c, copy=True)
    r_mask = np.ma.getmaskarray(m_r)
    n_mask = np.ma.getmaskarray(m_n)
    c_mask = np.ma.getmaskarray(m_c)
    masks = np.array([r_mask, n_mask, c_mask], dtype=object)
    N_r = m_r.count()
    N_n = m_n.count()
    N_c = m_c.count()
    counts = np.array([N_r, N_n, N_c])
    
    return m_r, m_n, m_c, counts, masks

def init_masses(m_r: np.ma, m_n: np.ma, m_c: np.ma, natural_units=True, c=1, verbosity=0):

    #m_unit = np.min([np.min(m_i) for m_i in (m_r.compressed(), m_n.compressed(), m_c.compressed()) if len(m_i) > 0])

    m = np.array([m_r.compressed(), m_n.compressed(), m_c.compressed()], dtype=object, copy=True)
    m_unit = np.min([np.min(m_i) for m_i in m if len(m_i) > 0])
    m_raw = m

    m *= (1. if unitful_masses else 1./m_unit) # Ensure m is provided in desired units
    if unitful_masses and not natural_units: 
        m      *= (1./c**2) # WIP
        m_unit *= (1./c**2)

    if verbosity > 2:
        print('m_unit:  ', m_unit)
        if verbosity > 3:
            print('m (raw):\n', m_raw)
            print('m (out):\n', m)
        else:
            print('m:\n', m)

    return m, m_unit

def init_densities(masks, p_t, normalized_subdens=True, densities_in=None):
    ## local DM densities for each species, assume equal mix for now.
    # TODO: More granular / nontrivial distribution of densities? Spacial dependence? Sampling?
    
    if normalized_subdens:
        p_loc = p_t/3.
    else:
        p_loc = p_t

    ## TODO: Accept more specific density profiles (WIP)
    if densities_in is not None:
        if all([len(dens_in) <= 1 for dens_in in densities_in]):
            p_r  = np.ma.masked_where(masks[0], np.full(densities_in[0], dtype=float), copy=True)
            p_n  = np.ma.masked_where(masks[1], np.full(densities_in[1], dtype=float), copy=True)
            p_c  = np.ma.masked_where(masks[2], np.full(densities_in[2], dtype=float), copy=True)
            np.ma.set_fill_value(p_r, 0.0)
            np.ma.set_fill_value(p_n, 0.0)
            np.ma.set_fill_value(p_c, 0.0)

            p = np.array([p_r.compressed()*p_loc, p_n.compressed()*p_loc, p_c.compressed()*p_loc], dtype=object, copy=True)
        else:
            p_r = np.array(densities_in[0])
            p_n = np.array(densities_in[1])
            p_c = np.array(densities_in[2])
            
            p = np.array([p_r*p_loc, p_n*p_loc, p_c*p_loc], dtype=object)
    else:
        N_r, N_n, N_c = [len(mask) for mask in masks]
        p_r  = np.ma.masked_where(masks[0], np.full(N_r, 1./max(1., N_r), dtype=float), copy=True)
        p_n  = np.ma.masked_where(masks[1], np.full(N_n, 1./max(1., N_n), dtype=float), copy=True)
        p_c  = np.ma.masked_where(masks[2], np.full(N_c, 1./max(1., N_c), dtype=float), copy=True)
        np.ma.set_fill_value(p_r, 0.0)
        np.ma.set_fill_value(p_n, 0.0)
        np.ma.set_fill_value(p_c, 0.0)

        p = np.array([p_r.compressed()*p_loc, p_n.compressed()*p_loc, p_c.compressed()*p_loc], dtype=object, copy=True)

    return p

## Define (initial) pi-axion dark matter mass-amplitudes for each species, optional units of [eV/c]
def init_amplitudes(m, p, m_unit=1, mass_units=True, natural_units=True, unitful_amps=unitful_masses, rescale_amps=False, h=1, c=1, verbosity=0):
    amps = np.array([np.array([np.sqrt(2 * p[s][i]) / m[s][i] if np.abs(m[s][i]) > 0. else 0. for i in range(len(m[s]))]) for s in range(len(m))], dtype=object, copy=True)
    amps_raw = np.copy(amps)

    #rescale_amps = mass_units if unitful_amps else not(mass_units)
    # normalize/rescale amplitudes by dividing by amp of pi_0?
    amps_to_units = np.sqrt((h/c)**3) # (TODO: WIP, double check these units)
    if rescale_amps:
        if unitful_amps:
            amps *= (1. / m_unit)          # convert to units of [m_unit] instead of [eV]
        else:
            amps *= m_unit                 # convert to units of [eV] instead of [m_unit]
            if not(natural_units):
                amps *= amps_to_units
    else:
        if unitful_amps and not(natural_units):
            amps *= amps_to_units

    if verbosity > 3:
        print('amps (raw):\n', amps_raw)
        print('amps (out):\n', amps)
    elif verbosity > 0:
        print('amps:\n', amps)
    return amps

# Sample global and local phases from normal distribution, between 0 and 2pi
def init_phases(masks, rng=None, sample_delta=True, sample_Theta=True, delta_in=None, Theta_in=None, cosine_form=True, verbosity=0, sample_dist='uniform'):
    #global d, Th
    N_r, N_n, N_c = [len(mask) for mask in masks]
    ## local phases for each species (to be sampled)
    if delta_in == None:
        d_raw = np.array([np.ma.masked_where(masks[0], np.zeros(N_r, dtype=float)).compressed(),
                          np.ma.masked_where(masks[1], np.zeros(N_n, dtype=float)).compressed(),
                          np.ma.masked_where(masks[2], np.zeros(N_c, dtype=float)).compressed()], dtype=object)
    else:
        d_raw = delta_in
    d = d_raw

    ## global phase for neutral complex species (to be sampled)
    if Theta_in == None:
        Th_raw = np.array([np.ma.masked_where(masks[0], np.zeros(N_r, dtype=float)).compressed(),
                           np.ma.masked_where(masks[1], np.zeros(N_n, dtype=float)).compressed(),
                           np.ma.masked_where(masks[2], np.zeros(N_c, dtype=float)).compressed()], dtype=object)
    else:
        Th_raw = Theta_in
    Th = Th_raw

    if verbosity > 3 and sample_dist == 'normal':
        print('delta (raw):\n', d_raw)
        print('Theta (raw):\n', Th_raw)

    d, Th = sample_phases(rng, d, Th, sample_delta, sample_Theta, distribution=sample_dist, cosine_form=cosine_form, verbosity=verbosity)

    return d, Th

mu  = np.pi      # mean
sig = np.pi / 3  # standard deviation
def sample_phases(rng, d_in, Th_in, sample_delta=True, sample_Theta=True, mean_delta=mu, mean_theta=mu, stdev_delta=sig, stdev_theta=sig, distribution='uniform', cosine_form=True, verbosity=0):

    mu_delta  = mu  if mean_delta  is None else mean_delta
    mu_theta  = mu  if mean_theta  is None else mean_theta
    sig_delta = sig if stdev_delta is None else stdev_delta
    sig_theta = sig if stdev_theta is None else stdev_theta

    # Shift local phase by pi to account for cosine/sine convention in EoM definitions
    shift_val = np.pi if cosine_form else 0.
    # Modulo 2pi to ensure value is within range
    if sample_delta:
        if distribution == 'uniform':
            d  = np.array([np.mod(rng.uniform(0, 2*np.pi, len(d_i)) + shift_val, 2*np.pi) for d_i in d_in], dtype=object)
        elif distribution == 'normal':
            #d  = np.array([np.ma.masked_where(mask[i], np.mod(rng.normal(mu_delta, sig_delta, len(mask)) + shift_val, 2*np.pi)).compressed() for i,mask in enumerate(masks)], dtype=object)
            d  = np.array([np.mod(rng.normal(mu_delta, sig_delta, len(d_i)) + shift_val, 2*np.pi) for d_i in d_in], dtype=object)
        else:
            d = d_in
    else:
        d = d_in
    if sample_Theta:
        if distribution == 'uniform':
            Th  = np.array([np.mod(rng.uniform(0, 2*np.pi, len(Th_i)) + shift_val, 2*np.pi) for Th_i in Th_in], dtype=object)
        elif distribution == 'normal':
            #Th = np.array([np.ma.masked_where(mask[i], np.mod(rng.normal(mu_theta, sig_theta, len(mask)) + shift_val, 2*np.pi)).compressed() for i,mask in enumerate(masks)], dtype=object)
            Th = np.array([np.mod(rng.normal(mu_theta, sig_theta, len(Th_i)) + shift_val, 2*np.pi) for Th_i in Th_in], dtype=object)
        else:
            Th = Th_in
    else:
        Th = Th_in

    if verbosity > 3:
        print('Sample delta?  ', sample_delta, '(%s distribution)' % distribution if sample_delta else '')
        if sample_delta and distribution == 'normal': print('  mu: %.2f  |  sigma: %.2f' % (mu_delta, sig_delta))
        print('delta (out):\n', d)
        print('Sample Theta?  ', sample_Theta,  '(%s distribution)' % distribution if sample_Theta else '')
        if sample_Theta and distribution == 'normal': print('  mu: %.2f  |  sigma: %.2f' % (mu_theta, sig_theta))
        print('Theta (out):\n', Th)
    elif verbosity > 0:
        print('delta:\n', d)
        print('Theta:\n', Th)

    return d, Th

if __name__ == '__main__':
    default_outdir = default_output_directory
    parser = argparse.ArgumentParser(description='Parse command line arguments.')
    parser.add_argument('--t',          type=int, default=300,  help='Ending time value')
    parser.add_argument('--tN',         type=int, default=300,  help='Number of timesteps')
    parser.add_argument('--k' ,         type=int, default=-1,   help='Max k-mode to include in calculations (-1 to assume kN)')
    parser.add_argument('--kN',         type=int, default=200,  help='Number of k-modes')
    parser.add_argument('--k_res',      type=float, default=1,  help='Stepsize of k-modes to sample')
    parser.add_argument('--seed',       type=int, default=None, help='RNG seed')

    parser.add_argument('--eps',        type=np.float64, default=1,     help='Millicharge value')
    parser.add_argument('--F',          type=np.float64, default=1e11,  help='F_pi coupling constant in [GeV]')
    parser.add_argument('--L3',         type=np.float64, default=1e11,  help='Lambda_3 coupling constant in [GeV]')
    parser.add_argument('--L4',         type=np.float64, default=1e11,  help='Lambda_4 coupling constant in [GeV]')
    parser.add_argument('--m_scale',    type=np.float64, default=1e-20, help='Mass scale of dQCD quarks in [eV]')
    parser.add_argument('--rho',        type=np.float64, default=0.4,   help='Local DM energy density, in [GeV/cm^3]')

    parser.add_argument('--sample_delta', action=argparse.BooleanOptionalAction,  default=True, help='Toggle whether local phases are randomly sampled')
    parser.add_argument('--sample_theta', action=argparse.BooleanOptionalAction,  default=True, help='Toggle whether global phases are randomly sampled')

    parser.add_argument('--A_0',        type=np.float64, default=1.0,   help='Photon field initial conditions')
    parser.add_argument('--Adot_0',     type=np.float64, default=1.0,   help='Photon field rate of change initial conditions')
    parser.add_argument('--A_pm',       type=int,        default=+1,    help='Polarization case (+1 or -1)')

    parser.add_argument('--use_natural_units', action=argparse.BooleanOptionalAction,  default=True, help='Toggle whether c=h=G=1')
    parser.add_argument('--use_mass_units',    action=argparse.BooleanOptionalAction,  default=True, help='Toggle whether calculations are done in units of quark mass (True) or eV (False)')
    parser.add_argument('--make_plots',        action=argparse.BooleanOptionalAction,  default=True, help='Toggle whether plots are made at the end of each run')
    parser.add_argument('--show_plots',        action=argparse.BooleanOptionalAction,  default=False, help='Toggle whether plots are displayed at the end of each run')
    parser.add_argument('--skip_existing',     action=argparse.BooleanOptionalAction,  default=False, help='Toggle whether phashes that exist in output dir should be skipped (False to force rerun/overwrite)')

    parser.add_argument('--verbosity',         type=int,  default=0,               help='From 0-9, set the level of detail in console printouts. (-1 to suppress all messages)')
    parser.add_argument('--save_output_files', action=argparse.BooleanOptionalAction, default=True, help='Toggle whether or not to save the results from this run')
    parser.add_argument('--config_name',       type=str,  default='default',       help='Descriptive name to give to this parameter case')
    parser.add_argument('--num_cores',         type=int,  default=1,               help='Number of parallel processing threads to use')
    parser.add_argument('--data_path',         type=str,  default=default_outdir,  help='Path to output directory where files should be saved')
    parser.add_argument('--int_method',        type=str,  default='RK45',          help='Numerical integration method, to be used by scipy solve_ivp')
    parser.add_argument('--num_samples',       type=int,  default=1,               help='Number of times to rerun a parameter set, except randomly sampled variables')
    parser.add_argument('--mem_per_core',      type=int,  default=0,               help='Amount of memory available to each parallelized node, in GB')

    parser.add_argument('--P', action=argparse.BooleanOptionalAction, default=True, help='Turn on/off the P(t) coefficient in the numerics')
    parser.add_argument('--B', action=argparse.BooleanOptionalAction, default=True, help='Turn on/off the B(t) coefficient in the numerics')
    parser.add_argument('--C', action=argparse.BooleanOptionalAction, default=True, help='Turn on/off the C_+/-(t) coefficient in the numerics')
    parser.add_argument('--D', action=argparse.BooleanOptionalAction, default=True, help='Turn on/off the D(t) coefficient in the numerics')

    parser.add_argument('--fit_F', action=argparse.BooleanOptionalAction, help='Toggle whether F_pi is determined from given mass & millicharge (True) or provided manually (False)')
    parser.add_argument('--scan_F',         type=int,  nargs=2,       help='Provide min and max values of F_pi values to search, in [log GeV] scale')
    parser.add_argument('--scan_F_N',       type=int,  default=3,     help='Provide number of values to search within specified F_pi range')
    parser.add_argument('--scan_mass',      type=int,  nargs=2,       help='Provide min and max values of mass scales to search, in [log eV] scale')
    parser.add_argument('--scan_mass_N',    type=int,  default=10,    help='Provide number of values to search within specified mass range')
    parser.add_argument('--scan_Lambda',    type=int,  nargs=2,       help='Provide min and max values of coupling constant scales to search, in [log GeV] units (L_3=L_4)')
    parser.add_argument('--scan_Lambda3',   type=int,  nargs=2,       help='Provide min and max values of L_3 values to search, in [log GeV]')
    parser.add_argument('--scan_Lambda4',   type=int,  nargs=2,       help='Provide min and max values of L_4 values to search, in [log GeV]')
    parser.add_argument('--scan_Lambda_N',  type=int,  default=10,    help='Provide number of values to search within specified Lambda range')
    parser.add_argument('--scan_Lambda3_N', type=int,  default=None,  help='Provide number of values to search within specified L_3 range')
    parser.add_argument('--scan_Lambda4_N', type=int,  default=None,  help='Provide number of values to search within specified L_4 range')
    parser.add_argument('--scan_epsilon',   type=int,  nargs=2,       help='Provide min and max values of millicharge scales to search, in [log] units')
    parser.add_argument('--scan_epsilon_N', type=int,  default=10,    help='Provide number of values to search within specified millicharge range')

    parser.add_argument('--dqm_c', type=list, nargs=6, default=[1.,1.,1.,1.,1.,1.], help='Provide scaling constants c1-c6 used to define dQCD quark species masses. None = random sample')

    args = parser.parse_args()
    main(args)
