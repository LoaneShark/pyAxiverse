import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.pyplot import subplot2grid
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import display, clear_output, HTML, Image
from pyparsing import line
from scipy.signal import spectrogram
from scipy.optimize import curve_fit
import dill
import json
import hashlib
import sys
import os
import ast
import glob

signstr = {1: '+', -1: '-', 0: '±'}
signtex = {1: '+', -1: '-', 0: '\pm'}
GeV = 1e9
default_output_directory='~/scratch'
scratch_output_directory='~/scratch'
version='v3.2'
# Fundamental constants
c = c_raw = np.float64(2.998e10)    # Speed of light       [cm/s]
h = h_raw = np.float64(4.136e-15)   # Planck's constant    [eV/Hz]
G = G_raw = np.float64(1.0693e-19)  # Newtonian constant   [cm^5 /(eV s^4)]
e = 0.3                             # Dimensionless electron charge
# Pointers to global variables
'''
params = None
t = t_span = t_num = t_step = t_sens = None
k_values = k_span = k_num = k_step = None
eps = None
F = None
L3 = L4 = None
l1 = l2 = l3 = l4 = None
A_0 = Adot_0 = A_pm = A_sens = None
m = m_u = m_0 = m_q = k_0 = t_0 = None
p = p_t = p_0 = amps = None
d = Th = None
qm = qc = dqm = eps_c = xi = None
seed = res_con = None
use_natural_units = use_mass_units = None
unitful_m = unitful_k = unitful_amps = dimensionful_p = None
rescale_m = rescale_k = rescale_amps = rescale_consts = None
unitful_c = unitful_h = unitful_G = None
L3_sc = L4_sc = F_sc = c_u = h_u = G_u = None
'''
# Formatting Colors
colordict = {
    'purple': '#b042f5'
}

## Set parameters of model for use in numerical integration
def init_params(params_in: dict, sample_delta=True, sample_theta=True, t_max=10, t_min=0, t_N=500,
                k_max=200, k_min=1, k_N=200):
    # Define global variables
    global params
    global t, t_span, t_num, t_step, t_sens
    global k_values, k_span, k_num, k_step
    global e
    global eps
    global F
    global c, h, G
    global L3, L4
    global l1, l2, l3, l4
    global A_0, Adot_0, A_pm, A_sens
    global m, m_u, m_0, m_q, k_0, t_0
    global p, p_t, p_0, amps
    global d, Th
    global qm, qc, dqm, eps_c, xi
    global seed, res_con
    global use_natural_units, use_mass_units
    global unitful_m, unitful_k, unitful_amps, dimensionful_p
    global rescale_m, rescale_k, rescale_amps, rescale_consts
    global unitful_c, unitful_h, unitful_G
    
    # t domain
    t_span = [t_min, t_max]  # Time range
    t_num  = t_N             # Number of timesteps
    t, t_step = np.linspace(t_span[0], t_span[1], t_num, retstep=True)
    t_sens = params_in['t_sens']
    # k domain
    k_span = [k_min, k_max] # Momentum range
    k_num  = k_N            # Number of k-modes
    k_values, k_step = np.linspace(k_span[0], k_span[1], k_num, retstep=True)
    k_0 = params_in['k_0']
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
    sig_d  = params_in['sig_d']  if 'sig_d' in params_in else np.pi / 3     # local phase standard deviation
    mu_Th  = params_in['mu_Th']  if 'mu_Th' in params_in else np.pi        # global phase mean
    sig_Th = params_in['sig_Th'] if 'sig_Th' in params_in else np.pi / 3  # global phase standard deviation
    force_resample = False
    if sample_delta and all([pkey not in params_in for pkey in ['d_r', 'd_n', 'd_c']]) and force_resample:
            d = [np.mod(np.random.normal(mu_d, sig_d, len(d_i)), 2*np.pi) for d_i in d]
    if sample_theta and all([Tkey not in params_in for Tkey in ['Th_r', 'Th_n', 'Th_c']]) and force_resample:
            Th = [np.mod(np.random.normal(mu_Th, sig_Th, len(Th_i)), 2*np.pi) for Th_i in Th]
            
    # rescaling and unit configuration
    int_method = params_in['int_method']
    use_natural_units = params_in['use_natural_units']
    use_mass_units    = params_in['use_mass_units']
    unitful_m      = params_in['unitful_m']
    rescale_m      = params_in['rescale_m']
    unitful_k      = params_in['unitful_k']
    rescale_k      = params_in['rescale_k']
    unitful_amps   = params_in['unitful_amps']
    rescale_amps   = params_in['rescale_amps']
    rescale_consts = params_in['rescale_consts']
    dimensionful_p = params_in['dimensionful_p']
    unitful_c = False if use_natural_units else c != 1
    unitful_h = False if use_natural_units else h != 1
    unitful_G = False if use_natural_units else G != 1

    # performance metrics
    num_cores    = params_in['num_cores']    if 'num_cores'    in params_in else 1
    mem_per_core = params_in['mem_per_core'] if 'mem_per_core' in params_in else None

    t_0 = 1./m_u if unitful_m else 1.

    # Turn off irrelevant constants
    if N_r <= 0 and N_n <= 0:
        L4 = -1                   # Turn off Lambda_4 if there are no surviving neutral (real and complex) species
    if N_c <= 0:
        L3 = -1                   # Turn off Lambda_3 if there are no surviving charged species

    #rescale_params(rescale_m, rescale_k, rescale_amps, rescale_consts, unitful_c=unitful_c, unitful_h=unitful_h, unitful_G=unitful_G)
            
    # Store for local use, and then return
    params = {'e': e, 'F': F, 'p_t': p_t, 'eps': eps, 'L3': L3, 'L4': L4, 'l1': l1, 'l2': l2, 'l3': l3, 'l4': l4,
              'A_0': A_0, 'Adot_0': Adot_0, 'A_pm': A_pm, 'amps': amps, 'd': d, 'Th': Th, 'h': h, 'c': c, 'G': G,
              'qm': qm, 'qc': qc, 'dqm': dqm, 'eps_c': eps_c, 'xi': xi, 'N_r': N_r, 'N_n': N_n, 'N_c': N_c, 'p_0': p_0,
              'm': m, 'm_r': m_r, 'm_n': m_n, 'm_c': m_c, 'p': p, 'p_r': p_r, 'p_n': p_n, 'p_c': p_c, 'm_0': m_0, 'm_q': m_q,
              'mu_d': mu_d, 'sig_d': sig_d, 'mu_Th': mu_Th, 'sig_Th': sig_Th, 'k_span': k_span, 'k_num': k_num, 'k_0': k_0,
              't_span': t_span, 't_num': t_num, 'A_sens': A_sens, 't_sens': t_sens, 'res_con': res_con, 'm_u': m_u, 't_u': t_0,
              'unitful_m': unitful_m, 'rescale_m': rescale_m, 'unitful_amps': unitful_amps, 'rescale_amps': rescale_amps, 
              'unitful_k': unitful_k, 'rescale_k': rescale_k, 'rescale_consts': rescale_consts, 'seed': seed, 'int_method': int_method,
              'use_natural_units': use_natural_units, 'use_mass_units': use_mass_units, 'dimensionful_p': dimensionful_p,
              'num_cores': num_cores, 'mem_per_core': mem_per_core}
    
    return params

# TODO: Make sure this actually works
def get_params():
    return params

# TODO: This is currently deprecated / unused -- is it worth fixing up?
def rescale_params(rescale_m, rescale_k, rescale_amps, rescale_consts, unitful_c=False, unitful_h=False, unitful_G=False, verbosity=0):
    global params
    global m, amps, k_values
    global m_0, k_0
    global L3, L4, F, c, h, G
    global L3_sc, L4_sc, F_sc, c_u, h_u, G_u
    is_natural_units = all([not(unitful_c), not(unitful_h), not(unitful_G)])
    
    # Values to use in order to ensure natural / dimensionful units
    c_u = c if unitful_c else 1.
    h_u = h if unitful_h else 1.
    G_u = G if unitful_G else 1.

    # Rescale all eV unit constants to unit mass
    L3_sc = abs(L3) if not rescale_consts else L3 / m_u
    L4_sc = abs(L4) if not rescale_consts else L4 / m_u
    F_sc  = abs(F)  if not rescale_consts else  F / m_u

prune_char = ['[',']','',' ','\n',None,[]]
str_split  = lambda str_in: str_in.replace('\n','').replace('\\n','').replace(',','').replace('\'','').replace('"','').replace('[','').replace(']','').split(' ')[1:-1]
str_to_arr = lambda str_in: [arr_val for arr_val in str_split(str_in) if len(arr_val) > 0]

# Helper functions to handle string to array conversion
def str_to_float_array(s):
    try:
        return [np.float64(x) for x in s.split()]
    except ValueError:
        return s

def str_to_int_array(s):
    try:
        return [int(x) for x in s.split()]
    except ValueError:
        return s

# Convert space-separated strings to lists of strings for specific keys
def convert_string_to_list(data, key):
    if key in data and isinstance(data[key], str):
        data[key] = data[key].split()

def parse_array_string(value):
    arrays = []
    parts = [subval.strip() for subval in value.split('array(')]
    for part in parts:
        array_dtype = part.replace(')','').replace(']','').split('dtype=')[1] if 'dtype=' in part else 'float64'
        if part:
            start_index = part.find('[')
            end_index = part.find(']')
            if start_index != -1 and end_index != -1:
                array_str = part[start_index+1:end_index]
                array_values = [x for x in array_str.split(',') if len(x) > 0]
                arrays.append(np.array(array_values, dtype=array_dtype))
    return arrays

def parse_value(value):
    value = value.replace('\n', '')
    
    # Handling lists of numpy arrays
    if value.startswith('[array'):
        return parse_array_string(value)
    # Handling list-like strings
    elif value.startswith('[') and value.endswith(']'):
        array_str = value[1:-1]
        array_values = [x for x in array_str.split(' ') if x.strip()]
        # Check if it's a list of strings
        if all("'" in val or '"' in val for val in array_values):
            return [x.strip("'\"") for x in array_values]
        elif all('.' in val for val in array_values):
            return str_to_float_array(array_str)
        else:
            return str_to_int_array(array_str)
    '''
    elif value.startswith('[') and value.endswith(']'):
        parsed_value = np.array([pval for pval in str_to_arr(value)])
        return parsed_value
    '''
    
    # Other types
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    '''
    try:
        parsed_value = ast.literal_eval(value)
        if isinstance(parsed_value, tuple):
            return np.array(parsed_value)
        return parsed_value
    except (ValueError, SyntaxError):
        return value
    '''

def parse_dictionary(data):
    for key, value in data.items():
        if isinstance(value, str):
            data[key] = parse_value(value)
    return data

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, int):
            return '%s' % obj
        return super(NumpyEncoder, self).default(obj)
    
    def decode(dct):
        if dct.get('__type__') == 'numpy.ndarray':
            data = dct['data']
            dtype = dct['dtype']
            return np.array(data, dtype=dtype)
        elif type(dct) is dict:
            return parse_dictionary(dct)
        else:
            return parse_value(dct)
    
# Generate a unique but reproduceable hash for the given parameter set
def get_parameter_space_hash(params_in, verbosity=0):
    
    phash = hashlib.sha1(json.dumps(params_in, sort_keys=True, ensure_ascii=True, cls=NumpyEncoder).encode()).hexdigest()

    if verbosity > 3:
        print('parameter space configuration hash:')
        print(phash)

    return phash

def get_rng(seed=None, verbosity=0):
    entropy_size = 4 # TODO: Probably should reduce to 8 or 4
    if seed is not None:
        rng_ss = np.random.SeedSequence(entropy=seed, pool_size=entropy_size)
    else:
        rng_ss = np.random.SeedSequence(pool_size=entropy_size)
    rng_seed = str(rng_ss.entropy)
    rng = np.random.default_rng(rng_ss)

    if verbosity > 3 or (verbosity >= 0 and seed is not None):
        print('rng_seed:', rng_seed)

    return rng, rng_seed

def save_coefficient_functions(functions, filename):
    """
    Save functions to a file using dill.
    
    Parameters:
    - functions (dict): A dictionary of functions to save. Key is the function name, value is the function object.
    - filename (str): The name of the file to save the functions to.
    """
    with open(filename, 'wb') as f:
        dill.dump(functions, f)

def load_coefficient_functions(filename):
    """
    Load functions from a file using dill.
    
    Parameters:
    - filename (str): The name of the file to load the functions from.
    
    Returns:
    - functions (dict): A dictionary of loaded functions. Key is the function name, value is the function object.
    """
    with open(filename, 'rb') as f:
        functions = dill.load(f)
    return functions

def save_results(output_dir_in, filename, params_in, results=None, plots=None, save_format='pdf', verbosity=0,
                 save_params=True, save_results=True, save_plots=True, plot_types=['amps', 'nums', 'resonance', 'alp'],
                 test_run=False, scratch_dir=scratch_output_directory, save_coefficients=False, P=None, B=None, C=None, D=None):
    
    # Don't save to longterm data for a test run
    output_dir = scratch_dir if test_run else output_dir_in

    # Resolve per-user paths to absolute system paths
    if output_dir[0] == '~':
        output_dir = os.path.join(os.path.expanduser('~'), '/'.join(output_dir.split('/')[1:]))
    file_list = []
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save parameters
    if save_params and params_in is not None:
        params_filename = os.path.join(output_dir, filename + '.json')
        with open(params_filename, 'w') as f:
            json.dump(params_in, f, sort_keys=True, indent=4, cls=NumpyEncoder, default=str)
        file_list.append(params_filename)
    
    # Save results
    if save_results and results is not None:
        results_filename = os.path.join(output_dir, filename + '.npy')
        np.save(results_filename, results)
        file_list.append(results_filename)

    # Save P(t), D(t), C(t), and B(t) definitions
    if save_coefficients:
        coeffs_filename = os.path.join(output_dir, filename + '_funcs.pkl')
        functions_in = {
            'P': P,
            'B': B,
            'C': C,
            'D': D
        }
        save_coefficient_functions(functions_in, coeffs_filename)
        file_list.append(coeffs_filename)
    
    # Save plots
    doc_formats = ['pdf', 'document', 'doc', 'all']
    tex_formats = ['tex', 'latex', 'svg', 'pgf', 'all'] # TODO
    img_formats = ['png', 'img', 'image', 'jpg', 'all']
    nbk_formats = ['notebook', 'nb', 'ipynb', 'jupyter', 'all']
    web_formats = ['html', 'web', 'all']
    if save_plots and plots is not None:
        if save_format == 'all' or any([save_format in fmt_set for fmt_set in [doc_formats, img_formats, nbk_formats, web_formats]]):
            plot_figs = [plots[p_type] for p_type in plot_types]
            if save_format in doc_formats:
                with PdfPages(os.path.join(output_dir, filename + '_plots.pdf')) as pdf:
                    for fig in plot_figs:
                        pdf.savefig(fig)
            if save_format in img_formats:
                for i, fig in enumerate(plot_figs):
                    fig.savefig(os.path.join(output_dir, filename + f'_plot_{i}.png'))
            if save_format in nbk_formats:
                # Display plots in the notebook (TODO)
                for fig in plot_figs:
                    display(fig)
            if save_format in web_formats:
                # Convert notebook to HTML and save (TODO)
                if True:
                    print('HTML not supported yet')
                else:
                    html_content = str(HTML('<h1>Simulation Plots</h1>'))
                    for fig in plot_figs:
                        #display(fig)
                        html_content += str(HTML(str(html_content) + '<img src="data:image/png;base64,{}">'.format(fig)))
                    with open(os.path.join(output_dir, filename + '_plots.html'), 'w') as f:
                        f.write(str(html_content))
        else:
            print(f'Incompatable file format: {save_format}')
    
    # Write paths to saved files to console
    if verbosity >= 0 and any([save_params, save_results, save_plots]):
            
        if len(file_list) <= 0 and verbosity >= 2:
            print('Done! (No files saved)')
        if verbosity >= 0:
            print(f'Results saved to {output_dir}')
        if verbosity > 0:
            
            # Append image filenames
            if save_format in doc_formats:
                file_list.append(os.path.join(output_dir, filename + '_plots.pdf'))
            if save_format in img_formats:
                for i, _ in enumerate(plot_figs):
                    file_list.append(os.path.join(output_dir, filename + f'_plot_{i}.png'))
            if save_format in web_formats:
                file_list.append(os.path.join(output_dir, filename + '_plots.html'))

            # Pretty print the list of files with their sizes
            fsizes = [os.path.getsize(file) for file in file_list]
            flen = len(filename) + 12
            print(f'{"Files saved:":{flen + 2}}  | (Total: {sizeof_fmt(sum(fsizes))})')
            for file, fsize in zip(file_list, fsizes):
                fname = file.split('/')[-1]
                print(f'  {fname:{flen}}  | {sizeof_fmt(fsize)}')

def load_multiple_results(output_dir, label, load_images=False, save_format='pdf'):
    """
    Parameters:
    - output_dir (str): The directory where the output files are saved.
    - label (str): The unique identifier for the simulation.
    - load_images (bool): Whether to load the saved plots or not.
    - save_format (str): The format in which the plots were saved. Options are 'pdf', 'png', 'notebook', 'html'.
    
    Returns:
    - all_params (list of dicts): List of input parameters for each simulation.
    - all_results (list of np.arrays): List of simulation results.
    - all_plots (list of lists or None): A list of lists containing matplotlib figures or images for each simulation.
    """
    
    file_dir  = os.path.join(os.path.expanduser(output_dir), label) if '~' in output_dir else os.path.join(output_dir, label)
    all_files = os.listdir(file_dir)
    
    relevant_files = [f for f in all_files if f.startswith(label) and f.endswith('.json')] # Assume input params are being saved for now, at least
    
    all_params = []
    all_results = []
    all_plots = []
    all_coeffs = []
    
    for filename in relevant_files:
        # Extract base name without extension
        base_name = filename.rsplit('.', 1)[0]
        
        # Load parameters
        params_filename = os.path.join(file_dir, base_name + '.json')
        with open(params_filename, 'r') as f:
            params = json.loads(f.read(), object_hook=NumpyEncoder.decode)
        all_params.append(params)
        
        # Load results
        results_filename = os.path.join(file_dir, base_name + '.npy')
        results = np.array(np.load(results_filename), dtype=object)
        all_results.append(results)

        # Load coefficient functions
        coeffs_filename = os.path.join(file_dir, base_name, '_funcs.pkl')
        if os.path.exists(coeffs_filename):
            all_coeffs.append(load_coefficient_functions(coeffs_filename))
        else:
            all_coeffs.append(None)
        
        # Load plots
        plots = []
        if load_images and save_format == 'png':
            i = 0
            while os.path.exists(os.path.join(file_dir, base_name + f'_plot_{i}.png')):
                plots.append(plt.imread(os.path.join(file_dir, base_name + f'_plot_{i}.png')))
                i += 1
        all_plots.append(plots)
    
    return all_params, all_results, all_plots, all_coeffs

def load_single_result(output_dir, filename, load_plots=False, save_format='pdf'):
    """
    Parameters:
    - output_dir (str): The directory where the output files are saved.
    - save_format (str): The format in which the plots were saved. Options are 'pdf', 'png', 'notebook', 'html'.
    
    Returns:
    - params (dict): The input parameters for the simulation.
    - results (np.array): The simulation results as a numpy array.
    - plots (list or None): A list of matplotlib figures if save_format is 'png', otherwise None.
    """
    
    # Load parameters
    params_filename = os.path.join(output_dir, filename + '.json')
    with open(params_filename, 'r') as f:
        params = dict(json.loads(f.read(), object_hook=NumpyEncoder.decode))
    
    # Load results
    results_filename = os.path.join(output_dir, filename + '.npy')
    results = np.array(np.load(results_filename),dtype=np.float64)
    
    # Load coefficient functions
    coeffs_filename = os.path.join(output_dir, filename + '_funcs.pkl')
    if os.path.exists(coeffs_filename):
        coeffs_dict = dict(load_coefficient_functions(coeffs_filename))
    else:
        coeffs_dict = None

    # Load plots
    plots = []
    if load_plots and save_format is not None :
        if save_format == 'png':
            i = 0
            while os.path.exists(os.path.join(output_dir, filename, f'_plot_{i}.png')):
                plots.append(plt.imread(os.path.join(output_dir, filename, f'_plot_{i}.png')))
                i += 1
    
    return params, results, plots, coeffs_dict

def load_all(input_str, output_root='~/scratch', version=version, load_images=False, save_format='pdf'):
    """
    Load all results given a full path to the output directory or just the simulation label.
    
    Parameters:
    - input_str (str): Either the full path to the output directory or just the simulation label.
    - output_root (str): Root directory for the outputs. Default is '~/scratch'.
    - version (str): Version of the simulation. Default is specified at the top of this file.
    - load_images (bool): Whether to load the saved plots or not.
    - save_format (str): The format in which the plots were saved. Options are 'pdf', 'png', 'notebook', 'html'.
    
    Returns:
    - all_params, all_results, all_plots as described in the load_multiple_results function.
    """
    
    # Check if the provided input_str is a directory path or just a label
    if os.path.isdir(input_str):
        output_dir  = os.path.join(input_str.split('/')[:-1])
        result_name = input_str.split('/')[-1]
    else:
        output_dir  = os.path.join(output_root, version)
        result_name = input_str
    
    return load_multiple_results(output_dir, result_name, load_images, save_format)

def load_single(input_str, label=None, phash=None, output_root='~/scratch', version=version, save_format='pdf', load_plots=False, verbosity=0):
    """
    Load results of a single run given a full filepath to any of the files associated with a run, or just the label and unique parameter hash.
    
    Parameters:
    - input_str (str): Either the full filepath to any of the files associated with a run or just the unique parameter hash.
    - label (str): The unique identifier for the simulation. Required if only the hash is provided.
    - output_root (str): Root directory for the outputs. Default is '~/scratch'.
    - version (str): Version of the simulation. Default is 'v2.8'.
    - save_format (str): The format in which the plots were saved. Options are 'pdf', 'png', 'notebook', 'html'.
    
    Returns:
    - params, results, plots as described in the load_results function.
    """
    
    # Check if the provided input_str is a file path or just a hash
    if os.path.isfile(input_str):
        output_dir, filename_with_ext = os.path.split(input_str)
        filename, _ = os.path.splitext(filename_with_ext)
    else:
        output_dir = output_root
        input_name, _ = os.path.splitext(input_str.split('/')[-1])
        input_split = input_name.split('_')
        input_label = label
        input_hash  = phash
        if (input_label is None) or (input_hash is None):
            if len(list(input_split)) > 1:
                input_split_first = input_split[:-1]
                input_split_last  = input_split[-1]
                if len(input_split_last) == 40:
                    input_hash = input_split_last
                    input_label = '_'.join(input_split_first)
                else:
                    input_label = '_'.join(input_split)
                    raise ValueError("If only the label is provided, the simulation parameter hash must also be specified.")
            else:
                if len(input_name) == 40:
                    input_hash = input_name
                else:
                    input_label = input_name
        # Throw error if insufficient information is provided to locate the specific file
        if input_label is not None:
            if input_hash is None:
                raise ValueError("If only the hash is provided, the simulation label must also be specified.")
        else:
            if input_label is None:
                raise ValueError("If only the label is provided, the simulation parameter hash must also be specified.")
        if verbosity > 3:
            print('input_str', input_str)
            print('output_dir_in', output_dir)
            print('output_dir split', )
            print('label:  ', label)
            print('phash:  ', phash)
            print('input_label:  ', input_label)
            print('input_phash:  ', input_hash)
        pathroot, pathrest = os.path.split(output_dir)
        output_dir = os.path.join(pathroot, pathrest, version, input_label)
        if verbosity > 3:
            print('output_dir', output_dir)
        filename = f"{input_label}_{input_hash}"
    
    return load_single_result(output_dir, filename, load_plots, save_format)

# TODO: Merge logic in loading file function to here
def parse_filename(filename):
    """
    Parse the given filename to extract the simulation result name and parameter space hash.
    
    Parameters:
    - filename (str): The filename to parse.
    
    Returns:
    - tuple: (result_label, result_phash)
    """
    # Remove directory and file extension from the filename
    base_filename = os.path.basename(filename)
    filename_without_ext, _ = os.path.splitext(base_filename)
    
    # Split the filename to extract the label and hash
    parts = filename_without_ext.split('_')
    result_phash = parts[-1]
    result_label = '_'.join(parts[:-1])
    
    return result_label, result_phash

def if_output_exists(directory, phash):
    """
    Check if any file in the given directory contains the provided phash.
    
    Args:
    - directory (str): The directory path to search in.
    - phash (str): The parameter hash to search for.
    
    Returns:
    - bool: True if a file with the phash exists, False otherwise.
    """
    
    # List all files in the directory
    if '~' in directory:
        directory = os.path.expanduser(directory)
    all_files = glob.glob(os.path.join(directory, '*'))
    
    # Check if any of the files contain the phash
    for file in all_files:
        if phash in file:
            return True
    return False

# Main function to load results and plot them, for a single given case
def plot_single_case(input_str, output_dir=default_output_directory, plot_res=True, plot_nums=True, plot_coeffs=True, plot_spectrum=True, k_samples_in=[], set_params_globally=False, tex_fmt=False, version=version):

    # Load results
    params, results, _, coeffs = load_single(input_str, output_root=output_dir, version=version)

    t_span = params['t_span']
    t_num  = params['t_num']
    k_span = params['k_span']
    k_num  = params['k_num']

    if set_params_globally:
        params = init_params(params, sample_delta=False, sample_theta=False, 
                             t_max=t_span[1], t_min=t_span[0], t_N=t_num, 
                             k_max=k_span[1], k_min=k_span[0], k_N=k_num)

    k_sens_in  = params.get('k_sens_arr', None)
    k_sens_arr = np.array(k_sens_in.split() if type(k_sens_in) is str else k_sens_in, dtype=np.float64)
    k_mean_in  = params.get('k_mean_arr', None)
    k_mean_arr = np.array(k_mean_in.split() if type(k_mean_in) is str else k_mean_in, dtype=np.float64)
    k_peak_in = params.get('k_peak_arr', None)
    k_peak_arr = np.array(k_peak_in.split() if type(k_peak_in) is str else k_peak_in, dtype=np.float64)

    # Define which k values should be plotted
    k_peak = np.max(k_sens_arr) # peak (running avg) value per k-mode
    k_mean = np.max(k_mean_arr) # mean value per k-mode
    k_rmax = np.max(k_peak_arr) # (raw) peak value per k-mode
    logscale_k = False
    k_values = np.linspace(k_span[0], k_span[1], k_num)
    k_samples = params.get('k_samples', [])
    times = np.linspace(t_span[0], t_span[1], t_num)
    if len(k_samples_in) <= 0:
        if len(k_samples) <= 0:
            if logscale_k:
                k_samples = np.geomspace(1, k_span[1], num=15)
            else:
                k_samples = [k_i for k_i, k_val in enumerate(k_values) if k_val in [0,1,10,50,100,150,200,500,k_peak,k_mean]]
    else:
        k_samples = k_samples_in

    # Extracting parameters from the dictionary
    unitful_m = params.get('unitful_m', None)
    rescale_m = params.get('rescale_m', None)
    unitful_k = params.get('unitful_k', None)
    rescale_k = params.get('rescale_k', None)
    unitful_amps = params.get('unitful_amps', None)
    rescale_amps = params.get('rescale_amps', None)
    rescale_consts = params.get('rescale_consts', None)

    # Calling get_units
    units = get_units(unitful_m, rescale_m, unitful_k, rescale_k, unitful_amps, rescale_amps, rescale_consts)

    # Plot results of numerical integration, as imported from file
    if plot_res:
        plot_amplitudes(params_in=params, units_in=units, results_in=results, k_samples=k_samples, times=times, tex_fmt=tex_fmt)

    # Plot occupation number of the photon field, as imported from file
    if plot_nums:
        plot_occupation_nums(params_in=params, units_in=units, results_in=results, numf=None, k_samples=k_samples, times=times, tex_fmt=tex_fmt)

    # Plot time-dependent oscillatory coefficients, as imported from file
    if plot_coeffs:
        P = coeffs['P'] if coeffs is not None else P_off
        B = coeffs['B'] if coeffs is not None else B_off
        C = coeffs['C'] if coeffs is not None else C_off
        D = coeffs['D'] if coeffs is not None else D_off
        plot_coefficients(params_in=params, units_in=units, P=P, B=B, C=C, D=D, k_samples=k_samples, times=times, tex_fmt=tex_fmt)
    
    if plot_spectrum:
        k_to_Hz_local = lambda ki, k0=params['k_0'], h=h_raw, c=c_raw: k_to_Hz(ki, k0, h, c)
        Hz_to_k_local = lambda fi, k0=params['k_0'], h=h_raw, c=c_raw: Hz_to_k(fi, k0, h, c)
        plot_resonance_spectrum(params_in=params, units_in=units, fwd_fn=k_to_Hz_local, inv_fn=Hz_to_k_local, tex_fmt=tex_fmt)

# k_ratio: apply [k_func] to each k mode and then return the ratio of the final vs. initial ampltidues (sensitive to a windowed average specified by [sens])
k_ratio = lambda func, t_sens, A_sens: np.array([k_f/k_i for k_f, k_i in zip(k_sens(func, t_sens), k_sens(func, -t_sens))])

# k_class: softly classify the level of resonance according to the final/initial mode amplitude ratio, governed by [func, t_sens, and A_sens]
k_class = lambda func, t_sens, A_sens, res_con: np.array(['damp' if k_r <= 0.9 else 'none' if k_r <= (1. + np.abs(A_sens)) else 'semi' if k_r <= res_con else 'res' for k_r in k_ratio(func, t_sens, A_sens)])

get_times = lambda params_in, times_in: times_in if times_in is not None else t if t is not None else np.linspace(params_in['t_span'][0], params_in['t_span'][1], params_in['t_num'])

## Identify the k mode with the greatest peak amplitude, and the mode with the greatest average amplitude
def get_peak_k_modes(results_in, k_values_in=None, write_to_params=False):
    global k_func, k_sens, k_ratio, k_class, k_peak, k_mean, tot_res
    t_num = len(results_in[0][0])
    k_values = k_values_in if k_values_in is not None else k_values if k_values is not None else None

    # k_func : apply [func] on the time-series for each k mode, e.g. max or mean
    k_func = lambda func: np.array([k_fval for k_fi, k_fval in enumerate([func(np.abs(results_in[k_vi][0][:])) for k_vi, k_v in enumerate(k_values)])])
    
    # k_sens : apply [k_func] but limit the time-series by [sens], e.g. sens = 0.1 to look at the first 10%. Negative values look at the end of the array instead.
    win_min  = lambda sens: int(t_num*(1./2)*np.abs(((1. - sens)*np.sign(sens) + (1. - sens))))  # min value / left endpoint (of the sensitivity window over which to average)
    win_max  = lambda sens: int(t_num*(1./2)*np.abs(((1. + sens)*np.sign(sens) + (1. - sens))))  # max value / right endpoint
    k_sens = lambda func, sens: np.array([k_fval for k_fi, k_fval in enumerate([func(np.abs(results_in[k_vi][0][win_min(sens):win_max(sens)])) for k_vi, k_v in enumerate(k_values)])])

    # k mode(s) with the largest contributions to overall number density growth
    k_peak = k_values[np.ma.argmax(k_func(max))]
    k_mean = k_values[np.ma.argmax(k_func(np.ma.mean))]
    
    # store max, all-time mean, and late-time mean for each k-mode locally, as well as resonance classifications
    if write_to_params:
        global params
        params['k_peak_arr']  = k_func(max)
        params['k_mean_arr']  = k_func(np.mean)
        params['k_sens_arr']  = k_sens(np.mean, t_sens)
        params['k_class_arr'] = k_class(np.mean, t_sens, A_sens, res_con)

        #t_res_i = np.ma.argmax(np.array(n_tot) > res_con)
        #t_res   = t[t_res_i]
        #params['t_res'] = t_res
        tot_res = 'resonance' if sum(k_ratio(np.ma.mean, t_sens, A_sens)) > res_con else 'none'
        params['res_class'] = tot_res
    
    return k_peak, k_mean

# Plot the amplitudes (results of integration)
def plot_amplitudes(params_in, units_in, results_in, k_samples=[], times=None, plot_Adot=True, tex_fmt=False):
    plt = make_amplitudes_plot(params_in, units_in, results_in, k_samples, times, plot_Adot, tex_fmt)
    plt.show()
    
def make_amplitudes_plot(params_in, units_in, results_in, k_samples=[], times_in=None, plot_Adot=True, tex_fmt=False):
    k_values = np.linspace(params_in['k_span'][0], params_in['k_span'][1], params_in['k_num'])
    k_peak, k_mean = get_peak_k_modes(results_in, k_values)
    signdict = signtex if tex_fmt else signstr
    fontsize = 16 if tex_fmt else 14
    if len(k_samples) <= 0:
        #k_samples = np.geomspace(1,len(k_values),num=5)
        k_samples = [i for i, k_i in enumerate(k_values) if k_i in [0,1,10,50,100,150,200,500,k_peak,k_mean]]
    times = get_times(params_in, times_in)

    xdim = 5
    if plot_Adot:
        ydim = 3 
    else:
        ydim = 2

    #fig = Figure(figsize=(4*xdim, 4*ydim))
    #plt.subplot2grid((ydim,xdim), (0,0), fig=fig, colspan=3)
    plt.figure(figsize=(4*xdim, 4*ydim))
    plt.subplot2grid((ydim,xdim), (0,0), colspan=3)

    #plot_colors = cm.get_cmap('plasma').jet(np.linspace(0,1,len(k_samples)))

    for k_idx, k_sample in enumerate(k_samples):
        k_s = int(k_sample)
        #print(results_in[k_s, 0])
        plt.plot(times, results_in[k_s][0], label='k='+str(k_values[k_s]))
    plt.title(r'Evolution of the mode function $A_{%s}(k)$' % signdict[0])
    plt.xlabel(r'Time $[%s]$' % units_in['t'])
    plt.ylabel(r'$A_{%s}(k)$' % signdict[0])
    plt.yscale('log')
    plt.legend()
    plt.grid()

    #plt.subplot(2,1,2)
    plt.subplot2grid((ydim,xdim), (1,0), colspan=3)
    plt.plot(times, [sum([np.abs(results_in[i][0][t_i])**2 for i in range(len(k_values))]) for t_i in range(len(times))])
    plt.title(r'Evolution of the (total) power for $A_{%s}$' % signdict[0])
    plt.xlabel(r'Time $[%s]$' % units_in['t'])
    plt.ylabel(r'$|A_{%s}|^2$' % signdict[0])
    plt.yscale('log')
    plt.grid()


    if plot_Adot:
        #plt.subplot(2,1,2)
        plt.subplot2grid((ydim,xdim), (2,0), colspan=3)
        plt.plot(times, [sum([results_in[i][1][t_i] for i in range(len(k_values))]) for t_i in range(len(times))], color='g', label='total')
        plt.plot(times, [results_in[list(k_values).index(k_mean)][1][t_i] for t_i in range(len(times))], color='y', label='k = %d (mean)' % k_mean)
        plt.plot(times, [results_in[list(k_values).index(k_peak)][1][t_i] for t_i in range(len(times))], color='orange', label='k = %d (peak)' % k_peak)
        plt.title(r'Evolution of the (total) change in amplitude for $A[%s]$' % signdict[0])
        plt.xlabel(r'Time $[%s]$' % units_in['t'])
        plt.ylabel(r'$\dot{A}_{%s}$' % signdict[0])
        #plt.yscale('log')
        plt.legend()
        plt.grid()

    #print(len(d[0]))

    textstr1, textstr2 = print_param_space(params_in, units_in)

    plt.subplot2grid((ydim,xdim), (max(0,ydim-3),3), rowspan=(3 if plot_Adot else 2))
    plt.text(0.15, 0 + (ydim-2)*0.2, textstr1, fontsize=fontsize)
    plt.axis('off')

    plt.subplot2grid((ydim,xdim), (max(0,ydim-3),4), rowspan=(3 if plot_Adot else 2))
    plt.text(0, 0 + (ydim-2)*0.2, textstr2, fontsize=fontsize)
    plt.axis('off')

    plt.tight_layout()
    
    return plt

# Amplitude getter functions for occupation number function
A_i     = lambda i, solns: solns[i][0]
Adot_i  = lambda i, solns: solns[i][1]
k_i     = lambda i, k_vals=None: k_vals[i]
#ImAAdot = lambda k, t: (1/2)*np.abs(np.cos(2*k*t))
ImAAdot = lambda k, t: -(1/2)
k_vals_p = lambda params: np.linspace(params['k_span'][0], params['k_span'][1], params['k_num'])

# Particle number for k_mode value at index i (TODO: Verify units in below equation)
n_k = lambda k, A, Adot, Im: (k**2 * np.abs(A)**2 + np.abs(Adot) - 2*k*Im(k))
# Wrapper function for the above, using only params and results as inputs
n_p = lambda i, params, solns, k_vals=None, t_in=None, n=n_k: n(k=k_i(i, k_vals=k_vals if k_vals is not None else k_vals_p(params)), 
                                                                A=A_i(i, solns), Adot=Adot_i(i, solns), Im=lambda k: ImAAdot(k, t=get_times(params, t_in)))

'''(Deprecated)
#k_to_w = np.float64(4.555e25) # 2πc/hbar [(Hz/eV)*(cm/s)]
w_p = lambda i, params: w(i, k_vals_p(params), params['m_u'])
w = lambda i, k_v, k_u, c=c_raw, h=h_raw: np.abs(k_v[i]*k_u*(2*np.pi/h))
#n_p = lambda i, params, solns: n(i, lambda j: w_p(j, params), solns)
#n = lambda i, w, solns: (w(i)/2) * (((np.square(np.abs(solns[i][1])))/(np.square(w(i)))) + np.square(np.abs(solns[i][0]))) - (1/2)
'''


# Plot occupation number results
def plot_occupation_nums(params_in, units_in, results_in, numf=None, k_samples=[], times=None, scale_n=False, tex_fmt=False):
    plt, _, _, _ = make_occupation_num_plots(params_in, units_in, results_in, numf, k_samples, times, scale_n, tex_fmt)
    plt.show()

sum_n_k = lambda n_in, k_v: np.sum([n_in(k) for k in k_v], axis=0)
sum_n_p = lambda n_in, p_in, sol_in, k_v, times: np.sum([n_p(i, p_in, sol_in, k_v, times, n=n_in) for i in range(len(k_v))], axis=0)

def make_occupation_num_plots(params_in, units_in, results_in, numf_in=None, k_samples_in=[], times_in=None, scale_n=False, write_to_params=False, tex_fmt=False):
    k_values = np.linspace(params_in['k_span'][0], params_in['k_span'][1], params_in['k_num'])
    k_peak, k_mean = get_peak_k_modes(results_in, k_values)
    fontsize = 16 if tex_fmt else 14
    if len(k_samples_in) <= 0:
        #k_samples = np.geomspace(1,len(k_values),num=5)
        k_samples = [k_i for k_i, k_val in enumerate(k_values) if k_val in [0,1,10,20,50,75,100,125,150,175,200,500,k_peak,k_mean]]
    else:
        k_samples = k_samples_in
    times = get_times(params_in, times_in)
    numf = numf_in if numf_in is not None else n_k
    
    plt.figure(figsize=(20, 9))

    plt.subplot2grid((2,5), (0,0), colspan=3)
    for k_sample in k_samples:
        k_s = int(k_sample)
        k_nums = n_p(k_s, params_in, results_in, k_values, times, n=numf)
        plt.plot(times, k_nums, label='k='+str(k_values[k_s]))
    plt.title(r'Occupation number per k mode', fontsize=16)
    plt.xlabel(r'Time $[%s]$' % units_in['t'])
    #plt.xlim(0,0.2)
    plt.ylabel(r'$n[k]/n_0$' if scale_n else r'$n[k]$')
    plt.yscale('log'); #plt.ylim(bottom = 0.1)
    plt.legend()
    plt.grid()

    n_tot = sum_n_p(numf, params_in, results_in, k_values, times)

    res_con = params_in['res_con']
    if scale_n:
        n_tot /= abs(n_tot[0])
        #n_tot += max(0, np.sign(n_tot[0]))  # TODO: Address this placeholder fix for negative n

    t_res_i = np.ma.argmax(np.array(n_tot) > res_con)
    t_res   = times[t_res_i]
    n_res   = n_tot[t_res_i]
    tot_res = 'resonance' if sum(k_ratio(np.ma.mean, params_in['t_sens'], params_in['A_sens'])) > res_con else 'none'
    if write_to_params:
        params_in['t_res'] = t_res
        params_in['res_class'] = tot_res
    #n_res = res_con*sum(k_sens(np.mean, -t_sens))

    #with plt.xkcd():
    plt.subplot2grid((2,5), (1,0), colspan=3)
    #fig,ax = plt.subplots()
    #plt.plot(np.ma.masked_where(t >= t_res, times), np.ma.masked_where(np.array(n_tot) > res_con*sum(k_sens(np.mean, -t_sens)), n_tot), label='none', color='grey')
    plt.plot(np.ma.masked_greater_equal(times, t_res), n_tot, label='none', color='grey')
    plt.plot(np.ma.masked_less(times, t_res), n_tot, label='resonance', color='blue')
    plt.title(r'Occupation Number (total)', fontsize=16)
    plt.xlabel(r'Time $[%s]$' % units_in['t'])
    #plt.xlim(0,0.1)
    plt.ylabel(r'$n$')
    plt.yscale('log')
    plt.legend()
    plt.grid()

    textstr1, textstr2 = print_param_space(params_in, units_in)

    plt.subplot2grid((2,5), (0,3), rowspan=2)
    plt.text(0.15, 0.05, textstr1, fontsize=fontsize)
    plt.axis('off')

    plt.subplot2grid((2,5), (0,4), rowspan=2)
    plt.text(0, 0.05, textstr2, fontsize=fontsize)
    plt.axis('off')

    plt.tight_layout()
    
    return plt, params_in, t_res, n_res

P_off = lambda t: np.float64(0)
B_off = lambda t: np.float64(0)
D_off = lambda t: np.float64(0)
C_off = lambda t, pm: np.float64(0)

Alpha = lambda t, k, k0, P, C, D, A_pm: ((C(t, A_pm)*(k*k0) + D(t)) / (1. + P(t))) + (k*k0)**2
Beta  = lambda t, B, P: B(t) / (1. + P(t))

# Plot time-dependent coefficient values of the model
def plot_coefficients(params_in, units_in, P=None, B=None, C=None, D=None, polarization=None, k_unit=None, k_samples=[], times=None, plot_all=True, tex_fmt=False):
    plt = make_coefficients_plot(params_in, units_in, P, B, C, D, polarization, k_unit, k_samples, times, plot_all, tex_fmt)
    plt.show()
    
def make_coefficients_plot(params_in, units_in, P_in=None, B_in=None, C_in=None, D_in=None, Cpm_in=None, k_unit=None, k_samples_in=[], times_in=None, plot_all=True, tex_fmt=False):
    global k_0
    k_values = np.linspace(params_in['k_span'][0], params_in['k_span'][1], params_in['k_num'])
    #k_peak, k_mean = get_peak_k_modes(results_in, k_values)
    fontsize = 16 if tex_fmt else 14
    P = P_in if P_in is not None else P_off
    B = B_in if B_in is not None else B_off
    D = D_in if D_in is not None else D_off
    C = C_in if C_in is not None else C_off
    Cpm = Cpm_in if Cpm_in is not None else params_in['A_pm']
    k0 = k_unit if k_unit is not None else params_in['k_0'] if 'k_0' in params_in else k_0 if k_0 is not None else 1.
    
    if len(k_samples_in) <= 0:
        k_samples = [i for i, k_i in enumerate(k_values) if k_i in [0,1,10,20,50,75,100,125,150,175,200,500,k_peak,k_mean]]
    else:
        k_samples = k_samples_in
    times = get_times(params_in, times_in)

    #fig = Figure(figsize = (20,9))
    #plt.subplot2grid((2,5), (0,0), fig=fig, colspan=3, rowspan=1)
    plt.figure(figsize = (20,9))
    plt.subplot2grid((2,5), (0,0), colspan=3, rowspan=1)

    for (c_t, c_label) in get_coefficient_values(params_in, P, B, C, D, times):
        plt.plot(times, c_t, label=c_label)

    plt.xlabel(r'Time $[%s]$' % units_in['t'])
    plt.yscale('log')
    plt.title(r'$\left(1 + P(t)\right)\left(\ddot{A}_{k,\pm} + k^2 A_{k,\pm}\right) + B(t)\dot{A}_{k,\pm} + \left(C_{\pm}(t) k + D(t)\right) A_{k,\pm} = 0 $', fontsize=16)
    plt.grid()
    plt.legend()
    
    # Compare Alpha(t) and Beta(t) for a sampling of k values
    #   for EoM of the form: A_k'' + [Beta]A_k' + [Alpha(k)]A_k = 0
    if plot_all:
        plt.subplot2grid((2,5), (1,0), colspan=3, rowspan=1)

        #k_samples = np.geomspace(1,len(k_values),num=5)
        k_samples = [i for i, k_i in enumerate(k_values) if k_i in [0,1,10,25,50,75,100,150,200,500]]

        if all([type(Alpha(times, k_values[k_s], k0, P, C, D, Cpm)) is np.float64 for k_s in k_samples]):
            label_a = r'[[$\alpha(t)$]]'
            alpha_s = np.ma.array(np.full(len(times), 0.0, dtype=np.float64), mask=True)
            plt.plot(times, alpha_s, label=label_a)
        else:
            for k_sample in k_samples:
                k_s = int(k_sample)
                alpha_s = Alpha(times, k_values[k_s], k0, P, C, D, Cpm)
                if type(alpha_s) is np.float64:
                    label_a = r'[[$\alpha_k(t)$  k=%d]]' % k_values[k_s]
                    alpha_s = np.ma.array(alpha_s + np.full(len(times), 0.0, dtype=np.float64), mask=True)
                    plt.plot(times, alpha_s, label=label_a)
                else:
                    label_a = r'$\alpha_k(t)$  k=%d' % k_values[k_s]
                    plt.plot(times, alpha_s, label=label_a)

        beta_s = Beta(times, P, B)
        if type(beta_s) is np.float64:
            label_b = r'[[$\beta(t)$]]'
            beta_s  = np.ma.array(beta_s + np.full(len(times), 0.0, dtype=np.float64), mask=True)
        else:
            label_b = r'$\beta(t)$'
        plt.plot(times, beta_s, label=label_b)

        plt.grid()
        plt.xlabel(r'Time $[%s]$' % units_in['t'])
        plt.yscale('log')
        plt.title(r'$\ddot{A}_{k,\pm} + \beta(t)\dot{A}_{k,\pm} + \alpha_{k}(t)A_{k,\pm} = 0$', fontsize=16)
        plt.legend()

    # Write parameter space configuration next to plot
    textstr1, textstr2 = print_param_space(params_in, units_in)

    plt.subplot2grid((2,5), (0,3), rowspan=2)
    plt.text(0.15, 0.05, textstr1, fontsize=fontsize)
    plt.axis('off')

    plt.subplot2grid((2,5), (0,4), rowspan=2)
    plt.text(0, 0.05, textstr2, fontsize=fontsize)
    plt.axis('off')

    plt.tight_layout()
    
    return plt

def get_coefficient_values(params_in, P, B, C, D, times_in=[]):
    global t
    func_vals = []
    times = times_in if len(times_in) > 0 else t
    for c_func, l_root, sign in zip([P, B, C, C, D], 
                                    ['P(t)', 'B(t)', 'C_{%s}(t)' % signstr[+1], 'C_{%s}(t)' % signstr[-1], 'D(t)'], 
                                    [0, 0, +1, -1, 0]):
        if l_root[0] == 'C':
            c_t = c_func(times, sign)
        else:
            c_t = c_func(times)

        if type(c_t) is np.float64:
            label   = r'[[$%s$]]' % l_root
            c_t     = np.ma.array(c_t + np.full(len(times), 0.0, dtype=float), mask=True)
            c_range = (np.nan, np.nan)
        else:
            label   = r'$%s$' % l_root
            c_range = (min(c_t), max(c_t))

        func_vals.append((c_t, label))
    return func_vals

def get_coefficient_ranges(params_in, P, B, C, D, k_samples, times_in=None):
    global t
    c_ranges = []
    times = times_in if len(times_in) > 0 else t
    
    for c_func, l_root, sign in zip([P, B, C, C, D], 
                                    ['P(t)', 'B(t)', 'C_{%s}(t)' % signstr[+1], 'C_{%s}(t)' % signstr[-1], 'D(t)'], 
                                    [0, 0, +1, -1, 0]):
        if l_root[0] == 'C':
            c_t = c_func(times, sign)
        else:
            c_t = c_func(times)

        if type(c_t) is np.float64:
            c_t     = np.ma.array(c_t + np.full(len(t), 0.0, dtype=float), mask=True)
            c_range = (np.nan, np.nan)
        else:
            label   = r'$%s$' % l_root
            c_range = (min(c_t), max(c_t))

        c_ranges.append(l_root[0] + '(t) range: [%.1e, %.1e]' % c_range + (' for + case ' if sign > 0 else ' for - case ' if sign < 0 else ''))
    
    return c_ranges

def print_coefficient_ranges(params_in, P_in=None, B_in=None, C_in=None, D_in=None, Cpm_in=None, k_unit=None, k_samples_in=[], times_in=None, print_all=False):
    
    if len(k_samples_in) <= 0:
        k_samples = [i for i, k_i in enumerate(k_values) if k_i in [0,1,10,20,50,75,100,125,150,175,200,500,k_peak,k_mean]]
    else:
        k_samples = k_samples_in
    P    = P_in   if P_in   is not None else P_off
    B    = B_in   if B_in   is not None else B_off
    D    = D_in   if D_in   is not None else D_off
    C    = C_in   if C_in   is not None else C_off
    Cpm  = Cpm_in if Cpm_in is not None else params_in['A_pm']
    k0   = k_unit if k_unit is not None else k_0
    times = get_times(params_in, times_in)
    
    for c_range_str in get_coefficient_ranges(params_in, P, B, C, D, k_samples, times):
        print(c_range_str)
    if print_all:
        print('------------------------------------------------------')
        if all([type(Alpha(times, k_values[k_s], k0, P, C, D, Cpm)) is np.float64 for k_s in k_samples]):
            print('Alpha(t) range: [%.1e, %.1e] for all k' % (np.nan, np.nan))
        else:
            for k_s in k_samples:
                if type(Alpha(times, k_values[k_s], k0, P, C, D, Cpm)) is not np.float64:
                    print('Alpha(t,k) range: [%.1e, %.1e] when k = %d' % (min(Alpha(times, k_values[k_s], k0, P, C, D, Cpm)), max(Alpha(times, k_values[k_s], k0, P, C, D, Cpm)), k_values[k_s]))
        if type(Beta(times, P, B)) is np.float64:
            print('Beta(t)    range: [%.1e, %.1e]' % (np.nan, np.nan))
        else:
            print('Beta(t)    range: [%.1e, %.1e]' % (min(Beta(times, P, B)), max(Beta(times, P, B))))

# E^2 = p^2c^2 + m^2c^4
# Assuming k, m are given in units of eV/c and eV/c^2 respectively
#k_to_Hz = lambda ki, mi=0, m_0=m0, e=e: 1/h * np.sqrt((ki*k0*e)**2 + ((mi*m_0 * e))**2)
k_to_Hz = lambda ki, k0, h, c: ki * ((k0*c) / (2*np.pi*h))
#Hz_to_k = lambda fi, mi=0, m_0=m0, e=e: 1/(e*k0) * np.sqrt((h * fi)**2 - ((mi*m_0 * e))**2)
Hz_to_k = lambda fi, k0, h, c: fi * ((h*2*np.pi) / (k0*c))

def plot_resonance_spectrum(params_in, units_in, fwd_fn, inv_fn, tex_fmt=False):
    plot = make_resonance_spectrum(params_in, units_in, fwd_fn, inv_fn, tex_fmt)
    plt.show()

def make_resonance_spectrum(params_in, units_in, fwd_fn, inv_fn, tex_fmt=False):
    k_values = np.linspace(params_in['k_span'][0], params_in['k_span'][1], params_in['k_num'])
    class_colors = {'none': 'lightgrey', 'damp': 'darkgrey', 'semi': 'blue', 'res': 'red'}
    res_con_in = params_in['res_con']

    t_sens = params_in['t_sens']
    A_sens = params_in['A_sens']
    
    plt.figure(figsize = (20,6))
    plt.suptitle(r'Resonance Classification')

    ax = plt.subplot2grid((2,4), (0,0), colspan=2, rowspan=2)
    plt.scatter(k_values, k_ratio(np.mean, t_sens, A_sens), c=[class_colors[k_c] if k_c in class_colors else 'orange' for k_c in k_class(np.mean, t_sens, A_sens, res_con_in)])
    plt.xlabel(r'$k$')
    axT = ax.secondary_xaxis('top', functions=(fwd_fn, inv_fn))
    #axT.set_xlabel(r'$f_{\gamma}$ [Hz]')
    axT.set_xlabel(r'$\nu$ [Hz]')
    plt.ylabel(r'Growth in $n_k$')
    plt.yscale('log')
    plt.grid()

    plt.subplot2grid((2,4), (0,2), colspan=2, rowspan=2)
    class_counts = [(np.array(k_class(np.mean, t_sens, A_sens, res_con_in)) == class_label).sum() for class_label in class_colors.keys()]
    plt.bar(class_colors.keys(),class_counts,color=class_colors.values())
    plt.xlabel(r'Classification')
    plt.ylabel(r'Count')
    plt.grid()

    plt.tight_layout()
    
    return plt

def plot_ALP_survey(params_in, verbosity=0, tex_fmt=False):
    plt.figure(figsize = (16,12))
    #plt.suptitle('ALP Survey Results')

    plt.subplot2grid((1,1), (0,0), colspan=1, rowspan=1)
    xmin, xmax = (1e-12, 1e7)   # Scale limits
    ymin, ymax = (1e-21, 2e-6)   # Scale limits
    res_color  = colordict['purple']
    plot_masses = True
    show_mass_ranges = False

    ## Interactive mode (WIP)
    with plt.ion():
        # Log-scaled axes
        ax = plt.gca()
        ax.minorticks_on()
        ax.set_xlabel(r'$m_a\quad[eV]$',fontsize=30)
        ax.set_xlim(xmin, xmax)
        ax.set_xscale('log')
        ax.set_ylabel(r'$|g_{a\gamma}|\quad[GeV^{-1}]$',fontsize=30)
        ax.set_ylim(ymin, ymax)
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        ax.set_zorder(3)
        ax.set_facecolor('none')
        #ax.grid('on')
        
        ## Primary data
        ax.set_zorder(3)
        ax.scatter(m_u, GeV/F, s=1000, c=res_color, marker='*')
        
        ## Secondary Data
        ax2 = plt.gcf().add_subplot()
        ax2.set_xscale('log')
        ax2.set_xlim(xmin, xmax)
        ax2.set_yscale('log')
        ax2.set_ylim(ymin, ymax)
        ax2.axis('off')
        ax2.set_zorder(2)
        # Millicharge line (WIP)
        #eps_logval = 2*np.log10(10*eps)-19.9
        eps_logval = lambda ep: (2.615)*np.log10(10*ep + 0.868)-(20.61)
        eps_scaleval = 2
        if verbosity > 8:
            print('10^eps: ', 10**(eps_logval(eps)))
            print('10^eps * scale: ', 10**(eps_logval(eps)) * eps_scaleval)
        eps_str = ('$\\varepsilon=%s$' % ('%d' if eps == 1 else '%.1f' if eps > 0.01 else '%.1e')) % eps
        ax2.plot(np.geomspace(xmin, xmax), np.geomspace((10**(eps_logval(eps)) * eps_scaleval), (10**(eps_logval(eps)+2.15) * eps_scaleval)), c='magenta', linestyle='solid', label=eps_str)
        # F_pi and Lambda lines (WIP)
        ax2.hlines(GeV / F, xmin, m_u, colors=res_color, linestyles='dashed', label='$F_{\pi}$')
        for Lval, Lidx in zip([L3, L4], ['3','4']):
            if Lval > 0:
                ax2.hlines(GeV / Lval, xmin, xmax, colors='black', linestyles='dashdot', label='$\Lambda_%s$' % Lidx)
        # m_i lines/blocks for each species (WIP)
        if plot_masses:
            for s_idx, m_s in enumerate(m):
                s_idx_str = '0' if s_idx == 0 else '\pi' if s_idx == 1 else '\pm' if s_idx == 2 else '\text{ERR}'
                for m_idx, m_i in enumerate(m_s):
                    m_plot_data = {'x': m_i,
                                'label': '$m_{(%s),%s}$' % (s_idx_str, str(m_idx)),
                                'color': res_color if m_i == m_u else 'black',
                                'style': 'dashed'}
                    if len(m_s) == 1 or m_i == m_u or not(show_mass_ranges):  # Plot a vertical line
                        ax2.vlines(m_plot_data['x'], ymin, ymax, colors=m_plot_data['color'], linestyles=m_plot_data['style'], label=m_plot_data['label'])
                    if len(m_s) > 1 and m_idx <= 0 and show_mass_ranges:         # Plot a horizontal line
                        ax2.axvspan(min(m_s), max(m_s), ymin=0, ymax=1, color='black', label='$m_{(%s)}$' % (s_idx_str), alpha=0.5, visible=show_mass_ranges)
        ax2.legend()
                        
        
        ## Background Image
        survey_img = image.imread('./tools/survey_img_crop.PNG')
        im_ax = ax.twinx()
        im_ax.axis('off')
        im_ax = im_ax.twiny()
        im_ax.axis('off')
        im_ax.imshow(survey_img, extent=[xmin, xmax, ymin, ymax], aspect='auto')
    
    return plt

# TODO: Make this function, which plots the result of a number of runs over a series of preferred metrics, for a given mass and density
def plot_parameter_space(tex_fmt=False):
    return None

logfit = lambda x, a, b, c: a*np.log10(10*x+b)+c

# Solve for F_pi given epsilon (millicharge) and ALP (unit) mass
def fit_Fpi(eps, m_scale, show_plots=False, verbosity=0, use_old_fit=False):
    if use_old_fit:
        # TODO: Fix for epsilon != 1?
        fit_res = fit_crude_epsilon_relation(pts_in=[(0.1,-19.9,eps), (0.5,-18.6,eps), (1,-17.9,eps)], plot_fit=show_plots, verbosity=verbosity)
        F_pi = logfit((m_scale, eps), a=fit_res[0], b=fit_res[1], c=fit_res[2])
    else:
        F_pi = 1./((4.38299e-20)*(eps**2)*(10)*(m_scale**(1/9))/(0.1**2))

    if verbosity >=2:
        print('solving for F_pi: %.1e GeV' % F_pi)

    return F_pi

# Fits a log-linear relation as an approximation to the pi-axion mass / F_pi coupling relationship
# TODO: Replace with more robust epsilon relation
def fit_crude_epsilon_relation(pts_in=[], plot_fit=True, verbosity=0):
    pts_x = [[x,eps] for x,_,eps in pts_in]
    pts_y = [[y,eps] for _,y,eps in pts_in]
    eps_vals = set([eps for _,_,eps in pts_in])
    
    xspace = np.linspace(0.01, 1.5, 150)
    if all([eps_i == 1 for eps_i in eps_vals]):
        popt, pcov = curve_fit(logfit, np.array(pts_x)[:,0],np.array(pts_y)[:,0],p0=(2,0,-19.9))
    else:
        print('TODO: eps != 1 not yet supported')
        return None

    if plot_fit:
        for eps_val in eps_vals:
            plt.plot([x for x,_,eps in pts_in if eps == eps_val], [y for _,y,eps in pts_in if eps == eps_val], marker='x', label=r'$\varepsilon = %.0e$' % eps_val, linewidth=0, ms=10)
            plt.plot(xspace, logfit(xspace, a=2, b=0, c=-19.9), linestyle='--')
            plt.plot(xspace, logfit(xspace, a=popt[0], b=popt[1], c=popt[2]), linestyle='-')

            plt.ylim(-21, -17)
            plt.xlim(0, 1.1)
            #plt.yscale('log')
            plt.grid()
            plt.show()

    if verbosity >= 6:
        print('F_pi = %.3f log_10(10x + %.3f) + %.3f' % (popt[0], popt[1], popt[2]))
    
    return popt, pcov

# Helper function to pretty print parameters of model alongside plots
def print_param_space(params, units_in):
    k_step_in = (params['k_span'][1] - params['k_span'][0] + 1.) / params['k_num']
    t_step_in = (params['t_span'][1] - params['t_span'][0] + 1.) / params['t_num']
    units = units_in.copy()
    for key in ['c', 'h', 'G', 'm', 'k', 'p', 'amp', 'Lambda', 'lambda', 'F', 't', 'Theta', 'delta', 'eps']:
        if key not in units_in or units_in[key] == 1:
            units[key] = ''
        elif key in ['Lambda','F'] and units_in[key] == 'm_u':
            units[key] = '\quad[%s]' % ('eV')
        elif key in ['k'] and units_in[key] == 'eV':
            units[key] = '\quad[%s]' % ('m_u')
        else:
            units[key] = '\quad[%s]' % (units_in[key])
    
    textstr1 = '\n'.join((
        r'$A_\pm(t = 0)=%.2g$' % (params['A_0'], ),
        r'$\dot{A}_\pm (t = 0)=%.2g$' % (params['Adot_0'], ),
        r'$\pm=%s$' % (signstr[params['A_pm']], ),
        '\n',
        r'$\varepsilon=%.0g$' % (params['eps'], ),
        r'$F_{\pi}=%.0g%s$' % (params['F'], units['F']),
        r'$c = h = G = 1$' if all([units[key] == '' for key in ['c', 'h', 'G']]) else '\n'.join([r'$%s=%.2g%s$' % (key, val, units[key]) for key, val in zip(['c', 'h', 'G'], [params['c'], params['h'], params['G']])]),
        '\n',
        r'$\lambda_1=%g%s$' % (params['l1'], units['lambda']),
        r'$\lambda_2=%g%s$' % (params['l2'], units['lambda']),
        r'$\lambda_3=%g%s$' % (params['l3'], units['lambda']),
        r'$\lambda_4=%g%s$' % (params['l4'], units['lambda']),
        r'$\Lambda_3=%.0g%s$' % (params['L3'], units['Lambda']) if params['L3'] > 0 else r'$[\Lambda_3=%s]$' % ('N/A'),
        r'$\Lambda_4=%.0g%s$' % (params['L4'], units['Lambda']) if params['L4'] > 0 else r'$[\Lambda_4=%s]$' % ('N/A'),
        '\n',
        r'$\Delta k=%.2g%s$' % (k_step_in, units['k']),
        r'$k \in [%d, %d]$' % (params['k_span'][0], params['k_span'][1]),
        '\n',
        r'$\Delta t=%g%s$' % (t_step_in, units['t']),
        r'$t \in [%d, %d]$' % (params['t_span'][0], params['t_span'][1]),
        ))
    m0_mask = np.ma.getmask(params['m'][0]) if np.ma.getmask(params['m'][0]) else np.full_like(params['m'][0], False)
    m1_mask = np.ma.getmask(params['m'][1]) if np.ma.getmask(params['m'][1]) else np.full_like(params['m'][1], False)
    m2_mask = np.ma.getmask(params['m'][2]) if np.ma.getmask(params['m'][2]) else np.full_like(params['m'][2], False)
    textstr2 = '\n'.join((
        r'$m_{q, dQCD} = [%s]\quad%.2g eV$' % (', '.join('%d' % q for q in np.array(params['qm']) / params['m_q']), params['m_q']),
        '' if units['m'] == 'eV' else r'$m_{u} = %.2e\quad[eV]$' % (params['m_u'], ),
        r'$m%s$' % units['m'],
        ' '.join([r'$m_{(0),%d}=%.2g$'   % (i+1, params['m'][0][i]*params['m_0'], ) for i in range(len(params['m'][0])) if not m0_mask[i]]),
        ' '.join([r'$m_{(\pi),%d}=%.2g$' % (i+1, params['m'][1][i]*params['m_0'], ) for i in range(len(params['m'][1])) if not m1_mask[i]]),
        ' '.join([r'$m_{(\pm),%d}=%.2g$' % (i+1, params['m'][2][i]*params['m_0'], ) for i in range(len(params['m'][2])) if not m2_mask[i]]),
        '\n',
        r'$\pi%s$' % units['amp'],
        ' '.join([r'$\pi_{(0),%d}=%.2g$'   % (i+1, params['amps'][0][i], ) for i in range(len(params['amps'][0])) if not m0_mask[i]]),
        ' '.join([r'$\pi_{(\pi),%d}=%.2g$' % (i+1, params['amps'][1][i], ) for i in range(len(params['amps'][1])) if not m1_mask[i]]),
        ' '.join([r'$\pi_{(\pm),%d}=%.2g$' % (i+1, params['amps'][2][i], ) for i in range(len(params['amps'][2])) if not m2_mask[i]]),
        '\n',
        r'$\rho\quad[eV/cm^3]$',
        ' '.join([r'$\rho_{(0),%d}=%.2g$'   % (i+1, params['p'][0][i]/params['p_0'], ) for i in range(len(params['p'][0])) if not m0_mask[i]]),
        ' '.join([r'$\rho_{(\pi),%d}=%.2g$' % (i+1, params['p'][1][i]/params['p_0'], ) for i in range(len(params['p'][1])) if not m1_mask[i]]),
        ' '.join([r'$\rho_{(\pm),%d}=%.2g$' % (i+1, params['p'][2][i]/params['p_0'], ) for i in range(len(params['p'][2])) if not m2_mask[i]]),
        '\n',
        ' '.join([r'$\delta_{(0),%d}=%.2g \pi$'   % (i+1, params['d'][0][i]/np.pi, ) for i in range(len(params['d'][0])) if not m0_mask[i]]),
        ' '.join([r'$\delta_{(\pi),%d}=%.2g \pi$' % (i+1, params['d'][1][i]/np.pi, ) for i in range(len(params['d'][1])) if not m1_mask[i]]),
        ' '.join([r'$\delta_{(\pm),%d}=%.2g \pi$' % (i+1, params['d'][2][i]/np.pi, ) for i in range(len(params['d'][2])) if not m2_mask[i]]),
        '\n',
        ' '.join([r'$\Theta_{(0),%d}=%.2g \pi$' % (i+1, params['Th'][0][i]/np.pi, ) for i in range(len(params['Th'][0])) if not m0_mask[i]]),
        ' '.join([r'$\Theta_{(\pi),%d}=%.2g \pi$' % (i+1, params['Th'][1][i]/np.pi, ) for i in range(len(params['Th'][1])) if not m1_mask[i]]),
        ' '.join([r'$\Theta_{(\pm),%d}=%.2g \pi$' % (i+1, params['Th'][2][i]/np.pi, ) for i in range(len(params['Th'][2])) if not m2_mask[i]]),
        ))
        
    return textstr1, textstr2

def get_units(unitful_m, rescale_m, unitful_k, rescale_k, unitful_amps, rescale_amps, rescale_consts, dimensionful_p=False,
              unitful_c=False, unitful_h=False, unitful_G=False, use_mass_units=True, use_natural_units=None, verbosity=0):
    global units, params
    is_natural_units = all([not(unitful_c), not(unitful_h), not(unitful_G)]) if use_natural_units is None else use_natural_units
    units = {'c': 1 if not unitful_c else 'cm/s', 'h': 1 if not unitful_h else 'eV/Hz', 'G': 1 if not unitful_G else 'cm^5/(eV s^4)', 
             'm': 'm_u' if not((unitful_m and not rescale_m) or (not unitful_m and rescale_m)) else 'eV/c^2' if unitful_c else 'eV',
             'k': 'm_u' if not rescale_k else 'eV/c' if unitful_c else 'eV',
             'p': 'eV/cm^3' if dimensionful_p else 'eV^4' if not(unitful_c or unitful_h) else 'eV^4/(hc)^3',
             'amp': 'm_u' if (rescale_amps and unitful_amps) else 'eV^2/m_u^2' if (rescale_amps and not unitful_amps) else 'eV' if unitful_amps and is_natural_units else 'eV/c', # 'eV (h^3 c)^(-1/2)'
             'Lambda': 'm_u' if rescale_consts else 'eV/c^2' if unitful_c else 'eV',
             'lambda': 1,
             'F': 'm_u' if rescale_consts else 'eV/c^2' if unitful_c else 'eV',
             't': '1/m_u',
             'Theta': 'π',
             'delta': 'π'}
    
    unit_args = (unitful_m, rescale_m, unitful_k, rescale_k, unitful_amps, rescale_amps, rescale_consts, dimensionful_p, unitful_c, unitful_h, unitful_G, use_mass_units, is_natural_units)

    if verbosity >=0:
        print_units(units, unit_args, verbosity=verbosity)

    return units

# Helper function to pretty print parameter values to console (with units)
pp_param = lambda x, x_u=1., n=2, pfx='  ', notn='e': str(pfx)+('\n'+str(pfx)).join(['m_(%s): %s' % \
            (s_label, ' | '.join([('%.'+('%d' % n)+str(notn)) % (x_s_i*x_u) for x_s_i in x_s]) if len(x_s) > 0 else 'N/A') for x_s, s_label in zip(x, ['0','π','±'])])

def print_units(units, unit_args, verbosity=0):
    unitful_m, rescale_m, unitful_k, rescale_k, unitful_amps, rescale_amps, rescale_consts, dimensionful_p, unitful_c, unitful_h, unitful_G, use_mass_units, use_natural_units = unit_args
    if verbosity > 2:
        print('use_mass_units: %5s' % str(use_mass_units), '||', 'use_natural_units:', use_natural_units)
        if verbosity > 3 and not use_natural_units:
            for c_sw, c_name, c_val in zip([unitful_c, unitful_h, unitful_G], ['c', 'h', 'G'], [c, h, G]):
                print('unitful_'+c_name+':', c_sw, '      | ', c_name+' =', str(1 if units[c_name] == 1 else '%.3e [%s]' % (c_val, units[c_name])))
        print('----------------------------------------------------')
    if verbosity > 3:
        print('unitful_masses:', '%5s' % str(unitful_m),      '| [m_u]' if not unitful_m else '| [%s]' % 'eV')
        print('rescale_m:     ', '%5s' % str(rescale_m),      ''        if not rescale_m else '| [%s] -> [%s]' % (('eV','m_u') if unitful_m    else ('m_u','eV')))
        print('unitful_k:     ', '%5s' % str(unitful_k),      '| [m_u]' if not unitful_k else '| [%s]' % 'eV')
        print('rescale_k:     ', '%5s' % str(rescale_k),      ''        if not rescale_k else '| [%s] -> [%s]' % (('eV','m_u') if unitful_k    else ('m_u','eV')))
        print('unitful_amps:  ', '%5s' % str(unitful_amps),   '| [eV]'  if unitful_amps  else '| [eV^2/m_u]')
        print('rescale_amps:  ', '%5s' % str(rescale_amps),   '' if not rescale_amps     else '| [%s] -> [%s]' % (('eV','m_u') if unitful_amps else ('eV^2/m_u','eV^2/m_u^2')))
        print('rescale_consts:', '%5s' % str(rescale_consts), '' if not rescale_consts   else '| [%s] -> [%s]' % (('eV','m_u') if unitful_m    else ('eV', 'eV/m_u')))
        print('----------------------------------------------------')

def print_params(units, m=None, p=None, amps=None, Th=None, d=None, m_q=None, m_0=None, m_u=None, natural_units=True, verbosity=0, precision=3):
    if verbosity >= 0:
        if m is not None:
            print('m_dQCD = %.0e [eV%s]' % (m_q, '' if natural_units else '/c^2'))
            if units['m'] == 'm_u': print('m_u = %.3e [eV%s]' % (m_u, '' if natural_units else '/c^2'))
            print('m [' + units['m'] + ']\n'      + pp_param(m, m_0, n=precision))
        if p is not None:
            print('rho [' + units['p'] + ']\n'    + pp_param(p, n=precision))
        if amps is not None:
            print('amp [' + units['amp'] + ']\n'  + pp_param(amps, n=precision))
        if Th is not None:
            print('Theta [π]\n'                   + pp_param(Th, 1. / np.pi, n=2, notn='f'))
        if d is not None:
            print('delta [π]\n'                   + pp_param(d,  1. / np.pi, n=2, notn='f'))

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

# Return sector of EM spectrum where given k-mode frequency resides, and any possible corresponding phenomenological EM observables
def get_frequency_class(k_mode_in, k_to_HZ, res_label, verbosity=0):
    # Known observable frequency ranges (Hz)
    FRB_values = [100e6, 5000e6]
    CMB_values = []              # TODO: Any possible connection to CMB features? Explore this.
    UIE_values = [1.58e13, 6e13] # TODO: Sharper UIE feature lines? This fitting within a weak spectrum currently.
    AXP_values = []              # TODO: Identify ranges of frequencies for AXP anomalous emissions
    GRB_values = [3e19, 3e21]    # TODO: Identify potential relation to afterglow features

    Hz_label = lambda f, pd=pd: pd.cut([f],
                                       [0, 300e6, 3e12, 480e12, 750e12, 30e15, 30e18, np.inf],
                                       labels=['Radio', 'Microwave', 'Infrared', 'Visible', 'UV', 'X-ray', 'Gamma ray'])
    
    Hz_class  = None
    obs_class = {'FRB': None, 'UIE': None, 'GRB': None, 'CMB': 'N/A', 'AXP': 'N/A', 'Afterglow':'N/A'}

    if 'res' in res_label:
        Hz_peak  = k_to_HZ(k_mode_in)
        Hz_class = Hz_label(Hz_peak)[0]
        if verbosity >= 0:
            print('peak resonance at k = %d corresponds to photon frequency at %.2e Hz (%s)' % (k_peak, Hz_peak, Hz_class))
        if Hz_peak >= FRB_values[0] and Hz_peak <= FRB_values[1]:
            obs_class['FRB'] = True
            if verbosity >= 0:
                print('possible FRB signal')
        if Hz_peak >= UIE_values[0] and Hz_peak <= UIE_values[1]:
            obs_class['UIE'] = True
            if verbosity >= 0:
                print('possible UIE signal')
        if Hz_peak >= GRB_values[0] and Hz_peak <= GRB_values[1]:
            obs_class['GRB'] = True
            if verbosity >= 0:
                print('possible GRB signal')
    
    return Hz_class, obs_class

# Returns a tuple containing the min/max resonant k-modes, and the resonance classification (broad-band / narrow-band / multi-band)
def get_resonance_band(k_values_in, k_class_arr, k_to_HZ, class_sens=0.1, verbosity=0):
    # Initialize lists to store start and end indices of resonant segments
    start_indices = []
    end_indices = []

    # Identify continuous segments with 'res' classification
    start_idx = None
    for i, label in enumerate(k_class_arr):
        if label == 'res' and start_idx is None:
            start_idx = i
        elif label not in ['res', 'semi'] and start_idx is not None:
            start_indices.append(start_idx)
            end_indices.append(i-1)
            start_idx = None
    if start_idx is not None:  # Handle case where last segment reaches the end
        start_indices.append(start_idx)
        end_indices.append(len(k_class_arr)-1)

    # Determine resonance classification
    res_count  = np.ma.masked_where(k_class_arr != 'res', k_class_arr, copy=True).count()
    semi_count = np.ma.masked_where(k_class_arr != 'semi', k_class_arr, copy=True).count()
    if not start_indices:  # No resonance segments found
        return None, None, None
    elif len(start_indices) == 1:  # Only one resonance segment
        if start_indices[0] == 0 and end_indices[0] == len(k_class_arr) - 1:
            classification = 'broad-band'
        elif (int(res_count) / int(len(k_class_arr))) >= float(class_sens):
            classification = 'broad-band'
        else:
            classification = 'narrow-band'
    else:  # Multiple resonance segments
        classification = 'multi-band'

    # Convert k-values to HZ and return results
    min_res_Hz = float(k_to_HZ(min(k_values_in[start_indices[0]:end_indices[-1] + 1])))
    max_res_Hz = float(k_to_HZ(max(k_values_in[start_indices[0]:end_indices[-1] + 1])))

    if verbosity >= 0:
        print('%s resonance detected from %.0g to %.0g Hz' % (classification, min_res_Hz, max_res_Hz))
    
    return min_res_Hz, max_res_Hz, classification