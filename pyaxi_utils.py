import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.pyplot import subplot2grid
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import display, clear_output, HTML, Image
from pyparsing import line
from scipy.signal import spectrogram
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
import sklearn.metrics
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
version='v3.2.8'
# Fundamental constants
c = c_raw = np.float64(2.998e10)    # Speed of light       [cm/s]
h = h_raw = np.float64(4.136e-15)   # Planck's constant    [eV/Hz]
G = G_raw = np.float64(1.0693e-19)  # Newtonian constant   [cm^5 /(eV s^4)]
e = 0.3                             # Dimensionless electron charge
# Formatting Colors
colordict = {
    'purple': '#b042f5'
}

## Set parameters of model for use in numerical integration
# TODO: Remove dependence on this function
def init_params(params_in: dict, sample_delta=True, sample_theta=True, t_max=10, t_min=0, t_N=500,
                k_max=200, k_min=1, k_N=200):
    
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
    inf_con = params_in['inf_con']
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

    # characteristic periods of oscillation for each species
    if 'T_u' in params_in:
        T_u = params_in['T_u']
        T_r = params_in['T_r']
        T_n = params_in['T_n']
        T_c = params_in['T_c']
    else:
        T_u, T_r, T_n, T_c = get_timescales(m, m_0, m_u=1)

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
    d      = params_in['d']   if 'd' in params_in else np.array([d_r, d_n, d_c], dtype=object)

    # global phase for complex species in (0, 2pi)
    Th_r    = params_in['Th_r'] if 'Th_r' in params_in else np.array([np.zeros(int(N_r))])
    Th_n    = params_in['Th_n'] if 'Th_n' in params_in else np.array([np.zeros(int(N_n))])
    Th_c    = params_in['Th_c'] if 'Th_c' in params_in else np.array([np.zeros(int(N_c))])
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
    int_method        = params_in['int_method']
    use_natural_units = params_in['use_natural_units']
    use_mass_units    = params_in['use_mass_units']
    unitful_m         = params_in['unitful_m']
    rescale_m         = params_in['rescale_m']
    unitful_k         = params_in['unitful_k']
    rescale_k         = params_in['rescale_k']
    unitful_amps      = params_in['unitful_amps']
    rescale_amps      = params_in['rescale_amps']
    rescale_consts    = params_in['rescale_consts']
    dimensionful_p    = params_in['dimensionful_p']
    unitful_c = False if use_natural_units else c != 1
    unitful_h = False if use_natural_units else h != 1
    unitful_G = False if use_natural_units else G != 1

    # performance metrics
    num_cores    = params_in['num_cores']    if 'num_cores'    in params_in else 1
    mem_per_core = params_in['mem_per_core'] if 'mem_per_core' in params_in else None

    t_0 = 1./m_u if unitful_m else 1.

    # Turn off irrelevant constants
    disable_P = params_in['disable_P'] if 'disable_P' in params_in else False
    disable_B = params_in['disable_B'] if 'disable_B' in params_in else False
    disable_C = params_in['disable_C'] if 'disable_C' in params_in else False
    disable_D = params_in['disable_D'] if 'disable_D' in params_in else False
    if N_r <= 0:
        disable_C = True
        if N_n <= 0:
            L4 = -1                   # Turn off Lambda_4 if there are no surviving neutral (real and complex) species
    if N_c <= 0:
        disable_D = True
        L3 = -1                   # Turn off Lambda_3 if there are no surviving charged species

    #rescale_params(rescale_m, rescale_k, rescale_amps, rescale_consts, unitful_c=unitful_c, unitful_h=unitful_h, unitful_G=unitful_G)
            
    # Store for local use, and then return
    params = {'e': e, 'F': F, 'p_t': p_t, 'eps': eps, 'L3': L3, 'L4': L4, 'l1': l1, 'l2': l2, 'l3': l3, 'l4': l4,
              'A_0': A_0, 'Adot_0': Adot_0, 'A_pm': A_pm, 'amps': amps, 'd': d, 'Th': Th, 'h': h, 'c': c, 'G': G,
              'qm': qm, 'qc': qc, 'dqm': dqm, 'eps_c': eps_c, 'xi': xi, 'N_r': N_r, 'N_n': N_n, 'N_c': N_c, 'p_0': p_0,
              'm': m, 'm_r': m_r, 'm_n': m_n, 'm_c': m_c, 'p': p, 'p_r': p_r, 'p_n': p_n, 'p_c': p_c, 'm_0': m_0, 'm_q': m_q,
              'mu_d': mu_d, 'sig_d': sig_d, 'mu_Th': mu_Th, 'sig_Th': sig_Th, 'k_span': k_span, 'k_num': k_num, 'k_0': k_0,
              't_span': t_span, 't_num': t_num, 'A_sens': A_sens, 't_sens': t_sens, 'res_con': res_con, 'm_u': m_u,
              't_u': t_0, 'T_n': T_n, 'T_r': T_r, 'T_c': T_c, 'T_u': T_u, 'inf_con': inf_con,
              'disable_P': disable_P, 'disable_B': disable_B, 'disable_C': disable_C, 'disable_D': disable_D,
              'unitful_m': unitful_m, 'rescale_m': rescale_m, 'unitful_amps': unitful_amps, 'rescale_amps': rescale_amps, 
              'unitful_k': unitful_k, 'rescale_k': rescale_k, 'rescale_consts': rescale_consts, 'seed': seed, 'int_method': int_method,
              'use_natural_units': use_natural_units, 'use_mass_units': use_mass_units, 'dimensionful_p': dimensionful_p,
              'num_cores': num_cores, 'mem_per_core': mem_per_core}
    
    return params


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
    entropy_size = 4
    if seed is not None:
        rng_ss = np.random.SeedSequence(entropy=int(seed), pool_size=entropy_size)
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
        params_in['config_name'] = str(os.path.basename(os.path.dirname(params_filename)))
        params_in['units'] = get_units_from_params(params_in)
        with open(params_filename, 'w') as f:
            with np.printoptions(threshold=np.inf):
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
                        plt.close(fig)
            if save_format in img_formats:
                for i, fig in enumerate(plot_figs):
                    fig.savefig(os.path.join(output_dir, filename + f'_plot_{i}.png'))
                    plt.close(fig)
            if save_format in nbk_formats:
                # Display plots in the notebook (TODO)
                for fig in plot_figs:
                    display(fig)
                    #plt.close(fig)
            if save_format in web_formats:
                # Convert notebook to HTML and save (TODO)
                if True:
                    print('HTML not supported yet')
                else:
                    html_content = str(HTML('<h1>Simulation Plots</h1>'))
                    for fig in plot_figs:
                        #display(fig)
                        html_content += str(HTML(str(html_content) + '<img src="data:image/png;base64,{}">'.format(fig)))
                        plt.close(fig)
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
            
            # Append filenames for generated plots
            if save_plots:
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

def load_multiple_results(output_dir, label, load_images=False, save_format='pdf', nested=False, include_debug=True, combined_folder=False, load_method='json'):
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
    load_pandas = load_method in ['pandas','pd']
    output_path = os.path.expanduser(output_dir) if '~' in output_dir else output_dir
    if nested:
        file_dirs  = [os.path.join(os.path.expanduser(output_path), sub_dir) for sub_dir in os.listdir(output_path) if not('debug' in sub_dir) or include_debug]
        all_files  = [os.path.join(sub_dir, nested_file) for sub_dir in file_dirs for nested_file in os.listdir(sub_dir)]
    else:
        file_dir   = os.path.join(os.path.expanduser(output_path), label)
        all_files  = [os.path.join(file_dir, file_name) for file_name in os.listdir(file_dir) if not('debug' in file_name) or include_debug]
    
    # List of tuples (filename, absolute path)
    relevant_files = [os.path.split(f) for f in all_files if (os.path.basename(f).startswith(label) or (label == 'all' or combined_folder)) and os.path.basename(f).endswith('.json')] # Assume input params are being saved for now, at least
    
    all_params  = []
    all_results = []
    all_plots   = []
    all_coeffs  = []
    
    for filepath, filename in relevant_files:
        # Extract base name without extension
        base_name = filename.rsplit('.', 1)[0]
        
        # Load parameters
        params_filename = os.path.join(filepath, base_name + '.json')
        if load_pandas:
            params = pd.read_json(params_filename, dtype=True, typ='series')
        else:
            with open(params_filename, 'r') as f:
                params = json.loads(f.read(), object_hook=NumpyEncoder.decode)
        params['file_name'] = base_name
        all_params.append(params)
        
        # Load results
        results_filename = os.path.join(filepath, base_name + '.npy')
        results = np.array(np.load(results_filename), dtype=object)
        all_results.append(results)

        # Load coefficient functions
        coeffs_filename = os.path.join(filepath, base_name, '_funcs.pkl')
        if os.path.exists(coeffs_filename):
            all_coeffs.append(load_coefficient_functions(coeffs_filename))
        else:
            all_coeffs.append(None)
        
        # Load plots
        plots = []
        if load_images and save_format == 'png':
            i = 0
            while os.path.exists(os.path.join(filepath, base_name + f'_plot_{i}.png')):
                plots.append(plt.imread(os.path.join(filepath, base_name + f'_plot_{i}.png')))
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

def load_all(output_root='~/scratch', version=version, load_images=False, save_format='pdf', include_debug=False):
    return load_case(input_str='all', output_root=output_root, version=version, load_images=load_images, save_format=save_format, include_debug=include_debug)

def load_case(input_str, output_root='~/scratch', version=version, load_images=False, save_format='pdf', include_debug=True, combined_folder=False):
    """
    Load all results for a given class of configuration parameters given a full path to the output directory or just the simulation label.
    
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
    
    return load_multiple_results(output_dir, result_name, load_images, save_format, nested=(input_str=='all'), include_debug=include_debug, combined_folder=combined_folder)

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

# TODO: Make logic in loading file refer to this function
def parse_filename(filename):
    """
    Parse the given filename to extract the simulation result name and parameter space hash.
    Basic structure of a filename is {config_label}_{parameter_space_hash}{file_extension}
    
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
def plot_single_case(input_str, output_dir=default_output_directory, plot_res=True, plot_nums=True, plot_coeffs=True, plot_spectrum=True, k_samples_in=[], set_params_globally=False, tex_fmt=False, add_colorbars=False, version=version):

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
    k_peak_in  = params.get('k_peak_arr', None)
    k_peak_arr = np.array(k_peak_in.split() if type(k_peak_in) is str else k_peak_in, dtype=np.float64)

    # Define which k values should be plotted
    k_peak = np.argmax(k_sens_arr) # peak (running avg) value per k-mode
    k_mean = np.argmax(k_mean_arr) # mean value per k-mode
    k_rmax = np.argmax(k_peak_arr) # (raw) peak value per k-mode
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
        plot_amplitudes(params_in=params, units_in=units, results_in=results, k_samples=k_samples, times=times, tex_fmt=tex_fmt, add_colorbars=add_colorbars)

    # Plot occupation number of the photon field, as imported from file
    if plot_nums:
        plot_occupation_nums(params_in=params, units_in=units, results_in=results, numf=None, k_samples=k_samples, times=times, tex_fmt=tex_fmt, add_colorbars=add_colorbars)

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
        plot_resonance_spectrum(params_in=params, units_in=units, results_in=results, fwd_fn=k_to_Hz_local, inv_fn=Hz_to_k_local, tex_fmt=tex_fmt)

# Helper function for below (NOTE: m_u may not be the same m_u as in other parts of the code -- look into this)
min_timescale = lambda m_min, m_u: 1./(np.min([m_min,1.]))*((2*np.pi)/(m_u))

# Characteristic timescales (minimum amount of time needed to capture full oscillations) by species
def get_timescales(m, m0, m_u=1., verbosity=0):
    # Assuming m_min is given in units of [m_u], else set rescaling relation in m_u arg
    t_min = lambda m: min_timescale(m, m_u)
    for i in range(m.shape[0]):
        m_min_r = np.min(m[0]*m0) if len(m[0]) > 0 else 0
        t_min_r = t_min(m_min_r)  if len(m[0]) > 0 else 0
        m_min_n = np.min(m[1]*m0) if len(m[1]) > 0 else 0
        t_min_n = t_min(m_min_n)  if len(m[1]) > 0 else 0
        m_min_c = np.min(m[2]*m0) if len(m[2]) > 0 else 0
        t_min_c = t_min(m_min_c)  if len(m[2]) > 0 else 0
    # find largest t_min to set as our characteristic time period
    T_min   = np.max(np.abs([t_min_r, t_min_n, t_min_c]))

    if verbosity >= 2:
        if verbosity >= 5:
            print('Characteristic timescales by species:')
            print(' -   reals: m_min = %.2f [m_u]  --->  T_r = %.2fπ [1/m_u]' % (m_min_r, np.abs(t_min_r/np.pi)))
            print(' - complex: m_min = %.2f [m_u]  --->  T_n = %.2fπ [1/m_u]' % (m_min_n, np.abs(t_min_n/np.pi)))
            print(' - charged: m_min = %.2f [m_u]  --->  T_c = %.2fπ [1/m_u]' % (m_min_c, np.abs(t_min_c/np.pi)))
            if verbosity >= 8:
                print(' -----------> T_min = %.2fπ [1/m_u]' % np.abs(T_min/np.pi))
        else:
            print('Characteristic timescale: T_min = %.2fπ [1/m_u]' % np.abs(T_min/np.pi))
        print('----------------------------------------------------')

    return T_min, t_min_r, t_min_n, t_min_c

# k_ratio: apply [k_func] to each k mode and then return the ratio of the final vs. initial amplitudes (sensitive to a windowed average specified by [sens])
# (DEPRECATED) see classify_resonance instead
k_ratio = lambda func, t_sens, A_sens, k_sens: np.array([k_f/k_i for k_f, k_i in zip(k_sens(func, t_sens), k_sens(func, -t_sens))])

# k_class: softly classify the level of resonance according to the final/initial mode amplitude ratio, governed by [func, t_sens, and A_sens]
# (DEPRECATED) see classify_resonance instead
# k_class = lambda func, t_sens, A_sens, res_con: np.array(['damp' if k_r <= 0.9 else 'none' if k_r <= (1. + np.abs(A_sens)) else 'soft' if k_r <= res_con else 'res' for k_r in k_ratio(func, t_sens, A_sens)])

get_times = lambda params_in, times_in: times_in if times_in is not None else np.linspace(params_in['t_span'][0], params_in['t_span'][1], params_in['t_num'])
get_kvals = lambda params_in, kvals_in: kvals_in if kvals_in is not None else np.linspace(params_in['k_span'][0], params_in['k_span'][1], params_in['k_num'])

# get indices for time-averaged windows, characterized by sensitivity
# (e.g. sens = 0.1 means shave off 10% of the time window; sign (+/-) of sens determines which endpoint of the window is returned)
win_L_N  = lambda sens, t_n: int(t_n*(1./2)*np.abs(((1. - sens)*np.sign(sens) + (1. - sens))))  # time window in [0, (1-sens)]
win_U_N  = lambda sens, t_n: int(t_n*(1./2)*np.abs(((1. + sens)*np.sign(sens) + (1. - sens))))  # time window in [sens, 1]

## Identify the k mode with the greatest peak amplitude, and the mode with the greatest average amplitude
# TODO: Update this logic to be more in line with classify_resonance
def get_peak_k_modes(params_in, results_in, k_values_in=None, write_to_params=False):

    t_num = len(results_in[0][0])
    k_values = k_values_in if k_values_in is not None else get_kvals(params_in, k_values_in)

    win_lower  = lambda sens, t_n=t_num: win_L_N(sens, t_n)  # anchor left endpoint at 0 (-/+ sens for L/R endpoints)
    win_upper  = lambda sens, t_n=t_num: win_U_N(sens, t_n)  # anchor right endpoint at t[N] (-/+ sens for L/R endpoints)
    
    # k_func : apply [func] on the time-series for each k mode, e.g. max or mean
    k_func = lambda func: np.array([k_fval for k_fi, k_fval in enumerate([func(np.abs(results_in[k_vi][0][:])) for k_vi, k_v in enumerate(k_values)])])

    # k_sens : apply [k_func] but limit the time-series by [sens], e.g. sens = 0.1 to skip the first 10% in calculating our time-averaged values
    k_sens = lambda func, sens: np.array([k_fval for k_fi, k_fval in enumerate([func(np.abs(results_in[k_vi][0][win_lower(sens):win_upper(sens)])) for k_vi, k_v in enumerate(k_values)])])

    # k mode(s) with the largest contributions to overall number density growth
    k_peak = k_values[np.ma.argmax(k_func(max))]
    k_mean = k_values[np.ma.argmax(k_func(np.ma.mean))]

    # store max, all-time mean, and late-time mean for each k-mode locally, as well as resonance classifications
    if write_to_params:
        # TODO: Update this to new logic
        params_in['k_peak_arr']  = k_func(max)
        params_in['k_mean_arr']  = k_func(np.mean)
        params_in['k_sens_arr']  = k_sens(np.mean, params_in['t_sens'])
        #params_in['k_class_arr'] = k_class(np.mean, t_sens, A_sens, res_con)

        # TODO: unify all of the different methods we use to classify resonance
        #tot_res = 'resonance' if sum(k_ratio(np.ma.mean, t_sens, A_sens)) > res_con else 'none'
        #params_in['res_class'] = tot_res
    
    return k_peak, k_mean

## Helper function for colorbar plotting
def get_colorbar_params(k_values_in):
    k_values = k_values_in
    cm_vals = np.linspace(0,1,len(k_values))

    # helper to normalize data into the [0.0, 1.0] interval.
    cm_norm_pri = colors.Normalize(vmin=np.min(cm_vals), vmax=np.max(cm_vals))
    # alternative to normalize data into the [-1.0, 1.0] interval.
    cm_norm_alt = colors.Normalize(vmin=-np.max(cm_vals), vmax=np.max(cm_vals))

    # choose a colormap
    #c_m = cm.viridis
    #c_m = cm.cividis_r
    #c_m = cm.cool
    c_m = cm.winter
    #c_m = cm.PuBuGn
    #c_m = cm.plasma
    #c_m = cm.hsv
    #c_m = cm.twilight
    #c_m = cm.CMRmap

    # create a ScalarMappable and initialize a data structure
    s_m_pri = cm.ScalarMappable(cmap=c_m, norm=cm_norm_pri)
    s_m_alt = cm.ScalarMappable(cmap=c_m, norm=cm_norm_alt)

    # in case we need to have different colobar normalization schema for the data vs the plotting function
    s_m_plt = s_m_pri
    cm_norm_plt = cm_norm_pri
    s_m_cbar = s_m_pri
    cm_norm_cbar = cm_norm_pri

    # Format text labels and tickmark locations for colorbar legend
    cbar_ticks = np.linspace(0, 1, 2)
    cbar_labels = [r'$%s$' % k_values[int(tick_val-1)] for tick_val in np.linspace(k_values[0], k_values[-1], len(cbar_ticks))]

    return c_m, cm_vals, cbar_ticks, cbar_labels, s_m_plt, cm_norm_plt, s_m_cbar, cm_norm_cbar

# Plot the amplitudes (results of integration)
def plot_amplitudes(params_in, units_in, results_in, k_samples=[], times=None, plot_Adot=True, plot_RMS=False, plot_avg=False, tex_fmt=False, add_colorbars=False):
    amp_plt = make_amplitudes_plot(params_in, units_in, results_in, k_samples, times, plot_Adot, plot_RMS, plot_avg, tex_fmt, add_colorbars)
    amp_plt.show()
    
def make_amplitudes_plot(params_in, units_in, results_in, k_samples_in=[], times_in=None, plot_Adot=True, plot_RMS=False, plot_avg=False, tex_fmt=False, add_colorbars=False, abs_amps=None, precision_limit=1e100):
    k_values = np.linspace(params_in['k_span'][0], params_in['k_span'][1], params_in['k_num'])
    k_peak, k_mean = get_peak_k_modes(params_in, results_in, k_values)
    plot_all_k = True if len(k_samples_in) == 1 and k_samples_in[0] < 0 else False
    if plot_all_k:
        k_samples = [i for i, k_i in enumerate(k_values)]
    elif len(k_samples_in) <= 0:
        #k_samples = np.geomspace(1,len(k_values),num=5)
        k_samples = [i for i, k_i in enumerate(k_values) if k_i in [0,1,10,50,100,150,200,500,k_peak,k_mean]]
    else:
        k_samples = k_samples_in
    
    signdict = signtex if tex_fmt else signstr
    fontsize = 16 if tex_fmt else 14
    times = get_times(params_in, times_in)
    t_step = float(params_in['t_span'][1] - params_in['t_span'][0]) / float(params_in['t_num'])

    xdim = 5
    if plot_Adot:
        ydim = 3 
    else:
        ydim = 2

    #fig = Figure(figsize=(4*xdim, 4*ydim))
    #plt.subplot2grid((ydim,xdim), (0,0), fig=fig, colspan=3)
    fig = plt.figure(figsize=(4*xdim, 4*ydim))
    ax1 = plt.subplot2grid((ydim,xdim), (0,0), colspan=3, fig=fig)

    if add_colorbars:
        c_m, cm_vals, cbar_ticks, cbar_labels, s_m_plt, cm_norm_plt, s_m_cbar, cm_norm_cbar = get_colorbar_params(k_values)

    for k_idx, k_sample in enumerate(k_samples):
        k_s = int(k_sample)
        #print(results_in[k_s, 0])

        if abs_amps is None:
            abs_amps = True if add_colorbars else False
        if abs_amps or add_colorbars:
            y = np.abs(results_in[k_s][0])
        else:
            y = results_in[k_s][0]

        if add_colorbars:
            plt.plot(times, y, label='k='+str(k_values[k_s]), linewidth=1, color=c_m(cm_norm_plt(cm_vals[k_idx])))
        else:
            plt.plot(times, y, label='k='+str(k_values[k_s]))

    if add_colorbars:
        cbar1 = plt.colorbar(s_m_cbar, label=r'$k$', cmap=c_m, norm=cm_norm_cbar, drawedges=False, location='right', fraction=0.02, pad=0, anchor=(0.0,0.1))
        cbar1.set_ticks(cbar_ticks)
        cbar1.set_ticklabels(cbar_labels)
    else:
        plt.legend()
    absbuff = '|' if abs_amps else ''
    plt.title(r'Evolution of the mode function $%sA_{%s}(k)%s$' % (absbuff, signdict[0], absbuff))
    plt.xlabel(r'Time $[%s]$' % units_in['t'])
    plt.ylabel(r'$%sA_{%s}(k)%s$' % (absbuff, signdict[0], absbuff))
    plt.yscale('log')
    plt.grid()

    amp_sq = lambda i, A_in, t_in: np.exp(2*np.log(np.abs(results_in[i][A_in][t_in]))) if results_in[i][A_in][t_in] < precision_limit else np.inf

    #plt.subplot(2,1,2)
    plt.subplot2grid((ydim,xdim), (1,0), colspan=3)
    plt.plot(times, [sum([amp_sq(i, 0, t_i) for i in range(len(k_values))]) for t_i in range(len(times))])
    if add_colorbars:
        # Add and then remove a blank colorbar so that all plots are lined up with those that use colorbar legends
        cbar2 = plt.colorbar(s_m_cbar, alpha=0, location='right', fraction=0.02, pad=0, anchor=(0.0,0.1), drawedges=False, ticks=[])
        cbar2.remove()
    plt.title(r'Evolution of the (total) power $|A_{%s}|^2$' % signdict[0])
    plt.xlabel(r'Time $[%s]$' % units_in['t'])
    plt.ylabel(r'$|A_{%s}|^2$' % signdict[0])
    plt.yscale('log')
    plt.grid()

    if plot_Adot:
        #plt.subplot(2,1,2)
        res_in_arr = np.array(results_in)
        A_dims = (res_in_arr.shape[0], res_in_arr.shape[-1])
        A_der  = np.reshape(np.delete(res_in_arr, 1, axis=1), A_dims)
        A_mean = np.reshape(A_der[list(k_values).index(k_mean)], (1, A_dims[1]))[0]
        A_peak = np.reshape(A_der[list(k_values).index(k_peak)], (1, A_dims[1]))[0]
        # Plot Amplitude rate of change
        plt.subplot2grid((ydim,xdim), (2,0), colspan=3)
        plt.plot(times, np.sum(A_der, axis=0), color='g', label=r'total')
        plt.plot(times, A_mean, color='y', label=r'$k$ = %d (mean)' % k_mean)
        plt.plot(times, A_peak, color='orange', label=r'$k$ = %d (peak)' % k_peak)
        plt.title(r'Evolution of the (total) change in amplitude for $A_{%s}$' % signdict[0])
        plt.xlabel(r'Time $[%s]$' % units_in['t'])
        plt.ylabel(r'$\dot{A}_{%s}$' % signdict[0])
        plt.yscale('symlog')
        if add_colorbars:
            # Add and then remove a blank colorbar so that all plots are lined up with those that use colorbar legends
            cbar3 = plt.colorbar(s_m_cbar, alpha=0, location='right', fraction=0.02, pad=0, anchor=(0.0,0.1), drawedges=False, ticks=[])
            cbar3.remove()
        
        if plot_avg:
            # TODO: Implement a decent average/convolution function (WIP)
            avg = lambda a_in, m, dt=t_step, N=len(k_values): np.array([np.convolve(a_in[ki], np.ones(N)/N, mode=m) for ki in a_in])
            plot_avg_modes = ['full', 'same', 'valid']
            for mode in plot_avg_modes:
                A_d_avg = avg(A_der, mode)
                avg_times = np.linspace(times[0], times[-1], num=A_d_avg.shape[-1])
                plt.plot(avg_times, A_d_avg, label=r'avg (%s)' % mode)
        if plot_RMS:
            # Plot Amplitude rate of change RMS / averaged values
            rms = lambda a_in, t_in=times: np.sqrt(np.trapz(a_in**2, t_in)/(t_in[-1] - t_in[0]))
            a_rms = lambda a_in, t_in=times, N=len(k_values): np.array([[rms(a_in[ki,:ti], t_in[:ti]) if ti > 1 else np.abs(a_in[ki,ti]) for ti, td in enumerate(t_in)] for ki in np.arange(N)])
            # Calculate RMS for A_dot using two different methods
            a_rms_1 = a_rms(np.reshape(np.sum(A_der, axis=0), (1, A_dims[1])), N=1)[0]
            a_rms_2 = np.sum(a_rms(A_der), axis=0)
            # Calculate for k_peak and k_mean
            a_rms_peak  = a_rms(np.reshape(A_der[list(k_values).index(k_mean)], (1, A_dims[1])), N=1)[0]
            a_rms_mean  = a_rms(np.reshape(A_der[list(k_values).index(k_peak)], (1, A_dims[1])), N=1)[0]

            plt.plot(times, a_rms_1, color='black', linestyle='-', label=r'total (1)')
            plt.plot(times, a_rms_2, color='black', linestyle=':', label=r'total (2)')
            plt.plot(times, a_rms_peak, color='yellow', linestyle='-', label=r'$k$ = %d (mean)' % k_mean)
            plt.plot(times, a_rms_mean, color='orange', linestyle='-', label=r'$k$ = %d (peak)' % k_peak)

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

# Set all values greater than cutoff to be np.inf
def A_limit(A_in, limit=1e100):
    A_in[A_in > limit] = np.inf
    return A_in

# Amplitude getter functions for occupation number function
A_i     = lambda i, solns: solns[i][0]
Adot_i  = lambda i, solns: solns[i][1]
k_i     = lambda i, k_vals=None: k_vals[i]
#ImAAdot = lambda k, t: (1/2)*np.abs(np.cos(2*k*t))
#ImAAdot = lambda k, t: -(1/2)
ImAAdot = lambda k, t: (1/(4*k))  # Temp workaround to ensure this is -(1/2) at t=0
k_vals_p = lambda params: np.linspace(params['k_span'][0], params['k_span'][1], params['k_num'])

# Particle number for k_mode value (TODO: Verify unitful scaling in below equation)
#n_k = lambda k, A, Adot, Im: (k**2 * np.abs(A)**2 + np.abs(Adot)**2 - 2*k*Im(k))
n_k = lambda k, A, Adot, Im: ((k/2)*(np.abs(A)**2 + (np.abs(Adot)**2 / k**2)) - (1/2))
# Wrapper function for the above, using only params and results as inputs
n_p = lambda i, params, solns, k_vals=None, t_in=None, n=n_k: \
    n(k=k_i(i, k_vals=k_vals if k_vals is not None else k_vals_p(params)),
      A=A_limit(A_i(i, solns), params['inf_con']), 
      Adot=A_limit(Adot_i(i, solns), params['inf_con']),
      Im=lambda k: ImAAdot(k, t=get_times(params, t_in)))

def binned_classifier(k_stat, N_bins, ln_rescon=2, return_dict=False):
    """
    The function `binned_classifier` classifies data based on statistical values and returns the
    classification label along with related ratios and baseline value.
    
    :param k_stat: The `k_stat` parameter in the `binned_classifier` function represents a list of
    values that are used to classify data into different categories based on certain conditions
    :param N_bins: The `N_bins` parameter in the `binned_classifier` function represents the total
    number of bins in the input `k_stat` array. It is used to iterate over the elements of the `k_stat`
    array and perform calculations based on the values within these bins
    :param ln_rescon: The `ln_rescon` parameter in the `binned_classifier` function represents the
    threshold value for determining significant changes in the data. It is used to classify the data
    into different categories based on whether the change exceeds this threshold, defaults to 2
    (optional)
    :param return_dict: The `return_dict` parameter in the `binned_classifier` function determines
    whether the function should return the classification results as a dictionary or as individual
    values, defaults to False (optional)
    :return: The function `binned_classifier` returns either a tuple or a dictionary depending on the
    value of the `return_dict` parameter.
    """
    b_baseline = k_stat[0]
    class_label = 'none' if b_baseline < ln_rescon else 'injection'
    b_max_val = 0.
    b_max_idx = 0
    for bidx in range(1, N_bins-1):
        bval = k_stat[bidx] - b_baseline
        if bval > b_max_val:
            b_max_val = bval
            b_max_idx = bidx
        # Exponential growth
        if bval >= ln_rescon:
            if class_label in ['none', 'damp', 'burst']:
                class_label = 'resonance'
        # Exponential decay
        elif bval <= -ln_rescon:
            if class_label in ['none', 'damp']:
                class_label = 'damp'
            else:
                class_label = 'burst'
        # No significant change
        else:
            if class_label in ['injection']:
                # Re-zero out initial reference number once energy injection has stabilized
                b_max_val = 0.
                b_max_idx = bidx
                b_baseline = k_stat[bidx]
        
    ratio_max = b_max_val
    ratio_final = k_stat[-1] - b_baseline
    
    if (ratio_final - ratio_max) <= -ln_rescon:
        class_label = 'burst'
    
    if return_dict:
        return {'label': class_label, 'ratio_final': ratio_final, 'ratio_max': ratio_max, 'baseline': b_baseline}
    else:
        return class_label, ratio_final, ratio_max, b_baseline

# Classify resonance based on ability to fit to step-function shape
def heaviside_classifier(t_in, n_in, res_con=1000, err_thresh=1, verbosity=0):

    # WIP: We can safely neglect/round up values less than 1 (to avoid div by 0 errors)
    n_in[n_in < 1] = 1.0
    
    n_ini = n_in[0]
    n_fin = np.max(n_in[-1])
    n_max = np.max(n_in)
    n_min = np.min(n_in)

    # TODO/WIP: Handle inf and/or NaN values gracefully
    if any(np.isinf(n_in)):

        ratio_m = np.inf
        ratio_f = (n_fin / max(abs(n_ini), 1)) if n_fin != np.inf and n_ini != np.inf else np.inf

        t_res_idx = np.argwhere((n_in / n_ini) >= res_con)[0]
        t_res = t_in[t_res_idx[0] if len(t_res_idx) > 0 else -1]
        t_max_idx = np.argwhere(np.isinf(n_in))[0]
        t_max = t_in[t_max_idx[0] if len(t_max_idx) > 0 else t_max_idx[0]]

        if not(np.isinf(n_max) and np.isinf(n_fin)) and n_max/n_fin > res_con: # if n_max != n_final, assume energy "burst"
            n_class = 'burst'
        else: # otherwise, runaway instability
            n_class = 'resonance'

        if verbosity >= 8:
            print('ratio_m: ', ratio_m)
            print('ratio_f: ', ratio_f)
            print('t_res: ', t_res)
            print('t_max: ', t_max)
        
        return n_class, ratio_f, ratio_m, t_res, t_max
    
    # Renormalize values to be ratio relative to initial number density in
    n_norm = n_in/n_ini
    n_max_norm = np.max(n_norm)

    n_fit = (n_norm) / (n_max_norm)
    if verbosity >= 8:
        print('t_in:', t_in.shape, t_in.dtype)
        print('n_in:', n_in.shape, n_in.dtype)
        print('n_fit:', n_fit.shape)

    n_fit_max = np.max(n_fit)
    if verbosity >= 8:
        print('n_max:', n_max)
        print('n_min:', n_min)
        print('n_ini:', n_ini)
        print('n_fin:', n_fin)

    x_obs = t_in
    y_obs = n_fit

    H = lambda x,a,b,c: a * (np.sign(x-b)) + c # Heaviside fitting function

    popt, pcov = curve_fit(H,x_obs,y_obs,p0=(1,0,0),bounds=(0, 1),x_scale='jac')
    if verbosity >= 3:
        print('fit = a: %.2f   b: %.2f   c: %.2f' % (popt[0], popt[1], popt[2]))
        print('pcov = \n %s' % pcov)

    # Plug optimized parameters into desired function
    predicted = H(x_obs,popt[0],popt[1],popt[2]) 

    # Use some metric to quantify fit quality (for now mean-squared error)
    lms_err = np.log10(sklearn.metrics.mean_squared_error(y_obs*n_max_norm, predicted*n_max_norm))
    #rms_err = np.sqrt(sklearn.metrics.mean_squared_error(y_obs, predicted))
    if verbosity >= 1:
        print('log-mean-squared error: %.2f' % lms_err)

    ratio_m = n_max / n_ini
    ratio_f = n_fin / n_ini
    t_res_idx = np.argwhere((n_fit / n_fit[0]) >= res_con)
    t_max_idx = np.argwhere(n_fit >= n_fit_max)
    t_res = t_in[t_res_idx[0] if len(t_res_idx) > 0 else -1]
    t_res = t_res if type(t_res) is np.float64 else t_res[0]
    t_max = t_in[0 if len(t_max_idx) == 0 else t_max_idx[0]]
    t_max = t_max if type(t_max) is np.float64 else t_max[0]
    if verbosity >= 8:
        print('ratio_m: ', ratio_m)
        print('ratio_f: ', ratio_f)
        print('t_res: ', t_res)
        print('t_max: ', t_max)

    if lms_err > err_thresh: # if bad fit, assume exponential growth
        if n_max/n_fin > res_con: # if n_max != n_final, assume energy "burst"
            n_class = 'burst'
        else: # otherwise, runaway instability
            n_class = 'resonance'
    else:
        if ratio_f > res_con: # if n_final > n_initial, assume energy injection
            if n_max/n_fin > res_con: # second check to catch energy "burst" form
                n_class = 'burst'
            else: # otherwise, "plateau" shaped energy injection
                n_class = 'injection'
        elif 1./ratio_f > res_con: # if n_initial >> n_final, assume overall damping (probably shouldn't happen)
            n_class = 'damp'
        else: # otherwise, no resonance
            n_class = 'none'
    if verbosity >= 8:
        print('res_class: ', n_class)

    return n_class, ratio_f, ratio_m, t_res, t_max

# Classify amplitude growth given occupation numbers
# Assume input has dimensions of (k x t)
def classify_resonance(params_in, nk_arr, k_span, method='heaviside', verbosity=5):
    # TODO: Either flesh out or remove the unimplemented methods beyond heaviside

    T_min = params_in['T_u'] if 'T_u' in params_in else get_timescales(np.array(params_in['m'], dtype=object), params_in['m_0'], m_u=1)[0]
    t_f   = params_in['t_span'][1]
    times = np.array(get_times(params_in, None))
    res_con = params_in['res_con']
    k_values = np.linspace(k_span[0], k_span[1], params_in['k_num'])
    n_tot = np.sum(nk_arr, axis=0)

    if method == 'binned':
        # Split the plot into 4 < N < 10 bins, where N is given by the number of characteristic
        # timescales included in the total integration time
        N_bins = int(min(max(np.floor(t_f/T_min), 4), 10))
        ln_rescon = max(1., np.log10(res_con))

        nk_binned = binned_statistic(times, np.log10(nk_arr), bins=N_bins, statistic='mean')
        class_res = pd.DataFrame([binned_classifier(nk_binned.statistic[k_i], N_bins, ln_rescon, return_dict=True) for k_i, _ in enumerate(k_values)])
        nk_class = np.array(class_res['label'])
        nk_ratios = np.array(zip(k_values, class_res['ratio_final'] + class_res['ratio_max'] + class_res['baseline']))

        nt_binned = binned_statistic(times, np.log10(n_tot), bins=N_bins, statistic='mean')
        tot_class, ratio_f, ratio_m, base_val = binned_classifier(nt_binned.statistic, N_bins, ln_rescon)
        
        t_res = times[np.where(np.log10(n_tot) >= base_val)][0]
        t_max = times[np.where(np.log10(n_tot) >= base_val+ratio_m)][0]
    elif method == 'peaks':
        # TODO: peak-to-peak classification method?
        return None
    elif method == 'heaviside':
        tot_class, ratio_f, ratio_m, t_res, t_max = heaviside_classifier(times, n_tot, res_con, verbosity=verbosity)
        nk_class_arr = np.array([heaviside_classifier(times, n_k, res_con, verbosity=0) for n_k in nk_arr])
        nk_class = nk_class_arr[:,0]
        nk_ratios = nk_class_arr[:,1:].astype(np.float64)
    elif method == 'window':
        # TODO: Flesh this out
        '''
        # (From plotting subroutine)
        #tot_res = 'resonance' if sum(k_ratio(np.ma.mean, win_sens, params_in['A_sens'])) > res_con else 'soft' if (t_idx_a < t_idx_lim and t_idx_a > 0) else 'none'
        '''
        return None
    elif method == 'cutoff':
        # TODO: Flesh this out
        # k_class: softly classify the level of resonance according to the final/initial mode amplitude ratio, governed by [func, t_sens, and A_sens]
        '''
        # (from k_class)
        k_class = lambda func, t_sens, A_sens, res_con: np.array(['damp' if k_r <= 0.9 else 'none' if k_r <= (1. + np.abs(A_sens)) else 'soft' if k_r <= res_con else 'res' for k_r in k_ratio(func, t_sens, A_sens)])
        # (From get_peak_k_modes)
        tot_res = 'resonance' if sum(k_ratio(np.ma.mean, t_sens, A_sens)) > res_con else 'none'
        '''
        return None
    elif method == 'RMS':
        # TODO: copy-paste RMS stuff from make_amplitudes_plot here
        return None
    elif method == 'avg':
        # TODO: copy-paste running average stuff from make_amplitudes_plot here
        return None
    else:
        print('Error: method=\'%s\' is not a valid option.' % method)
        return None
    # nk_class  : str   = [k]-dim array of classification labels for each mode
    # tot_class : str   = single classification label for n_total
    # nk_ratios : float = [k]-dim array of resonance strength quantified by numerical metrics
    # ratio_f   : float = single value quantifying resonance strength for n_total
    # ratio_m   : float = single value quantifying resonance strength for n_total
    # t_res     : float = time, in given units, when resonance begins
    # t_max     : float = time, in given units, of maximum total n value
    return nk_class, tot_class, nk_ratios, ratio_f, ratio_m, t_res, t_max

# Plot occupation number results
def plot_occupation_nums(params_in, units_in, results_in, numf=None, k_samples=[], times=None, scale_n=False, class_method='heaviside', tex_fmt=False, add_colorbars=False):
    num_plt, _, _, _ = make_occupation_num_plots(params_in, units_in, results_in, numf, k_samples, times, scale_n, class_method, tex_fmt, add_colorbars)
    num_plt.show()

sum_n_k = lambda n_in, k_v: np.sum([n_in(k) for k in k_v], axis=0)
sum_n_p = lambda n_in, p_in, sol_in, k_v, times: np.sum([n_p(k_i, p_in, sol_in, k_v, times, n=n_in) for k_i in range(len(k_v))], axis=0)

def make_occupation_num_plots(params_in, units_in, results_in, numf_in=None, k_samples_in=[], times_in=None, scale_n=True, class_method='heaviside', tex_fmt=False, add_colorbars=False, write_to_params=False):
    k_span = (params_in['k_span'][0], params_in['k_span'][1])
    k_values = get_kvals(params_in, None)
    k_peak, k_mean = get_peak_k_modes(params_in, results_in, k_values)
    fontsize = 16 if tex_fmt else 14

    plot_all_k = True if len(k_samples_in) == 1 and k_samples_in[0] < 0 else False
    if plot_all_k:
        k_samples = [k_i for k_i, k_val in enumerate(k_values)]
    elif len(k_samples_in) <= 0:
        #k_samples = np.geomspace(1,len(k_values),num=5)
        k_samples = [k_i for k_i, k_val in enumerate(k_values) if k_val in [0,1,10,20,50,75,100,125,150,175,200,500,k_peak,k_mean]]
    else:
        k_samples = k_samples_in

    times = get_times(params_in, times_in)
    t_sens = params_in['t_sens']
    A_sens = params_in['A_sens']
    res_con = params_in['res_con']
    numf = numf_in if numf_in is not None else n_k
    
    plt.figure(figsize=(20, 9))

    plt.subplot2grid((2,5), (0,0), colspan=3)

    if add_colorbars:
        c_m, cm_vals, cbar_ticks, cbar_labels, s_m_plt, cm_norm_plt, s_m_cbar, cm_norm_cbar = get_colorbar_params(k_values)

    for s_idx, k_sample in enumerate(k_samples):
        k_s = int(k_sample)
        nk_raw = n_p(k_s, params_in, results_in, k_values, times, n=numf)
        nk_plt = np.ma.masked_greater(nk_raw, params_in['inf_con'])

        if add_colorbars:
            plt.plot(times, nk_plt, label='k='+str(k_values[k_s]), linewidth=1, color=c_m(cm_norm_plt(cm_vals[s_idx])))
        else:
            plt.plot(times, nk_plt, label='k='+str(k_values[k_s]))
    
    plt.title(r'Occupation number per k mode', fontsize=16)
    plt.xlabel(r'Time $[%s]$' % units_in['t'])
    plt.xlim(-1, times[-1]+1)
    plt.ylabel(r'$n[k]$')
    plt.yscale('log')
    #plt.ylim(bottom = 0.1)
    if add_colorbars:
        cbar1 = plt.colorbar(s_m_cbar, label=r'$k$', cmap=c_m, norm=cm_norm_cbar, drawedges=False, location='right', fraction=0.02, pad=0, anchor=(0.0,0.1))
        cbar1.set_ticks(cbar_ticks)
        cbar1.set_ticklabels(cbar_labels)
    else:
        plt.legend()
    plt.grid()

    n_tot = sum_n_p(numf, params_in, results_in, k_values, times)
    n_tot_plt = np.ma.masked_where(np.isinf(n_tot), n_tot)
    
    nk_arr = np.array([n_p(k_i, params_in, results_in, k_values, times, n=numf) for k_i,_ in enumerate(k_values)])
    k_class_arr, tot_class, k_ratio_arr, ratio_f, ratio_m, t_res, t_max = classify_resonance(params_in, nk_arr, k_span, class_method)
    t_res = t_res[0] if type(t_res) is list else t_res
    print('tot_class: %s' % tot_class)
    print('ratio_f: %.2e' % ratio_f)
    print('ratio_m: %.2e' % ratio_m)
    print('t_res: %.3f' % t_res)
    print('t_max: %.3f' % t_max)

    if write_to_params:
        params_in['t_res'] = t_res
        params_in['res_class'] = tot_class
        params_in['res_ratio_f'] = ratio_f
        params_in['res_ratio_m'] = ratio_m
        params_in['k_class_arr'] = k_class_arr
        params_in['k_ratio_arr'] = k_ratio_arr
    #n_res = res_con*sum(k_sens(np.mean, -t_sens))
    n_res = n_tot[np.where(times >= t_res)][0]
    n_max = n_tot[np.where(times >= t_max)][0]
    print('n_res:', n_res)
    print('n_max:', n_max)

    #with plt.xkcd():
    plt.subplot2grid((2,5), (1,0), colspan=3)
    #fig,ax = plt.subplots()
    #plt.plot(np.ma.masked_where(t >= t_res, times), np.ma.masked_where(np.array(n_tot) > res_con*sum(k_sens(np.mean, -t_sens)), n_tot), label='none', color='grey')

    plt.plot(np.ma.masked_greater(times, min(t_res, t_max)), n_tot_plt, label='none', color='grey')
    plt.plot(np.ma.masked_greater(np.ma.masked_less(times, t_max), t_res), n_tot_plt, label='energy injection', color='orange')
    plt.plot(np.ma.masked_less(times, t_res), n_tot_plt, label='resonance', color='red')
    plt.title(r'Occupation Number (total)', fontsize=16)
    plt.xlabel(r'Time $[%s]$' % units_in['t'])
    plt.xlim(-1, times[-1]+1)
    plt.ylabel(r'$n/n_0$' if scale_n else r'$n$')
    plt.yscale('log')
    if add_colorbars:
        # Add and then remove a blank colorbar so that all plots are lined up with those that use colorbar legends
        cbar2 = plt.colorbar(s_m_cbar, alpha=0, location='right', fraction=0.02, pad=0, anchor=(0.0,0.1), drawedges=False, ticks=[])
        cbar2.remove()
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

    print('res  | t: %.2f    n = %.2e' % (t_res, n_res))
    print('max  | t: %.2f    n = %.2e' % (t_max, n_max))
    print('res condition: %s' % res_con)
    #print('t_sens =', win_sens)
    #print('k_ratio sum: ', sum(k_ratio(np.ma.mean, win_sens, params_in['A_sens'])))
    #print('t_lim = %.2f' % times[t_idx_lim])
    print('class = ', params_in['res_class'], '=>', tot_class)
    #print(n_tot)
    
    return plt, params_in, t_res, n_res

P_off = lambda t: np.float64(0)
B_off = lambda t: np.float64(0)
D_off = lambda t: np.float64(0)
C_off = lambda t, pm: np.float64(0)

Alpha = lambda t, k, k0, P, C, D, A_pm: ((C(t, A_pm)*(k*k0) + D(t)) / (1. + P(t))) + (k*k0)**2
Beta  = lambda t, B, P: B(t) / (1. + P(t))

# Plot time-dependent coefficient values of the model
def plot_coefficients(params_in, units_in, P=None, B=None, C=None, D=None, polarization=None, k_unit=None, k_samples=[], times=None, plot_all=True, tex_fmt=False):
    coeff_plt = make_coefficients_plot(params_in, units_in, P, B, C, D, polarization, k_unit, k_samples, times, plot_all, tex_fmt)
    coeff_plt.show()
    
def make_coefficients_plot(params_in, units_in, P_in=None, B_in=None, C_in=None, D_in=None, Cpm_in=None, k_unit=None, k_samples_in=[], times_in=None, plot_all=True, tex_fmt=False):
    k_values = np.linspace(params_in['k_span'][0], params_in['k_span'][1], params_in['k_num'])
    fontsize = 16 if tex_fmt else 14
    P = P_in if P_in is not None else P_off
    B = B_in if B_in is not None else B_off
    D = D_in if D_in is not None else D_off
    C = C_in if C_in is not None else C_off
    Cpm = Cpm_in if Cpm_in is not None else params_in['A_pm']
    k0 = k_unit if k_unit is not None else params_in['k_0'] if 'k_0' in params_in else 1.
    
    if len(k_samples_in) <= 0:
        k_samples = [i for i, k_i in enumerate(k_values) if k_i in [0,1,10,20,50,75,100,125,150,175,200,500]]
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
        plt.yscale('symlog')
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
    func_vals = []
    times = times_in if len(times_in) > 0 else get_times(params_in, times_in)
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
    c_ranges = []
    times = times_in if len(times_in) > 0 else get_times(params_in, times_in)
    
    for c_func, l_root, sign in zip([P, B, C, C, D], 
                                    ['P(t)', 'B(t)', 'C_{%s}(t)' % signstr[+1], 'C_{%s}(t)' % signstr[-1], 'D(t)'], 
                                    [0, 0, +1, -1, 0]):
        if l_root[0] == 'C':
            c_t = c_func(times, sign)
        else:
            c_t = c_func(times)

        if type(c_t) is np.float64:
            c_t     = np.ma.array(c_t + np.full(len(times), 0.0, dtype=float), mask=True)
            c_range = (np.nan, np.nan)
        else:
            label   = r'$%s$' % l_root
            c_range = (min(c_t), max(c_t))

        c_ranges.append(l_root[0] + '(t) range: [%.1e, %.1e]' % c_range + (' for + case ' if sign > 0 else ' for - case ' if sign < 0 else ''))
    
    return c_ranges

def print_coefficient_ranges(params_in, P_in=None, B_in=None, C_in=None, D_in=None, Cpm_in=None, k_unit=None, k_samples_in=[], times_in=None, print_all=False):
    k_values = get_kvals(params_in, None)
    if len(k_samples_in) <= 0:
        k_samples = [i for i, k_i in enumerate(k_values) if k_i in [0,1,10,20,50,75,100,125,150,175,200,500]]
    else:
        k_samples = k_samples_in
    P    = P_in   if P_in   is not None else P_off
    B    = B_in   if B_in   is not None else B_off
    D    = D_in   if D_in   is not None else D_off
    C    = C_in   if C_in   is not None else C_off
    Cpm  = Cpm_in if Cpm_in is not None else params_in['A_pm']
    k0   = k_unit if k_unit is not None else params_in['k_0']
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

def plot_resonance_spectrum(params_in, units_in, results_in, fwd_fn, inv_fn, numf_in=None, class_method='heaviside', tex_fmt=False, plot_max=False):
    res_plt = make_resonance_spectrum(params_in, units_in, results_in, fwd_fn, inv_fn, numf_in, class_method, tex_fmt, plot_max)
    res_plt.show()

def make_resonance_spectrum(params_in, units_in, results_in, fwd_fn, inv_fn, numf_in=None, class_method='heaviside', tex_fmt=False, plot_max=False):
    k_span = (params_in['k_span'][0], params_in['k_span'][1])
    k_values = get_kvals(params_in, None)
    class_colors = {'none': 'lightgrey', 'damp': 'darkgrey', 'burst':'purple', 'resonance': 'red'}
    res_con_in = params_in['res_con']

    t_sens = params_in['t_sens']
    A_sens = params_in['A_sens']
    times  = np.linspace(params_in['t_span'][0], params_in['t_span'][1], params_in['t_num'])

    nk_arr = np.array([n_p(k_i, params_in, results_in, k_values, times, n=numf_in if numf_in is not None else n_k) for k_i,_ in enumerate(k_values)])
    nk_class, tot_class, nk_ratios, ratio_f, ratio_m, t_res, t_max = classify_resonance(params_in, nk_arr, k_span, method=class_method)
    
    plt.figure(figsize = (20,6))
    plt.suptitle(r'Resonance Classification')

    ax = plt.subplot2grid((2,4), (0,0), colspan=2, rowspan=2)
    plt.scatter(k_values, nk_ratios[:,0], c=[class_colors[k_c] if k_c in class_colors else 'pink' for k_c in nk_class])
    if plot_max:
        plt.scatter(k_values, nk_ratios[:,1], c=[class_colors[k_c] if k_c in class_colors else 'pink' for k_c in nk_class], alpha=0.2)
    plt.xlabel(r'$k$%s' % (' [$m_u$]' if params_in['use_mass_units'] else ''))
    plt.xlim(left=-1, right=params_in['k_span'][1] + 1)
    
    plt.ylabel(r'Growth in $n_k$')
    plt.yscale('log') if np.any(np.isfinite(nk_ratios[:,0])) or np.any(np.isfinite(nk_ratios[:,1])) else plt.yscale('symlog')
    plt.grid()

    plt.subplot2grid((2,4), (0,2), colspan=2, rowspan=2)
    class_counts = [(nk_class == class_label).sum() for class_label in class_colors.keys()]
    plt.bar(class_colors.keys(),class_counts,color=class_colors.values())
    plt.xlabel(r'Classification')
    plt.ylabel(r'Count')
    plt.grid()

    plt.tight_layout()
    
    return plt

## Mass-coupling relations for pi-axion / QCD axion
# Solve for pi-axion decay constant, given desired photon coupling [GeV]
F_pi_from_g_x = lambda g_x, l1=1, eps=1: 2*l1*(eps**2) / g_x
g_x_from_F_pi = lambda F_pi, l1=1, eps=1: 2*l1*(eps**2) / F_pi
# NOTE: Adapted from Humberto's modification to AxionLimits notebook, but worth replacing with a more robust relation in the future
# assuming [GeV]^-1 as units
g_x_from_m_x  = lambda m_x, epsilon=1, lambda1=1, theta=1, alpha=(1./137): (8.7e-12)*(alpha)*(epsilon**2)*(theta)*(lambda1)*(m_x**(1/4))
m_x_from_g_x  = lambda g_x, epsilon=1, lambda1=1, theta=1, alpha=(1./137): ((g_x)/((8.7e-12)*(alpha)*(epsilon**2)*(theta)*(lambda1)))**4
# Solve for F_pi, given m_x, assuming above relations [GeV]
# NOTE: We still have a slight discrepancy b/w predicted values and the plotted trendline from Humberto's modification to AxionLimits notebook
F_pi_from_m_x = lambda m_x, l1=1, eps=1, theta=1, alpha=(1./137): F_pi_from_g_x(g_x_from_m_x(m_x, lambda1=l1, epsilon=eps, theta=theta, alpha=alpha), l1=l1, eps=eps)

# TODO: Rescale plot axes if current data doesn't fit in default range
def plot_ALP_survey(params_in, verbosity=0, tex_fmt=True, fit_coupling=False):
    tools_dir = os.path.abspath(os.path.join('./tools'))
    if tools_dir not in sys.path:
        sys.path.append(tools_dir)

    # Shade of purple chosen for visibility against existing plot colors
    res_color  = '#b042f5' if 'res' in params_in['res_class'] else 'grey'

    # Current values for model parameters
    m_u = params_in['m_u']
    F   = params_in['F']
    l1  = params_in['l1']
    eps = params_in['eps']
    m   = params_in['m']
    GeV = 1e9

    if fit_coupling:
        # fit the coupling from the mass, ignoring provided F_pi
        g_in = g_x_from_m_x(m_u, epsilon=eps, lambda1=l1)
    else:
        # use model's F_pi as decay constant
        g_in = g_x_from_F_pi(F_pi=F/GeV)

    m_min = np.min([np.max([np.min([np.min(m_i * 0.5e-1) for m_i in m if len(m_i) > 0]), 1.0e-22]), 1.0e-12])
    m_max = np.max([np.min([np.max([np.max(m_i * 0.5e+1) for m_i in m if len(m_i) > 0]), 1.0e5]), 1.0e-1])
    g_min = np.min([np.max([g_in * 0.5e-1, 1.0e-30]), 1.0e-20])
    g_max = np.max([np.min([g_in * 0.5e+1, 1.0e-4]), 1.0e-9])

    # Import plotting functions from AxionLimits and set up AxionPhoton plot
    from PlotFuncs import FigSetup, AxionPhoton, MySaveFig, BlackHoleSpins, FilledLimit, line_background
    fig,ax = FigSetup(Shape='Rectangular', ylab='$|g_{\pi\gamma\gamma}|$ [GeV$^{-1}$]', mathpazo=True,
                      m_min=m_min, m_max=m_max, g_min=g_min, g_max=g_max)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ## Populate standard AxionPhoton limits plot
    # Plot QCD axion lines and experimental bounds
    AxionPhoton.QCDAxion(ax,C_center=abs(5/3-1.92)*(44/3-1.92)/2,C_width=0.7,vmax=1.1,)
    AxionPhoton.Cosmology(ax)
    AxionPhoton.StellarBounds(ax)
    AxionPhoton.SolarBasin(ax)
    AxionPhoton.Haloscopes(ax,projection=True,BASE_arrow_on=False)
    AxionPhoton.Helioscopes(ax,projection=True)
    AxionPhoton.LSW(ax,projection=True)
    AxionPhoton.LowMassAstroBounds(ax,projection=True)

    # Dark matter astro/cosmo bounds:
    AxionPhoton.ALPdecay(ax,projection=True)
    AxionPhoton.NeutronStars(ax)
    AxionPhoton.AxionStarExplosions(ax)

    ## TODO: Verify we are using the appropriate equation to calculate the g_pi-m_pi relation
    # Reference Lines
    AxionPhoton.piAxion(ax,epsilon=1,lambda1=1,theta=1,label_mass=1e-2,C_logwidth=10,cmap='Greys',fs=18,rot = 6.0,
                    C_center=1,C_width=1.2,vmax=0.9)
    AxionPhoton.piAxion(ax,epsilon=0.5,lambda1=1,theta=1,label_mass=1e-2,C_logwidth=10,cmap='Greys',fs=18,rot = 6.0,
                    C_center=1,C_width=1.2,vmax=0.9)
    AxionPhoton.piAxion(ax,epsilon=0.01,lambda1=1,theta=1,label_mass=1e-2,C_logwidth=10,cmap='Greys',fs=18,rot = 6.0,
                    C_center=1,C_width=1.2,vmax=0.9)
    # This case
    AxionPhoton.piAxion(ax,epsilon=eps,lambda1=l1,theta=1,label_mass=1e-2,C_logwidth=10,cmap='GnBu',fs=18,rot = 6.0,
                    C_center=1,C_width=1.2,vmax=0.9)

    ## Plot star marker for this specific parameter configuration
    # Primary data
    ax1 = plt.gcf().add_subplot()
    ax1.set_xscale('log')
    ax1.set_xlim(xmin, xmax)
    ax1.set_yscale('log')
    ax1.set_ylim(ymin, ymax)
    ax1.axis('off')
    ax1.set_zorder(3)
    #g_u = GeV/F
    if fit_coupling:
        g_u = g_x_from_m_x(m_u, epsilon=eps, lambda1=l1)
    else:
        g_u = g_x_from_F_pi(F_pi=F/GeV, eps=eps, l1=l1)
    ax1.scatter(m_u, g_u, s=2000, c='white', marker='*')
    ax1.scatter(m_u, g_u, s=1000, c=res_color, marker='*')

    # Secondary Data
    ax2 = plt.gcf().add_subplot()
    ax2.set_xscale('log')
    ax2.set_xlim(xmin, xmax)
    ax2.set_yscale('log')
    ax2.set_ylim(ymin, ymax)
    ax2.axis('off')
    ax2.set_zorder(2)
    if len(m[0] > 1):
        for m_v in m[0][1:]:
            if fit_coupling:
                g_v = g_x_from_m_x(m_v, epsilon=eps, lambda1=l1)
            else:
                g_v = g_x_from_F_pi(F_pi=F/GeV, eps=eps, l1=l1)
            ax2.scatter(m_v, g_v, s=1000, c='grey', marker='*')
    
    return plt

logfit = lambda x, a, b, c: a*np.log10(10*x+b)+c

# Solve for F_pi given epsilon (millicharge) and ALP (unit) mass
# TODO: Find a more elegant / robust way to describe this relation
def fit_Fpi(eps_in, m_in, l1_in, fit_QCD=False, verbosity=0):

    ## Fit parameters to be equivalent to QCD axion case
    if fit_QCD:
        # QCD Axion cases
        KSVZ = 1.92
        DFSZ = 0.75
        g_a_from_m_a = lambda m_a, C_ag=KSVZ: 2e-10*C_ag*m_a

        ## Fit parameters to be equivalent to QCD axion case
        c1 = 0.5; c2 = 0.5
        # Solve for dark-quark mass, given desired ALP mass [eV]
        m_Im  = lambda m_a, F_pi=F_pi_from_m_x, eps=1, c1=c1, c2=c2: m_a**2 / (F_pi(m_a, eps=eps)*1e9*(c1+c2))
        # Solve for dark-quark mass, given desired ALP coupling [eV]
        m_Ig  = lambda g_a, F_pi=F_pi_from_g_x, eps=1, c1=c1, c2=c2: m_a**2 / (F_pi(g_a, eps=eps)*1e9*(c1+c2))
        # Solve for model-dependent constants
        C_ag = lambda l1=1, eps=1, a_e=(1./137): -4*np.pi*(l1/a_e)*eps**2
        z_ag = lambda m_a, g_a: 2*(m_a)*(1./(g_a * 10e10))
        
        # TODO: add functionality for the below to be more dynamic
        QCD_model = KSVZ
        
        m_a_in  = m_in   # target ALP mass
        if verbosity >= 4:
            print('Fitting pi-axiverse to QCD axion parameter space')
        # QCD Axion Parameters
        m_a = m_a_in
        g_a = g_a_from_m_a(m_a_in, C_ag=QCD_model)
        z_in = z_ag(m_a, g_a)
        C_in = QCD_model
        
        # Pi-Axiverse Parameters
        pin_g = True
        pin_m = False
        if pin_g:
            m_I_fit = m_Ig(g_a, F_pi=F_pi_from_g_x, eps=eps_in)
            #F_pi_fit = F_pi_from_g_x(g_a_from_m_a(m_a_in, C_ag=C_ag(eps=eps_in, l1=l1_in)), eps=eps_in, l1=l1_in)
            F_pi_fit = F_pi_from_g_x(g_a, eps=eps_in, l1=l1_in)
        elif pin_m:
            m_I_fit = m_Im(m_a_in, F_pi=F_pi_from_m_x, eps=eps_in)
            F_pi_fit = F_pi_from_m_x(m_a_in, eps=eps_in, l1=l1_in)
        else:
            m_I_fit = m_Im(m_a_in, F_pi=F_pi_from_m_x, eps=eps_in)
            F_pi_fit = F_pi_from_g_x(g_a, eps=eps_in, l1=l1_in)
        
        m_piaxi = np.sqrt(((c1 + c2)*m_I_fit)*(F_pi_fit*1e9))
        
        #g_piaxi = g_x_from_m_x(m_piaxi, epsilon=eps_in, lambda1=l1_in)
        #g_piaxi = 2 * l1_in/F_pi_fit * eps_in**2
        g_piaxi = g_x_from_F_pi(F_pi_fit, eps=eps_in, l1=l1_in)
        
        z_piaxi = z_ag(m_piaxi, g_piaxi)
        C_piaxi = C_ag(eps=eps_in, l1=l1_in)
        
        if verbosity >= 7:
            print('QCD    |    m_a  = %.1e [eV]' % m_a)
            print('       |    g_a  = %.1e [GeV^-1]' % g_a)
            print('       |    z_ag = %.1e' % z_in)
            print('       |    C_ag = %.1e' % C_in)
            print('pi-axi |    eps  = %.1e   ----->   F_pi = %.1e [GeV]   ----->   m_I = %.1e [eV]' % (eps_in, F_pi_fit, m_I_fit))
            print('       |    m_pi = %.1e [eV]' % m_piaxi)
            print('       |    g_pi = %.1e [GeV^-1]' % g_piaxi)
            print('       |    z_pi = %.1e' % z_piaxi)
            print('       |    C_pi = %.1e' % C_piaxi)
    else:
        F_pi_fit = F_pi_from_m_x(m_in, eps=eps_in, l1=l1_in)
    
    # Rescale return value to [eV]
    return F_pi_fit * 1e9

# TODO/WIP: Boolean check to determine whether given parameters conform to mass-coupling constraints, within [sens] orders of magnitude
def check_Fpi_fit(eps, m_u, l1, F_in, sens=2., fit_QCD=False, verbosity=0):

    F_fit = fit_Fpi(eps, m_u, l1, fit_QCD, verbosity=0)
    if verbosity >= 9:
        if fit_QCD:
            print('F_pi (QCD):  %.2e' % F_fit)
        else:
            print('F_pi (calc): %.2e' % F_fit)
    F_bool = np.abs(np.log10(F_fit) - np.log10(F_in)) <= sens

    return F_bool

# (Deprecated)
def fit_Fpi_old(eps, m_scale, show_plots=False, verbosity=0, use_old_fit=False):
    if use_old_fit:
        fit_res = fit_crude_epsilon_relation(pts_in=[(0.1,-19.9,eps), (0.5,-18.6,eps), (1,-17.9,eps)], plot_fit=show_plots, verbosity=verbosity)
        F_pi = logfit((m_scale, eps), a=fit_res[0], b=fit_res[1], c=fit_res[2])
    else:
        F_pi = 1./((4.38299e-20)*(eps**2)*(10)*(m_scale**(1/9))/(0.1**2))

    if verbosity >=2:
        print('solving for F_pi: %.1e GeV' % F_pi)

    return F_pi

# (Deprecated)
# Fits a log-linear relation as an approximation to the pi-axion mass / F_pi coupling relationship
# Should eventually replace with more robust epsilon relation
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

def get_units_from_params(params_in, verbosity=0):
    return get_units(unitful_m=params_in['unitful_m'], rescale_m=params_in['rescale_m'], unitful_k=params_in['unitful_k'], rescale_k=params_in['rescale_k'], \
                     unitful_amps=params_in['unitful_amps'], rescale_amps=params_in['rescale_amps'], rescale_consts=params_in['rescale_consts'], \
                     unitful_c=not(params_in['use_natural_units']), unitful_h=not(params_in['use_natural_units']), unitful_G=not(params_in['use_natural_units']), \
                     dimensionful_p=params_in['dimensionful_p'], use_mass_units=params_in['use_mass_units'], use_natural_units=params_in['use_natural_units'], \
                     verbosity=verbosity)

def get_units(unitful_m, rescale_m, unitful_k, rescale_k, unitful_amps, rescale_amps, rescale_consts, dimensionful_p=False,
              unitful_c=False, unitful_h=False, unitful_G=False, use_mass_units=True, use_natural_units=None, verbosity=0):
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
            print('peak resonance at k = %.1f corresponds to photon frequency at %.2e Hz (%s)' % (k_mode_in, Hz_peak, Hz_class))
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
        elif label not in ['res', 'soft', 'burst'] and start_idx is not None:
            start_indices.append(start_idx)
            end_indices.append(i-1)
            start_idx = None
    if start_idx is not None:  # Handle case where last segment reaches the end
        start_indices.append(start_idx)
        end_indices.append(len(k_class_arr)-1)

    # Determine resonance classification
    res_count   = np.ma.masked_where(k_class_arr != 'res',   k_class_arr, copy=True).count()
    burst_count = np.ma.masked_where(k_class_arr != 'burst', k_class_arr, copy=True).count()
    soft_count  = np.ma.masked_where(k_class_arr != 'soft',  k_class_arr, copy=True).count()
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

## Fine Structure Constant corrections
alpha_sm   = lambda t: 1./137
alpha_off  = lambda t: 1.
# (include sum over all surviving species, where lambda(3 or 4) and Lambda(3 or 4) are determined by neutral/charged species)
# TODO/WIP: This is incomplete (only have equation for diagonal terms, missing sign differences in assumed form for off-diagonal contributions)
piaxi_fs = lambda t, lambdas, Lambdas, e, eps, amps, masses, phases, charges, alpha=alpha_sm: \
    alpha(t) * (1 + (2*(e**2))*(eps**2)*np.sum([(l_i)/(L_i**2) * np.abs(amp_i)*np.abs(amp_j) * np.cos(m_i*t + d_i)*np.cos(m_j*t + d_j) \
        for amp_i, m_i, d_i, l_i, L_i, c_i in zip(amps, masses, phases, lambdas, Lambdas, charges) for amp_j, m_j, d_j, _, _, c_j in zip(amps, masses, phases, lambdas, Lambdas, charges) \
            if c_i == c_j]))
    #alpha(t) * (1 + (2*(e**2))*(eps**2)*np.sum([(l_i)/(L_i**2) * np.abs(amp_i)**2 * np.cos(m_i*t + d_i)**2 for amp_i, m_i, d_i, l_i, L_i in zip(amps, masses, phases, lambdas, Lambdas)]))

def get_fs_corrections(params_in):
    e_in  = params_in['e']
    l3_in = params_in['l3']
    l4_in = params_in['l4']
    L3_in = params_in['L3']
    L4_in = params_in['L4']
    m_in  = np.concatenate(params_in['m'], axis=None)
    d_in  = np.concatenate(params_in['d'], axis=None)
    amps_in  = np.concatenate(params_in['amps'], axis=None)
    eps_in = params_in['eps']
    Nr_in  = params_in['N_r']
    Nn_in  = params_in['N_n']
    Nc_in  = params_in['N_c']

    lambdas_in = np.array([l4_in]*Nr_in + [l4_in]*Nn_in + [l3_in]*Nc_in)
    Lambdas_in = np.array([L4_in]*Nr_in + [L4_in]*Nn_in + [L3_in]*Nc_in)
    charges_in = np.array([0]*Nr_in     + [0]*Nn_in     + [1]*Nc_in)

    alpha_corrected = lambda t: piaxi_fs(t, lambdas=lambdas_in, Lambdas=Lambdas_in, e=e_in, eps=eps_in, amps=amps_in, masses=m_in, phases=d_in, charges=charges_in)
    return alpha_corrected

def plot_fs_constant(params_in, verbosity=0, return_plot=False):
    alpha_e = get_fs_corrections(params_in)

    if verbosity >= 5:
        print('Fine Structure Constant Oscillations: ')
        print('    alpha_SM     = %.1e ' % alpha_sm(t=0))
        print('    alpha_e(t=0) = %.1e ' % alpha_e(t=0))

    # Show plot of alpha
    m_ref_in = params_in['m_u']
    fs_t = np.linspace(0, 3./m_ref_in, 100)
    plt.plot(fs_t, [alpha_sm(t) for t in fs_t], label=r'$\alpha_{SM}$')
    plt.plot(fs_t, [alpha_e(t)  for t in fs_t], label=r'$\alpha_{e}(t)$')
    plt.ylabel(r'$\alpha$')
    plt.xlabel(r'$t$')
    plt.grid()
    plt.legend()
    if return_plot:
        return plt
    else:
        plt.show()

## Coupling Constants
# Parity-Odd Interaction (triangle anomaly)
g_anomaly  = lambda F_pi, l1=1., eps=1., fs_in=alpha_sm, t=0: fs_in(t)*(eps**2)*(2*l1)/(F_pi)
# Scalar QED interactions and scattering couplings
g_coupling = lambda Li, li=1., eps=1., fs_in=alpha_off, t=0:  fs_in(t)*(eps**2)*(2*li)/(Li**2)

def get_coupling_constants(params_in, verbosity=0, use_corrected_fs=False):
    l1_in = params_in['l1']
    l2_in = params_in['l2']
    l3_in = params_in['l3']
    l4_in = params_in['l4']
    L3_in = params_in['L3']
    L4_in = params_in['L4']
    eps_in = params_in['eps']
    Fpi_in = params_in['F']
    alpha_e = get_fs_corrections(params_in) if use_corrected_fs else alpha_sm
    # Parity-Even
    g_1 = g_anomaly(F_pi=Fpi_in, l1=l1_in, eps=eps_in) if params_in['N_r'] > 0 else None
    g_2 = g_coupling(Li=1., li=l2_in, eps=eps_in)      if params_in['N_c'] > 0 else None
    g_3 = g_coupling(Li=L3_in, li=l3_in, eps=eps_in, fs_in=alpha_e) if params_in['N_c'] > 0 else None
    g_4 = g_coupling(Li=L4_in, li=l4_in, eps=eps_in, fs_in=alpha_e) if params_in['N_r'] + params_in['N_n'] > 0 else None

    if verbosity >= 7:
        print('g_anomaly = %s   |   triangle anomaly'   % ('%.1e [eV]' % g_1 if g_1 is not None else 'None        '))
        print('g_2       = %s   |   scalar QED'         % ('%.1e [eV]' % g_2 if g_2 is not None else 'None        '))
        print('g_3       = %s   |   charged scattering' % ('%.1e [eV]' % g_3 if g_3 is not None else 'None        '))
        print('g_4       = %s   |   neutral scattering' % ('%.1e [eV]' % g_4 if g_4 is not None else 'None        '))
    
    return g_1, g_2, g_3, g_4

cosmo_stability = lambda m_in, F_pi, eps: 1e-34 * (eps**4) * (m_in/(1e-5))**3 * (1e21/F_pi)**2

# Helper functions to calculate phase differences
def calc_local_phase_diffs(d, include_diags=False, verbosity=0):
    N_r = len(d[0])
    N_n = len(d[1])
    N_c = len(d[2])
    # real neutral, complex neutral, and charged species local phase differences
    d_r_diffs = [np.abs(d[0][i]-d[0][j]) for i in range(N_r) for j in range(i, N_r) if i != j]
    d_n_diffs = [np.abs(d[1][i]-d[1][j]) for i in range(N_n) for j in range(i, N_n) if i != j]
    d_c_diffs = [np.abs(d[2][i]-d[2][j]) for i in range(N_c) for j in range(i, N_c) if i != j]
    if include_diags:
        d_r_diffs += [0.0] * N_r
        d_n_diffs += [0.0] * N_n
        d_c_diffs += [0.0] * N_c

    # real, neutral (total), and charged species local phase differences
    d_real    = d_r_diffs
    d_neutral = (d_r_diffs + d_n_diffs + [np.abs(d[0][i]-d[1][j]) for i in range(N_r) for j in range(N_n)]) if N_r > 0 else d_n_diffs
    d_charged = d_c_diffs
    d_total   = d_r_diffs + d_n_diffs + d_c_diffs # TODO: Fix this

    # mean, variance, max, and min for each category
    mean_d_r = np.mean(d_real)    if N_r > 1 else None
    var_d_r  = np.var(d_real)     if N_r > 1 else None
    max_d_r  = np.max(d_real)     if N_r > 1 else None
    min_d_r  = np.min(d_real)     if N_r > 1 else None
    mean_d_n = np.mean(d_neutral) if N_r + N_n > 1 else None
    var_d_n  = np.var(d_neutral)  if N_r + N_n > 1 else None
    max_d_n  = np.max(d_neutral)  if N_r + N_n > 1 else None
    min_d_n  = np.min(d_neutral)  if N_r + N_n > 1 else None
    mean_d_c = np.mean(d_charged) if N_c > 1 else None
    var_d_c  = np.var(d_charged)  if N_c > 1 else None
    max_d_c  = np.max(d_charged)  if N_c > 1 else None
    min_d_c  = np.min(d_charged)  if N_c > 1 else None
    # and for all local phases in general
    mean_d_tot = np.mean(d_total) if (N_r + N_n + N_c) > 1 else None
    var_d_tot  = np.var(d_total)  if (N_r + N_n + N_c) > 1 else None
    max_d_tot  = np.max(d_total)  if (N_r + N_n + N_c) > 1 else None
    min_d_tot  = np.min(d_total)  if (N_r + N_n + N_c) > 1 else None
    
    if verbosity >= 3:
        print('mean and variance: |δ_i - δ_j|')
        print('  reals       | %s%s%s' % (('mean: %.2f π' % (mean_d_r/np.pi) if mean_d_r is not None else mean_d_r), ('       var: %.2f π' % (var_d_r/np.pi) if var_d_r is not None else ''),
                                       ('       range: [%.2f , %.2f] π' % ((min_d_r/np.pi), (max_d_r/np.pi)) if var_d_r is not None else '')))
        print('  neutrals    | %s%s%s' % (('mean: %.2f π' % (mean_d_n/np.pi) if mean_d_n is not None else mean_d_n), ('       var: %.2f π' % (var_d_n/np.pi) if var_d_n is not None else ''),
                                       ('       range: [%.2f , %.2f] π' % ((min_d_n/np.pi), (max_d_n/np.pi)) if var_d_n is not None else '')))
        print('  charged     | %s%s%s' % (('mean: %.2f π' % (mean_d_c/np.pi) if mean_d_c is not None else mean_d_c), ('       var: %.2f π' % (var_d_c/np.pi) if var_d_c is not None else ''),
                                       ('       range: [%.2f , %.2f] π' % ((min_d_c/np.pi), (max_d_c/np.pi)) if var_d_c is not None else '')))
        print('  total (WIP) | %s%s%s' % (('mean: %.2f π' % (mean_d_tot/np.pi) if mean_d_tot is not None else mean_d_tot), ('       var: %.2f π' % (var_d_tot/np.pi) if var_d_tot is not None else ''),
                                       ('       range: [%.2f , %.2f] π' % ((min_d_tot/np.pi), (max_d_tot/np.pi)) if var_d_tot is not None else '')))

    return (mean_d_r, var_d_r, min_d_r, max_d_r), (mean_d_n, var_d_n, min_d_n, max_d_n), (mean_d_c, var_d_c, min_d_c, max_d_c), (mean_d_tot, var_d_tot, min_d_tot, max_d_tot)

def calc_global_phase_diffs(Th, include_diags=False, verbosity=0):
    N_r = len(Th[0])
    N_n = len(Th[1])
    N_c = len(Th[2])
    # complex neutral and charged species global phases
    Th_n_diffs = [np.abs(Th[1][i]-Th[1][j]) for i in range(N_n) for j in range(i, N_n) if i != j]
    Th_c_diffs = [np.abs(Th[2][i]-Th[2][j]) for i in range(N_c) for j in range(i, N_c) if i != j]
    if include_diags:
        Th_n_diffs += [0.0] * N_n
        Th_c_diffs += [0.0] * N_c

    Th_neutral = Th_n_diffs
    Th_charged = Th_c_diffs
    Th_total   = Th_n_diffs + Th_c_diffs  # TODO: Fix this
    
    mean_Th_n = np.mean(Th_neutral) if N_n > 1 else None
    var_Th_n  = np.var(Th_neutral)  if N_n > 1 else None
    max_Th_n  = np.max(Th_neutral)  if N_n > 1 else None
    min_Th_n  = np.min(Th_neutral)  if N_n > 1 else None
    mean_Th_c = np.mean(Th_charged) if N_c > 1 else None
    var_Th_c  = np.var(Th_charged)  if N_c > 1 else None
    max_Th_c  = np.max(Th_charged)  if N_c > 1 else None
    min_Th_c  = np.min(Th_charged)  if N_c > 1 else None
    mean_Th_tot = np.mean(Th_total) if N_n + N_c > 1 else None
    var_Th_tot  = np.var(Th_total)  if N_n + N_c > 1 else None
    max_Th_tot  = np.max(Th_total)  if N_n + N_c > 1 else None
    min_Th_tot  = np.min(Th_total)  if N_n + N_c > 1 else None

    if verbosity >= 3:
        print('mean and variance: |θ_i - θ_j|')
        print('  neutrals    | %s%s%s' % (('mean: %.2f π' % (mean_Th_n/np.pi) if mean_Th_n is not None else mean_Th_n), ('       var: %.2f π' % (var_Th_n/np.pi) if var_Th_n is not None else ''),
                                       ('       range: [%.2f , %.2f] π' % ((min_Th_n/np.pi), (max_Th_n/np.pi)) if var_Th_n is not None else '')))
        print('  charged     | %s%s%s' % (('mean: %.2f π' % (mean_Th_c/np.pi) if mean_Th_c is not None else mean_Th_c), ('       var: %.2f π' % (var_Th_c/np.pi) if var_Th_c is not None else ''),
                                       ('       range: [%.2f , %.2f] π' % ((min_Th_c/np.pi), (max_Th_c/np.pi)) if var_Th_c is not None else '')))
        print('  total (WIP) | %s%s%s' % (('mean: %.2f π' % (mean_Th_tot/np.pi) if mean_Th_tot is not None else mean_Th_tot), ('       var: %.2f π' % (var_Th_tot/np.pi) if var_Th_tot is not None else ''),
                                       ('       range: [%.2f , %.2f] π' % ((min_Th_tot/np.pi), (max_Th_tot/np.pi)) if var_Th_tot is not None else '')))

    return (None, None, None, None), (mean_Th_n, var_Th_n, min_Th_n, max_Th_n), (mean_Th_c, var_Th_c, min_Th_c, max_Th_c), (mean_Th_tot, var_Th_tot, min_Th_tot, max_Th_tot)
