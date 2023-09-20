import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import display, HTML, Image
from scipy.signal import spectrogram
from scipy.optimize import curve_fit
#from collections import OrderedDict
import time
from datetime import timedelta
import json
import hashlib
from pathlib import Path
import os

signstr = {1: '+', -1: '-', 0: '±'}
color_purple = '#b042f5'
GeV = 1e9

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
    global m, m_u, m_0, m_q, k_0
    global p, p_t, p_0, amps
    global d, Th
    global qm, qc, dqm, eps_c, xi
    global seed, res_con
    
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
              'm': m, 'm_r': m_r, 'm_n': m_n, 'm_c': m_c, 'p': p, 'p_r': p_r, 'p_n': p_n, 'p_c': p_c, 'm_0': m_0, 'm_q': m_q,
              'mu_d': mu_d, 'sig_d': sig_d, 'mu_Th': mu_Th, 'sig_Th': sig_Th, 'k_span': k_span, 'k_num': k_num, 'k_0': k_0,
              't_span': t_span, 't_num': t_num, 'A_sens': A_sens, 't_sens': t_sens, 'res_con': res_con, 'm_u': m_u,
              'unitful_m': unitful_m, 'rescale_m': rescale_m, 'unitful_amps': unitful_amps, 'rescale_amps': rescale_amps, 
              'unitful_k': unitful_k, 'rescale_k': rescale_k, 'rescale_consts': rescale_consts, 'seed': seed}
    
    return params

def get_params():
    return params

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
# Generate a unique but reproduceable hash for the given parameter set
def get_parameter_space_hash(params_in):
    return hashlib.sha1(json.dumps(params_in, sort_keys=True, ensure_ascii=True, cls=NumpyEncoder).encode()).hexdigest()

def save_results(output_dir_in, filename, params_in, results=None, plots=None, save_format='pdf', verbosity=0,
                 save_params=True, save_results=True, save_plots=True, plot_types=['amps', 'nums', 'resonance', 'alp'],
                 test_run=False, scratch_dir='~/scratch'):
    
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
    if save_params and params_in != None:
        params_filename = os.path.join(output_dir, filename + '.json')
        file_list.append(params_filename)
        with open(params_filename, 'w') as f:
            json.dump(params_in, f, sort_keys=True, indent=4, cls=NumpyEncoder, default=str)
    
    # Save results
    if save_results and results != None:
        results_filename = os.path.join(output_dir, filename + '.npy')
        file_list.append(results_filename)
        np.save(results_filename, results)
    
    # Save plots
    doc_formats = ['pdf', 'document', 'doc', 'all']
    tex_formats = ['tex', 'latex', 'all'] # TODO
    img_formats = ['png', 'img', 'image', 'jpg', 'all']
    nbk_formats = ['notebook', 'nb', 'ipynb', 'jupyter', 'all']
    web_formats = ['html', 'web', 'all']
    if save_plots and plots != None:
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

def load_results(output_dir, filename, save_format='pdf'):
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
    params_filename = os.path.join(output_dir, filename, '.json')
    with open(params_filename, 'r') as f:
        params = json.load(f, cls=NumpyEncoder)
    
    # Load results
    results_filename = os.path.join(output_dir, filename, '.npy')
    results = np.load(results_filename)
    
    # Load plots
    plots = []
    if save_format == 'png':
        i = 0
        while os.path.exists(os.path.join(output_dir, filename, f'_plot_{i}.png')):
            plots.append(plt.imread(os.path.join(output_dir, filename, f'_plot_{i}.png')))
            i += 1
    
    return params, results, plots

## Identify the k mode with the greatest peak amplitude, and the mode with the greatest average amplitude
def get_peak_k_modes(results_in):
    global k_func, k_sens, k_ratio, k_class, k_peak, k_mean, tot_res

    # k_func : apply [func] on the time-series for each k mode, e.g. max or mean
    k_func = lambda func: np.array([k_fval for k_fi, k_fval in enumerate([func(np.abs(results_in[k_vi][0][:])) for k_vi, k_v in enumerate(k_values)])])
    
    # k_sens : apply [k_func] but limit the time-series by [sens], e.g. sens = 0.1 to look at the first 10%. Negative values look at the end of the array instead.
    win_min  = lambda sens: int(t_num*(1./2)*np.abs(((1. - sens)*np.sign(sens) + (1. - sens))))  # min value / left endpoint (of the sensitivity window over which to average)
    win_max  = lambda sens: int(t_num*(1./2)*np.abs(((1. + sens)*np.sign(sens) + (1. - sens))))  # max value / right endpoint
    k_sens = lambda func, sens: np.array([k_fval for k_fi, k_fval in enumerate([func(np.abs(results_in[k_vi][0][win_min(sens):win_max(sens)])) for k_vi, k_v in enumerate(k_values)])])
    
    # k_ratio: apply [k_func] to each k mode and then return the ratio of the final vs. initial ampltidues (sensitive to a windowed average specified by [sens])
    k_ratio = lambda func, t_sens, A_sens: np.array([k_f/k_i for k_f, k_i in zip(k_sens(func, t_sens), k_sens(func, -t_sens))])
    
    # k_class: softly classify the level of resonance according to the final/initial mode amplitude ratio, governed by [func, t_sens, and A_sens]
    k_class = lambda func, t_sens, A_sens: np.array(['damp' if k_r <= 0.9 else 'none' if k_r <= (1. + np.abs(A_sens)) else 'semi' if k_r <= res_con else 'res' for k_r in k_ratio(func, t_sens, A_sens)])

    # k mode(s) with the largest contributions to overall number density growth
    k_peak = k_values[np.ma.argmax(k_func(max))]
    k_mean = k_values[np.ma.argmax(k_func(np.ma.mean))]
    
    # store max, all-time mean, and late-time mean for each k-mode locally
    params['k_peak_arr']  = k_func(max)
    params['k_mean_arr']  = k_func(np.mean)
    params['k_sens_arr']  = k_sens(np.mean, t_sens)
    params['k_class_arr'] = k_class(np.mean, t_sens, A_sens)

    tot_res = 'resonance' if sum(k_ratio(np.ma.mean, t_sens, A_sens)) > res_con else 'none'
    params['class'] = tot_res
    
    return k_peak, k_mean

# Plot the amplitudes (results of integration)
def plot_amplitudes(params_in, units_in, k_samples=[], plot_Adot=True):
    plt = make_amplitudes_plot(params_in, units_in, k_samples, plot_Adot)
    plt.show()
    
def make_amplitudes_plot(params_in, units_in, results_in, k_samples=[], plot_Adot=True):
    if len(k_samples) <= 0:
        #k_samples = np.geomspace(1,len(k_values),num=5)
        k_samples = [i for i, k_i in enumerate(k_values) if k_i in [0,1,10,50,100,150,200,500,k_peak,k_mean]]
    times = t

    xdim = 5
    if plot_Adot:
        ydim = 3 
    else:
        ydim = 2

    plt.figure(figsize=(4*xdim, 4*ydim))

    #plt.subplot(2,1,1)
    plt.subplot2grid((ydim,xdim), (0,0), colspan=3)
    #plot_colors = cm.get_cmap('plasma').jet(np.linspace(0,1,len(k_samples)))
    for k_idx, k_sample in enumerate(k_samples):
        k_s = int(k_sample)
        #print(results_in[k_s, 0])
        plt.plot(times, results_in[k_s][0], label='k='+str(k_values[k_s]))
    plt.title('Evolution of the mode function $A_{'+signstr[0]+'}$(k)')
    plt.xlabel('Time [%s]' % units_in['t'])
    plt.ylabel('$A_{'+signstr[0]+'}$(k)')
    plt.yscale('log')
    plt.legend()
    plt.grid()

    #plt.subplot(2,1,2)
    plt.subplot2grid((ydim,xdim), (1,0), colspan=3)
    plt.plot(times, [sum([np.abs(results_in[i][0][t_i])**2 for i in range(len(k_values))]) for t_i in range(len(times))])
    plt.title('Evolution of the (total) power for $A_{'+signstr[0]+'}$')
    plt.xlabel('Time [%s]' % units_in['t'])
    plt.ylabel('$|A_{'+signstr[0]+'}|^2$')
    plt.yscale('log')
    plt.grid()


    if plot_Adot:
        #plt.subplot(2,1,2)
        plt.subplot2grid((ydim,xdim), (2,0), colspan=3)
        plt.plot(times, [sum([results_in[i][1][t_i] for i in range(len(k_values))]) for t_i in range(len(times))], color='g', label='total')
        plt.plot(times, [results_in[list(k_values).index(k_mean)][1][t_i] for t_i in range(len(times))], color='y', label='k = %d (mean)' % k_mean)
        plt.plot(times, [results_in[list(k_values).index(k_peak)][1][t_i] for t_i in range(len(times))], color='orange', label='k = %d (peak)' % k_peak)
        plt.title('Evolution of the (total) change in amplitude for A'+signstr[0])
        plt.xlabel('Time [%s]' % units_in['t'])
        plt.ylabel('$\dot{A}_{'+signstr[0]+'}$')
        #plt.yscale('log')
        plt.legend()
        plt.grid()

    #print(len(d[0]))

    textstr1, textstr2 = print_param_space(params_in, units_in)

    plt.subplot2grid((ydim,xdim), (max(0,ydim-3),3), rowspan=(3 if plot_Adot else 2))
    plt.text(0.15, 0.2, textstr1, fontsize=14)
    plt.axis('off')

    plt.subplot2grid((ydim,xdim), (max(0,ydim-3),4), rowspan=(3 if plot_Adot else 2))
    plt.text(0, 0.2, textstr2, fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    
    return plt

# Plot occupation number results
def plot_occupation_nums(params, units, results, numf, omega, k_samples=[], scale_n=True):
    plt = make_occupation_num_plots(params, units, results, numf, omega, k_samples, scale_n)
    plt.show()

sum_n_k = lambda n, w, solns, times: np.array([sum([n(i, w, solns)[t_i] for i in range(len(k_values))]) for t_i in range(len(times))])

def make_occupation_num_plots(params_in, units_in, results_in, numf, omega, k_samples_in=[], scale_n=True):
    global params
    if len(k_samples_in) <= 0:
        #k_samples = np.geomspace(1,len(k_values),num=5)
        k_samples = [i for i, k_i in enumerate(k_values) if k_i in [0,1,10,20,50,75,100,125,150,175,200,500,k_peak,k_mean]]
    else:
        k_samples = k_samples_in
    times = t
    
    plt.figure(figsize=(20, 9))

    plt.subplot2grid((2,5), (0,0), colspan=3)
    for k_sample in k_samples:
        k_s = int(k_sample)
        plt.plot(times, numf(k_s, omega, results_in), label='k='+str(k_values[k_s]))
    plt.title('Occupation number per k mode', fontsize=16)
    plt.xlabel('Time $[%s]$' % units_in['t'])
    #plt.xlim(0,0.2)
    plt.ylabel('n[k]/$n_0$' if scale_n else 'n[k]')
    plt.yscale('log'); #plt.ylim(bottom = 0.1)
    plt.legend()
    plt.grid()

    n_tot = sum_n_k(numf, omega, results_in, times)
    if scale_n:
        n_tot /= abs(n_tot[0])
        n_tot += max(0, np.sign(n_tot[0]))  # Placeholder fix for negative n

    t_res_i = np.ma.argmax(np.array(n_tot) > res_con)
    t_res   = t[t_res_i]
    n_res   = n_tot[t_res_i]
    params['res_class'] = tot_res
    params['t_res'] = t_res
    #n_res = res_con*sum(k_sens(np.mean, -t_sens))

    #with plt.xkcd():
    plt.subplot2grid((2,5), (1,0), colspan=3)
    #fig,ax = plt.subplots()
    #plt.plot(np.ma.masked_where(t >= t_res, times), np.ma.masked_where(np.array(n_tot) > res_con*sum(k_sens(np.mean, -t_sens)), n_tot), label='none', color='grey')
    plt.plot(np.ma.masked_greater_equal(times, t_res), n_tot, label='none', color='grey')
    plt.plot(np.ma.masked_less(times, t_res), n_tot, label='resonance', color='blue')
    plt.title('Occupation Number (total)', fontsize=16)
    plt.xlabel('Time $[%s]$' % units_in['t'])
    #plt.xlim(0,0.1)
    plt.ylabel('n')
    plt.yscale('log')
    plt.legend()
    plt.grid()

    textstr1, textstr2 = print_param_space(params_in, units_in)

    plt.subplot2grid((2,5), (0,3), rowspan=2)
    plt.text(0.15, 0.1, textstr1, fontsize=14)
    plt.axis('off')

    plt.subplot2grid((2,5), (0,4), rowspan=2)
    plt.text(0, 0.1, textstr2, fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    
    return plt, params

P_off = lambda t: np.float64(0)
B_off = lambda t: np.float64(0)
D_off = lambda t: np.float64(0)
C_off = lambda t, pm: np.float64(0)

Alpha = lambda t, k, k0, P, C, D, A_pm: ((C(t, A_pm)*(k*k0) + D(t)) / (1. + P(t))) + (k*k0)**2
Beta  = lambda t, B, P: B(t) / (1. + P(t))

# Plot time-dependent coefficient values of the model
def plot_coefficients(params_in, units_in, P=None, B=None, C=None, D=None,  k_samples=[]):
    plt = make_coefficients_plot(params_in, units_in, P_in=P, B_in=B, C_in=C, D_in=D, k_samples_in=k_samples)
    plt.show()
    
def make_coefficients_plot(params_in, units_in, P_in=None, B_in=None, C_in=None, D_in=None, Cpm_in=None, k_unit=None, k_samples_in=[], plot_all=True):
    global k_0
    P = P_in if P_in is not None else P_off
    B = B_in if B_in is not None else B_off
    D = D_in if D_in is not None else D_off
    C = C_in if C_in is not None else C_off
    Cpm = Cpm_in if Cpm_in is not None else params['A_pm']
    k0 = k_unit if k_unit is not None else k_0
    
    if len(k_samples_in) <= 0:
        k_samples = [i for i, k_i in enumerate(k_values) if k_i in [0,1,10,20,50,75,100,125,150,175,200,500,k_peak,k_mean]]
    else:
        k_samples = k_samples_in
    plt.figure(figsize = (20,9))
    times = t

    plt.subplot2grid((2,5), (0,0), colspan=3, rowspan=1)

    for (c_t, c_label) in get_coefficient_values(P, B, C, D, times):
        plt.plot(times, c_t, label=c_label)

    plt.xlabel('Time $[%s]$' % units_in['t'])
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
            alpha_s = np.ma.array(np.full(len(t), 0.0, dtype=np.float64), mask=True)
            plt.plot(times, alpha_s, label=label_a)
        else:
            for k_sample in k_samples:
                k_s = int(k_sample)
                alpha_s = Alpha(times, k_values[k_s], k0, P, C, D, Cpm)
                if type(alpha_s) is np.float64:
                    label_a = r'[[$\alpha_k(t)$  k=%d]]' % k_values[k_s]
                    alpha_s = np.ma.array(alpha_s + np.full(len(t), 0.0, dtype=np.float64), mask=True)
                    plt.plot(times, alpha_s, label=label_a)
                else:
                    label_a = r'$\alpha_k(t)$  k=%d' % k_values[k_s]
                    plt.plot(times, alpha_s, label=label_a)

        beta_s = Beta(times, P, B)
        if type(beta_s) is np.float64:
            label_b = r'[[$\beta(t)$]]'
            beta_s  = np.ma.array(beta_s + np.full(len(t), 0.0, dtype=np.float64), mask=True)
        else:
            label_b = r'$\beta(t)$'
        plt.plot(times, beta_s, label=label_b)

        plt.grid()
        plt.xlabel('Time $[%s]$' % units_in['t'])
        plt.yscale('log')
        plt.title(r'$\ddot{A}_{k,\pm} + \beta(t)\dot{A}_{k,\pm} + \alpha_{k}(t)A_{k,\pm} = 0$', fontsize=16)
        plt.legend()

    # Write parameter space configuration next to plot
    textstr1, textstr2 = print_param_space(params_in, units_in)

    plt.subplot2grid((2,5), (0,3), rowspan=2)
    plt.text(0.15, 0.1, textstr1, fontsize=14)
    plt.axis('off')

    plt.subplot2grid((2,5), (0,4), rowspan=2)
    plt.text(0, 0.1, textstr2, fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    
    return plt

def get_coefficient_values(P, B, C, D, times_in=[]):
    global params, t
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
            label   = '[[$' + l_root + '$]]'
            c_t     = np.ma.array(c_t + np.full(len(t), 0.0, dtype=float), mask=True)
            c_range = (np.nan, np.nan)
        else:
            label   = '$' + l_root + '$'
            c_range = (min(c_t), max(c_t))

        func_vals.append((c_t, label))
    return func_vals

def get_coefficient_ranges(P, B, C, D, k_samples, times_in=[]):
    global params, t
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
            label   = '$' + l_root + '$'
            c_range = (min(c_t), max(c_t))

        c_ranges.append(l_root[0] + '(t) range: [%.1e, %.1e]' % c_range + (' for + case ' if sign > 0 else ' for - case ' if sign < 0 else ''))
    
    return c_ranges

def print_coefficient_ranges(P_in=None, B_in=None, C_in=None, D_in=None, Cpm_in=None, k_unit=None, k_samples_in=[], print_all=False):
    global params
    if len(k_samples_in) <= 0:
        k_samples = [i for i, k_i in enumerate(k_values) if k_i in [0,1,10,20,50,75,100,125,150,175,200,500,k_peak,k_mean]]
    else:
        k_samples = k_samples_in
    P    = P_in   if P_in   is not None else P_off
    B    = B_in   if B_in   is not None else B_off
    D    = D_in   if D_in   is not None else D_off
    C    = C_in   if C_in   is not None else C_off
    Cpm  = Cpm_in if Cpm_in is not None else params['A_pm']
    k0   = k_unit if k_unit is not None else k_0
    times = t
    
    for c_range_str in get_coefficient_ranges(P, B, C, D, k_samples):
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

def plot_resonance_spectrum(params_in, units_in, fwd_fn, inv_fn):
    plt = make_resonance_spectrum(params_in, units_in, fwd_fn, inv_fn)
    plt.show()

def make_resonance_spectrum(params_in, units_in, fwd_fn, inv_fn):
    class_colors = {'none': 'lightgrey', 'damp': 'darkgrey', 'semi': 'blue', 'res': 'red'}
    
    plt.figure(figsize = (20,6))
    plt.suptitle('Resonance Classification')

    ax = plt.subplot2grid((2,4), (0,0), colspan=2, rowspan=2)
    plt.scatter(k_values, k_ratio(np.mean, t_sens, A_sens), c=[class_colors[k_c] if k_c in class_colors else 'orange' for k_c in k_class(np.mean, t_sens, A_sens)])
    plt.xlabel('$k$')
    axT = ax.secondary_xaxis('top', functions=(fwd_fn, inv_fn))
    axT.set_xlabel('$f_{\gamma}$ (Hz)')
    plt.ylabel('Growth in $n_k$')
    plt.yscale('log')
    plt.grid()

    plt.subplot2grid((2,4), (0,2), colspan=2, rowspan=2)
    class_counts = [(np.array(k_class(np.mean, t_sens, A_sens)) == class_label).sum() for class_label in class_colors.keys()]
    plt.bar(class_colors.keys(),class_counts,color=class_colors.values())
    plt.xlabel('Classification')
    plt.ylabel('Count')
    plt.grid()

    plt.tight_layout()
    
    return plt

def plot_ALP_survey(verbosity=0):
    plt.figure(figsize = (16,12))
    #plt.suptitle('ALP Survey Results')

    plt.subplot2grid((1,1), (0,0), colspan=1, rowspan=1)
    xmin, xmax = (1e-12, 1e7)   # Scale limits
    ymin, ymax = (1e-21, 2e-6)   # Scale limits
    res_color  = color_purple
    plot_masses = True
    show_mass_ranges = False

    ## Interactive mode (WIP)
    with plt.ion():
        # Log-scaled axes
        ax = plt.gca()
        ax.minorticks_on()
        ax.set_xlabel('$m_a\quad[eV]$',fontsize=30)
        ax.set_xlim(xmin, xmax)
        ax.set_xscale('log')
        ax.set_ylabel('$|g_{a\gamma}|\quad[GeV^{-1}]$',fontsize=30)
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

def plot_parameter_space():
    return None

def fit_epsilon_relation(pts_in=[], plot_fit=True):
    pts_x = [x for x,y in pts_in]
    pts_y = [y for x,y in pts_in]

    logfit = lambda t, a, b, c: a*np.log10(10*t+b)+c
    xspace = np.linspace(0.01, 1.5, 150)

    popt, pcov = curve_fit(logfit, np.array(pts_x),np.array(pts_y),p0=(2,0,-19.9))

    if plot_fit:
        plt.plot(pts_x, pts_y, marker='x', linewidth=0, ms=10, color='red')
        plt.plot(xspace, logfit(xspace, a=2, b=0, c=-19.9), color='blue')
        plt.plot(xspace, logfit(xspace, a=popt[0], b=popt[1], c=popt[2]), color='green')

        plt.ylim(-21, -17)
        plt.xlim(0, 1.1)
        plt.yscale('exp')
        plt.grid()
        plt.show()
    
    return popt

# Helper function to pretty print parameters of model alongside plots
def print_param_space(params, units_in, case='full'):
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
    
    if case == 'simple':
        textstr1 = '\n'.join((
                r'$A_\pm(t = 0)=%.2f$' % (params['A_0'], ),
                r'$\dot{A}_\pm (t = 0)=%.2f$' % (params['Adot_0'], ),
                r'$\pm=%s$' % (signstr[params['A_pm']], ),
                '\n',
                r'$\varepsilon=%.0e%s$' % (params['eps'], ),
                r'$F_{\pi}=%.0e%s$' % (params['F'], units['F']),
                '\n',
                r'$\lambda_1=%d%s$' % (params['l1'], units['lambda']),
                r'$\lambda_2=%d%s$' % (params['l2'], units['lambda']),
                r'$\lambda_3=%d%s$' % (params['l3'], units['lambda']),
                r'$\lambda_4=%d%s$' % (params['l4'], units['lambda']),
                r'$\Lambda_3=%.0e%s$' % (params['L3'], units['Lambda']),
                r'$\Lambda_4=%.0e%s$' % (params['L4'], units['Lambda']),
                '\n',
                r'$\Delta k=%.2f%s$' % (params['k_step'], units['k']),
                r'$k \in \left[%d, %d\\right]$' % (params['k_span[0]'], params['k_span'][1]),

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
                '' if units['m'] == 'eV' else r'$m_{u} = %.2e\quad[eV]$' % (m_u, ),
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

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)
