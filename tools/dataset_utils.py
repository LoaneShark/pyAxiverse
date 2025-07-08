## Scan log files for large batch runs and identify errored/missing ARGFILE lines
## (Optionally rename log files accordingly)

# TODO: Print total number of errored / missing / completed files
# TODO: Identify missing logs (bad SLURM output path?) by matching .json filenames to print statements in log files
# TODO: Write code to count up parameter comfigurations at the end, and ensure it matches ARGFILE.

import os
import shutil
import glob
import re
import numpy as np
import pandas as pd
import argparse
import matplotlib as mpl
import swifter

#np.warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

from swifter import set_defaults
set_defaults(progress_bar=False)

use_tex = True
if use_tex:
    mpl.use('agg')

import matplotlib.pyplot as plt
import seaborn as sns
from ..pyaxi_utils import load_all, load_case, get_units_from_params, classify_resonance, fit_Fpi, \
                        check_Fpi_fit, calc_local_phase_diffs, calc_global_phase_diffs
from ..pyaxi_utils import n_p, n_k, cosmo_stability


# LaTeX Formatting for Plots
if use_tex:
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
    })
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

    SMALL_SIZE  = 8
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 20

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    '''
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the x tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the y tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', labelsize=BIGGER_SIZE, titlesize=BIGGER_SIZE)  # fontsize of the figure title
    '''


## --------------------------------------------------------------------



# Set dtypes and print formats for each column
param_map = {
    "config_name":       [str,             None],
    "file_name":         [str,             None],
    "units":             ['object',        None],
    "A_0":               [np.float32,      None],
    "A_pm":              [np.int32,        None],
    "A_sens":            [np.float32,      None],
    "Adot_0":            [np.float32,      None],
    "F":                 [np.float64,      'sci'],
    "G":                 [np.float64,      'sci'],
    "L3":                [np.float64,      'sci'],
    "L4":                [np.float64,      'sci'],
    "N_c":               [np.int32,        None],
    "N_n":               [np.int32,        None],
    "N_r":               [np.int32,        None],
    "N_tot":             [np.int32,        None],
    "amps":              [np.ndarray,      'sci'],
    "c":                 [np.float32,      'sci'],
    "d":                 [np.ndarray,      'pi'],
    "Th":                [np.ndarray,      'pi'],
    "dqm":               [np.ndarray,      'sci'],
    "e":                 [np.float32,      None],
    "eps":               [np.float64,      'sci'],
    "eps_c":             [np.ndarray,      'int'],
    "g_pi":              [np.float64,      'sci'],
    "g_qed":             [np.float64,      'sci'],
    "g_3":               [np.float64,      'sci'],
    "g_4":               [np.float64,      'sci'],
    "h":                 [np.float64,      'sci'],
    "disable_P":         [bool,            None],
    "disable_B":         [bool,            None],
    "disable_C":         [bool,            None],
    "disable_D":         [bool,            None],
    "int_method":        [str,             None],
    "jupyter":           ['Int64',         None],
    "k_0":               [np.float32,      'sci'],
    "k_class_arr":       [np.ndarray,      None],
    "k_mean_arr":        [np.ndarray,      None],
    "k_num":             [np.int32,        'int'],
    "k_peak_arr":        [np.ndarray,      'sci'],
    "k_ratio_arr":       [np.ndarray,      'sci'],
    #"k_sens_arr":        [np.ndarray,      None],
    "k_span":            [np.ndarray,      'float'],
    "l1":                [np.int32,        None],
    "l2":                [np.int32,        None],
    "l3":                [np.int32,        None],
    "l4":                [np.int32,        None],
    "m":                 [np.ndarray,      'sci'],
    "m_0":               [np.float64,      'sci'],
    "m_c":               [np.ndarray,      'sci'],
    "m_n":               [np.ndarray,      'sci'],
    "m_q":               [np.float64,      'sci'],
    "m_r":               [np.ndarray,      'sci'],
    "m_u":               [np.float64,      'sci'],
    "mem_per_core":      ['Int64',         None],
    "mem_per_node":      ['Int64',         None],
    "mu_Th":             [np.float32,      'pi'],
    "mu_d":              [np.float32,      'pi'],
    "num_cores":         [np.int32,        None],
    "p":                 [np.ndarray,      'sci'],
    "p_0":               [np.float64,      'sci'],
    "p_c":               [np.ndarray,      'sci'],
    "p_n":               [np.ndarray,      'sci'],
    "p_r":               [np.ndarray,      'sci'],
    "p_t":               [np.float64,      'sci'],
    "parallel":          [bool,            None],
    "job_qos":           [str,             None],
    "job_partition":     [str,             None],
    "qc":                [np.ndarray,      'float'],
    "qm":                [np.ndarray,      'sci'],
    "res_band":          [np.ndarray,      'sci'],
    "res_band_class":    [str,             None],
    "res_class":         [str,             None],
    "res_con":           [np.float32,      None],
    "inf_con":           [np.float64,      None],
    "res_freq":          [str,             None],
    "res_freq_class":    ['object',        None],
    "res_freq_label":    [str,             None],
    "res_ratio_f":       [np.float64,      'sci'],
    "res_ratio_m":       [np.float64,      'sci'],
    "rescale_amps":      [bool,            None],
    "rescale_consts":    [bool,            None],
    "rescale_k":         [bool,            None],
    "rescale_m":         [bool,            None],
    "seed":              [str,             None],
    "sig_Th":            [np.float32,      'pi'],
    "sig_d":             [np.float32,      'pi'],
    "t_num":             [np.int32,        None],
    "t_res":             [np.float32,      'float'],
    "t_sens":            [np.float32,      'sci'],
    "t_span":            [np.ndarray,      'float'],
    "t_u":               [np.float64,      'sci'],
    "T_r":               [np.float32,      'float'],
    "T_n":               [np.float32,      'float'],
    "T_c":               [np.float32,      'float'],
    "T_u":               [np.float32,      'float'],
    "use_natural_units": [bool,            None],
    "use_mass_units":    [bool,            None],
    "dimensionful_p":    [bool,            None],
    "time_elapsed":      [str,             None],
    "unitful_amps":      [bool,            None],
    "unitful_k":         [bool,            None],
    "unitful_m":         [bool,            None],
    "xi":                [np.ndarray,      'float'],
}

param_tex = {
    "A_0":               r'$A_{0}$',
    "A_pm":              r'$A_{\pm}$',
    "Adot_0":            r'$\dot{A}_{\pm}$',
    "F":                 r'$F_{\pi}$',
    "G":                 r'$G$',
    "L3":                r'$\Lambda_{3}$',
    "L4":                r'$\Lambda_{4}$',
    "N_c":               r'$N_{c}$',
    "N_n":               r'$N_{n}$',
    "N_r":               r'$N_{r}$',
    "N_tot":             r'$N_{f}$',
    "amps":              r'$\pi_{i}$',
    "c":                 r'$c$',
    "d":                 r'$\delta_{\pi_{i}}$',
    "Th":                r'$\Theta_{\pi_{i}}$',
    "dqm":               r'$m_{q_{i}}$',
    "e":                 r'$e_{C}$',
    "eps":               r'$\varepsilon$',
    "eps_c":             r'$\xi_{\pm,\pi_{c,i}}$',
    "g_pi":              r'$g_{\pi\gammma\gamma}$',
    "g_qed":             r'$g_{QED}$',
    "g_3":               r'$g_{3}$',
    "g_4":               r'$g_{4}$',
    "h":                 r'$\hbar$',
    "k_0":               r'$k_{0}$',
    "k_class_arr":       [np.ndarray,      None],
    "k_mean_arr":        r'$\bar{n_{k}}$',
    "k_num":             r'$N_{\Delta k}$',
    "k_peak_arr":        r'$n_{k,max}$',
    "k_ratio_arr":       r'$n_{k,t_{f}}/n_{k,t_{0}}$',
    "l1":                r'$\lambda_{1}$',
    "l2":                r'$\lambda_{2}$',
    "l3":                r'$\lambda_{3}$',
    "l4":                r'$\lambda_{4}$',
    "m":                 r'$m_{\pi_{i}}$',
    #"m_0":               [np.float64,      'sci'],    # conversion factor from m_u to eV
    "m_c":               r'$m_{\pi_{c,i}}$',
    "m_n":               r'$m_{\pi_{n,i}}$',
    "m_q":               r'$\mathcal{O}(m_{I})$',
    "m_r":               r'$m_{\pi_{r,i}}$',
    "m_u":               r'$m_{0}$',
    "mu_Th":             r'$\mu_{\Theta_{i}}$',
    "mu_d":              r'$\mu_{\delta_{i}}$',
    "p":                 r'$\rho_{\pi_{i}}$',
    #"p_0":               [np.float64,      'sci'],    # conversion factor from m_u to eV
    "p_c":               r'$\rho_{\pi_{c,i}}$',
    "p_n":               r'$\rho_{\pi_{n,i}}$',
    "p_r":               r'$\rho_{\pi_{r,i}}$',
    "p_t":               r'$\rho_{DM}$',
    "qc":                r'$c_{i}$',
    "m_q":               r'$m_{I-III}$',
    "res_con":           r'$n_{res}$',
    "inf_con":           r'$n_{\inf}$',
    "res_freq":          r'$f_{res}$',
    "res_ratio_f":       r'$n_{k,t_{f}}/n_{k,t_{i}}$',
    "res_ratio_m":       r'$n_{k,t_{max}}/n_{k,t_{i}}$',
    "sig_Th":            r'$\sigma_{\Theta_{i}}$',
    "sig_d":             r'$\sigma_{\delta_{i}}$',
    "t_num":             r'$N_{\Delta t}$',
    "t_res":             r'$t_{res}$',
    "t_u":               r'$t_{0}$',
    "T_r":               r'$\tau_{r}$',
    "T_n":               r'$\tau_{n}$',
    "T_c":               r'$\tau_{c}$',
    "T_u":               r'$\tau_{0}$',
    "time_elapsed":      r'$T_{el.}$',
    "xi":                r'$\xi_{\pi_{c,i}}$',
}

dtype_map = {key: val[0] for key,val in param_map.items()}
print_fmt = {key: val[1] for key,val in param_map.items()}

#df[["A"]].describe().applymap(lambda x: f"{x:0.3f}")
def df_fmt(x, fmt=None):
    if pd.isna(x):
        return ''
    if type(x) in [float, np.float32, np.float64] or fmt in ['sci', 'pi', 'float']:
        if fmt == 'sci':
            return f'{x:.1e}'
        elif fmt == 'pi':
            xp = x/np.pi
            return f'{xp:0.1f}'
        else:
            return f'{x:0.1f}'
    elif type(x) in [int, np.int32, np.int64, 'Int64'] or fmt in ['int']:
        if x == 'None':
            return None
        else:
            return f'{x:d}'
    elif type(x) is str:
        if x.lower() == 'nan':
            return ''
        else:
            return x
    else:
        #print(x, type(x))
        return x

# WIP: Pretty printing dataframes
def df_pprint(df_in):
    df = df_in.copy()
    for param in df:
        df[param].apply(df_fmt, args=(print_fmt[param],))
    return df

def df_describe(df_in, use_pprint=False, include='all'):
    if use_pprint:
        return df_pprint(df_in.describe(include=include))
    else:
        return df_in.describe(include=include)



## Organize parameters
# Parameters related to computational complexity and runtime performance
performance_inputs = ['num_cores', 'jupyter', 'parallel', 'use_mass_units', 'use_natural_units', 'k_num', 't_num', 'N_r', 'N_c', 'N_n', 'N_tot', 'mem_per_core', 'mem_per_node', 'job_qos', 'k_span', 't_span']
performance_outputs = ['time_elapsed']
primary_sim_settings = ['k_span', 't_span', 'int_method', 'res_con', 'inf_con', 'k_num', 't_num']
secondary_sim_settings = ['A_sens', 't_sens']
tertiary_sim_settings = ['use_mass_units', 'use_natural_units', 'dimensionful_p', 'rescale_consts', 'unitful_amps', 'unitful_k', 'unitful_m', 'rescale_k', 'rescale_m', 'rescale_amps', 'rescale_consts']
sim_settings_parameters = primary_sim_settings + secondary_sim_settings + tertiary_sim_settings
# Input parameters
primary_inputs = ['F', 'm_q', 'eps', 'p_t', 'L4', 'L3']
secondary_inputs = ['A_0', 'Adot_0', 'A_pm', 'mu_Th', 'mu_d', 'sig_Th', 'sig_d']
tertiary_inputs = ['xi', 'eps_c', 'l1', 'l2', 'l3', 'l4']
constant_inputs = ['e', 'G', 'c', 'h']
input_parameters = primary_inputs + secondary_inputs + tertiary_inputs
# Dependent Parameters
primary_dependents = ['N_r', 'N_c', 'N_n', 'N_tot', 'disable_P', 'disable_B', 'disable_C', 'disable_D', 'qm', 'g_pi', 'm_u']
secondary_dependents = ['t_u', 'm_0', 'k_0', 'p_0', 'g_qed', 'g_3', 'g_4']
tertiary_dependents = ['dqm', 'm', 'm_r', 'm_n', 'm_c', 'p', 'p_r', 'p_n', 'p_c', 'amps', 'T_u', 'T_r', 'T_n', 'T_c']
dependent_parameters = primary_inputs + secondary_inputs + tertiary_inputs
k_dependents = ['k_class_arr', 'k_mean_arr', 'k_peak_arr', 'k_sens_arr', 'k_ratio_arr']
# Phase sampling correlation parameters
sampled_params = ['qc']
total_phase_dependents = ['mean_d_tot', 'var_d_tot', 'min_d_tot', 'max_d_tot', 'mean_Th_tot', 'var_Th_tot', 'min_Th_tot', 'max_Th_tot']
local_phase_dependents = ['mean_d_r', 'var_d_r', 'min_d_r', 'max_d_r', 'mean_d_n', 'var_d_n', 'min_d_n', 'max_d_n', 'mean_d_c', 'var_d_c', 'min_d_c', 'max_d_c']
global_phase_dependents = ['mean_Th_r', 'var_Th_r', 'min_Th_r', 'max_Th_r', 'mean_Th_n', 'var_Th_n', 'min_Th_n', 'max_Th_n', 'mean_Th_c', 'var_Th_c', 'min_Th_c', 'max_Th_c']
phase_parameters = total_phase_dependents + local_phase_dependents + global_phase_dependents
# Unused parameters
hidden_params = ['seed']
cosmetic_params = ['config_name', 'file_name', 'job_partition']
redundant_params = ['d', 'Th']
# Output/classification parameters
primary_outputs = ['res_class', 'res_freq', 'res_band_class']
secondary_outputs = ['res_band', 'res_freq_label', 'res_freq_class', 't_res', 'res_ratio_f', 'res_ratio_m']

## Parse list/dict-style nested parameter structure
df_arr_cols = ['qc', 'qm', 'xi', 'dqm', 'eps_c'] + \
            ['m_r', 'm_n', 'm_c', 'p_r', 'p_n', 'p_c'] + \
            ['res_band', 't_span', 'k_span']
            #['k_class', 'k_peak'] + \
df_subarrs = ['amps', 'd', 'Th'] # + ['m', 'p'] # + ['k_ratio']
df_dict_cols = ['res_freq_class']
df_arr_keys = {}
df_dict_keys = {}

# Helper functions
is_arr_param = lambda p_col: p_col in df_arr_cols or p_col in df_subarrs
is_dict_param = lambda p_col: p_col in df_dict_cols
is_vectorized_param = lambda p_col: is_arr_param(p_col) or is_dict_param(p_col)
get_vectorized_cols = lambda p_col: df_arr_keys[p_col] if is_arr_param(p_col) else df_dict_keys[p_col] if is_dict_param(p_col) else print(p_col)
vec_col_set = lambda set_in: ((set(df_arr_cols + df_dict_cols) | (set(df_arr_keys.keys()) - set(df_subarrs))) & set(set_in))
vectorized_param_set = lambda set_in: \
    ((set(set_in) | set([vec_val for vec_col in [get_vectorized_cols(vec_col_in) for vec_col_in in vec_col_set(set_in)] for vec_val in vec_col])) - (vec_col_set(set_in) | set(df_subarrs)))


# Create unique sets and split vector-shaped parameters into separate columns
performance_params = performance_inputs + performance_outputs + primary_inputs + \
                    primary_sim_settings + secondary_sim_settings + total_phase_dependents + ['config_name', 'job_partition']
time_param_set = vectorized_param_set(performance_params)

correlation_params = primary_inputs + secondary_inputs + \
                    primary_sim_settings + sampled_params + phase_parameters + \
                    primary_dependents + secondary_dependents + \
                    primary_outputs + secondary_outputs + ['config_name']
corr_param_set = vectorized_param_set(correlation_params)

disabled_params = hidden_params + k_dependents + constant_inputs + \
                tertiary_inputs + tertiary_sim_settings + tertiary_dependents + \
                cosmetic_params + redundant_params

combined_params = correlation_params + performance_params + disabled_params
#print(combined_params)
#combined_params = [combo_param for combo_param in combined_params if combo_param in df.columns]
#print(combined_params)
#print(combined_params)
combined_param_set = vectorized_param_set(combined_params)
#print(combined_param_set)


def df_push_dict(dict_in, col_in, sub_cols_in):
    for x in sub_cols_in:
        dict_in[x] = dict_in[col_in]
    return dict_in


## --------------------------------------------------------------------


def has_data_files(output_folder, uuid, verbosity=0):
    glob_path = os.path.join(output_folder, f'*{uuid}*')
    if verbosity >= 3:
        print('Searching for data files with UUID:', uuid)
        print('   ----->  ', glob_path)

    uuid_glob = glob.glob(glob_path)
    if verbosity >= 3:
        print(f'Found {len(uuid_glob)} data files')
    
    if len(uuid_glob) > 0:
        return True
    else:
        return False

def scan_log_files(directory, output_folder, scratch_folder=None, argfile_in=None, rename_in_place=False, 
                   output_errored=True, output_unlogged=True, chunked_outputs=True, remove_redundant=True, 
                   reseed_argfile=False, copy_successful=False, include_missing_uuids=False, 
                   max_line_num=22680, chunk_size=1000, verbosity=0):
    # Patterns to match the line number, UUID, and the 'Done!' message
    line_pattern = re.compile(r"LINE (\d+):\s*python")
    #uuid_pattern = re.compile(r"piaxiverse_main1_SU3_([a-f0-9]{40})")  # phash in log file
    uuid_pattern = re.compile(r"[a-zA-Z0-9|_]*_([a-f0-9]{40})")  # phash in log file
    done_pattern = re.compile(r"Done!")
    seed_pattern = re.compile(r"rng_seed:\s*(.*)$")
    argf_pattern = re.compile(r"ARGFILE:\s*(.*)$")

    unsuccessful_lines = []
    successful_lines = set()
    all_lines = []
    found_uuids = set()
    logfile_list = os.listdir(directory)
    all_jobs_successful = True
    seeds = np.full((max_line_num,), None)
    successful_uuids = np.full((max_line_num,), None)
    successful_logfiles = np.full((max_line_num,), None)
    n_places = max(int(np.floor(np.log10(max_line_num)) + 1), 3)
    print('max_line_num:', max_line_num)
    print('n_places: ', n_places)
    argfiles = set()

    scratch_directory = scratch_folder if scratch_folder is not None else os.path.join(directory, 'scratch')

    is_argfile_match = lambda argf1, argf2: argf1.replace('_seeded', '') == argf2.replace('_seeded', '')
    is_valid_argf = lambda argf: True if argf is None else is_argfile_match(os.path.basename(argf), os.path.basename(argfile_in)) if argfile_in is not None else True

    if verbosity >= 1:
        print('Searching for log files in: ', directory)
        print('Searching for data files in: ', output_folder)
        print('Matching argfiles to: ', argfile_in)

    if verbosity >= 5:
        print('--------------------------------------')

    # Scan directory with log files
    for filename in logfile_list:
        # Only process .txt files]
        if filename.endswith('.txt'): # and 'log' in filename:
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r') as file:
                    if verbosity >= 5:
                        print('Checking file: ', filename)
                    line_number = None
                    job_successful = False
                    uuid = None
                    seed = None
                    argfile = None

                    nlines = 0
                    for line in file:
                        # Reset state at the start of a new run, for multi-run log files
                        new_line_match = line_pattern.search(line)
                        if new_line_match:
                            if line_number is not None:
                                # Process the previous run
                                if is_valid_argf(argfile):
                                    all_lines.append(line_number)
                                    nlines += 1
                                    if job_successful and uuid:
                                        # Don't consider the run to be "successful" if its corresponding data files are missing
                                        if include_missing_uuids or has_data_files(output_folder, uuid, verbosity):
                                            found_uuids.add(uuid)
                                            successful_uuids[line_number-1] = uuid
                                            successful_lines.add(line_number)
                                            successful_logfiles[line_number-1] = filepath
                                    if not job_successful or (job_successful and not uuid and not include_missing_uuids):
                                        unsuccessful_lines.append(line_number)
                                        all_jobs_successful = False
                                        if reseed_argfile:
                                            if seeds[line_number-1] is None:
                                                seeds[line_number-1] = seed
                                    if job_successful:
                                        if include_missing_uuids:
                                            successful_lines.add(line_number)
                                            successful_logfiles[line_number-1] = filepath
                                        if reseed_argfile:
                                            seeds[line_number-1] = seed

                                # Reset variables for the next run
                                line_number = None
                                job_successful = False
                                uuid = None
                                seed = None

                        #print(line)

                        # Search for the line number pattern
                        if line_number is None:
                            if new_line_match:
                                line_number = int(new_line_match.group(1))
                                if verbosity >= 5:
                                    print(f'  LINE: {line_number}')
                        
                        # Search for given argfile
                        if argfile is None:
                            argfile_match = argf_pattern.search(line)
                            if argfile_match:
                                argfile = argfile_match.group(1).strip('\n')
                                argfiles.add(argfile)
                                if verbosity >= 5:
                                    print(f'  ARGFILE: {os.path.basename(argfile)}')
                        
                        # Search for RNG seed (to reproduce errored runs)
                        if seed is None:
                            seed_match = seed_pattern.search(line)
                            if seed_match:
                                seed = seed_match.group(1).strip('\n')
                                if verbosity >= 5:
                                    print(f'  SEED: {seed}')

                        # Check for the "Done!" line indicating successful completion
                        if done_pattern.search(line):
                            job_successful = True
                            if verbosity >= 5:
                                print(f'  DONE: {job_successful}')

                        # Search for the UUID pattern
                        if uuid is None:
                            uuid_match = uuid_pattern.search(line)
                            if uuid_match:
                                uuid = uuid_match.group(1).strip('\n')
                                if verbosity >= 5:
                                    print(f'  UUID: {uuid}')

                    # Process the last run in the file
                    if line_number:
                        if is_valid_argf(argfile):
                            all_lines.append(line_number)
                            nlines += 1
                            if job_successful and uuid:
                                # Don't consider the run to be "successful" if its corresponding data files are missing
                                if include_missing_uuids or has_data_files(output_folder, uuid, verbosity):
                                    found_uuids.add(uuid)
                                    successful_uuids[line_number-1] = uuid
                                    successful_lines.add(line_number)
                                    successful_logfiles[line_number-1] = filepath
                            if not job_successful or (job_successful and not uuid and not include_missing_uuids):
                                unsuccessful_lines.append(line_number)
                                # Don't flag total batch run as errored in N > 1 runs, the final one likely got cut off due to time
                                if nlines <= 1:
                                    all_jobs_successful = False
                                if reseed_argfile:
                                    # Prefer the successful run's seed to the errored one (legacy support workaround)
                                    if seeds[line_number-1] is None:
                                        seeds[line_number-1] = seed
                            if job_successful:
                                if include_missing_uuids:
                                    successful_lines.add(line_number)
                                if reseed_argfile:
                                    seeds[line_number-1] = seed
                            # Optionally rename log file
                            if rename_in_place:
                                nline_str = '%d' % line_number if nlines <= 1 else '%d-%d' % (np.min(all_lines), np.max(all_lines))
                                # TODO: How should we name log files with partial success rates?
                                new_filename = f'log_{nline_str}.txt' if all_jobs_successful else f'log_{nline_str}_ERR.txt'
                                new_filepath = os.path.join(directory, new_filename)
                                os.rename(filepath, new_filepath)
                                if verbosity >= 4:
                                    print(f'Renamed {filename} to {new_filename}')

            except Exception as e:
                print(f'  Error processing {filename}: {e}')
                raise(e)

    # Clean up redundant files (assuming they have been renamed in previous steps)
    # TODO: Adapt this to account for files with multiple runs in them.
    # TODO: Ensure the below is working correctly before fully enabling this functionality.
    if remove_redundant:
        redundant_log_files = []
        for filename in logfile_list:
            filename_base = filename.split('.txt')[0].replace('_ERR', '')
            if filename.split('.txt')[0].split('_')[-1] == 'ERR' and filename_base.join('.txt') in logfile_list:
                redundant_log_files.append(filename)
                # Move to recycling bin, presumably in a scratch storage space with a finite retention policy
                # FIXME: Currently unsure if the above logic is actually correct, disabled for now.
                if False:
                    old_filepath = os.path.join(directory, filename)
                    new_filepath = os.path.join(scratch_directory, 'recycle', filename)
                    os.rename(old_filepath, new_filepath)

    if reseed_argfile:
        if argfile_in is None:
            argfiles_base = set([os.path.basename(argf).replace('_seeded','') for argf in argfiles])
            if len(argfiles_base) > 2:
                print(f'Error reseeding argfile, log files in {directory} contain references to multiple base argfiles:')
                for argfile in argfiles_base:
                    print(f'    {argfile}')
            else:
                if len(argfiles) <= 0:
                    print('IndexError, no valid argfiles found!')
                for argfile_i in list(argfiles):
                    print(argfile_i,':',os.path.exists(argfile_i))
                    if os.path.exists(argfile_i):
                        argfile_in = argfile_i.replace('_seeded','')
                        break

        argfile_out = argfile_in
        
        if argfile_out is not None:
            argfile_out = f'{argfile_out}_seeded'
            #argfile_out = os.path.join(os.path.dirname(argfile_out), argfile_out)

            try:
                with open(argfile_in, 'r') as infile, open(argfile_out, 'w', newline='\n') as outfile:
                    for line_i, line in enumerate(infile):
                        line_seed = ('--seed %s' % seeds[line_i]) if seeds[line_i] is not None else ''
                        outline = line.strip('\n') + line_seed
                        outfile.write(outline + '\n')
            except Exception as e:
                print('Error while trying to read argfile: ', argfile_in)
                raise e
            
            if verbosity >= 1:
                print('Preseeded argfile generated: ', argfile_out)

    # Print results to console
    if output_errored or output_unlogged:
        unsuccessful_lines = sorted([line for line in set(unsuccessful_lines) if not(line in successful_lines)])
        successful_lines = sorted(list(successful_lines))
        all_lines = set(all_lines)
        total_errored = 0
        total_missing = 0

        if output_errored:
            if verbosity >= 1:
                print('\nUnsuccessful job LINE numbers:')
            if chunked_outputs:
                for i in range(0, int(np.ceil(float(max_line_num)/chunk_size))):
                    chunk_offset = (i*chunk_size)
                    minbound = chunk_offset + 1
                    maxbound = chunk_offset + chunk_size
                    if verbosity >= 1:
                        print(f'\n[{minbound} - {min(maxbound, max_line_num)}]')

                    errored_in_chunk = [line for line in unsuccessful_lines if line >= minbound and line <= maxbound]
                    errored_count = len(errored_in_chunk)
                    total_errored += errored_count
                    errored_count_str = '[N = %3s]' % str(errored_count) if errored_count > 0 else ''
                    if verbosity >= 1:
                        if len(unsuccessful_lines) == chunk_size:
                            print('ERRORED: [ALL]')
                        else:
                            print('ERRORED: %s' % errored_count_str, format_for_slurm_job_array(errored_in_chunk, offset=chunk_offset))

                    missing_in_chunk = [j for j in range(minbound, min(maxbound, max_line_num)+1) if j not in all_lines]
                    missing_count = len(missing_in_chunk)
                    total_missing += missing_count
                    missing_count_str = '[N = %3s]' % str(missing_count) if missing_count > 0 else ''
                    if verbosity >= 1:
                        if len(missing_in_chunk) == chunk_size:
                            print('MISSING: [ALL]')
                        else:
                            print('MISSING: %s' % missing_count_str, format_for_slurm_job_array(missing_in_chunk, offset=chunk_offset))
                    
                    if verbosity >= 1:
                        print('TO RERUN:', format_for_slurm_job_array(errored_in_chunk + missing_in_chunk, offset=chunk_offset))
            else:
                print(format_for_slurm_job_array(unsuccessful_lines, offset=0))

        # Display all output files in results folder without associated logs
        if output_unlogged:
            data_file_uuids = [data_filename.replace('.json','').split('_')[-1] for data_filename in os.listdir(output_folder) if '.json' in data_filename]
            data_file_root = '_'.join(os.listdir(output_folder)[0].replace('.json','').split('_')[:-1])
            unlogged = sorted([uuid for uuid in data_file_uuids if not('.npy' in uuid or '.pkl' in uuid or '.pdf' in uuid) and uuid not in found_uuids])
            total_unlogged = len(unlogged)
            if total_unlogged > 0:
                #if total_unlogged == total_missing:
                #    print('UNLOGGED: [ALL]')
                #else:
                if verbosity >= 1:
                    print('----------------------------------------------------------')
                    print('UNLOGGED: [N = %d]' % total_unlogged)
                    if verbosity >= 3:
                        print('          %s' % ('\n          '.join(unlogged)))
        
        # Print total counts for each category
        if verbosity >= 0:
            print('----------------------------------------------------------')
            print('TOTAL COMPLETED: %d' % len(successful_lines))
            print('TOTAL ERRORED: %d' % len(unsuccessful_lines))
            print('(TOTAL ERRORED: %d)' % total_errored)
            print('TOTAL MISSING: %d' % (max_line_num - (len(successful_lines) + len(unsuccessful_lines))))
            print('(TOTAL MISSING: %d)' % total_missing)
            if output_unlogged:
                print('TOTAL UNLOGGED: %d' % total_unlogged)
            print('----------------------------------------------------------')
            grand_total_count = len(successful_lines)+len(unsuccessful_lines)+(max_line_num - (len(successful_lines) + len(unsuccessful_lines)))
            print('GRAND TOTAL: %d' % grand_total_count)
            if output_unlogged:
                print('LOGGED + UNLOGGED: %d' % (len(successful_lines) + len(unsuccessful_lines) + total_unlogged))
            
            grand_total_count_2 = len(successful_lines) + total_errored + total_missing
            print('(GRAND TOTAL: %d)' % grand_total_count_2)
            if output_unlogged:
                print('(GRAND TOTAL + UNLOGGED: %d)' % (grand_total_count_2 + total_unlogged))
                print('(LOGGED + UNLOGGED: %d)' % (len(successful_lines) + total_errored + total_unlogged))
            if remove_redundant:
                print('----------------------------------------------------------')
                print('TOTAL REDUNDANT: %d' % len(redundant_log_files))
    
    if copy_successful:
        data_folder_name = os.path.basename(os.path.dirname(output_folder))

        new_log_dir = os.path.abspath(os.path.join(directory, '..', data_folder_name + '_clean'))
        if verbosity >= 1:
            print(('Copying successful log files from: %s\n' + 
                   '                               to: %s')  % (str(directory), str(new_log_dir)))
        if not os.path.exists(new_log_dir):
            os.makedirs(new_log_dir)

        new_data_dir = os.path.abspath(os.path.join(output_folder,'..', data_folder_name + '_clean'))
        if verbosity >= 1:
            print(('Copying successful data files from: %s\n' + 
                   '                                to: %s')  % (output_folder, new_data_dir))
        if not os.path.exists(new_data_dir):
            os.makedirs(new_data_dir)

        # Copy successful log files and data files to new directories with line numbers in filenames
        for idx, item in enumerate(zip(successful_uuids, successful_logfiles)):
            uuid, logfile = item
            if uuid is None or logfile is None:
                if verbosity >= 1:
                    print(f'Skipping line {idx + 1} due to missing UUID or logfile.')
                    print(f'  | uuid: {uuid}')
                    print(f'  | file: {logfile}')
                continue
            line_num = idx + 1
            
            old_logfile = logfile
            #new_logfile = data_folder_name + '_line%05d_%s.txt' % (line_num, uuid) if uuid is not None else '_line%05d.txt' % line_num
            new_logfile = os.path.join(new_log_dir, data_folder_name + '_line%05d.txt' % line_num)

            # Copy logfiles
            if verbosity >= 3:
                print('-------------------------------------------------------------')
            
            #for data_filename in os.listdir(output_folder):
            glob_path = os.path.join(output_folder,f'*{uuid}*')
            if verbosity >= 3:
                print('Searching for data files with UUID:', uuid)
                print('   ----->  ', glob_path)

            uuid_glob = glob.glob(glob_path)
            num_data_files = len(uuid_glob)
            if verbosity >= 3:
                print(f'Found {num_data_files} data files')

            if num_data_files >= 1:
                if verbosity >= 2:
                    print(f'Copying log file {old_logfile} to {new_logfile}')
                shutil.copy(old_logfile, new_logfile)

                for data_filename in uuid_glob:
                    #if data_filename.endswith('.json') and data_filename.startswith(data_folder_name) and uuid in data_filename:
                    #if uuid in data_filename:
                        data_file_base, data_file_ext = os.path.splitext(data_filename)
                        old_datafile = os.path.join(output_folder, data_filename)
                        #new_datafile = os.path.join(copy_to, data_filename.replace(data_folder_name, data_folder_name + '_line%05d' % line_num))
                        if '_' in data_file_base:
                            if 'func' or 'plot' in data_file_base:
                                data_file_suffix = '_' + '_'.join(data_file_base.split('_')[-2:])
                            else:
                                data_file_suffix = '_' + data_file_base.split('_')[-1]
                        else:
                            data_file_suffix = ''
                        #print('data_file_base:', data_file_base)
                        #print('old_datafile:', old_datafile)
                        #print('data_file_suffix:', data_file_suffix)
                        #print('data_file_replace:', data_folder_name + ('_line%05d_%s%s' % (line_num, data_file_suffix, data_file_ext)))

                        new_datafile = os.path.join(new_data_dir, data_filename.replace(data_file_base, data_folder_name + (f'_line%0{n_places}d%s' % (line_num, data_file_suffix))))
                        
                        # Copy data files
                        if verbosity >= 3:
                            print(f'Copying data file {old_datafile} to {new_datafile}')
                        shutil.copy(old_datafile, new_datafile)


def format_for_slurm_job_array(num_list, offset=0):
    # Do not attempt to format empty lists
    if len(num_list) < 1:
        #return num_list
        return ''
        #return "[None]"
    
    # Sort the list of numbers in ascending order
    num_list = np.array(sorted(num_list)) - offset
    
    result = []
    start = num_list[0]
    end = num_list[0]

    for i in range(1, len(num_list)):
        if num_list[i] == end + 1:
            # Continue the sequence
            end = num_list[i]
        else:
            # Sequence broken, append current range to the result
            if start == end:
                result.append(f'{start}')
            else:
                result.append(f'{start}-{end}')
            # Start a new sequence
            start = num_list[i]
            end = num_list[i]

    # Append the last range
    if start == end:
        result.append(f'{start}')
    else:
        result.append(f'{start}-{end}')

    # Join all the parts with commas
    return ','.join(result)



## --------------------------------------------------------------------



## Helper functions for data preprocessing
def prepare_data(config_name, output_root='~/scratch', version='v3.2.8', reclassify=None, recalc_phase_diffs=False, recalc_couplings=True, load_images=False, multithreaded=True, combined_folder=False):
    """Load and prepare data for future functions"""

    if config_name == 'all':
        params, results, _, coeffs = load_all(output_root, version, load_images=load_images)
    else:
        params, results, _, coeffs = load_case(config_name, output_root, version, load_images=load_images, combined_folder=combined_folder)

    # Convert to DataFrame
    params_df  = pd.DataFrame(params)

    # Free up memory
    del params

    # Extract unit mappings
    units_in = params_df.pop('units')
    units_df = pd.DataFrame(units_in.tolist(), index=params_df.index)
    
    # TODO: Temp fix, remove this when no longer needed
    #if int(version.replace('v','').split('.')[0]) <= 2:
    #    params_df['jupyter'].fillna(False, inplace=True)
    # TODO: Temp fix, remove this when no longer needed
    #if int(version.replace('v','').split('.')[0]) <= 3:
    #    if 'N_tot' not in params_df.keys():
    #        params_df['N_tot'] = params_df['N_r'].astype(np.int32)+params_df['N_n'].astype(np.int32)+params_df['N_c'].astype(np.int32)

    # Preprocess stored parameter values and populate dataframe
    for col, dtype in dtype_map.items():
        #print('col: %s   |  dtype: %s' % (col,dtype))
        if col in params_df:
            try:
                if dtype is np.ndarray:
                    try:
                        if multithreaded:
                            params_df[col] = params_df[col].swifter.apply(np.array, dtype=object)
                        else:
                            params_df[col] = params_df[col].apply(np.array, dtype=object)
                    except (np.VisibleDeprecationWarning):
                        print('Deprecation: %s' % col)
                        print(params_df[col])
                        if multithreaded:
                            params_df[col] = params_df[col].swifter.apply(np.array)
                        else:
                            params_df[col] = params_df[col].apply(np.array)
                else:
                    params_df[col] = params_df[col].astype(dtype)
            except (OverflowError):
                print('Overflow: (%s) - %s' % (col, dtype))
        else:
            print('Missing parameter: %s' % col)
    n_cols = [ncol for ncol in params_df if ncol not in dtype_map.keys()]
    if len(n_cols) > 0:
        print('Unspecified parameters: %s' % n_cols)
    
    # Dataframe containing results of numerics and data for plotting
    res_arr = np.array(results[0], dtype=np.float64)[:,0,:]
    results_df = pd.DataFrame(res_arr)
    print(results_df.values.shape)

    # Optionally rerun resonance classification
    if reclassify is not None:
        sols_arr  = np.array(results[0], dtype=np.float64)
        recalc_resonance = lambda x: reclassify_resonance(x, results_in=sols_arr)
        if multithreaded:
            params_df = params_df.swifter.apply(recalc_resonance, axis=1)
        else:
            params_df = params_df.apply(recalc_resonance, axis=1)

    # Free up memory
    del results
    
    if recalc_phase_diffs:
        #df_from_series = lambda s, cols_in: pd.DataFrame.from_dict(dict(zip(s.index, s.values)), dtype=np.float32)
        d_cols  = ['mean_d_r', 'var_d_r', 'min_d_r', 'max_d_r', 'mean_d_n', 'var_d_n', 'min_d_n', 'max_d_n', 'mean_d_c', 'var_d_c', 'min_d_c', 'max_d_c', 'mean_d_tot', 'var_d_tot', 'min_d_tot', 'max_d_tot']
        Th_cols = ['mean_Th_r', 'var_Th_r', 'min_Th_r', 'max_Th_r', 'mean_Th_n', 'var_Th_n', 'min_Th_n', 'max_Th_n', 'mean_Th_c', 'var_Th_c', 'min_Th_c', 'max_Th_c', 'mean_Th_tot', 'var_Th_tot', 'min_Th_tot', 'max_Th_tot']

        get_local_diffs = lambda x: [d_item for d_cat in calc_local_phase_diffs(x['d']) for d_item in d_cat]
        if multithreaded:
            local_diffs  = params_df.swifter.apply(get_local_diffs, axis=1)
        else:
            local_diffs  = params_df.apply(get_local_diffs, axis=1)
        for idx, d_col in enumerate(d_cols):
            params_df[d_col] = local_diffs.apply(lambda x: x[idx])

        get_global_diffs = lambda x: [Th_item for Th_cat in calc_global_phase_diffs(x['Th']) for Th_item in Th_cat]
        if multithreaded:
            global_diffs = params_df.swifter.apply(get_global_diffs, axis=1)
        else:
            global_diffs = params_df.apply(get_global_diffs, axis=1)
        for idx, Th_col in enumerate(Th_cols):
            params_df[Th_col] = global_diffs.apply(lambda x: x[idx])

        '''
        params_df['mean_d_r'] = local_diffs.apply(lambda x: pd.DataFrame(x[0][0], dtype=np.float32))
        params_df['var_d_r']  = local_diffs.apply(lambda x: pd.DataFrame(x[0][1], dtype=np.float32))
        params_df['min_d_r']  = local_diffs.apply(lambda x: pd.DataFrame(x[0][2], dtype=np.float32))
        params_df['max_d_r']  = local_diffs.apply(lambda x: pd.DataFrame(x[0][3], dtype=np.float32))
        print('--------------------------')
        print(params_df['mean_d_r'].describe())
        print('--------------------------')
        print(params_df['var_d_r'].describe())
        print('--------------------------')
        print(params_df['min_d_r'].describe())
        print('--------------------------')
        print(params_df['max_d_r'].describe())
        print('--------------------------')
    
        params_df['mean_d_n'] = pd.DataFrame(local_diffs.apply(lambda x: pd.DataFrame(x, dtype=np.float32)[1][0]), dtype=np.float32)
        params_df['var_d_n']  = pd.DataFrame(local_diffs.apply(lambda x: pd.DataFrame(x, dtype=np.float32)[1][1]), dtype=np.float32)
        params_df['min_d_n']  = pd.DataFrame(local_diffs.apply(lambda x: pd.DataFrame(x, dtype=np.float32)[1][2]), dtype=np.float32)
        params_df['max_d_n']  = pd.DataFrame(local_diffs.apply(lambda x: pd.DataFrame(x, dtype=np.float32)[1][3]), dtype=np.float32)
    
        params_df['mean_d_c'] = pd.DataFrame(local_diffs.apply(lambda x: pd.DataFrame(x, dtype=np.float32)[2][0]), dtype=np.float32)
        params_df['var_d_c']  = pd.DataFrame(local_diffs.apply(lambda x: pd.DataFrame(x, dtype=np.float32)[2][1]), dtype=np.float32)
        params_df['min_d_c']  = pd.DataFrame(local_diffs.apply(lambda x: pd.DataFrame(x, dtype=np.float32)[2][2]), dtype=np.float32)
        params_df['max_d_c']  = pd.DataFrame(local_diffs.apply(lambda x: pd.DataFrame(x, dtype=np.float32)[2][3]), dtype=np.float32)
        
        params_df['mean_d_tot'] = pd.DataFrame(local_diffs.apply(lambda x: pd.DataFrame(x, dtype=np.float32)[3][0]), dtype=np.float32)
        params_df['var_d_tot']  = pd.DataFrame(local_diffs.apply(lambda x: pd.DataFrame(x, dtype=np.float32)[3][1]), dtype=np.float32)
        params_df['min_d_tot']  = pd.DataFrame(local_diffs.apply(lambda x: pd.DataFrame(x, dtype=np.float32)[3][2]), dtype=np.float32)
        params_df['max_d_tot']  = pd.DataFrame(local_diffs.apply(lambda x: pd.DataFrame(x, dtype=np.float32)[3][3]), dtype=np.float32)

        
        params_df['mean_Th_r'] = global_diffs.apply(lambda x: x[0][0])
        params_df['var_Th_r']  = global_diffs.apply(lambda x: x[0][1])
        params_df['min_Th_r']  = global_diffs.apply(lambda x: x[0][2])
        params_df['max_Th_r']  = global_diffs.apply(lambda x: x[0][3])
        
        params_df['mean_Th_n'] = global_diffs.apply(lambda x: x[1][0])
        params_df['var_Th_n']  = global_diffs.apply(lambda x: x[1][1])
        params_df['min_Th_n']  = global_diffs.apply(lambda x: x[1][2])
        params_df['max_Th_n']  = global_diffs.apply(lambda x: x[1][3])
        
        params_df['mean_Th_c'] = global_diffs.apply(lambda x: x[2][0])
        params_df['var_Th_c']  = global_diffs.apply(lambda x: x[2][1])
        params_df['min_Th_c']  = global_diffs.apply(lambda x: x[2][2])
        params_df['max_Th_c']  = global_diffs.apply(lambda x: x[2][3])
                                                  
        params_df['mean_Th_tot'] = global_diffs.apply(lambda x: x[3][0])
        params_df['var_Th_tot']  = global_diffs.apply(lambda x: x[3][1])
        params_df['min_Th_tot']  = global_diffs.apply(lambda x: x[3][2])
        params_df['max_Th_tot']  = global_diffs.apply(lambda x: x[3][3])
        '''

    if recalc_couplings:
        from ..pyaxi_utils import g_anomaly, g_coupling, alpha_off, alpha_sm
        # Coupling constants for triangle diagram, scalar QED, charged scattering, and neutral scattering interactions respectively
        get_g_pi  = lambda x: g_anomaly(x['F']/1e9,   l1=x['l1'], eps=x['eps'], fs_in=alpha_sm)  if x['N_r'] > 0 else None
        get_g_qed = lambda x: g_coupling(1,           li=x['l2'], eps=x['eps'], fs_in=alpha_off) if x['N_c'] > 0 else None
        get_g_3   = lambda x: g_coupling(x['L3']/1e9, li=x['l3'], eps=x['eps'], fs_in=alpha_sm)  if x['N_c'] > 0 else None
        get_g_4   = lambda x: g_coupling(x['L4']/1e9, li=x['l4'], eps=x['eps'], fs_in=alpha_sm)  if x['N_r'] + x['N_n'] > 0 else None
        if multithreaded:
            params_df['g_pi']  = params_df.swifter.apply(get_g_pi, axis=1)
            params_df['g_qed'] = params_df.swifter.apply(get_g_qed, axis=1)
            params_df['g_3']   = params_df.swifter.apply(get_g_3, axis=1)
            params_df['g_4']   = params_df.swifter.apply(get_g_4, axis=1)
        else:
            params_df['g_pi']  = params_df.apply(get_g_pi, axis=1)
            params_df['g_qed'] = params_df.apply(get_g_qed, axis=1)
            params_df['g_3']   = params_df.apply(get_g_3, axis=1)
            params_df['g_4']   = params_df.apply(get_g_4, axis=1)
    
    # TODO: Remove these fixes when no longer needed
    if 'N_tot' not in params_df.columns:
        get_N_tot = lambda x: x['N_r'] + x['N_n'] + x['N_c']
        if multithreaded:
            params_df['N_tot'] = params_df.swifter.apply(get_N_tot, axis=1)
        else:
            params_df['N_tot'] = params_df.apply(get_N_tot, axis=1)

    if 'job_qos' not in params_df.columns:
        if multithreaded:
            params_df['job_qos'] = params_df.swifter.apply(lambda x: None, axis=1)
        else:
            params_df['job_qos'] = params_df.apply(lambda x: None, axis=1)

    if 'job_partition' not in params_df.columns:
        if multithreaded:
            params_df['job_partition'] = params_df.swifter.apply(lambda x: None, axis=1)
        else:
            params_df['job_partition'] = params_df.apply(lambda x: None, axis=1)

    if 'mem_per_node' not in params_df.columns:
        if multithreaded:
            params_df['mem_per_node'] = params_df.swifter.apply(lambda x: -1, axis=1)
        else:
            params_df['mem_per_node'] = params_df.apply(lambda x: -1, axis=1)

    # Dataframe containing the functions corresponding to oscillating coefficients B(t), C+/-(t), D(t), and P(t)
    coeffs_df  = pd.DataFrame(coeffs)

    # Free up memory
    del coeffs
    
    return params_df, results_df, coeffs_df, units_df

## Postprocess units and prepare units formatting dict
def get_units_fmt(units_df, verbosity=0):
    # TODO: Group runs by unit choices if more than one permutation exists? Maybe p-hashing the dict?
    unique_units = {col: len(units_df[col].unique()) <= 1 for col in units_df.columns}
    all_unique   = np.all([unique_units[key] for key in unique_units.keys()])
    # Assume all runs included use the same units for now
    if all_unique:
        unique_units = {col: units_df[col].unique()[0] for col in units_df.columns}
    else:
        unique_units = {col: units_df[col].unique()[0] for col in units_df.columns if unique_units[col]}
        if verbosity >= 1:
            print('Non-unique units: ', [col for col in unique_units.keys() if not unique_units[col]])

    # WIP: Organize mapping of units for each variable
    # TODO: Double check this and make sure units are being handled properly throughout
    #       the simulation
    # TODO: Double check the below overrides (taken from print_param_space() in pyaxi_utils)
    #       make sense and are okay to do, I forget why the code does this exactly...
    units_map = {}
    for u_key in ['m','m_c','m_n','m_r']:
        units_map[u_key] = unique_units['m']
    for u_key in ['g_pi', 'g_qed', 'g_3', 'g_4']:
        units_map[u_key] = 'GeV'
    for u_key in ['F']:
        units_map[u_key] = unique_units['F']
        if unique_units['F'] == 'm_u':
            units_map[u_key] = 'eV' # TODO: Check this! (F_pi value is always saved in eV)
        else:
            units_map[u_key] = unique_units['F']
    for u_key in ['L3', 'L4']:
        if unique_units['Lambda'] == 'm_u':
            units_map[u_key] = 'eV' # TODO: Check this! (Lambda values are always saved in eV)
        else:
            units_map[u_key] = unique_units['Lambda']
    for u_key in ['l1', 'l2', 'l3', 'l4']:
        units_map[u_key] = unique_units['lambda']
    for u_key in ['amps']:
        units_map[u_key] = unique_units['amp']
    for u_key in ['t_span','t_sens','t_res']:
        units_map[u_key] = unique_units['t']
    for u_key in ['k_span']:
        if unique_units['k'] == 'eV':
            units_map[u_key] = 'm_u' # TODO: Check this! (k values are always saved in m_u)
        else:
            units_map[u_key] = unique_units['k']
    for u_key in ['p','p_t','p_c','p_n','p_r']:
        units_map[u_key] = unique_units['p']
    for u_key in ['Th','mu_Th','sig_Th']:
        units_map[u_key] = unique_units['Theta']
    for u_key in ['d','mu_d','sig_d']:
        units_map[u_key] = unique_units['delta']
    for u_key in ['c','h','G']:
        units_map[u_key] = unique_units[u_key]
    for u_key in ['m_q', 'm_u']:
        units_map[u_key] = 'eV'
    #param_units = {key: unique_units[units_map[key]] for key in units_map.keys()}
    param_units = units_map

    units_fmt = lambda key, prefix='', units=param_units: '' if key not in units.keys() else ('[%s%s]' % (prefix, units[key])) if units[key] != 1 else ''

    return units_fmt

def get_param_dataframes(verbosity=0):
    ## Correlation analysis and dataframe postprocessing
    pd.options.display.max_columns = None

    # Convert dictionary to DataFrame for ease of manipulation
    df_in = params_data.copy(deep=True)

    # Expand array-like features into discrete columns
    df = df_in.copy(deep=True)
    for col in df_arr_cols:
        columns = ['%s_%d' % (col, ex_idx) for ex_idx, _ in enumerate(df[col][0])]
        df_arr_keys[col] = columns
        for col_x in columns:
            print_fmt[col_x] = print_fmt[col]
            dtype_map[col_x] = dtype_map[col]
        df = pd.concat([df, df.pop('%s' % col).apply(pd.Series).add_prefix('%s_' % col)], axis=1)
    for col in df_subarrs:
        columns = ['%s_%d' % (col, ex_idx) for ex_idx, _ in enumerate(df[col][0])]
        df_arr_keys[col] = columns
        for col_x in columns:
            print_fmt[col_x] = print_fmt[col]
            dtype_map[col_x] = dtype_map[col]
        df = pd.concat([df, df.pop('%s' % col).apply(pd.Series).add_prefix('%s_' % col)], axis=1)
        for sub_idx, sub_col in enumerate(df[columns]):
            sub_columns = ['%s_%d' % (sub_col, ex_idx) for ex_idx, _ in enumerate(df[sub_col][0])]
            df_arr_keys[sub_col] = sub_columns
            for col_x in sub_columns:
                print_fmt[col_x] = print_fmt[sub_col]
                dtype_map[col_x] = dtype_map[sub_col]
            df = pd.concat([df, df.pop('%s' % sub_col).apply(pd.Series).add_prefix('%s_' % sub_col)], axis=1)
    # Do the same for dicts
    for col in df_dict_cols:
        col_data = {'%s_%s' % (col, key): val for key,val in df[col][0].items()}
        columns = col_data.keys()
        for col_x in columns:
            print_fmt[col_x] = print_fmt[col]
            dtype_map[col_x] = dtype_map[col]
        df = pd.concat([df, df.pop('%s' % col).apply(pd.Series, index=columns)], axis=1)
        df_dict_keys[col] = columns

    # Descriptive Statistics
    if verbosity >= 5:
        print(df.describe())
        print('-----------------------------------------------------------------')

    '''
    if False: # Temp. prune parameters not found in older datasets
        new_params = ['res_freq', 'use_natural_units', 'res_band', 'use_mass_units', 'res_band_class']
        for new_param in new_params:
            if new_param not in df.columns:
                if new_param in corr_param_set:
                    corr_param_set.remove(new_param)
                if new_param in time_param_set:
                    time_param_set.remove(new_param)
                if new_param in combined_param_set:
                    combined_param_set.remove(new_param)
    '''

    full_df = df[sorted(combined_param_set)].copy(deep=True)
    #print(full_df)
    corr_df = df[sorted(corr_param_set)].copy(deep=True)
    time_df = df[sorted(time_param_set)].copy(deep=True)

    # Identify any free parameters not being properly classified
    unclassified_params = set(df_in.columns) - set(combined_params)
    if len(unclassified_params) > 0:
        if verbosity >= 1:
            print('-----------------------------------------------------------------')
            print('Unclassified parameters: ')
            for col in unclassified_params:
                print('%s  : %s' % (col, df_in[col].dtype))

    # Correlation Analysis
    if verbosity >= 1:
        print('\nModel Parameters and Correlations')
        print(df_describe(corr_df))
        #print(df_pprint(corr_df))
        #correlation_matrix = corr_df.corr(numeric_only=True)
        #print(df_pprint(correlation_matrix))
        print('-----------------------------------------------------------------')

    # Complexity Analysis
    if verbosity >= 2:
        print('\nTime Complexity Parameters and Correlations')
        print(df_describe(time_df))
        #time_corr_matrix = time_df.corr(numeric_only=True)
        #print(df_pprint(time_corr_matrix))
        print('-----------------------------------------------------------------')
    
    return corr_df, time_df, full_df

def reclassify_resonance(series_in, results_in, method_in=None):
    """Reclassify tot_class, nk_class, and ratios for a given Series."""
    n_p_local = n_p
    n_k_local = n_k
    k_span = series_in['k_span']
    t_span = series_in['t_span']
    
    k_values = np.linspace(k_span[0], k_span[1], int(series_in['k_num']))
    times    = np.linspace(t_span[0], t_span[1], int(series_in['t_num']))
    nk_arr = np.array([n_p_local(k_i, series_in, results_in, k_values, times, n=n_k_local) for k_i,_ in enumerate(k_values)])
    
    nk_class, tot_class, nk_ratios, ratio_f, ratio_m, t_res, t_max = classify_resonance(series_in, nk_arr, k_span, method=method_in)
    series_in['k_class_arr'] = nk_class
    series_in['k_ratio_arr'] = nk_ratios
    series_in['res_class']   = tot_class
    series_in['res_ratio_f'] = ratio_f
    series_in['res_ratio_m'] = ratio_m
    series_in['t_res']       = t_res
    
    return series_in

def plot_heatmaps(df):
    """Plot heatmaps of k_class_arr, k_ratio_arr, and k_peak_arr over multiple runs."""

    get_k_ratio = lambda arr, i: np.asarray(list(filter(None, map(lambda x: list(filter(None, x.strip().split(' ')))[i], arr.strip().replace(']','').strip().split('[')[1:]))), dtype=np.float64)
    #k_ratio_ser = lambda arr: pd.Series({'ratio': , 'ratio_max': , 't_res': , 't_max': })
    
    ## Assuming the arrays are already in the correct format
    # TODO: handle inf values in k_ratio_arr
    k_class_arr = df['k_class_arr']
    #k_ratio_arr = df['k_ratio_arr'].apply(lambda x: x[:,0])
    k_ratio_arr = df['k_ratio_arr'].apply(lambda x: get_k_ratio(x, 0))
    k_peak_arr  = df['k_ratio_arr'].apply(lambda x: get_k_ratio(x, 1))
    #t_res_arr   = df['k_ratio_arr'].apply(lambda x: get_k_ratio(x, 2))
    #t_max_arr   = df['k_ratio_arr'].apply(lambda x: get_k_ratio(x, 3))
    k_spans     = df['k_span']

    # Determine the global k-range
    global_k_min = min([span[0] for span in k_spans])
    global_k_max = max([span[1] for span in k_spans])
    total_k_modes = max([len(arr) for arr in k_class_arr])  # maximum number of k-modes
    
    # Convert string values to unique integers for plotting
    #unique_values = np.unique(np.concatenate(k_class_arr))
    unique_values = ['damp', 'none', 'burst', 'resonance']
    # TEMP/TODO: Remove 'aliases' logic when no longer needed
    class_aliases = {'semi':'burst', 'soft':'burst', 'injection':'burst', 'res':'resonance'}
    num_runs   = len(k_class_arr)
    num_values = len(k_class_arr[0])
    value_to_int = {value: idx for idx, value in enumerate(unique_values)}
    
    #k_class_arr = [[value_to_int[val] for val in row] for row in k_class_arr]
    #k_sens_arr  = [[val for val in row] for row in k_sens_arr]
    #k_peak_arr  = [[val for val in row] for row in k_peak_arr]
    
    if True:
        # Initialize combined heatmaps
        combined_k_class = np.full((len(k_class_arr), total_k_modes), np.nan)
        combined_k_ratio = np.full((len(k_ratio_arr), total_k_modes), np.nan)
        combined_k_peak  = np.full((len(k_peak_arr),  total_k_modes), np.nan)
        
        k_ratio_min = k_ratio_max = k_peak_min = k_peak_max = None

        # Fill in the combined heatmaps
        for idx, (k_class, k_ratio, k_peak, k_span) in enumerate(zip(k_class_arr, k_ratio_arr, k_peak_arr, k_spans)):
            start_idx = int((k_span[0] - global_k_min) * total_k_modes / (global_k_max - global_k_min))
            end_idx = start_idx + len(k_class)
            #print(k_class)
            combined_k_class[idx, start_idx:end_idx] = [value_to_int[val] if val in value_to_int.keys() else value_to_int[class_aliases[val]] for val in k_class]
            combined_k_ratio[idx, start_idx:end_idx] = k_ratio
            combined_k_peak[idx, start_idx:end_idx]  = k_peak
            
            k_ratio_min = min(k_ratio) if k_ratio_min is None else min(k_ratio_min, min(k_ratio))
            k_ratio_max = max(k_ratio) if k_ratio_max is None else max(k_ratio_max, max(k_ratio))
            k_peak_min  = min(k_peak)  if k_peak_min  is None else min(k_peak_min, min(k_peak))
            k_peak_max  = max(k_peak)  if k_peak_max  is None else max(k_peak_max, max(k_peak))

        # Plot the combined heatmaps
        plt.figure(figsize=(10, 12))
        ax1 = plt.subplot(3, 1, 1)
        im1 = plt.imshow(combined_k_class, aspect='auto', cmap='viridis')
        plt.title('k_class_arr')
        plt.ylabel('N')
        plt.xlabel('normalized k')

        ax2 = plt.subplot(3, 1, 2)
        im2 = plt.imshow(combined_k_ratio, aspect='auto', cmap='autumn')
        plt.title('k_ratio_arr')
        plt.ylabel('N')
        plt.xlabel('normalized k')

        ax3 = plt.subplot(3, 1, 3)
        im3 = plt.imshow(combined_k_peak, aspect='auto', cmap='autumn')
        plt.title('k_peak_arr')
        plt.ylabel('N')
        plt.xlabel('normalized k')
        
        # Add colorbars to indicate the mapping from integers back to string values
        cbar1 = plt.colorbar(im1, ax=ax1, ticks=list(value_to_int.values()))
        cbar1.set_ticklabels(list(value_to_int.keys()))

        cbar2 = plt.colorbar(im2, ax=ax2, ticks=np.linspace(k_ratio_min, k_ratio_max, 3))
        #cbar2.set_ticklabels(list(value_to_int.keys()))

        cbar3 = plt.colorbar(im3, ax=ax3, ticks=np.linspace(k_peak_min, k_peak_max, 3))
        #cbar3.set_ticklabels(list(value_to_int.keys()))
        

        plt.tight_layout()
        plt.show()
        
    else:
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        im1 = axs[0].imshow(k_class_arr, aspect='auto', cmap='viridis')
        axs[0].set_title('k_class_arr heatmap')
        axs[0].set_xlabel('Index')
        axs[0].set_ylabel('Run')

        im2 = axs[1].imshow(k_ratio_arr, aspect='auto', cmap='autumn')
        axs[1].set_title('k_ratio_arr heatmap')
        axs[1].set_xlabel('Index')
        axs[1].set_ylabel('Run')

        im3 = axs[2].imshow(k_peak_arr, aspect='auto', cmap='autumn')
        axs[2].set_title('k_peak_arr heatmap')
        axs[2].set_xlabel('Index')
        axs[2].set_ylabel('Run')

        # Add colorbars to indicate the mapping from integers back to string values
        cbar1 = plt.colorbar(im1, ax=axs[0], ticks=list(value_to_int.values()))
        cbar1.set_ticklabels(list(value_to_int.keys()))

        cbar2 = plt.colorbar(im2, ax=axs[1], ticks=list(value_to_int.values()))
        cbar2.set_ticklabels(list(value_to_int.keys()))

        cbar3 = plt.colorbar(im3, ax=axs[2], ticks=list(value_to_int.values()))
        cbar3.set_ticklabels(list(value_to_int.keys()))

        plt.tight_layout()
        plt.show()

def plot_scatter(df):
    """Plot m_r vs F and p_t vs Lambda4 scatter plots over multiple runs, labeled by res_class."""
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    
    unique_classes = df['res_class'].unique()
    
    for res_cls in unique_classes:
        subset = df[df['res_class'] == res_cls]
        
        axs[0].scatter(subset['F'], subset['m_r'].str[0], label=f'Res Class {res_cls}')
        axs[1].scatter(subset['L4'], subset['p_t'], label=f'Res Class {res_cls}')
    
    axs[0].set_title('m_r vs F')
    axs[0].set_xlabel('F')
    axs[0].set_ylabel('m_r')
    
    axs[1].set_title('p_t vs Lambda4')
    axs[1].set_xlabel('Lambda4 (L4)')
    axs[1].set_ylabel('p_t')
    
    axs[0].legend()
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

def make_pairplot(df_in, make_pairplot=True, make_heatmap=False, make_histograms=False, 
                  use_seaborn=True, pp_mode='kde',
                  save_plots=False, save_dir='./plots'):
    # pp_mode must be one of: 'kde', 'scatter', or 'mixed' pairplot display kinds

    log_params  = ['p_0', 'p_t', 'F', 'L3', 'L4', 'm_q', 'eps', 'res_ratio_f', 'res_ratio_m', 'm_0', 'k_0', 'm_u', 'g_pi', 'qm_0', 'qm_1', 'qm_2', 'res_con', 'inf_con']
    #log_params  += ['t_u']
    dict_params = []
    keep_params = ['res_class', 'config_name', 'units', 'res_band_class']

    #plot_params = ['F', 'L4', 'm_u', 'p_t', 'eps']
    plot_params = ['g_pi', 'L4', 'm_u', 'p_t', 'eps']
    fixed_params = {}

    # Filter and rescale data for plotting
    filter_nonunique  = True
    filter_sampled    = True
    rescale_logparams = True
    infs_to_nans = True
    #with pd.option_context("mode.copy_on_write", True):
    plot_data = df_in.copy(deep=True)

    def drop_col(col_in, plot_data_in=plot_data, plot_params_in=plot_params):
        plot_data_in.drop(col_in, inplace=True, axis=1)
        if col_in in plot_params_in:
            plot_params_in.remove(col_in)

    if infs_to_nans:
        plot_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in plot_data.columns:
        if rescale_logparams and col in log_params:
            #print(col)
            plot_data[col] = np.log10(plot_data[col])
        if filter_sampled:
            if col in sampled_params and col not in keep_params:
                drop_col(col)
        if filter_nonunique:
            if len(plot_data[col].unique()) <= 1 and col not in keep_params:
                if col in input_parameters or col in sim_settings_parameters:
                    fixed_params[col] = plot_data[col].unique()[0]
                    drop_col(col)

    # Print out which classes of runs are included in final dataset
    if conf_name == 'all':
        print('Including data from the following configurations:')
        dataset_cases = plot_data['config_name'].unique()
        for data_case in dataset_cases:
            print('    |  %s' % data_case)

    # TODO/WIP: Histograms
    # TODO: Stacked plots with color code for resonance status?
    if make_histograms:
        for column in plot_data.columns:
            #print(plot_data[column])
            x = plot_data[column]
            if column in log_params:
                title = 'log %s' % column
            else:
                title = '%s' % column
            fig_hst = plt.hist(x,bins='auto')
            plt.xlabel(title)
            plt.ylabel('N')
            plt.show()
    else:
        fig_hst = None

    # TODO: Scatter plots?

    # TODO/WIP: Pairplots
    #           WARNING: Takes a long time / lots of space if large numbers are not properly rescaled
    if make_pairplot:
        if use_seaborn:
            pp_fig_name = 'seaborn'
            # temp fix for seaborn version
            import warnings
            warnings.filterwarnings('ignore')
            
            # Format color scheme
            hue_order = ['res_class']
            palette = {'resonance': 'purple', 'burst': 'orange', 'none': 'teal', 'damp': 'grey'}
            
            # Format axes to show units
            fmt_col = lambda c_in: '%s   %s' % (c_in,units_fmt(c_in, prefix='log ' if c_in in log_params else ''))
            plot_data_units = plot_data.rename(columns={col: fmt_col(col) for col in plot_params})
            plot_params_units = [fmt_col(col) for col in plot_params]
            
            # kind = 'kde', 'scatter', 'hist', or 'reg'
            pp_kws  = {'levels':5} if pp_mode == 'kde' else {'alpha':0.35}
            pp_kind = pp_mode if pp_mode != 'mixed' else 'scatter'
            pp = sns.pairplot(plot_data_units, x_vars=plot_params_units, y_vars=plot_params_units, dropna=True, kind=pp_kind, diag_kind='kde', hue='res_class', palette=palette, plot_kws=pp_kws)
            if pp_mode == 'mixed':
                pp.map_offdiag(sns.kdeplot, levels=4, hue_order=['res_class'], palette=palette)
            pp.figure.suptitle('Pairplot: case = %s   (N = %d)' % (conf_name, len(plot_data.index)), y=1.05)

            # Plot unlabelled distribution as well
            if True:
                #pp_kws  = {'levels':5} if pp_mode == 'kde' else {'alpha':0.35}
                #pp_kind = pp_mode if pp_mode != 'mixed' else 'scatter'
                pp_bare = sns.pairplot(plot_data_units, x_vars=plot_params_units, y_vars=plot_params_units, dropna=False, kind=pp_kind, diag_kind='kde', plot_kws=pp_kws)
                if pp_mode == 'mixed':
                    pp_bare.map_offdiag(sns.kdeplot, levels=4)
                pp_bare.figure.suptitle('Pairplot: case = %s   (N = %d)' % (conf_name, len(plot_data.index)), y=1.05)

            print('\nSimulation settings:')
            for key, val in fixed_params.items():
                if key in sim_settings_parameters:
                    print('%20s :  %10s  [%s]' % (key, str(val), units_fmt(key)))
            
            print('Fixed params:')
            for key, val in fixed_params.items():
                if key in input_parameters:
                    print('%20s :  %10s  %s' % (key, str('%.1f' % val) if key in log_params else df_fmt(val, fmt=print_fmt[key]), units_fmt(key, prefix='log ' if key in log_params else '')))
        else:
            pp_fig_name = 'corner'
            # TODO: Implement corner.corner (no seaborn) plotting options
            raise ValueError

        #print('Units: ')
        #print(param_units)
    else:
        pp_fig_name = 'None'
        fig_pp = None

    # TODO: Heatmap for correlation matrix
    if make_heatmap:
        correlation_matrix = plot_data.corr(numeric_only=True)
        fig_hmp = sns.heatmap(correlation_matrix, annot=True)
    else:
        fig_hmp = None

    if np.any([make_histograms, make_pairplot, make_heatmap]):
        print('')
        plt.show()
    
    if save_plots:
        fig_list = [fig_hst, fig_pp, fig_hmp]
        fig_names = ['histogram', pp_fig_name, 'heatmap']
        for fig, name in zip(fig_list, fig_names):
            if fig is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                file_name = os.path.join(save_dir, f'{name}.png')
                fig.savefig(file_name)
    
    return fig_hst, fig_pp, fig_hmp



## --------------------------------------------------------------------



if __name__ == '__main__':
    # Path to where log files are stored
    log_default = os.path.expanduser('~/projects/pyAxiverse/logs/piaxi_main1/')
    # Path to where data files are stored
    output_default = os.path.expanduser('~/projects/pyAxiverse/data/piaxiverse_main1_SU3/')
    # Path to scratch storage space (recycling bin for redundant log files)
    scratch_default = os.path.expanduser('~/scratch/pyAxiverse/logs/scratch/piaxi_main1/')
    # Path to locally stored argfile (necessary if this path is different from the original path used when the script was first run)
    argfile_default = os.path.expanduser('~/projects/pyAxiverse/ARGFILES/piaxiverse_main1_SU3')

    # Parse command line args
    parser = argparse.ArgumentParser(description='Parse command line arguments.')
    parser.add_argument('--log_dir',      type=str, default=log_default,      help='Path to directory where log files are saved')
    parser.add_argument('--output_dir',   type=str, default=output_default,   help='Path to directory where data files are saved')
    parser.add_argument('--scratch_dir',  type=str, default=scratch_default,  help='Path to scratch directory, where deleted files should be sent')
    parser.add_argument('--argfile_path', type=str, default=argfile_default,  help='Path to the exact argfile of choice')
    parser.add_argument('--copy_clean_results', action='store_true', default=False, help='Toggle whether or not to copy successful files to a new directory')
    parser.add_argument('--plot_distribution', action='store_true', default=False, help='Toggle whether or not to plot marginalized distributions of search parameters')
    parser.add_argument('--plot_data_dir', type=str, default=None, help='Specify a different directory to use for distribution plotting data source')
    parser.add_argument('--plot_data_ver', type=str, default=None, help='Specify a version string or sub-directory name to use for distribution plotting data source')
    parser.add_argument('--verbosity', type=int, default=1, help='Output verbosity. -1 for output suppression.')
    args = parser.parse_args()

    log_dir_in = os.path.expanduser(args.log_dir)
    output_dir_in = os.path.expanduser(args.output_dir)
    scratch_dir_in = os.path.expanduser(args.scratch_dir)
    argfile_path_in = os.path.expanduser(args.argfile_path)
    copy_successful = bool(args.copy_clean_results)
    plot_distribution = bool(args.plot_distribution) or bool(args.plot_data_dir) or bool(args.plot_data_ver)

    verbosity = args.verbosity

    # Set output_unsuccessful_lines to True if you want to print the unsuccessful job LINE numbers
    scan_log_files(log_dir_in, output_dir_in, scratch_dir_in, rename_in_place=False, output_errored=True, 
                   argfile_in=argfile_path_in, reseed_argfile=True, remove_redundant=False, include_missing_uuids=False,
                   copy_successful=copy_successful, max_line_num=22680, verbosity=verbosity) # 22680


    if plot_distribution:
        # name of config identifier/subfolder from which to import data. Use 'all' for all (non-debug) datasets of a given version number.
        plot_data_dir = args.plot_data_dir if args.plot_data_dir is not None else output_dir_in
        conf_name = os.path.basename(os.path.dirname(plot_data_dir))
        if verbosity >= 1:
            print('----------------------------------------------------------------')
            print(f'Plotting distribution from folder: {conf_name}')
        #conf_name='all'

        version   = args.plot_data_ver if args.plot_data_ver is not None else ''
        #version   = 'v3.2.8'

        reclass_method = None
        #reclass_method = 'heaviside'

        params_data, results_data, coeffs_data, units_df = prepare_data(config_name=conf_name, output_root='./data', version=version, reclassify=reclass_method, 
                                                                        recalc_phase_diffs=True, recalc_couplings=True, load_images=False, combined_folder=True)
        units_fmt = get_units_fmt(units_df, verbosity=verbosity)

        corr_df, time_df, full_df = get_param_dataframes(verbosity=verbosity)

        histogram, pairplot, heatmap = make_pairplot(corr_df, make_pairplot=True, make_heatmap=False, make_histograms=False, 
                                                     use_seaborn=True, save_plots=False, save_dir=f'./plots/{conf_name}',
                                                     use_tex=False)