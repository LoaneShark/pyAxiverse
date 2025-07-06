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
import argparse

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
    parser.add_argument('--argfile_path', type=str, default=argfile_default, help='Path to the exact argfile of choice')
    parser.add_argument('--copy_clean_results', action='store_true', default=False, help='Toggle whether or not to copy successful files to a new directory')
    parser.add_argument('--verbosity', type=int, default=1, help='Output verbosity. -1 for output suppression.')
    args = parser.parse_args()

    log_dir_in = os.path.expanduser(args.log_dir)
    output_dir_in = os.path.expanduser(args.output_dir)
    scratch_dir_in = os.path.expanduser(args.scratch_dir)
    argfile_path_in = os.path.expanduser(args.argfile_path)
    copy_successful = args.copy_clean_results

    verbosity = args.verbosity

    # Set output_unsuccessful_lines to True if you want to print the unsuccessful job LINE numbers
    scan_log_files(log_dir_in, output_dir_in, scratch_dir_in, rename_in_place=False, output_errored=True, 
                   argfile_in=argfile_path_in, reseed_argfile=True, remove_redundant=False, include_missing_uuids=False,
                   copy_successful=copy_successful, max_line_num=22680, verbosity=verbosity) # 22680
