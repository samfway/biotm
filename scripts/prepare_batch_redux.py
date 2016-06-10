#!/usr/bin/env python

import argparse
from numpy import save as np_save
from numpy import load as np_load
from os import path, system

from biotm.misc.util import BalancedKFold


def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input-file', help='Input data matrix', required=True)
    args.add_argument('-l', '--input-labels', help='Input labels', required=True)
    args.add_argument('-o', '--output-dir', help='Output directory', default='./')
    args.add_argument('-s', '--scripts-dir', help='Scripts directory', default='./')
    args.add_argument('-d', '--dims', help='List of dimensions', default='5, 10, 25, 50, 100')
    args.add_argument('-k', '--num-folds', help='Number of CV folds', type=int, required=True)
    args.add_argument('-n', '--cv-iters', help='CV iterations', type=int, default=1)
    args.add_argument('-m', '--methods', help='Dimensionality reduction methods (comma-sep)',
                      default='svd,nmf,plsa,lda')
    args.add_argument('-S', '--max-scripts', help='Limit number of scripts', type=int, default=1000)
    args.add_argument('-t', '--tag', help='What to call the output files', type=str, default='OUTPUT')
    args = args.parse_args()
    return args


def make_bash_script(script_name, data_matrix_file, labels_file, output_dir,
                     training_file, testing_file, num_dims, cv_id, methods):
    output = open(script_name, 'w')
    output.write('#!/bin/bash\n')
    output.write('/Users/sawa6416/Projects/biotm/scripts/batch_dim_redux.py -i %s -l %s -o %s -r %s -t %s -k %d -n %d -m "%s"\n' %
                  (data_matrix_file, labels_file, output_dir, training_file, testing_file,
                   num_dims, cv_id, methods))
    system('chmod u+x %s' % script_name)  # make it executable.
    output.close()
    return


def add_to_bash_script(script_name, data_matrix_file, labels_file, output_dir,
                     training_file, testing_file, num_dims, cv_id, methods):
    output = open(script_name, 'a')
    output.write('/Users/sawa6416/Projects/biotm/scripts/batch_dim_redux.py -i %s -l %s -o %s -r %s -t %s -k %d -n %d -m "%s"\n' %
                  (data_matrix_file, labels_file, output_dir, training_file, testing_file,
                   num_dims, cv_id, methods))
    output.close()
    return

def make_launch_script(scripts_dir, n):
    output = open(path.join(scripts_dir, 'launch.sh'), 'w')
    output.write('#!/bin/bash\n')
    output.write('#PBS -N AG\n')
    output.write('#PBS -joe\n')
    output.write('#PBS -t 0-%d\n' % n)
    output.write('#PBS -q long8gb\n')
    output.write('#PBS -l pmem=8gb\n')
    output.write('#PBS -l nodes=1:ppn=1\n')
    output.write('%s/${PBS_ARRAYID}.sh\n' % scripts_dir)
    output.close()
    system('chmod u+x %s' % path.join(scripts_dir, 'launch.sh')) 


def make_results_launch_script(scripts_dir, output_dir, tag):
    filename = path.join(scripts_dir, 'results_launch.sh')
    output = open(filename, 'w')
    output.write('#!/bin/bash')
    output.write('#PBS -N mkfig')
    output.write('#PBS -joe')
    output.write('#PBS -q long8gb')
    output.write('#PBS -l pmem=8gb')
    output.write('#PBS -l nodes=1:ppn=4')
    output.write('#PBS -m e')
    output.write('/Users/sawa6416/Projects/biotm/scripts/process_batch.py -i %s -d "5,10,25,50" -k 10 -n 20 -l "alc" -o /Users/sawa6416/Projects/biotm/results/alc.pdf -t %s' % (output_dir, tag))
    output.close()
    system('chmod u+x %s' % (filename))


if __name__=="__main__":
    args = interface()

    data_matrix_file = args.input_file
    labels_file = args.input_labels
    output_dir = args.output_dir
    
    dim_steps = [int(x) for x in args.dims.split(',')]
    output_dir = args.output_dir
    scripts_dir = args.scripts_dir
    tag = args.tag

    labels = np_load(labels_file) 
    cv_folds = BalancedKFold(labels, args.num_folds, n_iter=args.cv_iters)

    i = 0
    for k, (training, testing) in enumerate(cv_folds):
        file_prefix = path.join(output_dir, 'CV_%d_' % k)
        training_file = file_prefix+'training.npy'
        np_save(training_file, training)
        testing_file = file_prefix+'testing.npy'
        np_save(testing_file, testing)
        
        for num_dims in dim_steps:
            if i >= args.max_scripts:
                ind = i % args.max_scripts
                script_file = path.join(scripts_dir, '%d.sh' % ind)
                add_to_bash_script(script_file, data_matrix_file, labels_file, output_dir,
                                 training_file, testing_file, num_dims, k, args.methods)
            else:
                script_file = path.join(scripts_dir, '%d.sh' % i)
                make_bash_script(script_file, data_matrix_file, labels_file, output_dir,
                                 training_file, testing_file, num_dims, k, args.methods)
            i += 1

    # Create everything
    num_scripts = min(i-1, args.max_scripts-1)
    make_launch_script(scripts_dir, num_scripts) 
    make_results_launch_script(scripts_dir, output_dir, tag)

