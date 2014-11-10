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
                      default='svd,nmf,plsa,lda,slda')
    args = args.parse_args()
    return args


def make_bash_script(script_name, data_matrix_file, labels_file, output_dir,
                     training_file, testing_file, num_dims, cv_id, methods):
    output = open(script_name, 'w')
    output.write('#!/bin/bash\n')
    output.write('batch_dim_redux.py -i %s -l %s -o %s -r %s -t %s -k %d -n %d -m "%s"\n' %
                  (data_matrix_file, labels_file, output_dir, training_file, testing_file,
                   num_dims, cv_id, methods))
    system('chmod u+x %s' % script_name)  # make it executable.
    return


def make_launch_script(scripts_dir, n):
    output = open(path.join(scripts_dir, 'launch.sh'))
    output.write('#!/bin/bash\n')
    output.write('#PBS -N AG\n')
    output.write('#PBS -joe\n')
    output.write('#PBS -t 0-%d\n' % n)
    output.write('#PBS -q long8gb\n')
    output.write('#PBS -l pmem=8gb\n')
    output.write('#PBS -l nodes=1:ppn=4\n')
    output.write('%s/${PBS_ARRAYID}.sh\n' % scripts_dir)
    output.close()


if __name__=="__main__":
    args = interface()

    data_matrix_file = args.input_file
    labels_file = args.input_labels
    output_dir = args.output_dir
    
    num_dims = args.num_dims
    prefix = 'CV_%d' % (args.cv_fold_id)
    techniques = get_methods(args.methods)
    output_dir = args.output_dir
    scripts_dir = args.scripts_dir

    labels = np_load(labels_file) 
    cv_folds = BalancedKFold(labels, args.num_folds, n_iter=args.cv_iters)

    for i, (training, testing) in enumerate(cv_folds):
        script_file = path.join(scripts_dir, '%d.sh' % i)
        file_prefix = path.join(output_dir, 'CV_%d_' % i)
        training_file = file_prefix+'training.npy'
        np_save(training_file, training)
        testing_file = file_prefix+'testing.npy'
        np_save(testing_file, testing)
        make_bash_script(script_file, output_dir, data_matrix_file, labels_file,
                         training_file, testing_file, args.methods)

    make_launch_script(scripts_dir, len(cv_folds))
