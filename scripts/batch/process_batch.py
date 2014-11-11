#!/usr/bin/env python

# mpl backend
import matplotlib
matplotlib.use('Agg')

import argparse
from numpy import save as np_save
from numpy import load as np_load
from numpy import array, zeros
from os import path, system

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from biotm.scripts.batch.util import get_methods
from biotm.plotting.grouped_box import make_grouped_box
import matplotlib.pyplot as plt
import brewer2mpl

def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input-dir', help='Input results directory', required=True)
    args.add_argument('-d', '--dims', help='List of dimensions', default='5, 10, 25, 50, 100')
    args.add_argument('-k', '--num-folds', help='Number of CV folds', type=int, required=True)
    args.add_argument('-n', '--cv-iters', help='CV iterations', type=int, default=1)
    args.add_argument('-m', '--methods', help='Dimensionality reduction methods (comma-sep)',
                      default='svd,nmf,plsa,lda,slda')
    args.add_argument('-l', '--label', help='Value being predcicted', required='True')
    args = args.parse_args()
    return args


if __name__=="__main__":
    args = interface()

    cv_dir = args.input_dir
    techniques = get_methods(args.methods)
    techniques = [t[1].upper() for t in techniques]
    num_folds = args.num_folds * args.cv_iters
    dim_steps = [int(d) for d in args.dims.split(',')]
    metadata_values = np_load(path.join(cv_dir, 'labels.npy'))
    metadata_category = args.label

    mdl = RandomForestClassifier()
    qual = accuracy_score
    plot_data = []
    cv_scores = zeros(num_folds)

    for technique in techniques:
        tech_data = []
        for d in dim_steps:
            for c in xrange(num_folds):            
                prefix = 'CV_%d_%s_%d_' % (c, technique, d)
                prefix = path.join(cv_dir, prefix)
                
                X_test = np_load(prefix + 'txd_test_matrix.npy')
                X_train = np_load(prefix + 'txd_training_matrix.npy')

                ind_prefix = path.join(cv_dir, 'CV_%d_' % (c))
                test = np_load(ind_prefix + 'testing.npy')
                train = np_load(ind_prefix + 'training.npy')
                
                y_test = metadata_values[test]
                y_train = metadata_values[train]

                mdl.fit(X_train, y_train)
                y_pred = mdl.predict(X_test)
                
                print prefix

                cv_scores[c] = qual(y_test, y_pred)
            tech_data.append(cv_scores.copy())
        plot_data.append(tech_data)

    np_save(path.join(cv_dir, 'plot_data.npy'), array(plot_data))
    data_matrix = np_load(path.join(cv_dir, 'data_matrix.npy'))

    for technique in techniques[:1]:
        tech_data = []
        for d in dim_steps:
            for c in xrange(num_folds):            
                prefix = 'CV_%d_%s_%d_' % (c, technique, d)
                prefix = path.join(cv_dir, prefix)

                ind_prefix = path.join(cv_dir, 'CV_%d_' % (c))
                test = np_load(ind_prefix + 'testing.npy')
                train = np_load(ind_prefix + 'training.npy')

                X_test = data_matrix[test]
                X_train = data_matrix[train]
                
                y_test = metadata_values[test]
                y_train = metadata_values[train]

                mdl.fit(X_train, y_train)
                y_pred = mdl.predict(X_test)
                
                cv_scores[c] = qual(y_test, y_pred)
            tech_data.append(cv_scores.copy())
        plot_data.append(tech_data)

fig, ax = plt.subplots(figsize=(10, 8), dpi=80)
plot_data = array(plot_data)        
names = techniques + ["None"]
points = [str(step) for step in dim_steps]
make_grouped_box(ax, plot_data, names, xticklabels=points)
ax.set_ylabel('Accuracy')
ax.set_xlabel('Number of Dimensions')
ax.set_title('Predicting "%s"' % (metadata_category))
plt.savefig(metadata_category+'_c.pdf')
