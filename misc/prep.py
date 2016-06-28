#!/usr/bin/env python

from biotm.parse.fileio import load_dataset, parse_mapping_file_to_dataframe, parse_mapping_file_to_dict
from biotm.plotting.grouped_box import make_grouped_box
from biotm.misc.util import BalancedKFold
from os import path, system
from numpy.random import randint, permutation
from scipy.stats import ranksums, f_oneway
from numpy import load as np_load
from collections import Counter

# Scikit imports
from sklearn.cross_validation import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, accuracy_score, f1_score

import numpy as np

num_folds = 10
num_iters = 10

def ag_prep(metadata_category, remove_labels, working_dir=None, prefix='AG_'):
    if working_dir is None:
        working_dir = '/Users/sawa6416/Projects/biotm/analysis/predicting_covariates/prepped/' + prefix + metadata_category
  
    scripts_dir = working_dir + '/scripts/'
    output_dir = working_dir + '/output/'
    system('mkdir -p %s' % (working_dir))
    system('mkdir -p %s' % (scripts_dir))
    system('mkdir -p %s' % (output_dir)) 

    otu_fp = open(otu_file, 'rU')
    map_fp = open(mapping_file, 'rU')
    data_matrix, sample_ids, taxonomy, labels, label_legend = load_dataset(otu_fp, map_fp, metadata_category, None, convert_labels=True)
    cats, mapping_dict = parse_mapping_file_to_dict(open(mapping_file, 'rU'))

    original_size = len(sample_ids)
    assert(len(sample_ids) == len(data_matrix))
    data_matrix = np.array(data_matrix)

    print "Data matrix dimensions: ", data_matrix.shape
    print "Number of Sample IDs: " , len(sample_ids)
    print "Number of Labels: ", len(labels)
    print "Sum of all counts: ", data_matrix.sum()
    print ""


    ### Select Only Stool Samples
    original_size = len(sample_ids)
    keep = np.array([mapping_dict[s]['BODY_SITE'] == 'UBERON:feces' for s in sample_ids])
    data_matrix = data_matrix[keep, :]
    sample_ids = sample_ids[keep]
    labels = labels[keep]
    print 'Removing non-STOOL samples... Removed %d samples.' % (original_size - len(sample_ids))
    print ""

    ### What labels do we have?
    print 'Breakdown of values:'
    counts = Counter(labels)
    for i, l in enumerate(label_legend):
        print '+ %d \t %s ' % (counts[i], l)
    print ""

    ### Remove NA and None
    original_size = len(sample_ids)
    #remove_labels = ['NA', 'no_data', 'Occasionally (1-2 times/week)']
    keep = np.array([i for i, l in enumerate(labels) if label_legend[l] not in remove_labels])
    data_matrix = data_matrix[keep, :]
    sample_ids = sample_ids[keep]
    labels = labels[keep]
    print 'Removing %s... Removed %d samples.' % (','.join(remove_labels), original_size - len(sample_ids))
    print ""

    ### Combine and delete labels
    for combine in combine_labels:
        ''' Assuming that combine is a list of lists ''' 
        new_label = label_legend.index(combine[0])  # Give everything the same label as the first element
        for i in xrange(len(labels)):
            if label_legend[labels[i]] in combine:
                labels[i] = new_label
        label_legend[new_label] = '+'.join(combine)

    ### Clean-up
    nonzero_labels = []
    nonzero_label_legend = []
    for i, l in enumerate(labels):
        if l not in nonzero_labels:
            nonzero_labels.append(l)
            nonzero_label_legend.append(label_legend[l])

    # original index -> index in nonzero_labels
    for i in xrange(len(labels)):
        labels[i] = nonzero_labels.index(labels[i])

    label_legend = nonzero_label_legend
    print 'Breakdown of values:'
    label_counts = Counter(labels)
    for i, l in enumerate(label_legend):
        print '+ %d \t %s ' % (label_counts[i], l)
    print ""

    ### Remove rare words 
    num_samples = data_matrix.shape[0]
    totals = (data_matrix > 0)
    x = totals.sum(axis=0)
    min_threshold = 5

    keep = [i for i in xrange(len(x)) if x[i] > min_threshold]

    original_size = data_matrix.shape[1]
    data_matrix = data_matrix[:,keep]
    print 'Removing dimensions with less than %d hits... Removed %d dimensions.' % (min_threshold, original_size-data_matrix.shape[1])
    print 'New shape:', data_matrix.shape
    print ""

    ### Remove empty samples
    totals = (data_matrix > 0)
    x = totals.sum(axis=1)
    min_threshold = 100
    keep = [i for i in xrange(len(x)) if x[i] > min_threshold]
    previous_num_samples = data_matrix.shape[0]
    data_matrix = data_matrix[keep, :]
    sample_ids = sample_ids[keep]
    labels = labels[keep]
    print 'Removing empty samples... Removed %d samples.' % (previous_num_samples-data_matrix.shape[0])
    print ""

    label_legend = nonzero_label_legend
    print 'Breakdown of values:'
    label_counts = Counter(labels)
    for i, l in enumerate(label_legend):
        print '+ %d \t %s ' % (label_counts[i], l)
    print ""

    # Save off everything
    labels = np.array(labels)
    np.save(path.join(output_dir, 'data_matrix'), data_matrix)
    np.save(path.join(output_dir, 'sample_ids'), sample_ids)
    np.save(path.join(output_dir, 'labels'), labels)


    # Prepare batch job
    print 'Preparing batch job...'
    cmd = '/Users/sawa6416/Projects/biotm/scripts/prepare_batch_redux.py -i %s -l %s -o %s -s %s -k %d -n %d -t %s' % (path.join(output_dir, 'data_matrix.npy'),
                                                                                                                       path.join(output_dir, 'labels.npy'),
                                                                                                                       output_dir,
                                                                                                                       scripts_dir,
                                                                                                                       num_folds,
                                                                                                                       num_iters
                                                                                                                       metadata_category)
    system(cmd)

    print 'Done!'
