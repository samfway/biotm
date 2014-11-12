#!/usr/bin/env python

import argparse
from numpy import load as np_load
from numpy import save as np_save
from os import path

from sklearn.decomposition import TruncatedSVD, KernelPCA, FastICA, MiniBatchDictionaryLearning, NMF
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from biotm.topic_models.plsa.plsa import plsa
from biotm.topic_models.r_lda.slda import slda, lda
from biotm.scripts.batch.util import get_methods


def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input-file', help='Input data matrix', required=True)
    args.add_argument('-l', '--input-labels', help='Input labels', required=True)
    args.add_argument('-o', '--output-dir', help='Output directory', default='./')
    args.add_argument('-r', '--training-vector', help='Train indices', required=True)
    args.add_argument('-t', '--testing-vector', help='Test indices', required=True)
    args.add_argument('-k', '--num-dims', help='Number of dimensions', type=int, required=True)
    args.add_argument('-n', '--cv-fold-id', help='CV fold id', type=int, required=True)
    args.add_argument('-m', '--methods', help='Dimensionality reduction methods (comma-sep)',
                      default='svd,nmf,plsa,lda,slda')
    args = args.parse_args()
    return args


if __name__=="__main__":
    args = interface()
    
    num_dims = args.num_dims
    prefix = 'CV_%d' % (args.cv_fold_id)
    techniques = get_methods(args.methods)
    output_dir = args.output_dir

    data_matrix = np_load(args.input_file)
    labels = np_load(args.input_labels)
    testing = np_load(args.testing_vector)
    training = np_load(args.training_vector)

    test_labels = labels[testing]
    training_labels = labels[training]
    test_matrix = data_matrix[testing, :]
    training_matrix = data_matrix[training, :]

    for technique, technique_name in techniques:
        dim_redux = technique(n_components=num_dims)
        file_label = '%s_%s_%s_' % (prefix,
                                    technique_name,
                                    str(num_dims))
        file_prefix = path.join(output_dir, file_label)

        txd_training_matrix = dim_redux.fit_transform(training_matrix, training_labels)
        txd_test_matrix = dim_redux.transform(test_matrix)
        np_save(file_prefix + 'txd_test_matrix', txd_test_matrix)
        np_save(file_prefix + 'txd_training_matrix', txd_training_matrix)
