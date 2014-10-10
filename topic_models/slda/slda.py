#!/usr/bin/env python

from convert_qiime_to_slda import create_slda_dataset
from subprocess import call
from os.path import join as path_join

""" 
    Naive Python Interface to SLDA.

    The `slda` class provides an interface to C. Brown's implementation
    of SLDA (https://github.com/chbrown/slda).  This used to be linked
    by David Blei's website.
""" 
SLDA_EXE = 'slda'
SLDA_SETTINGS = '/Users/samway/Documents/Work/TopicModeling/slda/settings.txt'
SLDA_INIT = 'random'

class slda:
    def __init__(self, n_components=2, temp_dir='./', alpha=1.0):
        self.num_topics = n_components
        self.temp_dir = temp_dir 
        self.alpha = alpha
        self.model_file = None


    def fit(self, X, y=None):
        """ Run estimation """
        # 1 - Create an SLDA dataset from the supplied matrices
        # 2 - Run estimation
        # 3 - Save link to model
        data_file, labels_file = create_slda_dataset(X, y, self.temp_dir) 
        call([SLDA_EXE, 'est', data_file, labels_file, SDLA_SETTINGS, 
              self.alpha, self.num_topics, SLDA_INIT, self.temp_dir])
        self.model_file = path_join(self.temp_dir, 'final.model')

        
    def transform(self, X, y=None):
        """ Run inference """
        # Check to make sure there is a model file (that you've ran fit())
        if self.model_file is None:
            raise ValueError('Attempt to use a model before training it!')
        data_file, labels_file = create_slda_dataset(X, y, self.temp_dir) 
        call([SLDA_EXE, 'inf', data_file, labels_file, SDLA_SETTINGS, 
              self.model_file, self.temp_dir])
        gamma_file = path_join(self.temp_dir, 'inf-gamma.dat')
        Xbar = array([[float(x) for x in line.strip().split()]for line in open(gamma_file, 'rU')])
        return Xbar
        

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    
