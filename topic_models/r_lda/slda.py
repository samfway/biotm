#!/usr/bin/env python

from biotm.topic_models.slda.convert_qiime_to_slda import create_slda_dataset
from tempfile import mkdtemp
from subprocess import call
from os import system
from numpy import array
from os.path import join as path_join

""" 
    Naive Python Interface to SLDA.
    The `slda` class provides an interface to R's lda package
""" 
SLDA_CMD = 'Rscript /Users/sawa6416/Projects/biotm/topic_models/r_lda/lda.r ' \
           '--source_dir /Users/sawa6416/Projects/biotm/topic_models/r_lda/ '\
           '-i %s -l %s -a %s -m %s -s %s -o %s -k %d'
SLDA_SCRATCH = '/Users/sawa6416/Tools/slda/scratch/'

class slda:
    def __init__(self, n_components=2, alpha=0.1):
        self.num_topics = n_components
        self.temp_dir = mkdtemp(dir=SLDA_SCRATCH, prefix='slda')
        self.alpha = alpha
        self.model_file = None


    def __del__(self):
        if 'slda' in self.temp_dir:
            system("rm -rf %s" % (self.temp_dir))


    def fit(self, X, y=None):
        """ Run estimation """
        # 1 - Create an SLDA dataset from the supplied matrices
        # 2 - Run estimation
        # 3 - Save link to model
        self.model_file = path_join(self.temp_dir, 'final.model')
        system('rm -f %s' % (self.model_file))
        data_file, labels_file = create_slda_dataset(X, y, path_join(self.temp_dir,'')) 

        cmd = SLDA_CMD % (data_file, labels_file, "slda", "est", self.model_file, 
                          self.temp_dir, self.num_topics)
        system(cmd)

        system('sleep 3')  # Allow for files to be written... 
        system('rm -f %s' % (data_file))
        system('rm -f %s' % (labels_file))

        
    def transform(self, X, y=None):
        """ Run inference """
        # Check to make sure there is a model file (that you've ran fit())
        if self.model_file is None:
            raise ValueError('Attempt to use a model before training it!')
        data_file, labels_file = create_slda_dataset(X, y, path_join(self.temp_dir, ''))

        cmd = SLDA_CMD % (data_file, labels_file, "slda", "inf", self.model_file,
                          self.temp_dir, self.num_topics)
        system(cmd)
        system('sleep 3')

        results_file = path_join(self.temp_dir, 'tc.out')
        Xbar = array([[float(x) for x in line.strip().split()]for line in open(results_file, 'rU')])
        return Xbar
        

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    
