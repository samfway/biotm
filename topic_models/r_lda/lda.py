#!/usr/bin/env python

from biotm.topic_models.slda.convert_qiime_to_slda import create_slda_dataset
from tempfile import mkdtemp
from subprocess import call
from os import system
from numpy import array
from os.path import join as path_join

""" 
    THIS IS VERY MUCH A COPY-PASTE OF THE SLDA WRAPPER WITH
    "SLDA" REPLACED WITH "LDA".  A COMBINED SCRIPT WOULD BE
    MUCH CLEANER.. 

    In due time.. 

    Naive Python Interface to LDA.
    The `lda` class provides an interface to R's lda package
""" 
#LDA_CMD = 'Rscript /Users/samway/Documents/Work/TopicModeling/biotm/topic_models/r_lda/lda.r ' \
#           '--source_dir /Users/samway/Documents/Work/TopicModeling/biotm/topic_models/r_lda/ ' \
#           '-i %s -l %s -a %s -m %s -s %s -o %s'
#LDA_SCRATCH = '/Users/samway/Documents/Work/TopicModeling/slda/scratch/'
LDA_CMD = 'Rscript /Users/sawa6416/Projects/biotm/topic_models/r_lda/lda.r ' \
           '--source_dir /Users/sawa6416/Projects/biotm/topic_models/r_lda/ '\
           '-i %s -l %s -a %s -m %s -s %s -o %s -k %d -w %d'
LDA_SCRATCH = '/Users/sawa6416/Tools/slda/scratch/'

class lda:
    def __init__(self, n_components=2, alpha=0.1):
        self.num_topics = n_components
        self.temp_dir = mkdtemp(dir=LDA_SCRATCH, prefix='lda')
        self.alpha = alpha
        self.model_file = None
        self.vocab_size = -1


    def __del__(self):
        if 'lda' in self.temp_dir:
            system("rm -rf %s" % (self.temp_dir))


    def fit(self, X, y=None):
        """ Run estimation """
        # 1 - Create an LDA dataset from the supplied matrices
        # 2 - Run estimation
        # 3 - Save link to model
        self.model_file = path_join(self.temp_dir, 'final.model')
        self.vocab_size = X.shape[1]

        system('rm -f %s' % (self.model_file))
        data_file, labels_file = create_slda_dataset(X, y, path_join(self.temp_dir,'')) 

        cmd = LDA_CMD % (data_file, labels_file, "lda", "est", 
                         self.model_file, self.temp_dir, self.num_topics, self.vocab_size)
        system(cmd)

        system('sleep 3')  # Allow for files to be written... 
        system('rm -f %s' % (data_file))
        system('rm -f %s' % (labels_file))

        
    def transform(self, X, y=None):
        """ Run inference """
        # Check to make sure there is a model file (that you've ran fit())
        if self.model_file is None:
            raise ValueError('Attempt to use a model before training it!')
        if X.shape[1] != self.vocab_size:
            raise ValueError('Matrix to be transformed of wrong shape!')

        data_file, labels_file = create_slda_dataset(X, y, path_join(self.temp_dir, ''))
        cmd = LDA_CMD % (data_file, labels_file, "lda", "inf", self.model_file, 
                         self.temp_dir, self.num_topics, self.vocab_size)
        system(cmd)
        system('sleep 3')

        results_file = path_join(self.temp_dir, 'tc.out')
        Xbar = array([[float(x) for x in line.strip().split()]for line in open(results_file, 'rU')])
        return Xbar
        

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    
