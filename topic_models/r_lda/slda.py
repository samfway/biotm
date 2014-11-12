#!/usr/bin/env python

from biotm.topic_models.slda.convert_qiime_to_slda import create_slda_dataset
from tempfile import mkdtemp
from subprocess import call
from os import system
from numpy import array
from os.path import join as path_join
from time import sleep

""" 
    Naive Python Interface to SLDA.
    The `slda` class provides an interface to R's lda package
""" 
LDA_CMD = 'Rscript /Users/sawa6416/Projects/biotm/topic_models/r_lda/lda.r ' \
           '--source_dir /Users/sawa6416/Projects/biotm/topic_models/r_lda/ '\
           '-i %s -l %s -a %s -m %s -s %s -o %s -k %d -w %d'
LDA_SCRATCH = '/Users/sawa6416/Tools/slda/scratch/'


def safe_open(filename, mode, num_retries=10, pause=3):
    """ Return file handle, if not available, wait 
        and try again. Give up after num_retries. """
    for i in xrange(num_retries):
        try:
            handle = open(filename, mode)
            break
        except IOError:
            print 'Snooze..'
            sleep(pause)
    else:
        raise IOError('Failed to safe-open %s' % (filename))
    return handle


class base_lda:
    def __init__(self, n_components=2, alpha=0.1, method='lda'):
        self.method = method
        self.num_topics = n_components
        self.num_words = -1
        self.temp_dir = mkdtemp(dir=LDA_SCRATCH, prefix=method)
        self.alpha = alpha
        self.model_file = None
        self.vocab_size = -1


    def __del__(self):
        if self.method in self.temp_dir:
            system("rm -rf %s" % (self.temp_dir))


    def fit(self, X, y=None):
        """ Run estimation """
        # 1 - Create an SLDA dataset from the supplied matrices
        # 2 - Run estimation
        # 3 - Save link to model
        self.model_file = path_join(self.temp_dir, 'final.model')
        self.num_words = X.shape[1]

        system('rm -f %s' % (self.model_file))
        data_file, labels_file = create_slda_dataset(X, y, path_join(self.temp_dir,'')) 

        cmd = LDA_CMD % (data_file, labels_file, self.method, "est", self.model_file, 
                          self.temp_dir, self.num_topics, self.num_words)
        system(cmd)

        sleep(3)
        system('rm -f %s' % (data_file))
        system('rm -f %s' % (labels_file))

        
    def transform(self, X, y=None):
        """ Run inference """
        # Check to make sure there is a model file (that you've ran fit())
        if self.model_file is None:
            raise ValueError('Attempt to use a model before training it!')
        if X.shape[1] != self.num_words:
            raise ValueError('Matrix to be processed of improper dimension.')
            
        data_file, labels_file = create_slda_dataset(X, y, path_join(self.temp_dir, ''))
        results_file = path_join(self.temp_dir, 'tc.out')

        sleep(3)
        system('rm -f %s' % (results_file))  # Make sure results file doesn't exist already
        cmd = LDA_CMD % (data_file, labels_file, self.method, "inf", self.model_file,
                          self.temp_dir, self.num_topics, self.num_words)
        system(cmd)
        sleep(3)

        Xbar = array([[float(x) for x in line.strip().split()] for line in safe_open(results_file, 'rU')])
        return Xbar
        

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
   
class lda(base_lda):
    def __init__(self, n_components=2, alpha=0.1):
        base_lda.__init__(self, n_components, alpha, method='lda')
    
class slda(base_lda):
    def __init__(self, n_components=2, alpha=0.1):
        base_lda.__init__(self, n_components, alpha, method='slda')
    
