import numpy as np
cimport numpy as np


""" Probabilistic Latent Semantic Analysis
    
    This .pyx file serves as the interfacing layer between
    the biotm PLSA (plsa.py) code and plsa_em_code.c, a
    C implementation of the EM algorithm. 

    nonzero() function and process of interfacing Python
    and C code based on Mathieu Blondel's PLSA implementation.
    Original post: http://www.mblondel.org/journal/2010/06/13/lsa-and-plsa-in-python/
    Git repo:  http://www.mblondel.org/code/plsa.git

    Mathieu's code also served as a template for setting
    up the Python/C interface.
""" 

""" Interfaces to C code """
cdef extern from "plsa_em_code.c":
    double _plsa_em(double *X,
                    unsigned int X_size,
                    double *p_w_z,
                    double *p_z_d,
                    double *p_d,
                    unsigned int num_words,
                    unsigned int num_docs,
                    unsigned int num_topics,
                    unsigned int folding,
                    double min_delta_l,
                    unsigned int max_em_iter)

cdef extern from "plsa_em_code.c":
    double _log_likelihood(double *X,
                           unsigned int X_size,
                           double *p_w_z,
                           double *p_z_d,
                           double *p_d,
                           unsigned int num_words,
                           unsigned int num_docs,
                           unsigned int num_topics)

cdef extern from "plsa_em_code.c":
    void _em_e_step(double *p_z_wd,
                    double *p_w_z,
                    double *p_z_d,
                    unsigned int num_words,
                    unsigned int num_docs,
                    unsigned int num_topics)

cdef extern from "plsa_em_code.c":
    void _em_m_step(double *X,
                unsigned int X_size,
                double *p_z_wd,
                double *p_w_z,
                double *p_z_d,
                unsigned int num_words,
                unsigned int num_docs,
                unsigned int num_topics,
                unsigned int folding)

cdef extern from "plsa_em_code.c":
    void normalize2d(double *c_array, unsigned int rows,
                     unsigned int cols)


def nonzero(td):
    """ Convert a sparse matrix td to a Nx3 matrix where N is the number of
        non-zero elements. The 1st column is the word count, the 2nd is the
        word index and the 3rd column is the document index.

        Inputs:
          -td: term-document matrix. 

        Returns:
          -a sparse representation of the matrix, stored as a 
           fortran array.  For example:
         
        >>> X = zeros((10, 10))
        >>> X[1][2] = 3
        >>> X[4][5] = 6
        >>> nonzero(X)
            array([[ 3.,  1.,  2.],
                   [ 6.,  4.,  5.]])

        NOTE: Elements of the returned array are (val, row, col)
    """
    rows, cols = td.nonzero()
    vals = td[rows,cols]
    if "scipy.sparse" in str(vals.__class__): vals = vals.toarray()
    return np.asfortranarray(np.vstack((vals, rows, cols)).T) 


def plsa_em(X,
            np.ndarray[np.float64_t, ndim=2, mode='c']p_w_z,
            np.ndarray[np.float64_t, ndim=2, mode='c']p_z_d,
            np.ndarray[np.float64_t, ndim=1, mode='c']p_d,
            folding=False,
            min_delta_l=0.0001,
            max_em_iter=10000):
    """ Prepare data to be passed into external C function
        to run expectation maximization algorithm for PLSA. 
        
        Inputs:
          -X:  term document matrix
          -p_w_z:  Distribution of words given topics P(w|z)
          -p_z_d:  Distribution of topics given docs P(z|d)
          -p_d:  Probabilities for each document
          -folding: (Boolean) whether or not to perform folding.
                    In folding, P(w|z) is held fixed and only
                    P(z|d) is optimized.  
          -min_delta_l:  Minimum change for log-likelihood.
          -max_em_iter:  Maximum number of iterations in 
                         the EM algorithm. 
    """

    cdef np.ndarray[np.float64_t, ndim=2, mode='fortran'] nonzero_X
    nonzero_X = nonzero(X).astype(np.float64)

    num_words = p_w_z.shape[0]
    num_docs = p_z_d.shape[1]
    num_topics = p_z_d.shape[0]

    return _plsa_em(<double *>nonzero_X.data,
                    <unsigned int>len(nonzero_X),
                    <double *>p_w_z.data,
                    <double *>p_z_d.data,
                    <double *>p_d.data,
                    <unsigned int>num_words,
                     <unsigned int>num_docs,
                     <unsigned int>num_topics,
                     <unsigned int>folding,
                     <double>min_delta_l,
                     <unsigned int>max_em_iter)


def em_e_step(np.ndarray[np.float64_t, ndim=3, mode='c']p_z_wd,
              np.ndarray[np.float64_t, ndim=2, mode='c']p_w_z,
              np.ndarray[np.float64_t, ndim=2, mode='c']p_z_d):
    """ Performs the E-step of the EM algorithm.  
        This function serves an interface to the corresponding
        function for the E-step written in C.  To run the full
        EM algorithm, use plsa_em() above.  
    """
    num_words = p_w_z.shape[0]
    num_docs = p_z_d.shape[1]
    num_topics = p_z_d.shape[0]

    _em_e_step(<double *>p_z_wd.data,
               <double *>p_w_z.data,
               <double *>p_z_d.data,
               <unsigned int> num_words,
               <unsigned int> num_docs,
               <unsigned int> num_topics)


def em_m_step(X, 
              np.ndarray[np.float64_t, ndim=2, mode='c']p_w_z,
              np.ndarray[np.float64_t, ndim=2, mode='c']p_z_d,
              np.ndarray[np.float64_t, ndim=3, mode='c']p_z_wd,
              folding=False): 
    """ Performs the E-step of the EM algorithm.  
        This function serves an interface to the corresponding
        function for the M-step written in C.  To run the full
        EM algorithm, use plsa_em() above.  
    """ 
    cdef np.ndarray[np.float64_t, ndim=2, mode='fortran'] nonzero_X
    nonzero_X = nonzero(X).astype(np.float64)

    num_words = p_w_z.shape[0]
    num_docs = p_z_d.shape[1]
    num_topics = p_z_d.shape[0]

    _em_m_step(<double *>nonzero_X.data,
               <unsigned int>len(nonzero_X),
               <double *>p_z_wd.data,
               <double *>p_w_z.data,
               <double *>p_z_d.data,
               <unsigned int> num_words,
               <unsigned int> num_docs,
               <unsigned int> num_topics,
               <unsigned int> folding)


def log_likelihood(X,
                   np.ndarray[np.float64_t, ndim=2, mode='c']p_w_z,
                   np.ndarray[np.float64_t, ndim=2, mode='c']p_z_d,
                   np.ndarray[np.float64_t, ndim=1, mode='c']p_d):
    """ Performs calculation of log-liklihood.  
        This function serves an interface to the corresponding
        function written in C.
    """ 
    cdef np.ndarray[np.float64_t, ndim=2, mode='fortran'] nonzero_X
    nonzero_X = nonzero(X).astype(np.float64)
    num_words = p_w_z.shape[0]
    num_docs = p_z_d.shape[1]
    num_topics = p_z_d.shape[0]

    return _log_likelihood(<double *>nonzero_X.data,
                           <unsigned int>len(nonzero_X),
                           <double *>p_w_z.data,
                           <double *>p_z_d.data,
                           <double *>p_d.data,
                           <unsigned int>num_words,
                           <unsigned int>num_docs,
                           <unsigned int>num_topics)


def normalize(np.ndarray[np.float64_t, ndim=2, mode='c']p_w_z):
    """ Performs normalization of a conditional probability matrix.
        This function serves as an interface to the corresponding
        function written in C.
    """
    num_words = p_w_z.shape[0]
    num_topics = p_w_z.shape[1]
    normalize2d(<double *>p_w_z.data, 
                <unsigned int>num_words,
                <unsigned int>num_topics) 
