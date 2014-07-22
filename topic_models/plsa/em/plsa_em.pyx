import numpy as np
cimport numpy as np


""" nonzero() function and process of interfacing Python
    and C code based on Mathieu Blondel's PLSA implementation.
    Original post: http://www.mblondel.org/journal/2010/06/13/lsa-and-plsa-in-python/
    Git repo:  http://www.mblondel.org/code/plsa.git
""" 

cdef extern from "plsa_em_code.c":
    double _plsa_em(unsigned int *X,
                    unsigned int X_size,
                    double *p_w_z_best,
                    double *p_z_d_best,
                    double *p_d,
                    unsigned int num_words,
                    unsigned int num_docs,
                    unsigned int num_topics,
                    unsigned int folding,
                    double min_delta_l,
                    unsigned int max_em_iter)

cdef extern from "plsa_em_code.c":
    double _log_likelihood(unsigned int *X,
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
    void _em_e_step(double *p_z_wd,
                double *p_w_z,
                double *p_z_d,
                unsigned int num_words,
                unsigned int num_docs,
                unsigned int num_topics)

cdef extern from "plsa_em_code.c":
    void normalize_axis1(double *c_array, unsigned int rows,
                         unsigned int cols)


def nonzero(td):
    """
    Convert a sparse matrix td to a Nx3 matrix where N is the number of non-zero
    elements. The 1st column is the word count, the 2nd is the word index
    and the 3rd column is the document index.
    """
    rows, cols = td.nonzero()
    vals = td[rows,cols]
    if "scipy.sparse" in str(vals.__class__): vals = vals.toarray()
    return np.asfortranarray(np.vstack((vals, rows, cols)).T) 


def plsa_em(X,
            np.ndarray[np.float64_t, ndim=2, mode='c']p_w_z,
            np.ndarray[np.float64_t, ndim=2, mode='c']p_z_d,
            np.ndarray[np.float64_t, ndim=1, mode='c']p_d,
            folding,
            min_delta_l=0.0001,
            max_em_iter=10000):
    """ Prepare data to be passed into external C function """ 

    cdef np.ndarray[np.uint32_t, ndim=2, mode='fortran'] nonzero_X
    nonzero_X = nonzero(X).astype(np.uint32)

    num_words = p_w_z.shape[0]
    num_docs = p_z_d.shape[1]
    num_topics = p_z_d.shape[0]

    _plsa_em(<unsigned int *>nonzero_X.data,
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


def em_e_step(p_z_wd, p_w_z, p_z_d):
    pass

def em_m_step(X, p_w_z, p_z_d, p_z_wd, folding=False): 
    pass 

def log_likelihood(X,
                   np.ndarray[np.float64_t, ndim=2, mode='c']p_w_z,
                   np.ndarray[np.float64_t, ndim=2, mode='c']p_z_d,
                   np.ndarray[np.float64_t, ndim=1, mode='c']p_d):
    """ C implementation of log_likelihood calculations """ 
    cdef np.ndarray[np.uint32_t, ndim=2, mode='fortran'] nonzero_X
    nonzero_X = nonzero(X).astype(np.uint32)
    num_words = p_w_z.shape[0]
    num_docs = p_z_d.shape[1]
    num_topics = p_z_d.shape[0]

    return _log_likelihood(<unsigned int *>nonzero_X.data,
                           <unsigned int>len(nonzero_X),
                           <double *>p_w_z.data,
                           <double *>p_z_d.data,
                           <double *>p_d.data,
                           <unsigned int>num_words,
                           <unsigned int>num_docs,
                           <unsigned int>num_topics)

def normalize(np.ndarray[np.float64_t, ndim=2, mode='c']p_w_z):
    num_words = p_w_z.shape[0]
    num_topics = p_w_z.shape[1]
    normalize_axis1(<double *>p_w_z.data, 
                    <unsigned int>num_words,
                    <unsigned int>num_topics) 
