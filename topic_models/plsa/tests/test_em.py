#!/usr/bin/env python
"""Unit tests for expectation maximization."""

from biotm.topic_models.plsa.plsa import log_likelihood, em_e_step, em_m_step

from unittest import TestCase, main
from numpy import array, zeros

class logLikelihoodTests(TestCase):
    """ Test parsing of log-likelihood calculation. """
    def setUp(self):
        # 3docs, 4words, 5topics

        """ SAM SAM SAM... there's a glaring flaw here, which 
            is that the rows sum to one instead of the columns!
        
            These test cases will still work fine, technically, 
            but should be updated as soon as possible.  
        """ 

        self.X = array([[1, 2, 3, 4],
                        [1, 1, 1, 1],
                        [7, 4, 2, 8]])  # (3,4)
        self.p_w_z = array([[0.1, 0.7, 0.0, 0.1, 0.1],
                            [0.1, 0.1, 0.6, 0.1, 0.1],
                            [0.1, 0.1, 0.1, 0.6, 0.1],
                            [0.1, 0.1, 0.1, 0.2, 0.5]])  # (4,5)
        self.p_z_d = array([[0.10, 0.20, 0.70],
                            [0.30, 0.60, 0.10],
                            [0.20, 0.20, 0.60],
                            [0.20, 0.40, 0.40],
                            [0.10, 0.50, 0.40]])  # (5,3)
        self.p_d = array([0.2, 0.2, 0.6])  # (3,1)


    def test_log_likelihood(self):
        """ Make sure the calculation of negative
            log-likelihood is correct. """ 
        ll = log_likelihood(self.X, self.p_w_z, 
                            self.p_z_d, self.p_d)
        self.assertAlmostEqual(ll, -76.0858, places=4)
        """ This number was calculated more or less by 
            hand as a test case.  """ 


    def test_e_step(self):
        num_docs, num_words = self.X.shape
        _,num_topics = self.p_w_z.shape

        p_z_wd = zeros((num_topics, num_words, num_docs))
        em_e_step(p_z_wd, self.p_w_z, self.p_z_d) 

        error1 = p_z_wd[:,:,0] - \
        array([[ 0.04,   0.05263158,  0.05263158,  0.06666667],
               [ 0.84,   0.15789474,  0.15789474,  0.2       ],
               [ 0.,     0.63157895,  0.10526316,  0.13333333],
               [ 0.08,   0.10526316,  0.63157895,  0.26666667],
               [ 0.04,   0.05263158,  0.05263158,  0.33333333]])
        error1 = error1.sum()
    
        error2 = p_z_wd[:,:,1] - \
        array([[ 0.03773585,  0.06896552,  0.05128205,  0.04651163],
         [ 0.79245283,  0.20689655,  0.15384615,  0.13953488],
         [ 0.,          0.4137931,   0.05128205,  0.04651163],
         [ 0.0754717,   0.13793103,  0.61538462,  0.18604651],
         [ 0.09433962,  0.17241379,  0.12820513,  0.58139535]])
        error2 = error2.sum()

        error3 = p_z_wd[:,:,2] - \
        array([[ 0.31818182,  0.13461538,  0.16666667,  0.16666667],
         [ 0.31818182,  0.01923077,  0.02380952,  0.02380952],
         [ 0.,          0.69230769,  0.14285714,  0.14285714],
         [ 0.18181818,  0.07692308,  0.57142857,  0.19047619],
         [ 0.18181818,  0.07692308,  0.0952381,   0.47619048]])
        error3 = error3.sum()
        self.assertAlmostEqual(error1+error2+error3, 0., places=7)

    
    def test_m_step(self):
        num_docs, num_words = self.X.shape
        _,num_topics = self.p_w_z.shape

        p_w_z = self.p_w_z.copy()
        p_z_d = self.p_z_d.copy()
        p_z_wd = zeros((num_topics, num_words, num_docs))

        em_e_step(p_z_wd, p_w_z, p_z_d)
        em_m_step(self.X, p_w_z, p_z_d, p_z_wd)

        error_p_w_z  = p_w_z - array([
            [0.442698731175421,   0.616127152710356,                   0,   0.167750075743908,   0.171734582815257],
            [0.136878906447827,   0.095715470817479,   0.651773242410349,   0.077068501707515,   0.071445178452401],
            [0.104194207728943,   0.107773954047137,   0.095692966838378,   0.429063071329947,   0.058166808091669],
            [0.316228154647808,   0.180383422425029,   0.252533790751273,   0.326118351218630,   0.698653430640674]])
        error_p_w_z = error_p_w_z.sum()
        self.assertAlmostEqual(0., error_p_w_z, places=7)
    
        error_p_z_d = p_z_d - array([ 
            [0.056982456140351,   0.051123761371753,   0.211066711066711],
            [0.242947368421053,   0.323182604869975,   0.121061478204335],
            [0.211228070175439,   0.127896695659326,   0.199895342752486],
            [0.325192982456140,   0.253708464902122,   0.202242202242202],
            [0.163649122807018,   0.244088473196824,   0.265734265734266]])
        error_p_z_d = error_p_z_d.sum()
        self.assertAlmostEqual(0., error_p_z_d, places=7)


if __name__ == '__main__':
    main()
