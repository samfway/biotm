#include <stdio.h>
#include <time.h>
#include "plsa_em_code.h"

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
{
    double *p_z_wd;
    double l_current, l_old;
    unsigned int i, j, k, em_iter, amount;

    // Seed the random number generator
    srand(time(NULL));

    p_z_wd = (double *) calloc(num_topics*num_words*num_docs,
                               sizeof(double));

    // Initialize p_z_d and (potentially) p_w_z 
    randomize(p_z_d, num_topics*num_docs);
    normalize2d(p_z_d, num_topics, num_docs);
    if (!folding)
    {
        randomize(p_w_z, num_words*num_topics);
        normalize2d(p_w_z, num_words, num_topics);
    }

    l_current = _log_likelihood(X, X_size, p_w_z,
                                p_z_d, p_d, num_words,
                                num_docs, num_topics);
    l_old = 0.0; 

    for (em_iter=0; em_iter<max_em_iter; em_iter++)
    {
    
        l_old = l_current;     
        _em_e_step(p_z_wd, p_w_z, p_z_d, num_words,
                   num_docs, num_topics);
        _em_m_step(X, X_size, p_z_wd, p_w_z, p_z_d,
                   num_words, num_docs, num_topics,
                   folding);
        l_current = _log_likelihood(X, X_size, p_w_z,
                                p_z_d, p_d, num_words,
                                num_docs, num_topics);

        if (abs(l_current-l_old) < abs(l_old)*min_delta_l)
            break;
    }

    // Clean up
    free(p_z_wd); 

    return l_current;
}

void _em_e_step(double *p_z_wd,
                double *p_w_z,
                double *p_z_d,
                unsigned int num_words,
                unsigned int num_docs,
                unsigned int num_topics)
{
    unsigned int i, j, k;
    double total_prob; 

    // Calculate P(z | w,d)
    for (i=0; i<num_docs; i++)
    {
        for (j=0; j<num_words; j++)
        {
            for (k=0; k<num_topics; k++)
            {
                // (sj, sk, i, j, k) (i*sj*sk + j*sk + k)
                p_z_wd[C3_(num_words, num_docs, k, j, i)] = 
                    p_w_z[C_(num_topics, j, k)] 
                  * p_z_d[C_(num_docs, k, i)]; 
            }
        }
    }

    // Normalize 
    for (i=0; i<num_docs; i++)
    {
        for (j=0; j<num_words; j++)
        {
            total_prob = 0.0;
            for (k=0; k<num_topics; k++)
            {
                total_prob += p_w_z[C_(num_topics, j, k)] 
                    * p_z_d[C_(num_docs, k, i)];
            }

            if (total_prob > 0.0)
            {
                for (k=0; k<num_topics; k++)
                {
                    p_z_wd[C3_(num_words, num_docs, k, j, i)] 
                        /= total_prob;
                }
            }
            else
            {
                for (k=0; k<num_topics; k++)
                {
                    p_z_wd[C3_(num_words, num_docs, k, j, i)] = 0.0;
                }
            }
        }
    }
}

void _em_m_step(double *X,
                unsigned int X_size,
                double *p_z_wd,
                double *p_w_z,
                double *p_z_d,
                unsigned int num_words,
                unsigned int num_docs,
                unsigned int num_topics,
                unsigned int folding)
{
    unsigned int i, j, k, doc, word; 
    double total_count, word_count; 

    if (!folding)
    {
        // Update P(w|z) if NOT folding in
        bzero(p_w_z, num_words*num_topics*sizeof(double));

        for (i=0; i<X_size; i++)
        {
            word_count = X[i];
            doc = (unsigned int)X[F_(X_size, i, 1)];
            word = (unsigned int)X[F_(X_size, i, 2)];
            for (k=0; k<num_topics; k++)
            {
                p_w_z[C_(num_topics, word, k)] += word_count *
                    p_z_wd[C3_(num_words, num_docs, k, word, doc)]; 
            }
        }
        normalize2d(p_w_z, num_words, num_topics);
    }

    // Update P(z|d) 
    bzero(p_z_d, num_topics*num_docs*sizeof(double));
    for (i=0; i<X_size; i++)
    {
        word_count = X[i];
        doc = (unsigned int)X[F_(X_size, i, 1)];
        word = (unsigned int)X[F_(X_size, i, 2)];

        for (k=0; k<num_topics; k++)
        {
            p_z_d[C_(num_docs, k, doc)] += word_count * 
                p_z_wd[C3_(num_words, num_docs, k, word, doc)];
        }
    }

    normalize2d(p_z_d, num_topics, num_docs);
}


// Randomize the elements of an array
void randomize(double *array, unsigned int num_elements)
{
    unsigned int i;
    double temp;

    for (i=0; i<num_elements; i++)
    {
        array[i] = (double)rand() / RAND_MAX;
    }
}

/*  Normalize a 2D conditional probability matrix.
    For example, p_w_z = P(w|z).  For given topic z,
    the probabilities of the words w must sum to one. 
   
    To normalize, you would want to call...
      normalize2d(p_w_z, num_words, num_topics);
    and this will ensure that the probability of all
    words given a topic sums to one.

    NOTE: 
    p_w_z is a 1D array, though with size num_words*num_topics
*/
void normalize2d(double *c_array, unsigned int rows,
                     unsigned int cols)
{
    unsigned int r, c;
    double sum;

    for (c=0; c<cols; c++)
    {
        sum = 0.0; 
        for (r=0; r<rows; r++)
        {
            sum += c_array[C_(cols, r, c)];
        }
        
        if (sum > 0.0)
        {
            for (r=0; r<rows; r++)
            {
                c_array[C_(cols, r, c)] /= sum; 
            }
        }
    }

}

double _log_likelihood(double *X,
                       unsigned int X_size,
                       double *p_w_z,
                       double *p_z_d,
                       double *p_d,
                       unsigned int num_words,
                       unsigned int num_docs,
                       unsigned int num_topics)
{
    unsigned int i, k, word, doc;
    double log_likelihood = 0.0;
    double inner_sum, p_dw, word_count;

    for (i=0; i<X_size; i++)
    {
        word_count = X[i];
        doc = (unsigned int)X[F_(X_size, i, 1)];
        word = (unsigned int)X[F_(X_size, i, 2)];
        inner_sum = 0.0; 

        for (k=0; k<num_topics; k++)
        {
            inner_sum += p_w_z[C_(num_topics, word, k)] 
                         * p_z_d[C_(num_docs, k, doc)];
        }
        p_dw = inner_sum * p_d[doc];
        if (p_dw > 0.0)
        {
            log_likelihood += word_count * log(p_dw);
        }
    }

    return log_likelihood;
}

