#include <stdio.h>
#include <time.h>

#define C_(cols, i, j) (i*cols + j)
#define C3_(rows, cols, i, j, k) (i*rows*cols + j*cols + k)
#define F_(rows, i, j) (j*rows + i)
#define SWAP(a, b, temp) temp=a; a=b; b=temp;

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
                unsigned int max_em_iter);

double _log_likelihood(unsigned int *X,
                       unsigned int X_size,
                       double *p_w_z,
                       double *p_z_d,
                       double *p_d,
                       unsigned int num_words,
                       unsigned int num_docs,
                       unsigned int num_topics);

void _em_e_step(double *p_z_wd,
                double *p_w_z,
                double *p_z_d,
                unsigned int num_words,
                unsigned int num_docs,
                unsigned int num_topics);

void _em_m_step(unsigned int *X,
                unsigned int X_size,
                double *p_z_wd,
                double *p_w_z,
                double *p_z_d,
                unsigned int num_words,
                unsigned int num_docs,
                unsigned int num_topics,
                unsigned int folding); 

void randomize(double *array, unsigned int num_elements);
void normalize_axis1(double *c_array, unsigned int rows,
                     unsigned int cols);

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
{
    double *p_w_z, *p_z_d, *p_z_wd;
    double l_current, l_old;
    unsigned int i, j, k, em_iter, amount;

    p_w_z = (double *) calloc(num_words*num_topics, sizeof(double));
    p_z_d = (double *) calloc(num_topics*num_docs, sizeof(double));
    p_z_wd = (double *) calloc(num_topics*num_words*num_docs,
                               sizeof(double));

    
    randomize(p_z_d, num_topics*num_docs);
    normalize_axis1(p_z_d, num_topics, num_docs);

    for (em_iter=0; em_iter<max_em_iter; em_iter++)
    {

    }

    free(p_w_z);
    free(p_z_d);
    free(p_z_wd); 
    
    return 2.0;
}


// Randomize the elements of an array
void randomize(double *array, unsigned int num_elements)
{
    unsigned int i;
    for (i=0; i<num_elements; i++)
    {
        array[i] = rand(); 
    }
}

/*  Normalize a 2D conditional probability matrix
    For example, p_w_z = P(w|z).  For given topic z,
    the probabilities of the words w must sum to one. 
   
    p_w_z is a 1D array, though with size num_words*num_topics
    To normalize, you would want to call...
      : 
      normalize_axis1(p_w_z, num_words, num_topics);
      : 
*/
void normalize_axis1(double *c_array, unsigned int rows,
                     unsigned int cols)
{
    unsigned int r, c;
    double sum;

    for (r=0; r<rows; r++)
    {
        sum = 0.0; 
        for (c=0; c<cols; c++)
        {
            sum += c_array[C_(cols, r, c)];
        }
        
        if (sum > 0.0)
        {
            for (c=0; c<cols; c++)
            {
                c_array[C_(cols, r, c)] /= sum; 
            }
        }
    }

}

double _log_likelihood(unsigned int *X,
                       unsigned int X_size,
                       double *p_w_z,
                       double *p_z_d,
                       double *p_d,
                       unsigned int num_words,
                       unsigned int num_docs,
                       unsigned int num_topics)
{
    unsigned int i, k, word_count, word, doc;
    double log_likelihood = 0.0;
    double inner_sum, p_dw;

    for (i=0; i<X_size; i++)
    {
        word_count = X[i];
        doc = X[F_(X_size, i, 1)];
        word = X[F_(X_size, i, 2)];
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

