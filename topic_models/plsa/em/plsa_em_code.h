#include <stdio.h>
#include <time.h>


// Public Module Constants
#define C_(cols, i, j) (i*cols + j)
#define C3_(sj, sk, i, j, k) (i*sj*sk + j*sk + k)
#define F_(rows, i, j) (j*rows + i)
#define SWAP(a, b, temp) temp=a; a=b; b=temp;


// Public Module Function Prototypes
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
                unsigned int max_em_iter);

double _log_likelihood(double *X,
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

void _em_m_step(double *X,
                unsigned int X_size,
                double *p_z_wd,
                double *p_w_z,
                double *p_z_d,
                unsigned int num_words,
                unsigned int num_docs,
                unsigned int num_topics,
                unsigned int folding); 

void randomize(double *array, unsigned int num_elements);
void normalize2d(double *c_array, unsigned int rows,
                 unsigned int cols);
