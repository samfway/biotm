#!/usr/bin/env python
from numpy import array
import argparse
from argparse import RawTextHelpFormatter
from biotm.parse.fileio import load_dataset 
from biotm.parse.util import convert_labels_to_int

""" 
    This code can be used to format python matrices/labels into
    the input format required by the C implementation of the SLDA 
    algorithm, the one found on David Blei's website.  

    Using the --validation flag, a set of matrices training/testing matrices is
    created.  Otherwise, by default, the program converts the matrix and all 
    labels into a format suitable for the C program. 
"""

def write_matrix_to_slda_file(data_matrix, output_file):
    """ Format matrix for input to SLDA """
    output = open(output_file, 'w')
    N = data_matrix.shape[1]  # Number of dimensions
    for row in data_matrix:
        to_write = [ '%d:%d'%(k, row[k]) for k in xrange(N) if row[k] > 0 ]
        output.write('%d %s\n' % (len(to_write), ' '.join(to_write)))
    output.close()


def create_slda_dataset(data_matrix, labels, output_prefix, sample_ids=None):
    """ Reformat matrix+labels for the SLDA C implementation.  
        Creates a labels file, a data file, and an (optional) sample_ids file.
    
        Data file is of the format:
            <M> <term_1>:<count> <term_2>:<count> ... <term_N>:<count>
        NOTE: M is the number of unique, non-zero elements in the row
              In other words, the number of <term>:<count> pairs to follow.
    """ 
    # 1)  Create labels file: one label per line 
    output_name = output_prefix + 'labels.txt'
    output = open(output_name, 'w')
    labels_file = output_name
    if labels is not None:
        #unique_labels, label_indices = convert_labels_to_int(labels)
        #output.write('\n'.join([str(l) for l in label_indices]))
        output.write('\n'.join([str(int(l)) for l in labels]))
    else:
        labels = [0]*len(data_matrix)
        output.write('\n'.join([str(l) for l in labels]))
    output.close()
    
    # 2) Create data file for SLDA Format: 
    output_name = output_prefix + 'data.txt' 
    write_matrix_to_slda_file(data_matrix, output_name)
    data_file = output_name

    # 3) Create sample id file: one per line (NOT NEEDED BY SLDA)
    if sample_ids is not None:
        output_name = output_prefix + 'sample_ids.txt'
        output = open(output_name, 'w')
        output.write('\n'.join(sample_ids))
        output.close() 

    # 4) Word list
    output_name = output_prefix + 'words.txt'
    output = open(output_name, 'w')
    for i in xrange(data_matrix.shape[1]):
        output.write(str(i) + '\n')
    output.close()

    # 5) Formatted response
    output_name = output_prefix + 'doc_info.txt'
    output = open(output_name, 'w')
    for i in xrange(data_matrix.shape[0]):
        output.write(str(i) + '\t' + str(labels[i])  + '\n')
    output.close()
    
    return data_file, labels_file

