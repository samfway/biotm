#!/usr/bin/env python

__author__ = "Sam Way"
__copyright__ = "Copyright 2014, The Clauset Lab"
__license__ = "BSD"
__maintainer__ = "Sam Way"
__email__ = "samfway@gmail.com"
__status__ = "Development"

from biotm.parse.util import convert_labels_to_int, custom_cast

from numpy import asarray, array
from pandas import read_csv

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from biom.parse import parse_biom_table


def load_dataset(data_matrix_fp, mapping_fp, 
                 metadata_category, metadata_value=None):
    """ Parse and prepare data for processing. """
    categories, mapping_dict = parse_mapping_file_to_dict(mapping_fp)
    sample_ids, taxonomy, data_matrix = parse_otu_matrix(data_matrix_fp)

    # Determine which sample ids were in the mapping file
    ids_in_mapping_file = [] 
    for i, sample_id in enumerate(sample_ids):
        if sample_id in mapping_dict:
            ids_in_mapping_file.append(i)
    ids_in_mapping_file = array(ids_in_mapping_file)
    
    # Select found sample_ids and corresponding data
    sample_ids = sample_ids[ids_in_mapping_file]
    data_matrix = data_matrix[:,ids_in_mapping_file]
    data_matrix = data_matrix.transpose()

    # Obtain labels + legend for selected samples
    labels = [mapping_dict[sample_id][metadata_category] for sample_id 
              in sample_ids]
    label_legend, labels = convert_labels_to_int(labels)

    return data_matrix, sample_ids, taxonomy, labels, label_legend 


def parse_mapping_file_to_dataframe(mapping_fp, sep='\t'):
    """ Parse a standard mapping file into a pandas
        dataframe object that can be queried like a database.

        Inputs:
          -mapping_fp: open file pointer to mapping file.
    
        Returns:
          -df: data frame (table) of metadata for each 
               SampleID.
    """
    df = read_csv(mapping_fp, sep=sep)
    col_names = df.columns.tolist()
    if not col_names[0] == '#SampleID':
        raise ValueError('File does not appear to be a valid'
                         ' mapping file!')

    col_names[0] = 'SampleID'  # Remove the '#'
    df.columns = col_names
    return df
 

def parse_mapping_file_to_dict(mapping_fp):
    """ Parse a standard mapping file into a dictionary
        relating each SampleID to its metadata values
        
        Inputs:
          -mapping_fp: open file pointer to mapping file.

        Returns:
          -categories:  metadata categories from parsed file
          -mapping_dict:  a dictionary of metadata indexed
                          by sample ids.  Each element is
                          itself a dictionary indexed by
                          metadata categories. 
    """ 
    mapping_dict = {} 

    header_line = mapping_fp.readline()
    header_pieces = header_line.strip().split('\t')
    if header_pieces[0] != '#SampleID':
        raise ValueError('This does not appear to be a valid mapping file')
    
    categories = header_pieces[1:]
    num_pieces = len(header_pieces)
    
    for line in mapping_fp:
        line = line.strip()
        if not line:
            continue
        line_pieces = line.split('\t')
        if len(line_pieces) > 1:
            sample_id = line_pieces[0] 
            num_categories = len(line_pieces[1:])
            mapping_dict[sample_id] = dict(zip(categories[:num_categories], line_pieces[1:]))

    return categories, mapping_dict


def parse_otu_matrix(biom_fp):
    """ Parses a (dense) OTU matrix from a biom file.
        Outputs: Dense OTU matrix, list of sample ids
    """
    # Parse the OTU table into a dense matrix
    otu_table = parse_biom_table(biom_fp)
    # Obtain a dense matrix (sparse eventually?)
    otu_matrix = otu_table._data.toarray()

    taxonomy = [otu_table.observation_metadata[i]['taxonomy'] for i 
        in xrange(len(otu_table.observation_metadata))]

    return array(otu_table.sample_ids),\
           array(taxonomy), otu_matrix


def load_biom_table(biom_fp):
    """ Returns actual biom table. """
    return parse_biom_table(biom_fp)
