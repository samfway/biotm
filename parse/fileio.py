#!/usr/bin/env python

__author__ = "Sam Way"
__copyright__ = "Copyright 2014, The Clauset Lab"
__license__ = "BSD"
__maintainer__ = "Sam Way"
__email__ = "samfway@gmail.com"
__status__ = "Development"

from biotm.parse.util import convert_labels_to_int, custom_cast

from numpy import asarray, array, matrix, hstack
from pandas import read_csv

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from biom.parse import parse_biom_table


def load_dataset(data_matrix_fp,
                 mapping_fp,
                 metadata_category,
                 metadata_value=None,
                 convert_labels=True):
    """ Load a standard microbiome dataset. 
        
        Inputs:
        data_matrix_fp - open file pointer to otu matrix file.
        mapping_fp - open file pointer to standard mapping file.
        metadata_category - metadata category to extract.
        metadata_value - for converting to binary labels.
        convert_labels - convert labels to integers.  
                         key/legend returned as label_legend.

        Returns:
        data_matrix - OTU matrix from file
        sample_ids - sample ids corresponding to order of data_matrix.
        taxonomy - taxonomy strings for each OTU
        labels - metadata values for each sample id
        label_legend - contains key to convert integer
                       labels to the original values
                       in the mapping file.
         
    """
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

    if convert_labels:
        label_legend, labels = convert_labels_to_int(labels)
    elif metadata_value is not None:
        label_legend = None 
        labels = [l == metadata_value for l in labels]
    else:
        label_legend = None

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
            values = [custom_cast(l) for l in line_pieces[1:]]
            mapping_dict[sample_id] = dict(zip(categories[:num_categories], values))

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


def get_metadata_matrix(mapping_fp, sample_ids, columns):
    """ Parse columns of a mapping file into a matrix 
        
        Inputs:
        mapping_fp - open mapping file to read.
        sample_ids - order of samples to maintain.
        columns - columns to extract from the mapping file.

        Returns:
        A matrix that can be stacked with a data matrix. 
    """
    categories, mapping_dict = parse_mapping_file_to_dict(mapping_fp)
    metadata = []

    if isinstance(columns, basestring):
        columns = [columns]

    for c in columns:
        vector = get_metadata_list(mapping_dict, sample_ids, c)
        metadata.append(vector)
    metadata = matrix(metadata).transpose()
    return metadata
    

def get_metadata_list(mapping_dict, sample_ids, metadata_category):
    """ Grab a list of metadata values in the order of 
        sample ids supplied.

        Inputs:
        mapping_dict - a dictionary-like object linking sample ids
                       to a metadata value.  The supplied list of 
                       sample ids must all be keys in this dict.
                       Addition to being present in the dictionary,
                       each entry must have a value for the supplied,
                       desired metadata category.
        sample_ids - sample ids corresponding to the desired order of
                     your data.  These must all be present as keys 
                     in the mapping_dict.
        metadata_category - metadata_category to extract

        Returns:
        A list of metadata values for the supplied metadata category
        in the order specified by the supplied list of sample ids. 
    """ 

    values = []
    for sample_id in sample_ids:
        if sample_id not in mapping_dict:
            raise KeyError("Sample ID (%s) not found in the supplied "
                             "mapping dictionary!" % (sample_id))
            
        if metadata_category not in mapping_dict[sample_id]:
            raise KeyError("Metadata category (%s) not found "
                           "in the supplied mapping dictionary "
                           "for sample id (%s)!" % 
                           (metadata_category, sample_id))
            
        values.append(mapping_dict[sample_id][metadata_category])
    return values


def add_control_variables(category_names, mapping_fp, data_matrix, sample_ids,
                          convert_category=None):
    """ Add control variables to data matrix.
        
        Inputs:
        category_names - (iterable) list of metadata category names
                         to be added as control variables to the 
                         data matrix.
        mapping_fp - open mapping file from which to extract metadata.
        data_matrx - OTU matrix to which we'll be adding the 
                     control variables.
        sample_ids - order of the samples in the data_matrix. 
        convert_category - list of booleans specifying whether or not
                           to convert a list of values to categorical
                           values. 

        Returns:
        1) A new data_matrix, with the desired control variables
        added as additional columns.
        2) A list of label legends for metadata categories 
        that were converted to numeric.
    """
    if convert_category is None:
        if isinstance(category_names, basestring):
            convert_category = [False]
        else:
            convert_category = [False]*len(category_names)
    else:
        if isinstance(convert_category, bool):
            if isinstance(category_names, basestring) or \
                len(category_names) == 1:
                convert_category = [convert_category]
            else:
                raise ValueError("convert_category should be "
                                 "the same length as the number "
                                 "of desired categories!")
        else:
            if len(category_names) != len(convert_category):
                raise ValueError("convert_category should be "
                                 "the same length as the number "
                                 "of desired categories!")
        
    md_matrix = get_metadata_matrix(mapping_fp, sample_ids, category_names)
    legends = []

    for i, c in enumerate(convert_category):
        if c:
            labels = array(md_matrix[:, i]).ravel()
            label_legend, labels = convert_labels_to_int(labels)
            legends.append(label_legend)
            md_matrix[:, i] = matrix(labels).transpose()

    return hstack([data_matrix, matrix(md_matrix, dtype=float)]), legends
