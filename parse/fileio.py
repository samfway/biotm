#!/usr/bin/env python

__author__ = "Sam Way"
__copyright__ = "Copyright 2014, The Clauset Lab"
__license__ = "BSD"
__maintainer__ = "Sam Way"
__email__ = "samfway@gmail.com"
__status__ = "Development"

from numpy import asarray, array
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from biom.parse import parse_biom_table


def load_dataset(data_matrix_fp, mapping_fp, 
                 metadata_category, metadata_value=None):
    """ Parse and prepare data for processing. """
    categories, mapping_dict = parse_mapping_file_to_dict(mapping_fp)
    sample_ids, data_matrix = parse_otu_matrix(data_matrix_fp)

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

    return data_matrix, sample_ids, labels, label_legend 


def parse_mapping_file_to_dict(mapping_fp):
    """ Parse a standard mapping file 
        
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
    return array(otu_table.sample_ids), otu_matrix


def custom_cast(s):
    """ Convert to int/float/string in that order of preference """
    for cast_func in (int, float, str):
        try:
            return cast_func(s)
        except ValueError:
            pass
    raise BaseException('Could not cast as number/string!')


def convert_labels_to_int(labels):
    """ Convert a list of labels to indices """ 
    if not len(labels):
        raise ValueError("Nothing to convert!")

    if isinstance(labels[0], basestring):
        label_legend = list(set(labels))
        converted_labels = [ label_legend.index(l) for l in labels ]
    elif is_iterable(labels[0]):  # Multiple labels, handle each one individually
        num_labels = len(labels[0])
        label_legend = []
        converted_labels = []
    
        # Get the unique sets of labels for each index
        for i in xrange(num_labels):
            temp_labels = [l[i] for l in labels]
            label_legend.append(list(set(temp_labels)))

        # Apply mapping to each label
        for label in labels:
            converted_label = [ leg.index(l) for leg, l in 
                                zip(label_legend, label) ]
            converted_labels.append(converted_label)
    else:  # Not a string, not a list, go for ints
        converted_labels = [int(x) for x in labels] 
        label_legend = list(set(labels))

    return label_legend, array(converted_labels)
