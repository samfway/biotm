#!/usr/bin/env python

__author__ = "Sam Way"
__copyright__ = "Copyright 2014, The Clauset Lab"
__license__ = "BSD"
__maintainer__ = "Sam Way"
__email__ = "samfway@gmail.com"
__status__ = "Development"

from numpy import array


def extract_from_taxa_string(tag, taxa_string):
    """ Extract a field from a taxonomy string.

        Inputs:
          tag - the field to be extracted.
          taxa_string - ';'-separated taxa string.

        Returns:
          Taxonomy field corresponding to the tag.
          (None if tag is not in taxa_string)

        For example if the taxa_string is
        'Root;k__Bacteria;p__Bacteroidetes'
        and the tag is 'p__', this function will 
        return 'Bacteroidetes'.
    """
    if tag in taxa_string:
        pieces = taxa_string.split(';')
        for piece in pieces:
            if tag in piece:
                return piece.replace(tag, '')
    return None


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
