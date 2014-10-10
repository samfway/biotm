#!/usr/bin/env python
"""Unit tests for parser support libraries."""

from numpy import ones, matrix

from biotm.parse.fileio import parse_mapping_file_to_dict
from biotm.parse.fileio import get_metadata_matrix
from biotm.parse.fileio import add_control_variables

from unittest import TestCase, main
from StringIO import StringIO

class mappingFileTests(TestCase):
    """ Test parsing of mapping file. """

    def test_parsing_mapping_dict(self):
        mapping_fp = StringIO("#SampleID\tfname\tlname\n" 
                              "s09\tSam\tWay\n"
                              "s10\tDelicious\tCoffee\n")

        mapping_cats, mapping_dict = parse_mapping_file_to_dict(mapping_fp)

        self.assertEqual(set(mapping_cats), set(['fname', 'lname']))
        self.assertEqual(set(mapping_dict.keys()), set(['s09', 's10']))
        self.assertEqual(mapping_dict['s09']['fname'], 'Sam')
        self.assertEqual(mapping_dict['s10']['lname'], 'Coffee')


class controlVariableTests(TestCase):
    """ Test functionality of adding columns from a mapping
        file as control variables to an OTU matrix. """    

    def test_md_matrix(self):
        mapping_fp = StringIO("#SampleID\tfname\tlname\n" 
                              "s08\tLaura\tNorris\n"
                              "s09\tSam\tWay\n"
                              "s10\tDelicious\tCoffee\n")   
        sample_ids = ['s08', 's09']
        v = get_metadata_matrix(mapping_fp, sample_ids, ['lname'])
        desired = ['Norris', 'Way']
       
        for i in xrange(2):
            self.assertEqual(desired[i], v[i][0])

        
    def test_add_cv(self):
        X = ones((2, 3))
        mapping_fp = StringIO("#SampleID\tfname\tlname\tINT\tFLOAT\n" 
                              "s08\tLaura\tNorris\t9\t11.86\n"
                              "s09\tSam\tWay\t4\t17.87\n"
                              "s10\tDelicious\tCoffee\t1\t1.9\n")   
        sample_ids = ['s08', 's09']
        X, ll = add_control_variables(['lname', 'INT', 'FLOAT'], mapping_fp,
                                  X, sample_ids,
                                  convert_category=[True, False, False])

        self.assertEqual(X[0,5], 11.86)
        self.assertEqual(X[1,5], 17.87)
        self.assertEqual(ll[0][int(X[0,3])], 'Norris')
        self.assertEqual(6, X.shape[1])


if __name__ == '__main__':
    main()
