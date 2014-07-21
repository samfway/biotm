#!/usr/bin/env python
"""Unit tests for parser support libraries."""

from biotm.parse.fileio import parse_mapping_file_to_dict

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
    

if __name__ == '__main__':
    main()
