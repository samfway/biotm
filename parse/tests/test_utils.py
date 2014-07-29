#!/usr/bin/env python
"""Unit tests for parser support libraries."""

from biotm.parse.util import extract_from_taxa_string 

from unittest import TestCase, main

class utilTests(TestCase):
    """ Test parsing of mapping file. """

    def test_taxa_parsing(self):
        temp = 'Root;k__Bacteria;p__Bacteroidetes;c__Bacteroidia' \
               ';o__Bacteroidales;f__Prevotellaceae'

        p = extract_from_taxa_string('p__', temp)
        self.assertEqual('Bacteroidetes', p)
    
        k = extract_from_taxa_string('k__', temp)
        self.assertEqual('Bacteria', k)

        none_thing = extract_from_taxa_string('ZORB', temp)
        self.assertEqual(none_thing, None)

if __name__ == '__main__':
    main()
