#!/usr/bin/env python

import click
from biotm.parse.fileio import load_dataset 
from sklearn.svm import SVC


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-m', '--mapping-file', help="Filepath to mapping file.", required=True)
@click.option('-i', '--otu-file', help="Filepath to otu file.", required=True)
@click.option('-c', '--metadata-category', help="Metadata category to extract.", required=True)
@click.option('-v', '--metadata-value', help="(Optional) Metadata value for boolean evaluation.")
def load_data(mapping_file, otu_file, metadata_category, metadata_value):
    """ Basic template for loading a dataset.  Makes sure that data 
        can be loaded from an biom + mapping file pair and passed 
        into a scikit-learn classifier.  
    """ 
    otu_fp = open(otu_file, 'rU')
    map_fp = open(mapping_file, 'rU')

    data_matrix, sample_ids, labels, label_legend = \
        load_dataset(otu_fp, map_fp, metadata_category, metadata_value)
    
    clf = SVC()
    clf.fit(data_matrix, labels)
    print 'It worked!'


if __name__ == '__main__':
    load_data()
