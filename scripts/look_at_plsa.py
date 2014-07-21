#!/usr/bin/env python

import click
from biotm.parse.fileio import load_dataset 
import matplotlib.pyplot as plt
from numpy import array

from biotm.topic_models.plsa import plsa


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-m', '--mapping-file', help="Filepath to mapping file.", required=True)
@click.option('-i', '--otu-file', help="Filepath to otu file.", required=True)
@click.option('-c', '--metadata-category', help="Metadata category to extract.", required=True)
@click.option('-v', '--metadata-value', help="(Optional) Metadata value for boolean evaluation.")
def have_a_look(mapping_file, otu_file, metadata_category, metadata_value):
    otu_fp = open(otu_file, 'rU')
    map_fp = open(mapping_file, 'rU')

    data_matrix, sample_ids, labels, label_legend = \
        load_dataset(otu_fp, map_fp, metadata_category, metadata_value)

    relevant_indices = array([i for i,v in enumerate(labels) if label_legend[v] != 'NA'])
    data_matrix = data_matrix[relevant_indices, :]
    sample_ids = sample_ids[relevant_indices]
    labels = labels[relevant_indices]
    
    tm = plsa(n_iter=1, n_components=5)
    p_z_d = tm.fit_transform(data_matrix, labels)
    plt.scatter(p_z_d[0,:], p_z_d[1,:], c=labels)
    plt.show()


if __name__ == '__main__':
    have_a_look()
