#!/usr/bin/env python

import matplotlib.pyplot as plt
import brewer2mpl

from numpy.random import random
from numpy import arange, array

def main():
    num_groups = 3
    num_time_points = 4
    num_samples_per_time_point = 10
    labels = ['Group1', 'Group2', 'Group3']
    xticklabels = ['One', 'Two', 'Three', 'Four']

    data = random((num_groups, num_time_points,
                   num_samples_per_time_point))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    make_grouped_box(ax, data, labels, xticklabels=xticklabels)
    plt.show()


def color_bp(bp, color):
    c = array(color) * 0.5
    c = tuple(c)

    for x in bp['boxes']: 
        plt.setp(x, color=c)
        x.set_facecolor(color)
    for x in bp['medians']:
        plt.setp(x, color=c)
    for x in bp['whiskers']: 
        plt.setp(x, color=c)
    for x in bp['fliers']: 
        plt.setp(x, color=c)
    for x in bp['caps']:
        plt.setp(x, color=c)


def make_grouped_box(ax, data, labels=None, colors=None,
                     xticklabels=[], width=0.9, legend_pos=0):
    if labels and len(data) != len(labels):
        raise ValueError('Number of labels must match ',
                         'size of data matrix.')

    if colors and len(colors != len(labels)):
        raise ValueError('Number of colors must match ',
                         'size of data matrix.')

    num_groups = len(labels)
    extra_groups = 2
    num_points = data.shape[1]
    base_range = arange(num_points)*(num_groups+extra_groups) + 1
    xticks = base_range + (num_groups/2)

    if not colors:
        num_colors = max(3, num_groups)
        colors = brewer2mpl.get_map('Set2', 
                                    'qualitative', 
                                    num_colors).mpl_colors

    for group_index, group in enumerate(data):
        positions = base_range + group_index 
        bp = ax.boxplot(data[group_index].transpose(),
                        positions=positions,
                        widths=width,
                        patch_artist=True)
        color_bp(bp, colors[group_index])

    if labels:
        handles = []
        for group_index in xrange(num_groups):
            temp = plt.Line2D(range(1), range(1), 
                              linewidth=2,
                              color=colors[group_index])
            handles.append(temp)
        """
        for group_index in xrange(num_groups):
            temp, = plt.plot([1,1],
                             linewidth=4,
                             marker='o',
                             color=colors[group_index])       
            handles.append(temp)
        """
        plt.legend(handles, labels, numpoints=1,
                   loc=legend_pos)
        for handle in handles:
            handle.set_visible(False)
    
    ax.set_xlim(0,(num_groups+extra_groups)*num_points)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

if __name__ == '__main__':
    main()
