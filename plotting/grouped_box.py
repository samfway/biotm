#!/usr/bin/env python

import matplotlib.pyplot as plt
import brewer2mpl

from numpy.random import random
from numpy import arange, array, mean 

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
    make_grouped_box(ax, data, labels, xticklabels=xticklabels,
                     legend_pos='lower right')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    make_separated_box(ax, data, labels, xticklabels=xticklabels,
                     legend_pos='lower right')
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


def make_separated_box(ax, data, labels=None, colors=None,
                       xticklabels=[], width=0.9, legend_pos=0,
                       dot_mean=False, mean_color='w'):
    if labels and len(data) != len(labels):
        raise ValueError('Number of labels must match ',
                         'size of data matrix.')

    if colors and len(colors) != len(labels):
        raise ValueError('Number of colors must match ',
                         'size of data matrix.')

    num_groups = len(labels)
    num_points = data.shape[1]

    if not colors:
        num_colors = max(3, num_groups)
        colors = brewer2mpl.get_map('Set2', 
                                    'qualitative', 
                                    num_colors).mpl_colors
    current_pos = 0
    xticks = []
    xlabels = []

    for i in xrange(num_groups):
        color = colors[i]
        for j in xrange(num_points):
            bp = ax.boxplot(data[i][j], positions=[current_pos],
                            widths=[width], patch_artist=True)
            xticks.append(current_pos)
            xlabels.append(xticklabels[j]) 
            color_bp(bp, color)
            if dot_mean:
                means = [mean(data[i][j])]
                ax.plot([current_pos], means, linestyle='None', 
                    marker='o', markerfacecolor=mean_color,
                    markeredgecolor='k')
            current_pos += 1 
        current_pos += 2

    if labels:
        legend_hack(ax, labels, colors, legend_pos)

    ax.set_xlim(-1,current_pos-2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

def make_grouped_box(ax, data, labels=None, colors=None,
                     xticklabels=[], width=0.9, legend_pos=0,
                     dot_mean=False, mean_color='w'):
    if labels and len(data) != len(labels):
        raise ValueError('Number of labels must match ',
                         'size of data matrix.')

    if colors and len(colors) != len(labels):
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
        if dot_mean:
            means = [mean(x) for x in data[group_index]]
            ax.plot(positions, means, linestyle='None', 
                marker='o', markerfacecolor=mean_color,
                markeredgecolor='k')

    if labels:
        legend_hack(ax, labels, colors, legend_pos)
    
    ax.set_xlim(0,(num_groups+extra_groups)*num_points)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)


def legend_hack(ax, labels, colors, legend_pos):
    """ Hack a legend onto a plot. 
    """ 
    handles = []
    for i, l in enumerate(labels):
        temp = plt.Line2D(range(1), range(1), 
                          linewidth=2,
                          color=colors[i])
        handles.append(temp)
    plt.legend(handles, labels, numpoints=1,
               loc=legend_pos)
    for handle in handles:
        handle.set_visible(False)


if __name__ == '__main__':
    main()
