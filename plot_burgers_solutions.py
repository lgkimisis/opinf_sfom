"""
Script for plotting the 1D Burgers' simulation data, the predictions by the OpInf-sFOM,
as well as the reprojected data on a basis of the training data.
"""

import numpy as np
import matplotlib.pyplot as plt

times = np.load('./times_plot_data.npy')
x = np.load('./x_plot_data.npy')
Q = np.load('./soln_plot_data.npy')
reprojected_soln_smooth = np.load('./soln_smooth_plot_data.npy')
Qreglob = np.load('./soln_reglob_plot_data.npy')


def plot_panels(filename=None):
    X, T = np.meshgrid(x, times, indexing='ij')
    
    fig, axs = plt.subplots(1, 4, figsize=(14, 6), gridspec_kw={"width_ratios":[0.33, 0.33, 0.33, 0.025]})

    axs[0].set_title('Snapshot data', size=14)
    axs[1].set_title('OpInf-sFOM predictions', size=14)
    axs[2].set_title(r'Reprojected data ($r=10$)', size=14)

    dat = axs[0].contourf(X, T, Q, 20, cmap='viridis') 
    axs[0].hlines(np.max(times)/2, np.min(x),np.max(x),linestyle = 'dashed', color = 'white')
    axs[0].set_ylabel(r"$t$", size=18, usetex=True)
    axs[0].set_yticks([0, 3, 6, 9, 12, 15, 18])
    axs[0].text(1.1, 4.5, 'Training regime',
                ha="center", va="center", rotation=90,
                fontsize=14, color='white')
    axs[0].annotate('',
                xy=(0.5, 9),
                xycoords='data', 
                xytext=(0.5, 0),
                # va='center',
                ha='center',
                arrowprops=dict(arrowstyle= "<->", lw=1, color='white'))

    axs[0].text(1.1, 13.5, 'Prediction regime',
                ha="center", va="center", rotation=90,
                fontsize=14, color='white')
    axs[0].annotate('',
                xy=(0.5, 18),
                xycoords='data', 
                xytext=(0.5, 9),
                # va='center',
                ha='center',
                arrowprops=dict(arrowstyle= "<->", lw=1, color='white'))

    axs[1].contourf(X, T, reprojected_soln_smooth, 20, cmap='viridis') 
    axs[1].vlines(5, np.min(times),np.max(times),linestyle = 'dashed', color='white')
    axs[1].text(2.5, 16, 'OpInf\nsubdomain', fontsize=14, ha='center', color='white')
    axs[1].text(7.5, 16, 'sFOM\nsubdomain', fontsize=14, ha='center', color='white')
    
    axs[2].contourf(X, T, Qreglob, 20, cmap='viridis') 
    fig.colorbar(dat, cax=axs[3], location='right')
    
    for ax in axs[:3]:
        ax.set_xticks([0, 5, x[-1]])
        ax.set_xticklabels([0, 5, 10])
        ax.set_xlabel(r"$z$", size=18, usetex=True)
        ax.set_yticks([0, 3, 6, 9, 12, 15, 18])

    for ax in axs[1:3]:
        ax.set_yticklabels([])

    
    if filename:
        plt.savefig(f'{filename}.png', dpi=200)
    
    plt.show()

plot_panels(filename='burgers_solutions_panels')