"""
Script for plotting the singular values of 1D Burgers' snapshot data.
The singular values of the complete domain, the OpInf subdomain and the sFOM subdomain are separately plotted.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter, FuncFormatter


svals_opinf = np.load('svals_opinf_burgers.npy')[:40]
svals_sfom = np.load('svals_sfom_burgers.npy')[:40]
svals_global = np.load('svals_global_burgers.npy')[:40]
sval_range = np.arange(1, 41)


fig, ax = plt.subplots(figsize=(6, 4)) 
    
ax.semilogy(sval_range, svals_opinf/svals_opinf[0], linewidth=3)

ax.semilogy(sval_range, svals_sfom/svals_sfom[0],
            color='tab:red', alpha=0.7, linewidth=3)

ax.semilogy(sval_range, svals_global/svals_global[0],
            ls='--', color='#3c3c3c', linewidth=3, label='Complete domain')

ax.set_xlabel(r'Singular value $\#$', size=14)
ax.set_ylabel('Normalized singular values', size=14)


ax.annotate('sFOM subdomain' + r'$= [5, 10]$',
            xy=(26, 0.0007),
            xycoords='data', 
            xytext=(29, 0.1),
            # va='center',
            ha='center',
            color='tab:red',
            alpha=0.7,
            fontsize=14,
            weight='roman',
            arrowprops=dict(arrowstyle= "-", lw=1.5, color='tab:red', alpha=0.7))

ax.annotate('OpInf subdomain' + r'$= [0, 5)$',
            xy=(25, 0.00000000007),
            xycoords='data', 
            xytext=(29, 0.00000002),
            # va='center',
            ha='center',
            color='tab:blue',
            fontsize=14,
            weight='roman',
            arrowprops=dict(arrowstyle= "-", color='tab:blue'))

ax.annotate('Full domain',
            xy=(23, 0.0005),
            xycoords='data', 
            xytext=(19, 0.000005),
            # va='center',
            ha='center',
            color='#3c3c3c',
            fontsize=14,
            weight='roman',
            arrowprops=dict(arrowstyle= "-", color='#3c3c3c'))

plt.tight_layout()

plt.savefig('sval_decay_burgers.png', dpi=200)
plt.savefig('sval_decay_burgers.pdf')

plt.show()