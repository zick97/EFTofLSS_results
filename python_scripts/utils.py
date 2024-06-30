import os, sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import yaml

from getdist import loadMCSamples, plots, mcsamples, MCSamples
from tensiometer_dev.tensiometer import *

def log_transform(chain, params, labels):
    # Take the logarithm of the parameters: this way, the KL decomposition simply becomes a regular
    # power-law decomposition making it is easier to build the highest-variance modes
    log_params = []
    log_labels = []
    # Iterate over the parameters and their labels
    for name, label in zip(params, labels):
        p = chain.getParams()
        method = getattr(p, name)
        chain.addDerived(np.log(method), name='log_'+name, label='\\log '+label)
        log_params += ['log_'+name]
        log_labels += ['\\log '+label]
    return log_params, log_labels

# Plot the fractional Fisher Matrix, along with the effective number of parameters
# Arguments:
#   - prior_chain = MCSamples() object containing the prior distribution chain
#   - posterior_chain = MCSamples() object containing the posterior distribution chain
#   - params, labels = string arrays containing parameters' names and labels
#   - log = whether to apply a logarithmic transformation to the input chains
#   - print_improvement = whether to print the improvement factor of the posterior over the prior
#   - norm = whether to use the CPCA decomposition for the Fisher Matrix, in order to have rows and columns 
#     with normalized sum
def plot_frac_fisher(prior_chain=MCSamples(), posterior_chain=MCSamples(), params=[], labels=[], 
                     print_improvement=False, norm=True, figsize=None, savename=None):
    # Effective number of parameters
    print(f'N_eff \t= {gaussian_tension.get_Neff(posterior_chain, prior_chain=prior_chain, param_names=params):.5}\n')
    KL_param_names = params
    
    # Compute the KL modes
    KL_eig, KL_eigv, KL_param_names = gaussian_tension.Q_UDM_KL_components(prior_chain, posterior_chain, 
                                                                        param_names=KL_param_names)

    # Compute the fractional Fisher matrix
    KL_param_names, KL_eig, fractional_fisher, _ = gaussian_tension.Q_UDM_fisher_components(prior_chain, posterior_chain, 
                                                                                            KL_param_names, 
                                                                                            which='1')
    # Print the improvement factor
    if print_improvement:
        with np.printoptions(precision=2, suppress=True):
            if any(eig < 1. for eig in KL_eig):
                print('Improvement factor over the prior:\n', KL_eig)
                print('Discarding error units due to negative values.')
            else:
                print('Improvement factor over the prior in E.U.:\n', np.sqrt(KL_eig-1))
    # Use alternative version (normalized column sum)
    # Eigenvalues and eigenvectors of the KL-decomposed Fisher matrix are not changed
    if norm:
        dict = gaussian_tension.linear_CPCA_chains(prior_chain, posterior_chain, param_names=params)
        fractional_fisher = dict['CPCA_var_contributions']

    # Plot (showing values and names)
    if figsize == None:
        figsize = int(len(params)/1.6)
    plt.figure(figsize=(figsize, figsize))
    im1 = plt.imshow(fractional_fisher, cmap='viridis')
    num_params = len(fractional_fisher)
    # The following loop is used to display the fractional fisher values inside the cells
    for i in range(num_params):
        for j in range(num_params):
            if fractional_fisher[j,i]>0.5:
                col = 'k'
            else:
                col = 'w'
            plt.text(i, j, np.round(fractional_fisher[j,i],2), va='center', ha='center', color=col)
    plt.xlabel('KL mode (error improvement)');
    plt.ylabel('Parameters');
    ticks  = np.arange(num_params)
    if any(eig < 1. for eig in KL_eig):
        labels = [str(t+1)+'\n'+str(l) for t,l in zip(ticks, np.round(KL_eig, 2))]
    else:
        labels = [str(t+1)+'\n'+str(l) for t,l in zip(ticks, np.round(np.sqrt(KL_eig-1), 2))]
    plt.xticks(ticks, labels, horizontalalignment='center', rotation=30);
    labels = ['$'+posterior_chain.getParamNames().parWithName(name).label+'$' for name in KL_param_names]
    plt.yticks(ticks, labels, horizontalalignment='right');

    # Save the plot as a pdf if savename is provided
    if savename is not None:
        plt.savefig('figures/'+savename, format='pdf', bbox_inches='tight')

    return KL_eig, KL_eigv, KL_param_names

# Create a nice-looking table to store multiple values from arrays with the same length.
# Values on the same row have the same index in the lists. 
def create_table(*lists):
    # Ensure all lists have the same length
    list_lengths = [len(lst) for lst in lists]
    if len(set(list_lengths)) != 1:
        return "Error: lists are not of the same length"

    # Combine all lists into one list of tuples
    combined_lists = list(zip(*lists))

    # Find the maximum length for each column (that is, each individual list)
    max_lengths = [max(len(str(item)) for item in list) for list in lists]

    # Create the table
    table = ''
    for row in combined_lists:
        # Add elements for each row, adjusting the width for each column
        table += ' | '.join(f'{str(item):<{max_len}}' for item, max_len in zip(row, max_lengths)) + '\n'

    return table

# Alternative to numpy.where(), meant to be used with non-numpy arrays and string arrays.
def find_index(list, condition):
    # List comprehension is the faster procedure
    return [i for i, elem in enumerate(list) if condition(elem)]

# Short utility function to print the Maximum Variance Contribution for each parameter of a given list.
def print_max(dict1={}, dict2={}, pars=[]):
    print('Maximum Variance Contribution')
    print('---------------------------------------------')
    print('First Chain', end='')
    print(' '*(22-len('First Chain')), end='|')
    print(' '*(22-len('Second Chain')), end='')
    print('Second Chain')
    for par in pars:
        idx = find_index(pars, lambda p: p == par)[0]
        print(f"{par} : {max(dict1['CPCA_var_contributions'][idx][:]):.2}", end='')
        # Print blank spaces to align the two columns
        print(' '*(15-len(par)), end='|')
        print(' '*(15-len(par)), end='')
        print(f"{par} : {max(dict2['CPCA_var_contributions'][idx][:]):.2}")