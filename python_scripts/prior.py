import os, sys
# Define chain folder: change it as you need in the function calls
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml

from getdist import loadMCSamples, plots, mcsamples, MCSamples

def get_type(type=""):
    # Get the correct type string
    # Find the first occurrence of ' character
    start = type.find("'")
    # Find the last occurrence of ' character
    end = type.rfind("'")
    # Extract the word between the ' characters
    if start != -1 and end != -1:
        extracted_word = type[start + 1:end]
        return extracted_word
    else:
        print('No word found within single quotes.')

from scipy.stats import norm, truncnorm
# from classy import Class
from tqdm import tqdm
import re
class priorChain():
    def __init__(self, n=10000, root_dir='', chain_name=''):
        self.root_dir = root_dir
        self.chain_name = chain_name
        self.n = n
        # Initialize various useful arrays
        self.names, self.labels, self.params = [], [], {}
        self.cosmo_names, self.cosmo_labels = [], []
        self.dv_names, self.dv_labels = [], []
        self.nuisance_names, self.nuisance_labels = [], []
        self.cosmo_prior = None
    
    def get_names(self):
        names, labels = [], []
        file = self.root_dir+self.chain_name+'.paramnames'
        try:
            with open(file, 'r') as f:
                lines = f.readlines()[:]
                for l, line in enumerate(lines):
                    # Split each line into words using space-tab-space as a delimiter
                    words = line.strip().split(' \t ')
                    if len(words) == 2:
                        names.append(words[0])
                        labels.append(words[1])
                    else:
                        print(f'Something went wrong in line {l}.')
                self.names, self.labels = names, labels
                return names, labels
        
        except FileNotFoundError:
            print(f'File not found: {file}')
            return None
        
    def build_flat(self, array=[]):
        low, high, scale = array[1], array[2], array[4]
        if (low == None) | (high == None):
            print('Need to expplicit both boundary values for the flat distribution.')
        return np.random.uniform(low=low, high=high, size=self.n)
        
    def build_gauss(self, array=[], mean=0, sigma=0.1):
        low, high, scale = array[1], array[2], array[4]
        mean /= scale
        sigma /= scale
        # Truncated Gaussian
        if (low != None) | (high != None):
            if low == None: low = -np.inf
            if high == None: high = np.inf
            X = truncnorm((low - mean) / sigma, (high - mean) / sigma, loc=mean, scale=sigma)
            X = X.rvs(self.n)
        else:
            X = norm.rvs(size=self.n, loc=mean, scale=sigma)
        return X
    
    # Function that implements Class to compute the value of the prior on sigma8
    # The idea is to call the same Class instance used in the EFT predictions, passing
    # as 'cosmology' parameters each line of the cosmo_prior chains previously generated
    def get_sigma8(self, size=10000):
        M = Class()
        M.set({'output': 'mPk', 'P_k_max_h/Mpc': 10, 'z_max_pk': 1})

        p = self.cosmo_prior.getParams()
        # Faster than .append()
        sigma8_array = np.empty(size)  
        for i in tqdm(range(size), desc='Computing sigma8: ', bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'):
            try:
                omega_b = p.omega_b[i] * 10**(-2)
                omega_cdm = p.omega_cdm[i]
                h = p.h[i]
                lnA_s = getattr(p, 'ln10^{10}A_s')[i]
                n_s = p.n_s[i]
            except NameError:
                print('Class is missing one or more parameters to compute sigma_8.')

            cosmo = {'omega_b': omega_b, 'omega_cdm': omega_cdm, 'h': h, 'ln10^{10}A_s': lnA_s, 'n_s': n_s}
            M.set(cosmo)
            M.compute()
            sigma8_array[i] = M.sigma8()
        return sigma8_array
             
    def get_params(self):
        names, _ = self.get_names()
        params = {}
        root_dir = self.root_dir+'/log.param'
        try:
            with open(root_dir, mode='r') as f:
                f_content = f.read()
                for name in names:
                    prior_array = []
                    pattern = rf"data\.parameters\['{re.escape(name)}'\]\s+=\s+\[(.*?)\]"
                    prior_pattern = rf"#data\.prior\['{re.escape(name)}'\]\s+=\s+\[(.*?)\]"
                    match = re.search(pattern, f_content)
                    if match:
                        # Extract the array values
                        array = match.group(1).split(',')
                    else:
                        print(f'No match found for {name}.')
                    prior_match = re.search(prior_pattern, f_content)
                    if prior_match:
                        prior_array = prior_match.group(1).split(',')
                    num_array = [np.float32(x.strip()) if x.strip() != 'None' else None for x in array[:5]]
                    num_array.extend([array[5]])
                    num_array.extend(prior_array)
                    params[name] = num_array
            f.close()
        except FileNotFoundError:
            print(f'File not found at the following path: {root_dir}')
        self.params = params
        return self.params
    
    def get_param_limits(self):
        if not len(self.params):
            print('Loading input file.')
            self.get_params()
        param_limits = {}
        for name in self.names:
            try:
                dist = self.params[name][6]
                dist = get_type(dist)
                if (self.params[name][1] != None) & (self.params[name][2] != None) & (dist == 'flat'):
                    param_limits[name] = (self.params[name][1], self.params[name][2])
            except IndexError:
                type = get_type(self.params[name][5])
                if type == 'nuisance':
                    param_limits[name] = (self.params[name][1], self.params[name][2])
        return param_limits

    # Build the chain using scipy
    def get_cosmo_prior(self, ignore_rows=0.3):
        params = self.get_params()
        ranges, chain_array = {}, []
        for name, label in zip(self.names, self.labels):
            if name in self.cosmo_names: pass
            else:
                num_array = params[name][:5]
                type = params[name][5]
                type = get_type(type)
                # num_array[3] = expected sigma -> if it is 0 the parameter is not varying
                if (type == 'cosmo') & (num_array[3] != 0):
                    self.cosmo_names.append(name)
                    self.cosmo_labels.append(label)
                    ranges[name] = (num_array[1], num_array[2])
                    dist = params[name][6]
                    dist = get_type(dist)
                    if dist == 'gaussian':
                        try:
                            mean = np.float32(params[name][7])
                            sigma = np.float32(params[name][8])
                        except IndexError:
                            print(f'Gaussian distribution for {name} missing mean or sigma.')
                        chain_array.append(self.build_gauss(array=num_array, mean=mean, sigma=sigma))
                    if dist == 'flat':
                        chain_array.append(self.build_flat(array=num_array))
                elif (type == 'cosmo') & (num_array[3] == 0):
                    print(f'{name} is not varying. Check the input parameter file.')
                
        self.cosmo_prior = MCSamples(samples=np.transpose(chain_array), 
                                    names=self.cosmo_names, labels=self.cosmo_labels, 
                                    ignore_rows=ignore_rows,
                                    ranges=ranges)
        return self.cosmo_prior
    
    def get_dv_prior(self, include_class=False):
        while not self.cosmo_prior:
            print('Running get_cosmo_prior() sampling.')
            self.get_cosmo_prior()
        try:
            p = self.cosmo_prior.getParams()
            print('Prior samples successfully loaded.')
        except ValueError:
            print('No MCSamples instance to compute derived parameters.')
        for name, label in zip(self.names, self.labels):
            if name in self.dv_names: pass
            else:
                type = self.params[name][5]
                type = get_type(type)
                if (type == 'derived'):
                    if (name == 'Omega_m'):
                        self.cosmo_prior.addDerived((p.omega_b/100 + p.omega_cdm)/p.h**2, name=name, label=label)
                    if (name == 'Omega_Lambda'):
                        self.cosmo_prior.addDerived((1. - (p.Omega_cdm + p.Omega_k + (p.omega_b/100)/0.68**2)), 
                                                    name=name, label=label)
                    if (name == 'A_s'):
                        if hasattr(p, 'ln10^{10}A_s'):
                            lnA_s = getattr(p, 'ln10^{10}A_s')
                            scale = self.params[name][4]
                            self.cosmo_prior.addDerived(np.exp(lnA_s)*10**(-10)/scale, name='A_s', label='10^{-9}A_{s }')
                        else:
                            print(f'Something went wrong with the parameter {name}.')
                    if (name == 'sigma8') & (include_class == True):
                        self.cosmo_prior.addDerived(self.get_sigma8(size=len(self.cosmo_prior[0])), name=name, label=label)
                    # Update at the end to check for repeating names
                    self.dv_names.append(name)
                    self.dv_labels.append(label)
        return self.cosmo_prior
        
    def get_nuisance_prior(self, config_name='', ignore_rows=0.3):
        self.get_params()
        ranges = {}
        file = yaml.full_load(open(self.root_dir+config_name+'.yaml', 'r'))
        chain_array, eft_params = [], []
        # Since the data file could contain up to 4 sky cuts, we build a prior distribution 
        # for each sky cut
        index = 0
        for name, label in zip(self.names, self.labels):
            num_array = self.params[name][:5]
            type = self.params[name][5]
            type = get_type(type)
            if type == 'nuisance':
                self.nuisance_names.append(name)
                self.nuisance_labels.append(label)
                ranges[name] = (self.params[name][1], self.params[name][2])
                eft_name = name.split('_')[0]
                dist = file['eft_prior'][eft_name]['type']
                if dist == 'flat':
                    chain_array.append(self.build_flat(array=num_array))
                if dist == 'gauss':
                    mean = file['eft_prior'][eft_name]['mean'][index]
                    sigma = file['eft_prior'][eft_name]['range'][index]
                    if len(file['eft_prior'][eft_name]['mean']) - 1 > index: index += 1
                    chain_array.append(self.build_gauss(num_array, mean=mean, sigma=sigma))
        nuisance_prior = MCSamples(samples=np.transpose(chain_array), 
                                    names=self.nuisance_names, labels=self.nuisance_labels,
                                    ignore_rows=ignore_rows, 
                                    ranges=ranges)
        return nuisance_prior
        
# Check that the chains fit inside the prior range
def range_check(posterior, prior):
    p = posterior.getParams()
    params = prior.get_params()
    param_list = []
    for name in prior.names:
        method = getattr(p, name)
        #print(f'{name} chain = [{method.min():.4}-{method.max():.4}]')
        #print(f'{name} range = [{params[name][1]:.4}-{params[name][2]:.4}]')
        if (params[name][1] <= method.min()) & (params[name][2] > method.max()):
            #print('INSIDE the prior range.\n')
            pass
        else:
            param_list.append([name])
            #print('OUTSIDE the prior range.\n')
            
    if not len(param_list):
        return print('Chains generated by Monte Python respect the prior ranges.')
    else:
        return print('The following parameters exceed the prior range:', param_list)