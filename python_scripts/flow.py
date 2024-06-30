from getdist import MCSamples, plots
from tensiometer import mcmc_tension, gaussian_tension, utilities
from tensiometer.mcmc_tension import DiffFlowCallback
from tensorflow import keras
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import matplotlib.pyplot as plt
import numpy as np

class DiffFlow(DiffFlowCallback, ReduceLROnPlateau):
    def __init__(self, chain1=MCSamples(), chain2=MCSamples(), params=[], labels=[]):
        # Initialize the parameters
        self.params = params
        self.labels = labels
        # Initialize the model to an empty list
        self.is_trained = False
        # Initialize the chains in order to match the parameter names given in input
        pars = chain1.getParams()
        samples1 = np.empty(shape=(len(self.params), len(chain1[0])))
        for i, name in enumerate(self.params):
            # Get the parameter array, using getattr() to get the attribute with the name given by the string,
            # i.e. getattr(pars, "name") is equivalent to pars.name
            exec(f'p{i}_array1 = getattr(pars, "{name}")')
            # Insert the samples 
            samples1[i] = eval(f'p{i}_array1')
        # Do the same for the second chain
        pars = chain2.getParams()
        samples2 = np.empty(shape=(len(self.params), len(chain2[0])))
        for i, name in enumerate(self.params):
            exec(f'p{i}_array2 = getattr(pars, "{name}")')
            # Insert the samples 
            samples2[i] = eval(f'p{i}_array2')

        # Create the MCSamples objects
        self.chain1 = MCSamples(samples=samples1.T, 
                     loglikes=chain1.loglikes, 
                     weights=chain1.weights, 
                     names=self.params, 
                     labels=self.labels,
                     settings={'ignore_rows':0.})

        # Do the same for the second chain
        self.chain2 = MCSamples(samples=samples2.T, 
                     loglikes=chain2.loglikes, 
                     weights=chain2.weights, 
                     names=self.params, 
                     labels=self.labels,
                     settings={'ignore_rows':0.})

    def plot_chains(self, savename=None, **kwargs):
        g = plots.get_subplot_plotter(width_inch=(len(self.params)/2)+5)
        g.settings.alpha_filled_add = 0.3
        g.settings.title_limit_fontsize = 0.8
        g.triangle_plot([self.chain1, self.chain2], params=self.params, filled=True, title_limit=1)
        # Save the plot as a PDF
        if savename is not None:
            plt.savefig('figures'+savename, format='pdf', bbox_inches='tight')
    
    def build_diff(self, **kwargs):
        # Raise an error if at least one of the self.params is not in one of the chains
        for p in self.params:
            if not (p in self.chain1.paramNames.list() or p in self.chain2.paramNames.list()):
                raise ValueError(f'Parameter {p} is not in one of the chains.')
        # Define the list of delta parameters (different names)
        self.delta_params = list(map(lambda p: 'delta_'+p, self.params))
        # Get the parameter difference chain
        self.diff_chain = mcmc_tension.parameter_diff_chain(self.chain1, self.chain2, boost=2)
        
        return self.diff_chain

    def plot_diff(self, savename=None, **kwargs):
        # Define Markers to 0 to get a more precise graphical vision of the following plot
        markers = {}
        for param in self.delta_params:
            markers[param] = 0.
        # Plot the parameter difference chain
        g = plots.get_subplot_plotter(width_inch=(len(self.params)/2)+5)
        g.settings.alpha_filled_add = 0.3
        g.settings.title_limit_fontsize = 0.8
        g.triangle_plot([self.diff_chain], params=self.delta_params, markers=markers, filled=True, title_limit=1)
        # Save the plot as a PDF
        if savename is not None:
            plt.savefig('figures'+savename, format='pdf', bbox_inches='tight')

    def train(self, eta, batch_size, epochs, steps_per_epoch):
        callbacks = [ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1),
                     EarlyStopping(monitor='loss', patience=8, verbose=1)]
        # Compile the Normalizing Flow (MAF) model
        dfc = DiffFlowCallback(self.diff_chain, feedback=1, learning_rate=eta)
        # Train the Normalizing Flow (MAF) model
        dfc.train(batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
        self.model = dfc
        self.is_trained = True
        print('Training completed.')
        return self.model

    def plot_distributions(self, savename=None, **kwargs):
        if not self.is_trained:
            raise ValueError('The model has not been trained yet.')
        # Transform the difference chain into a numpy array
        X_sample = np.array(self.model.dist_learned.sample(10000))
        # Convert the numpy array into a MCSamples object
        diff_flow = MCSamples(samples=X_sample, names=self.delta_params, label='Learned distribution')
        # Get the Gaussian approximation of the difference chain
        gaussian_approx = gaussian_tension.gaussian_approximation(self.diff_chain)
        gaussian_approx.label = 'Gaussian approximation'
        self.diff_chain.label = 'Difference chain'
        colors=['orange', 'dodgerblue', 'k']
        g = plots.get_subplot_plotter(width_inch=int(len(self.delta_params)/2)+5)
        g.settings.alpha_filled_add = 0.3
        g.settings.title_limit_fontsize = 0.8
        g.settings.num_plot_contours = 2
        g.triangle_plot([self.diff_chain, diff_flow, gaussian_approx], params=self.delta_params,
                        filled=False, markers={_p:0 for _p in self.delta_params},
                        colors=colors, diag1d_kwargs={'colors':colors})
        # Save the plot as a PDF
        if savename is not None:
            plt.savefig('figures'+savename, format='pdf', bbox_inches='tight')

    def estimate_shift(self, n=10, step=10000):
        if not self.is_trained:
            raise ValueError('The model has not been trained yet.')
        # Due to initialization, there is some variance in the result: we then take the mean of n estimates
        for i in range(n):
            # Estimate the shift
            shift_P_1, shift_low_1, shift_hi_1 = self.model.estimate_shift(step=step)
            # If it is the first estimate, initialize the variables
            if i == 0:
                exact_shift_P_1 = shift_P_1
                exact_shift_low_1 = shift_low_1
                exact_shift_hi_1 = shift_hi_1
            # Otherwise, add the estimate to the previous ones
            else:
                exact_shift_P_1 += shift_P_1
                exact_shift_low_1 += shift_low_1
                exact_shift_hi_1 += shift_hi_1
        # Take the mean
        exact_shift_P_1 /= n
        exact_shift_low_1 /= n
        exact_shift_hi_1 /= n
        
        print(f'Considering {self.delta_params} parameters...')
        print(f'Shift probability = {exact_shift_P_1:.5f} + {exact_shift_hi_1-exact_shift_P_1:.5f} - {exact_shift_P_1-exact_shift_low_1:.5f}')
        # Convert the result to effective number of sigmas
        print(f'          n_sigma = {utilities.from_confidence_to_sigma(exact_shift_P_1):.3f}')
