import warnings
import time
import json
import os
import argparse

import numpy as np
import tensorflow as tf


from simulate import load, load_simulated_data_generator
from probabilistic_word_embeddings.estimation import map_estimate, mean_field_vi
from probabilistic_word_embeddings.embeddings import Embedding


# TODO seed does not work on embeddngs for some reason...
#    - If you run fix seed in run_multiple_experiments (u have to do this manually in code) it will give u the same result for all experiements
#    - If repeat above you get different result.
#  # TODO We dont need to calculate p for true every loop.
# TODO OPTION TO save indiviual runs when running multiple

def sigmoid(x:float): # might be a good idea to have this in a common library.
  return 1 / (1 + np.exp(-x))

def calculate_p_2(vi:str, wi:str, theta:object, V:int):
    rho_v = theta[vi, :]          # $\rho_v = \theta_v$
    alpha_w = theta[wi + V, :]    # $\alpha_w = \theta_{V + w}$
    eta = rho_v.dot(alpha_w)      # $\eta = \rho_v^T \alpha_w$
    p = sigmoid(eta)              # Run it through the link function $\sigma; p = \sigma(\eta)$
    return p



def run_convergence_experiment(data_json_path, estimator, estimator_args:dict, embedding_args:dict, increment:int, embedding=Embedding, save_path=None):
    """
    Runs multiple estimators for incrementally larger data sets. Outputs a json file that summarizes the results.
        The purpose is to use this json as input to plotting functions.
    Currently only supports simulated data.

    Args:

        estimator: function : example map_estimate
        estimator_settings : arguments to the estimator
        embedding_args : 

        (Optional)
        embedding: if not specified will use default Embedding

    Output:
        if save_
    """ 
    if save_path:
        if not file_exists_check(save_path):
            return

    

    data_json = load(data_json_path)

    N = len(data_json['data'])
    vocabulary = data_json['vocabulary']
    vocab = list(vocabulary.keys())
    true_theta = np.array(data_json['theta'])
    V = len(vocabulary)

    assert increment < N, 'increment ({increment}) needs to be smaller than the number of observations ({N})'

    e_true = embedding(vocabulary=set(vocab), dimensionality=np.shape(true_theta)[1])
    e_true[vocab] = true_theta[:V]
    e_true[[w + "_c" for w in vocab]] = true_theta[V:]
    
    words_and_contexts = list(vocabulary.keys()) + [w + '_c' for w in list(vocabulary.keys())]

    seed = embedding_args.get('seed', None)
    batch_size = estimator_args.get('batch_size', None)
    if batch_size is None:
        warnings.warn('No batch_size specified in estimator args.')
    elif batch_size > N: 
        warnings.warn(f'batch_size ({batch_size}) larger than the number of observations ({N}). Using batch_size=None')
        batch_size=None
    elif batch_size > increment: # This causes VI to crash.
        warnings.warn(f'batch_size ({batch_size}) larger than increment {increment}. Will use batch_size=None until i*increment > batch_size')
    
    results = []
    for size in range(0, N + increment, increment):
        print('size', size)
        current_size = min(size, N)
        e = embedding(vocabulary=set(vocabulary.keys()), **embedding_args)

        if batch_size is None or batch_size > current_size:
            generator = load_simulated_data_generator(data_json_path, batch_size=None, max_num_observations=current_size)
        else:
            generator = load_simulated_data_generator(data_json_path, batch_size=batch_size, max_num_observations=current_size)

        start_time = time.time()
        if size > 0: # only run estimator if we have data.
            estimate = estimator(e, data_generator=generator, N=current_size, **estimator_args) 
        end_time = time.time()

        results_current = {'data_size':None, 'correlation':None, 'runtime':None, 'theta': None}
        theta_estimated = e[words_and_contexts].numpy()
        if estimator.__name__ == 'mean_field_vi' and size > 0: #Todo probably remove and sample using saved embeddings
            # VI gives us standard deviation of parameters and so we can visualize variations per run: theta + N(0,1)*std
            assert len(estimate) == 2, f'Unexpected output for {estimator.__name__}. Expected 2 outputs (mean, std).'

            N_resamples = 20 # TODO: hard coded setting...
            std_estimated =  estimate[1][words_and_contexts].numpy()
            rng = np.random.default_rng(seed=seed)
            noise = rng.standard_normal(size=(theta_estimated.shape[0], theta_estimated.shape[1], N_resamples)) #

            p_corr_resampled = []
            rmse_resampled = []
            for i in range(N_resamples):
                theta_resampled = theta_estimated + noise[:,:,i] * std_estimated
                e_resampled = embedding(vocabulary=set(vocab), dimensionality=np.shape(theta_resampled)[1])
                e_resampled[vocab] = theta_resampled[:V]
                e_resampled[[w + "_c" for w in vocab]] = theta_resampled[V:]
                corr_i, resampled_p, true_p = p_correlation(vocabulary, e_resampled, e_true)
                p_corr_resampled.append(corr_i)
                rmse_resampled.append(np.sqrt(np.mean((np.array(resampled_p)-np.array(true_p))**2)))
            
            results_current['std_theta'] = std_estimated.tolist()
            results_current['p_corr_resampled'] = p_corr_resampled
            
        
        #theta_estimated = e[words_and_contexts].numpy()
        corr, estimated_p, true_p = p_correlation(vocabulary, e, e_true) # TODO We dont need to calculate p for true every loop.
        rmse = np.sqrt(np.mean((np.array(estimated_p)-np.array(true_p))**2))#calculate_rmse(vocabulary, e, e_true)

        results_current.update({'data_size':current_size, 'correlation':corr, 'rmse':rmse, 'runtime':end_time - start_time, 'theta':theta_estimated.tolist(), 'p_estimate': estimated_p, 'p_true':true_p}) # Store store_estimated_p etc?
        results.append(results_current) 

    output_dict = {}
    output_dict['results'] = results
    output_dict['data_json_path'] = data_json_path
    output_dict['seed'] = seed
    output_dict['estimator'] = estimator.__name__
    output_dict['true_theta'] = true_theta.tolist()

    # Seed testing : 
    e = embedding(vocabulary=set(vocabulary.keys()), **embedding_args)
    output_dict['e'] = list(e[['word1','word1_c']].numpy()[0,:])
    output_dict['ec'] = list(e[['word1','word1_c']].numpy()[1,:])

    if save_path:
        with open(save_path, 'w') as file:
            json.dump(output_dict, file, indent=4)
            print(f'Saved to: {save_path}') # add check file exists
    

    return output_dict


def run_multiple_experiments(data_json_path, estimator, estimator_args, embedding_args, increment, embedding, N_experiments, save_path=None):
    
    if save_path:
        if not file_exists_check(save_path):
            return
    
    seed = embedding_args.get('seed', None)
    
    all_results = []
    for experiment in range(N_experiments):
        print(f"Running experiment {experiment + 1} of {N_experiments}")
        
        if seed is not None:
            new_seed = int("%s%s"%(seed, experiment))
        else:
            new_seed = None

        results = run_convergence_experiment(
            data_json_path=data_json_path,
            estimator=estimator,
            estimator_args=estimator_args,
            embedding_args=embedding_args if seed is None else {**embedding_args, 'seed': new_seed}, # replace seed without updating original dict.
            increment=increment,
            embedding=embedding,
            save_path=None  # TODO option to save each run?
        )
        all_results.append(results)

    output_dict = {}
    output_dict['all_results'] = all_results
    output_dict['data_json_path'] = data_json_path
    output_dict['estimator'] = estimator.__name__

    if save_path:
        with open(save_path, 'w') as file:
            json.dump(output_dict, file, indent=4)
            print(f"Saved to: {save_path}")

    return output_dict

      
# --- Library ---
def calculate_p(vi, wi, e):
    rho_v = e[vi]          # $\rho_v = \theta_v$
    alpha_w = e[wi + '_c']    # $\alpha_w = \theta_{V + w}$
    #eta = rho_v.dot(alpha_w)      # $\eta = \rho_v^T \alpha_w$
    eta = tf.tensordot(rho_v, alpha_w, axes=1)
    p = sigmoid(eta)              # Run it through the link function $\sigma; p = \sigma(\eta)$
    return p


def p_correlation(vocabulary, e_1, e_2):
    """
    Calculate the p for theta_1 and theta_2 and then the calculate the correlations of the two.
    Note: Make sure the thetas are sorted correctly and include the context (word_c) words.
    """
    words = list(vocabulary.keys())
    
    #V = len(words)
    store_p1 = []
    store_p2 = []
    for v in words:
        for w in words:
            if v != w: # TODO: comment
                p1 = calculate_p(v, w, e_1)
                p2 = calculate_p(v, w, e_2)
                store_p1.append(p1)
                store_p2.append(p2)

    corr = np.corrcoef(store_p1, store_p2)[1,0]
    return corr, store_p1, store_p2


def calculate_rmse(vocabulary, e_1, e_2):
    """
    Calculate the p for theta_1 and theta_2 and then the calculate the correlations of the two.
    Note: Make sure the thetas are sorted correctly and include the context (word_c) words.
    """
    words = list(vocabulary.keys())
    
    #V = len(words)
    store_p1 = []
    store_p2 = []
    for v in words:
        for w in words:
            if v != w: # TODO: comment
                p1 = calculate_p(v, w, e_1)
                p2 = calculate_p(v, w, e_2)
                store_p1.append(p1)
                store_p2.append(p2)

    #corr = np.corrcoef(store_p1, store_p2)[1,0]
    store_1 = np.array(store_p1)
    store_2 = np.array(store_p2)
    rmse = np.sqrt(np.mean((store_1-store_2)**2))
    return rmse

def file_exists_check(file_path:str):
    if os.path.isfile(file_path):
        response = input(f"The file '{file_path}' already exists. Do you want to continue? (y/n): ").strip().lower()
        if response == 'y':
            return True
        else:
            return False
    return True

# --- Plotting ---

import matplotlib.pyplot as plt

def plot_experiment_results(all_results, rmse=False, log_scale=False):
    """
    TODO: fig/ax input.

    Plots the correlation from multiple experiments against the data size.

    Args:
    - all_results (list of dicts): output of run_multiple_experiments

    """
    if 'all_results' not in all_results: #In the case of single experiment.
        print('added all results')
        all_results = {'all_results':[all_results]}

    plt.figure(figsize=(10, 6))  

    colors = plt.cm.tab10(np.linspace(0, min(1, len(all_results['all_results'])/10 ), len(all_results['all_results'])))

    for idx, result in enumerate(all_results['all_results']):
        if 'results' in result:
            data_sizes = [experiment['data_size'] for experiment in result['results']]
            if rmse:
                correlations = [experiment['rmse'] for experiment in result['results']]
            else:
                correlations = [experiment['correlation'] for experiment in result['results']]
            color = colors[idx] 
            
            y = np.log(correlations) if log_scale else correlations
            plt.plot(data_sizes, y, marker='o', linestyle='-', linewidth=2, color=color, label=f'Experiment {idx+1}')

            # VI: if 'p_corr_resampled' data is available
            if any('p_corr_resampled' in experiment for experiment in result['results']) and not (rmse or log_scale):
                # Assuming the length of 'p_corr_resampled' is consistent within each experiment
                num_resamples = min(len(experiment['p_corr_resampled']) for experiment in result['results'] if 'p_corr_resampled' in experiment)
                for i in range(num_resamples):
                    resampled_corrs = [experiment['p_corr_resampled'][i] for experiment in result['results'] if 'p_corr_resampled' in experiment]

                    y = np.log(resampled_corrs) if log_scale else resampled_corrs
                    plt.plot(data_sizes, y, linestyle='--', linewidth=1, color=color, alpha=0.3)

    plt.xlabel('Data Size')
    y_label = ("Log " if log_scale else "") + ("RMSE" if rmse else "Correlation")
    plt.ylabel(y_label)
    title =  y_label + ' vs. Data Size ' + ("" if 'data_json_path' not in all_results else '%s'%all_results['data_json_path'])
    title += "" if 'estimator' not in all_results else ' %s'%all_results['estimator']
    plt.title(title)

    """
    if rmse:
        plt.ylim(0.0, None)
    else:
        plt.ylim(0, 1)
    """
    #plt.legend()
    plt.grid(True)
    plt.show()


def plot_average_over_runs(all_results, rmse=True):
    """
    Args:
    - all_results: output of run_multiple_experiments
    """
    if 'all_results' not in all_results: 
        all_results = {'all_results': [all_results]}

    plt.figure(figsize=(10, 6))

    all_data_sizes = sorted(set(experiment['data_size'] for result in all_results['all_results'] for experiment in result['results']))

    avg_correlations = []
    std_correlations = []

    for data_size in all_data_sizes:
        if rmse: #TODO name change *correlations to metric or w/e
            correlations = [experiment['rmse'] for result in all_results['all_results'] for experiment in result['results'] if experiment['data_size'] == data_size]
            print(correlations)
        else:
            correlations = [experiment['correlation'] for result in all_results['all_results'] for experiment in result['results'] if experiment['data_size'] == data_size]
            print(correlations)


        avg_corr = np.mean(correlations)
        std_corr = np.std(correlations, ddof=1)

        avg_correlations.append(avg_corr)
        std_correlations.append(std_corr)

    line_color = 'royalblue'

    plt.plot(all_data_sizes, avg_correlations, label='Average Correlation', color=line_color, marker='o', linestyle='-')

    #plt.plot(all_data_sizes, np.array(avg_correlations) + np.array(std_correlations), label='Mean ± STD', color=line_color, linestyle='--')
    #plt.plot(all_data_sizes, np.array(avg_correlations) - np.array(std_correlations), color=line_color, linestyle='--')

    plt.fill_between(all_data_sizes, np.subtract(avg_correlations, std_correlations), np.add(avg_correlations, std_correlations), color=line_color, alpha=0.2, label='Standard Deviation')

    plt.xlabel('Data Size')
    plt.ylabel('Correlation')
    title = 'Average Correlation and Standard Deviation' + ("" if 'data_json_path' not in all_results else '\n%s'%all_results['data_json_path'])
    title += "" if 'estimator' not in all_results else '\n%s'%all_results['estimator']
    plt.title(title)
    if rmse:
        plt.ylim(0.0, None)
    else:
        plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_average_over_runs_multiple(*all_results_list, rmse=True, legend_labels=None, extra_title=''):
    """
    Args:
    - all_results_list: Multiple 'all_results', each being the output of run_multiple_experiments.
    - legend_labels: Optional list of strings to use as labels in the legend. Should match the number of all_results_list.
    """
    plt.figure(figsize=(10, 6))

    # Prepare distinct colors and markers for up to 5 different all_results
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    markers = ['o', '^', '*', 's', 'p']  # Circle, Triangle, Star, Square, Pentagon

    if legend_labels and len(legend_labels) != len(all_results_list):
        raise ValueError("Length of legend_labels must match the number of all_results provided.")

    for index, all_results in enumerate(all_results_list[:5]):  # Limit to the first 5 all_results if more are provided
        if 'all_results' not in all_results: 
            all_results = {'all_results': [all_results]}
        
        all_data_sizes = sorted(set(experiment['data_size'] for result in all_results['all_results'] for experiment in result['results']))

        avg_correlations = []
        std_correlations = []

        for data_size in all_data_sizes:
            if rmse:
                correlations = [experiment['rmse'] for result in all_results['all_results'] for experiment in result['results'] if experiment['data_size'] == data_size]
            else:
                correlations = [experiment['correlation'] for result in all_results['all_results'] for experiment in result['results'] if experiment['data_size'] == data_size]

            avg_corr = np.mean(correlations)
            std_corr = np.std(correlations, ddof=1)

            avg_correlations.append(avg_corr)
            std_correlations.append(std_corr)

        # Use custom or default legend label
        legend_label = legend_labels[index] if legend_labels else f'Run #{index+1}'

        # Plot each all_results with a unique color and marker
        plt.plot(all_data_sizes, avg_correlations, label=legend_label, color=colors[index], marker=markers[index], linestyle='-', markersize=9)
        plt.fill_between(all_data_sizes, np.subtract(avg_correlations, std_correlations), np.add(avg_correlations, std_correlations), color=colors[index], alpha=0.2)

    text_size=16

    plt.xlabel('Data Size', fontsize=text_size)

    y_label =("RMSE" if rmse else "Correlation")
    plt.ylabel(y_label, fontsize=text_size)
    title =  'Averaged '+ y_label + ' vs. Data Size' #+ #("" if 'data_json_path' not in all_results else '%s'%all_results['data_json_path'])
    title += extra_title
    #plt.ylabel('Correlation' if not rmse else 'RMSE')  # Adjust label based on the metric
    plt.title(title, fontsize=text_size+2)
    plt.ylim(0, None if rmse else 1)
    #plt.xlim(-1000, None)
    plt.legend()

    
    plt.legend(fontsize=text_size)
    plt.xticks(fontsize=text_size-1)  
    plt.yticks(fontsize=text_size-1)  # Set y-axis tick labels size

    plt.grid(True)
    plt.show()


def fix_single_experiment_case(all_results):
    """
    supposedly all results object
    """
    if 'all_results' not in all_results: #In the case of single experiment.
        all_results = {'all_results':[all_results]}

        try:
            all_results['data_json_path'] = all_results['all_results'][0]['results'][0]['data_json_path']
            all_results['estimator'] = all_results['all_results'][0]['results'][0]['estimator']
        except:
            pass
    return all_results

def plot_parameter_magnitude(all_results):
    """
    all results or just one result.
    """
    def magnitude_correction(theta):
        """
        In order to transform a randomly generated embedding to the one that minimizes a spherical prior,
        you need to do the following magnitude correction
        """
        assert np.shape(theta)[0] % 2 == 0
        V = int(np.shape(theta)[0]/2)
        rho = np.array(theta[0:V])
        alpha = np.array(theta[V:])

        # Frobenius Norm
        eps = (np.linalg.norm(alpha, ord='fro') / np.linalg.norm(rho, ord='fro'))**(1/2) # 1/2 if norm^2, 1/4 otherwise
        #print(eps)
        return np.concatenate((eps*rho, alpha/eps), axis=0) # corrected theta
    
    all_results = fix_single_experiment_case(all_results)

    plt.figure(figsize=(10, 6)) 

    colors = plt.cm.tab10(np.linspace(0, min(1, len(all_results['all_results'])/10 ), len(all_results['all_results'])))


    for idx, result in enumerate(all_results['all_results'][0:1]): #[0:1] remove [0:1].
        true_theta = np.array(result['true_theta'])
        true_theta = magnitude_correction(true_theta)
        if 'results' in result: # skip meta information (file name etc)
            data_sizes = [experiment['data_size'] for experiment in result['results']]
            

            if True: # dont need correction on the estimated
                corrected_theta = [magnitude_correction(experiment['theta']) for experiment in result['results']]
                avg_thetas = [np.mean(np.abs(theta)) for theta in corrected_theta]
            else: # TODO remove after testing above.
                avg_thetas = [np.mean(np.abs(experiment['theta'])) for experiment in result['results']] # avg of parameters

            
            color = colors[idx] 
            plt.plot(data_sizes, avg_thetas, marker='o', linestyle='-', linewidth=2, color=color, label=f'Experiment {idx+1}')

            if 'std_theta' in result['results'][0]: # VI
                avg_std = [np.mean(np.abs(experiment['std_theta'])) for experiment in result['results']]
                plt.fill_between(data_sizes, np.array(avg_thetas) - np.array(avg_std), np.array(avg_thetas) + np.array(avg_std), alpha=0.2)
                #plt.plot(data_sizes, np.array(avg_thetas) + np.array(avg_std), color=color, linestyle='--', alpha=0.75)
                #plt.plot(data_sizes, np.array(avg_thetas) - np.array(avg_std), color=color, linestyle='--', alpha=0.75)


    avg_true_theta = np.mean(np.abs(true_theta))
    plt.plot([data_sizes[0], data_sizes[-1]], [avg_true_theta, avg_true_theta], linestyle='-', linewidth=2, color='r', label='True')

    plt.xlabel('Data Size')
    plt.ylabel('mean(abs(θ))')
    title = 'Parameter Magnitude ' + ("" if 'data_json_path' not in all_results else '%s'%all_results['data_json_path'])
    title += "" if 'estimator' not in all_results else ' %s'%all_results['estimator']
    plt.title(title)
    #plt.ylim(0, 1)
    #plt.legend()
    plt.grid(True)
    plt.show()

def plot_scatter_p(all_results):

    i = -1
    all_results = fix_single_experiment_case(all_results)
    for idx, result in enumerate(all_results['all_results'][0:1]): #[0:1] remove [0:1].
        print(result.keys())
        
        estimated_p = np.array(result['results'][i]['p_estimate'])
        true_p = np.array(result['results'][i]['p_true'])
        data_size = result['results'][i]['data_size']

    plt.scatter(true_p, estimated_p)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel('True p')
    plt.ylabel('Estimated p')
    plt.title(f'data size: {data_size}')
    plt.grid(True)
    plt.show()

# ---- MAIN -----

def main_run_experiment():


    # python experiment_convergence.py --data_json_path test5k_d2_s08.json --increment 1000 --save_path results.json --estimator_type map --batch_size 100 --epochs 10
    #python experiment_convergence.py --data_json_path test5k_d2_s08.json --increment 1000 --save_path results.json --estimator_type map --batch_size 100 --epochs 10

    parser = argparse.ArgumentParser(description="Convergence experiment with incremental dataset sizes.")
    parser.add_argument('--data_json_path', type=str, required=True, help='Path to the simulated data JSON file')
    parser.add_argument('--save_path', type=str, required=False, help='Path where the results JSON will be saved. If not provided, results will be printed.')
    parser.add_argument('--increment', type=int, required=True, help='Incremental step size for data.')
    parser.add_argument('--embedding_dimension', type=int, default=2, help='Dimensionality of the embedding ')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for the estimators')
    parser.add_argument('--estimator_type', type=str, choices=['map','map_estimate','mean_field_vi', 'vi'], default='map', help='Type of estimator to use.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training ')

    args = parser.parse_args()

    embedding_args = {
        'dimensionality': args.embedding_dimension,
        'lambda0':0.0,
        'seed':1
    }

    estimator_args = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'evaluate': False, #required
        'model': 'sgns' #required
    }

    if args.estimator_type in ['vi', 'mean_field_vi']:
        from probabilistic_word_embeddings.estimation import mean_field_vi as estimator
    elif args.estimator_type in ['map', 'map_estimate']:
        from probabilistic_word_embeddings.estimation import map_estimate as estimator

    # Run the experiment
    results = run_convergence_experiment(
        data_json_path=args.data_json_path,
        estimator=estimator,
        estimator_args=estimator_args,
        embedding_args=embedding_args,
        increment=args.increment,
        save_path=args.save_path
    )
    if args.save_path:
        print('res', results)
        #print(f"Results saved to: {args.save_path}")
    else:
        print("Results:")
        print(results)

    
def main_run_multiple():
    parser = argparse.ArgumentParser(description="Run multiple convergence experiments with incremental dataset sizes.")
    parser.add_argument('--data_json_path', type=str, required=True, help='Path to the simulated data JSON file')
    parser.add_argument('--save_path', type=str, required=False, help='path where the results JSON will be saved')
    parser.add_argument('--increment', type=int, required=True, help='Incremental step size for data.')
    parser.add_argument('--embedding_dimension', type=int, default=2, help='Dimensionality of the embedding.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for the estimators')
    parser.add_argument('--estimator_type', type=str, choices=['map','map_estimate','mean_field_vi', 'vi'], default='map', help='Type of estimator to use.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training ')
    parser.add_argument('--N_experiments', type=int, default=5, help='Number of experiments to run')

    args = parser.parse_args()

    embedding_args = {
        'dimensionality': args.embedding_dimension,
        'lambda0': 0.0,
        'seed': 1
    }

    words_to_fix_rotation = [f'word{i}' for i in range(args.embedding_dimension-1)] # if (args.embedding_dimension == 2) else None #Ugly hard code :)
    estimator_args = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'evaluate': False,  # Required for the estimator
        'model': 'sgns' , # Required for the estimator
    }
    
    if args.estimator_type in ['vi', 'mean_field_vi']:
        estimator = mean_field_vi
        estimator_args['words_to_fix_rotation'] = words_to_fix_rotation
    elif args.estimator_type in ['map', 'map_estimate']:
        estimator = map_estimate

    # Run multiple experiments
    all_results = run_multiple_experiments(
        data_json_path=args.data_json_path,
        estimator=estimator,
        estimator_args=estimator_args,
        embedding_args=embedding_args,
        increment=args.increment,
        embedding=Embedding,
        N_experiments=args.N_experiments,
        save_path=args.save_path
    )

    return all_results


def load(file_path:str):
    # Side note: can use ijson for more effiecient, incremental data loading.
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


if __name__ == '__main__':

    """
    This part is a bit of mess...

    Example/Test Usage,
        this script is not suited to run entirely from command line, because of complicated arguments.
    """

    multiple_runs = True
    plot_only = False
    plot_vi_parameters=False
    plot_scatter = False


    json_path = 'test_vi_240.json'

    if plot_only:
        if False:
            print('Plotting')
            json_ = load(json_path)
            #plot_average_and_std_rmse(json_)
            #plot_experiment_results(json_, rmse=True, log_scale=False)
            #plot_experiment_results(json_['all_results'][0])
            plot_average_over_runs(json_, rmse=True)
            #plot_experiment_results(json_['all_results'][0])
        else: 
            #json_map = load('results_map.json')
            #json_vi = load('results_vi1.json')

            json_1 = load('map_d2_100.json')
            json_2 = load('vi_d2_100.json')
            json_3 = load('map_d5_100.json')
            plot_average_over_runs_multiple(json_1, json_2, json_3 ,rmse=True, legend_labels=['MAP', 'VI', 'MAPd5'], extra_title='\n 10 runs, $d = 2$')#json_map, json_vi)
    elif plot_vi_parameters:
        print('Plotting')
        json_vi = load(json_path)
        plot_parameter_magnitude(json_vi)
    elif plot_scatter:
        json_ = load(json_path)
        plot_scatter_p(json_)
    else:
        if multiple_runs:
            print('Multiple Runs')
            #python experiment_convergence.py --data_json_path test5k_d2_s08.json --increment 1000 --save_path results.json --estimator_type map --batch_size 100 --epochs 10 --N_experiments 5
            all_results = main_run_multiple()
            print(all_results)
            plot_experiment_results(all_results)

        else:
            print('Single Runs')
            #python experiment_convergence.py --data_json_path test5k_d2_s08.json --increment 1000 --save_path results.json --estimator_type map --batch_size 100 --epochs 10
            results = main_run_experiment()
