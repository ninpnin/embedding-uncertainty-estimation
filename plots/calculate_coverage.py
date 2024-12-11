import os
import json
import argparse
import numpy as np
from parsing_library import *

DATA_SIZES = [1000, 2000, 5000, 10000, 20000, 50000, 100000]

def calculate_coverage(data_folder, fit_folder, output_path, experiment_id, estimator, fit_folder_base_name, base_data_name):
    results = {}  
    output_path = os.path.join(output_path, fit_folder_base_name % experiment_id)
    fit_folder = os.path.join(fit_folder, estimator ,fit_folder_base_name % experiment_id)
    print('folder info:')
    print('fit_folder', fit_folder)
    print('output_path', output_path)
    print('fit_folder_base_name', fit_folder_base_name)
    print('base_data_name', base_data_name)
    for size in DATA_SIZES:
        print(f"data size: {size}  experiment: {experiment_id}")

        data_path = os.path.join(data_folder, base_data_name % experiment_id)
        with open(data_path) as f:
            data = json.load(f)

        true_theta = np.array(data['theta'])
        vocabulary = data['vocabulary']
        word_pairs = list(itertools.combinations(vocabulary, 2))
        total_pairs = len(word_pairs) * 2

        #fit_folder_size = os.path.join(fit_folder, estimator ,fit_folder_base_name % experiment_id)

        if estimator == 'vi':
            fit = load_fit_by_size(size, fit_folder)
            estimated_word_vectors, estimated_context_vectors = extract_word_and_context_vectors_vi(fit.variational_sample_pd, vocabulary)
            num_samples = estimated_word_vectors.shape[2]
        elif estimator == 'hmc':
            fit = load_fit_by_size(size, fit_folder)
            estimated_word_vectors = fit['word_vectors']
            estimated_context_vectors = fit['context_vectors']
            num_samples = estimated_word_vectors.shape[2]

        theta_samples = []
        for i in range(num_samples):
            word_vectors = estimated_word_vectors[:, :, i]
            context_vectors = estimated_context_vectors[:, :, i]
            theta = np.vstack((word_vectors, context_vectors))
            theta_samples.append(theta)

        covered_count = 0
        for word1, word2 in word_pairs:
            vi = vocabulary[word1]
            wi = vocabulary[word2]
            true_p = calculate_p(vi, wi, true_theta, len(vocabulary))

            eta_samples = [calculate_p(vi, wi, theta, len(vocabulary)) for theta in theta_samples]
            lower_ci, upper_ci = credible_interval(eta_samples)

            if lower_ci <= true_p <= upper_ci:
                covered_count += 1

            true_p = calculate_p(wi, vi, true_theta, len(vocabulary))

            eta_samples = [calculate_p(wi, vi, theta, len(vocabulary)) for theta in theta_samples]
            lower_ci, upper_ci = credible_interval(eta_samples)

            if lower_ci <= true_p <= upper_ci:
                covered_count += 1

        coverage_percentage = (covered_count / total_pairs) * 100
        results[size] = coverage_percentage  

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"coverage_{experiment_id}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", required=True)
    parser.add_argument("--fit_folder", required=True,)
    parser.add_argument("--output_path", required=True, help="Path to save the coverage results.")
    parser.add_argument("--experiment_id", type=int, required=True)
    parser.add_argument("--estimator", required=True, help="one of (vi, hmc, laplace, gibbs)")
    parser.add_argument("--fit_folder_base_name", required=True, help="Base name format for fit folder.")
    parser.add_argument("--base_data_name", required=True, help="Base name format for data files.")

    args = parser.parse_args()
    calculate_coverage(
        args.data_folder,
        args.fit_folder,
        args.output_path,
        args.experiment_id,
        args.estimator,
        args.fit_folder_base_name,
        args.base_data_name
    )
