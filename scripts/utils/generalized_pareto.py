from scipy.stats import genpareto
import matplotlib.pyplot as plt
import numpy as np

def suggest_M(S:int) -> int:
    # M is the number of largest samples to use to fit pareto distribution
    # From PSIS-paper: "M is empirically set as min(S/5, 3√S)", where S is the number of samples from the surrogate
    return min(int(S/5), int(3*S**(0.5)))


def fit_generalized_pareto_on_tail(rs_importance_ratios, M=None, return_all_parameters=False):
    """
    Fit generalized pareto on upper tail of weights (rs)
    Source: https://arxiv.org/pdf/1802.02538.pdf
    """

    if M is None:
        M = suggest_M(len(rs_importance_ratios))

    rs_M_largest = rs_importance_ratios[::-1][0:M]
    (k_shape, loc, scale) = genpareto.fit(rs_M_largest)
    
    if return_all_parameters:
        return (k_shape, loc, scale)
    else:
        return k_shape
    
def psis_diagnostic_results(k_shape:float, verbose=True) -> int:
    """
    k_shape : the Generalized Pareto shape

    returns diagnostic_status_code:int
    -1 : k < 0 - unclear what this means but the pareto distribution is the same as a uniform distribution in this case. Not mentioned in the paper but it's a valid shape for the Pareto.
     1 : If k < 0.5 - We conclude the variational approximation q is close enough to the true density
     2 : If 0.5 < k < 0.7 - It indicates the variational approximation q is not perfect but still useful.
     3 : If k > 0.7 - Indicates VI has failed.
    """

    if k_shape<0:
        diagnostic_status_code = -1
        print('Negative GPD shape k_hat = %.4f. Upper tail of weights follow a uniform distribution. What does this mean?'%k_shape)
    elif k_shape >= 0 and k_shape<=0.5:
        diagnostic_status_code = 1
        print('k < 0.5 (k = %.4f) We conclude the variational approximation q is close enough to the true density'%k_shape)
    elif k_shape > 0.7:
        diagnostic_status_code = 3
        print('0.7 < k (k = %.4f) Indicates VI has failed'%k_shape)
    elif k_shape > 0.5:
        diagnostic_status_code = 2
        print('0.5 < k < 0.7 (k = %.4f) Indicates the variational approximation q is not perfect but still useful'%k_shape)
    
    return diagnostic_status_code


def plot_generalized_pareto(shape, loc, scale):

    x = np.linspace(genpareto.ppf(0.01, shape, loc, scale),
                genpareto.ppf(0.99, shape, loc, scale), 1000)

    plt.plot(x, genpareto.pdf(x, shape, loc, scale), 'r-', label='genpareto pdf')
    plt.title('genpareto pdf')


if __name__ == '__main__':
    # Example usage
    from tensorflow_probability import distributions as tfd
    from tensorflow.random import set_seed
    set_seed(123)
    run_stability_test = True #repeat to see stability of k_hat


    # generate some mock data
    k, loc, scale = 0.6, 3, 2
    gpd = tfd.GeneralizedPareto(loc=loc, scale=scale, concentration=k)
    print('parameters: (k(shape), loc, scale)')
    print('original parameters:', (k, loc, scale))
    samples = gpd.sample(10000, seed=123)

    full_fit = fit_generalized_pareto_on_tail(samples, M = len(samples), return_all_parameters=True) #M = len(samples) just to see the fit on all of the data
    print('full fit parameters:', full_fit)

    tail_fit = fit_generalized_pareto_on_tail(samples, return_all_parameters=True) #M determined according to paper
    print('tail fit parameters:', tail_fit)
    psis_diagnostic_results(tail_fit[0])

    if run_stability_test:
        full_fit_k_shapes = []
        tail_fit_k_shapes = []

        for i in range(100):
            samples = gpd.sample(1000)

            full_fit = fit_generalized_pareto_on_tail(samples, M=len(samples), return_all_parameters=True)
            full_fit_k_shapes.append(full_fit[0])

            tail_fit = fit_generalized_pareto_on_tail(samples, return_all_parameters=True)
            tail_fit_k_shapes.append(tail_fit[0])
        
        print(f'Repeated full fit k: {np.mean(full_fit_k_shapes)} ({np.std(full_fit_k_shapes)})')
        print(f'Repeated tail fit k: {np.mean(tail_fit_k_shapes)} ({np.std(tail_fit_k_shapes)})')