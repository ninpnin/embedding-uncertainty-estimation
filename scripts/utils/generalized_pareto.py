from scipy.stats import genpareto
import matplotlib.pyplot as plt
import numpy as np

def suggest_M(S:int) -> int:
    # M is the number of largest samples to use to fit pareto distribution
    # From PSIS-paper: "M is empirically set as min(S/5, 3√S)", where S is the number of samples from the surrogate
    return min(int(S/5), int(3*S**(0.5)))


def fit_generalized_pareto_on_tail(rs_importance_ratios, M=None, return_all_parameters=False,  method="MLE"):
    """
    Fit generalized pareto on upper tail of weights (rs)
    Source: https://arxiv.org/pdf/1802.02538.pdf

    method : "MM" or "MLE" or "NEW"
    """
    if method is None:
        method = "MLE"

    if M is None:
        M = suggest_M(len(rs_importance_ratios))

    rs_M_largest = rs_importance_ratios[::-1][0:M]
    
    if method.lower()=='new':
        k_shape = zhang_stephens_k(rs_M_largest)
        loc, scale = None, None
    else:
        (k_shape, loc, scale) = genpareto.fit(rs_M_largest, method=method)
    
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


def zhang_stephens_k(X):

    """
    https://sci-hub.se/https://www.tandfonline.com/doi/abs/10.1198/tech.2009.08017?role=button&needAccess=true&journalCode=utch20
    See primarily (5), (6) and (7).

    Zhang and Stephens estimate of the shape parameter (k^{hat}_{NEW}) of the Pareto distribution. 
    
    """

    n = len(X)

    m = 20 + int(np.sqrt(n)) # m = 20 + [√n], where [x] denotes the largest integer smaller than or equal to x
    X_star = sorted(X)[int(n/4+0.5)] # Let X^* = X([n/4+0.5]) be the first quartile of the sample data.
    X_order_statistics_n = np.max(X) # X_{(n)} is the n:th order_statistics of X - which is the max of X.

    #print(m , X_star, X_order_statistics_n)

    thetas = np.array([])
    for j in range(m):
        theta_j = 1/X_order_statistics_n + (1 - np.sqrt(m/(j+0.5)) ) / (3*X_star) # +0.5 instead of -0.5 due to index start at 0.
        thetas = np.append(thetas, theta_j)
    assert len(thetas) == m , 'there should be m thetas'
    assert (thetas < 1/(X_order_statistics_n)).all() , 'thetas should be less than 1/X(n)'

    def calculate_l(theta, X, n):
        # l(θ)
        #k = -1/n * np.sum(np.log(1-theta*X)) # Note: not the same as shape parameter of Pareto.
        k = -np.mean(np.log(1-theta*X))
        l = n*(np.log(theta/k) + k - 1)
        return l

    ll = np.array([])
    for theta_j in thetas:
        ll = np.append(ll, calculate_l(theta_j, X, n))
    assert len(ll) == m , 'there should be m l(theta)'
    print(max(ll))

    ww = np.array([])
    for j in range(m):
        denominator = np.sum(np.exp(ll[j]-ll))
        ww = np.append(ww, 1/denominator)
    assert len(ww) == m , 'there should be m ww'

    theta_new = np.sum(thetas*ww)
    #k_new = 1/n*np.sum(np.log(1-theta_new*X))
    k_new = -np.mean(np.log(1-theta_new*X))
    return k_new



if __name__ == '__main__':
    # Example usage
    from tensorflow_probability import distributions as tfd
    from tensorflow.random import set_seed
    set_seed(123)
    run_stability_test = True #repeat to see stability of k_hat
    run_jackknife = True


    # generate some mock data
    method = 'NEW'
    k, loc, scale = 0.9, 0, 1
    gpd = tfd.GeneralizedPareto(loc=loc, scale=scale, concentration=k)
    print('parameters: (k(shape), loc, scale)')
    print('original parameters:', (k, loc, scale))
    samples = gpd.sample(1000, seed=1231) # seed 1231 gives poor full fit estimate

    print('alt k: ', (np.mean(samples)**2/(np.std(samples)**2-1)/2)) #X_bar^2 /( S^2 - 1) / 2

    full_fit = fit_generalized_pareto_on_tail(samples, M = len(samples), return_all_parameters=True, method=method) #M = len(samples) just to see the fit on all of the data
    print('full fit parameters:', full_fit)

    tail_fit = fit_generalized_pareto_on_tail(samples, return_all_parameters=True,  method=method) #M determined according to paper
    print('tail fit parameters:', tail_fit)
    psis_diagnostic_results(tail_fit[0])

    
    if run_stability_test:
        print('\nAverage over new samples')
        full_fit_k_shapes = []
        tail_fit_k_shapes = []
      
        for i in range(100):
            new_samples = gpd.sample(1000)

            full_fit = fit_generalized_pareto_on_tail(new_samples, M=len(new_samples), return_all_parameters=True,  method=method)
            full_fit_k_shapes.append(full_fit[0])

            tail_fit = fit_generalized_pareto_on_tail(new_samples, return_all_parameters=True,  method=method)
            tail_fit_k_shapes.append(tail_fit[0])
            
        print(f'Repeated full fit k: {np.mean(full_fit_k_shapes)} ({np.std(full_fit_k_shapes)})')
        print(f'Repeated tail fit k: {np.mean(tail_fit_k_shapes)} ({np.std(tail_fit_k_shapes)})')

   
    if run_jackknife:
        print('\nJackknife')
        full_fit_jackknife = []
        tail_fit_jackknife = []
      
        for i in range(len(samples)):
            jackknife_samples = [x for j,x in enumerate(samples) if j!=i] 

            full_fit = fit_generalized_pareto_on_tail(jackknife_samples, M=len(samples), return_all_parameters=True,  method=method)
            full_fit_jackknife.append(full_fit[0])

            tail_fit = fit_generalized_pareto_on_tail(jackknife_samples, return_all_parameters=True,  method=method)
            tail_fit_jackknife.append(tail_fit[0])
            
        print(f'Jackknife full fit k: {np.mean(full_fit_jackknife)} ({np.std(full_fit_jackknife)})')
        print(f'Jackknife tail fit k: {np.mean(tail_fit_jackknife)} ({np.std(tail_fit_jackknife)})')