data {
  int<lower=0> N;
  int<lower=0, upper=1> y[N];
  real x[N];
}

parameters {
  real beta; // slope
  real alpha; // intercept
}

transformed parameters {
  real eta[N];
  for (i in 1:N)
    eta[i] = beta * x[i] + alpha;
}

model {
  y ~ bernoulli_logit(eta);
}
