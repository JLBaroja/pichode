data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> y;
  array[N] int<lower=0, upper=1> r_left;
  array[N] int<lower=0, upper=1> r_right;
}

parameters {
  real<lower=1, upper=5000> tau;
}

model {
  real q = 1 / tau;
  real I_left = 0.5;
  real I_right = 0.5;

  // Uniform(1, 1000) prior is implicit in the bounds above.

  for (t in 1:N) {
    real p_left = I_left / (I_left + I_right);

    y[t] ~ bernoulli(p_left);

    I_left = (1 - q) * I_left + q * r_left[t];
    I_right = (1 - q) * I_right + q * r_right[t];
  }
}
