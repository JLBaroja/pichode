data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> y;
  array[N] int<lower=0, upper=1> r_left;
  array[N] int<lower=0, upper=1> r_right;
}

parameters {
  real<lower=2, upper=5000> tau;
  real log_relative_strength;
}

transformed parameters {
  real<lower=0> A0 = tau * exp(log_relative_strength);
}

model {
  real q = 1 / tau;
  real I_left = q * A0 * 0.5;
  real I_right = q * A0 * 0.5;

  // Priors
  log_relative_strength ~ normal(0, 1);

  for (t in 1:N) {
    real p_left = I_left / (I_left + I_right);

    y[t] ~ bernoulli(p_left);

    I_left = (1 - q) * I_left + q * r_left[t];
    I_right = (1 - q) * I_right + q * r_right[t];
  }
}
