data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> y;
  array[N] int<lower=0, upper=1> r_left;
  array[N] int<lower=0, upper=1> r_right;
}

parameters {
  real<lower=1, upper=5000> tau_fast;
  real<lower=tau_fast, upper=5000> tau_slow;
  real<lower=0, upper=1> w;
}



model {
  real q_fast = 1 / tau_fast;
  real q_slow = 1 / tau_slow;

  real I_left_fast = 0.5;
  real I_right_fast = 0.5;
  real I_left_slow = 0.5;
  real I_right_slow = 0.5;

  for (t in 1:N) {
    real I_left = (1 - w) * I_left_fast + w * I_left_slow;
    real I_right = (1 - w) * I_right_fast + w * I_right_slow;

    y[t] ~ bernoulli(I_left / (I_left + I_right));

    I_left_fast = (1 - q_fast) * I_left_fast + q_fast * r_left[t];
    I_right_fast = (1 - q_fast) * I_right_fast + q_fast * r_right[t];
    I_left_slow = (1 - q_slow) * I_left_slow + q_slow * r_left[t];
    I_right_slow = (1 - q_slow) * I_right_slow + q_slow * r_right[t];
  }
}
