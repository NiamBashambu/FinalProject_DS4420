# ============================================================
# Model 2: Bayesian Hierarchical Model for Transfer Premiums
# Gibbs sampler implemented manually (no pre-built modeling packages)
# Dependencies: MCMCpack (rinvgamma), mvnfast (rmvn)
# ============================================================

library(MCMCpack)

# --------------------------------------------------
# Load data exported from Model 1 Python pipeline
# Each row is one transfer with:
#   log_premium  = log(actual_fee) - log(predicted_value)
#   selling_league, buying_league, player_pos, player_age
# --------------------------------------------------
df <- read.csv("../data/processed/model2_input.csv")

# Discretize age into bands: U21, 21-25, 26-29, 30+
df$age_band <- cut(df$player_age, 
                   breaks = c(0, 20, 25, 29, Inf),
                   labels = c("U21", "21-25", "26-29", "30+")
                   )

# Convert categorical columns to integer indices for array lookup
corridor_idx <- as.integer(as.factor(df$league_pair))
position_idx <- as.integer(as.factor(df$player_pos))
age_idx      <- as.integer(as.factor(df$age_band))

# Store the number of groups and observations
n <- nrow(df)
C <- length(unique(df$league_pair))
P <- length(unique(df$player_pos))
A <- length(unique(df$age_band))

# The dependent variable
y <- df$log_premium

# --------------------------------------------------
# 2. Hyperprior settings and MCMC storage
#    Priors:
#      alpha     ~ Normal(0, prior_var_alpha)   global intercept
#      gamma_c   ~ Normal(0, tau2_corridor)     corridor effects
#      delta_p   ~ Normal(0, tau2_position)     position effects
#      phi_a     ~ Normal(0, tau2_age)          age band effects
#      sigma2    ~ InvGamma(a0, b0)             observation variance
#      tau2_*    ~ InvGamma(a1, b1)             group-level variances
# --------------------------------------------------

# Hyperparameters for inverse-gamma priors
a0 <- 2; b0 <- 1    # InvGamma shape/scale for sigma2 (observation variance)
a1 <- 2; b1 <- 1    # InvGamma shape/scale for tau2 terms (group variances)

# Prior variance for the global intercept
prior_var_alpha <- 4

# MCMC settings
n_iter <- 10000
burn_in <- floor(n_iter * 0.18)

# Pre-allocate storage matrices
no_samples <- n_iter
samples_alpha <- matrix(0, nrow = no_samples, ncol = 1)
samples_gamma <- matrix(0, nrow = no_samples, ncol = C)  # C corridor effects each iteration
samples_delta <- matrix(0, nrow = no_samples, ncol = P)  # P position effects each iteration
samples_phi <- matrix(0, nrow = no_samples, ncol = A)  # A age band effects each iteration
# Below are scalars on each iter
samples_sigma_squared <- matrix(0, nrow = no_samples, ncol = 1)  
samples_tau2_corridor <- matrix(0, nrow = no_samples, ncol = 1) 
samples_tau2_position <- matrix(0, nrow = no_samples, ncol = 1)  
samples_tau2_age <- matrix(0, nrow = no_samples, ncol = 1) 

# Initialize parameters
alpha <- 0
gamma <- rep(0, C)
delta <- rep(0, P)
phi <- rep(0, A)
sigma_squared <- 1
tau2_corridor <- 1
tau2_position <- 1
tau2_age <- 1


# --------------------------------------------------
# 3. Gibbs sampler
# --------------------------------------------------
for(i in 1:n_iter) {
  # Residuals:
  resid_alpha <- y - gamma[corridor_idx] - delta[position_idx] - phi[age_idx]
  
  # Posterior variance and mean
  v_alpha <- 1/(n/sigma_squared + (1/prior_var_alpha))
  m_alpha <- v_alpha * sum(resid_alpha /sigma_squared)
  
  # Draw alpha from its full conditional
  alpha <- rnorm(1, m_alpha, sqrt(v_alpha))
  
  # Store
  samples_alpha[i] <- alpha
  
  # Draw gamma (corridor effects)
  for(c in 1:C) {
    # Which transfers belong to corridor c?
    in_c <- which(corridor_idx == c)
    
    # Residuals (for these transfers only):
    resid_c <- y[in_c] - alpha - delta[position_idx[in_c]] - phi[age_idx[in_c]]
    
    # Posterior variance and mean
    n_c <- length(in_c)
    v_c <- 1/(n_c/sigma_squared + (1/tau2_corridor))
    m_c <- v_c * sum(resid_c /sigma_squared)
    
    # draw gamma[c]
    gamma[c] <- rnorm(1, m_c, sqrt(v_c))
  }
  # Store
  samples_gamma[i, ] <- gamma
  
  # Draw delta (position effects)
  for(p in 1:P) {
    # Which transfers belong to position p?
    in_p <- which(position_idx == p)
    
    # Residuals (for these transfers only):
    resid_p <- y[in_p] - alpha - gamma[corridor_idx[in_p]] - phi[age_idx[in_p]]
    
    # Posterior variance and mean
    n_p <- length(in_p)
    v_p <- 1/(n_p/sigma_squared + (1/tau2_position))
    m_p <- v_p * sum(resid_p /sigma_squared)
    
    # draw delta[p]
    delta[p] <- rnorm(1, m_p, sqrt(v_p))
  }
  # Store
  samples_delta[i, ] <- delta
  
  
  # Draw phi (age effects)
  for(a in 1:A) {
    # Which transfers belong to age a?
    in_a <- which(age_idx == a)
    
    # Residuals (for these transfers only):
    resid_a <- y[in_a] - alpha - gamma[corridor_idx[in_a]] - delta[position_idx[in_a]]
    
    # Posterior variance and mean
    n_a <- length(in_a)
    v_a <- 1/(n_a/sigma_squared + (1/tau2_age))
    m_a <- v_a * sum(resid_a /sigma_squared)
    
    # draw phi[a]
    phi[a] <- rnorm(1, m_a, sqrt(v_a))
  }
  # Store
  samples_phi[i, ] <- phi
  
  # Draw sigma_squared (observation variance)
  # Full residuals: y minus ALL four effects
  resid_all <- y - alpha - gamma[corridor_idx] - delta[position_idx] - phi[age_idx]
  
  # InvGamma posterior parameters
  shape_sigma <- (2*a0 + n) /2
  scale_sigma <- (2*b0 + sum(resid_all^2)) / 2
  
  sigma_squared <- rinvgamma(1, shape_sigma, scale_sigma)
  
  # Store
  samples_sigma_squared[i] <- sigma_squared
  
  # Draw tau2_corridor (corridor group variance)
  shape_tau_c <- (2*a1 + C) /2
  scale_tau_c <- (2*b1 + sum(gamma^2)) / 2
  
  tau2_corridor <- rinvgamma(1, shape_tau_c, scale_tau_c)
  
  # Store
  samples_tau2_corridor[i] <- tau2_corridor
  
  # Draw tau2_position
  shape_tau_p <- (2*a1 + P) /2
  scale_tau_p <- (2*b1 + sum(delta^2)) / 2
  
  tau2_position <- rinvgamma(1, shape_tau_p, scale_tau_p)
  
  # Store
  samples_tau2_position[i] <- tau2_position
  
  # Draw tau2_age
  shape_tau_a <- (2*a1 + A) /2
  scale_tau_a <- (2*b1 + sum(phi^2)) / 2
  
  tau2_age <- rinvgamma(1, shape_tau_a, scale_tau_a)
  
  # Store
  samples_tau2_age[i] <- tau2_age
}
  

# --------------------------------------------------
# 4. Convergence diagnostics
# --------------------------------------------------

# Trace plots for key parameters
plot(samples_alpha, type = "l", main = "Trace: alpha (global intercept)",
     xlab = "Iteration", ylab = "alpha")

plot(samples_sigma_squared, type = "l", main = "Trace: sigma2 (observation variance)",
     xlab = "Iteration", ylab = "sigma2")

plot(samples_gamma[, 1], type = "l",
     main = paste("Trace: gamma_1 (corridor:", levels(as.factor(df$league_pair))[1], ")"),
     xlab = "Iteration", ylab = "gamma_1")

plot(samples_tau2_corridor, type = "l",
     main = "Trace: tau2_corridor",
     xlab = "Iteration", ylab = "tau2_corridor")

# ACF plots
acf(samples_alpha, main = "ACF: alpha")
acf(samples_sigma_squared, main = "ACF: sigma2")
acf(samples_gamma[, 1], main = "ACF: gamma_1")
acf(samples_tau2_corridor, main = "ACF: tau2_corridor")

# --------------------------------------------------
# 5. Apply burn-in and (optionally) thinning
# --------------------------------------------------
burn_idx <- (burn_in + 1):n_iter

post_alpha <- samples_alpha[burn_idx]
post_gamma <- samples_gamma[burn_idx, ]
post_delta <- samples_delta[burn_idx, ]
post_phi <- samples_phi[burn_idx, ]
post_sigma2 <- samples_sigma_squared[burn_idx]
post_tau2_corridor <- samples_tau2_corridor[burn_idx]
post_tau2_position <- samples_tau2_position[burn_idx]
post_tau2_age <- samples_tau2_age[burn_idx]

# --------------------------------------------------
# 6. Posterior summaries
# --------------------------------------------------

# Helper function: summarize a vector of posterior samples
summarize_posterior <- function(samples) {
  c(mean = mean(samples),
    sd   = sd(samples),
    q05  = quantile(samples, 0.05),
    q95  = quantile(samples, 0.95))
}

# Global intercept — the average log-premium across all transfers
cat("=== Global intercept (alpha) ===\n")
print(summarize_posterior(post_alpha))

# Observation variance
cat("\n=== Observation variance (sigma2) ===\n")
print(summarize_posterior(post_sigma2))

# Corridor effects — the key result for the project
cat("\n=== Corridor effects (gamma) ===\n")
corridor_labels <- levels(as.factor(df$league_pair))
corridor_summary <- t(apply(post_gamma, 2, summarize_posterior))
rownames(corridor_summary) <- corridor_labels
print(round(corridor_summary, 4))

# Position effects
cat("\n=== Position effects (delta) ===\n")
position_labels <- levels(as.factor(df$player_pos))
position_summary <- t(apply(post_delta, 2, summarize_posterior))
rownames(position_summary) <- position_labels
print(round(position_summary, 4))

# Age band effects
cat("\n=== Age band effects (phi) ===\n")
age_labels <- levels(as.factor(df$age_band))
age_summary <- t(apply(post_phi, 2, summarize_posterior))
rownames(age_summary) <- age_labels
print(round(age_summary, 4))

# Group-level variances
cat("\n=== Group-level variances (tau2) ===\n")
cat("Corridor:", round(summarize_posterior(post_tau2_corridor), 4), "\n")
cat("Position:", round(summarize_posterior(post_tau2_position), 4), "\n")
cat("Age band:", round(summarize_posterior(post_tau2_age), 4), "\n")


# --------------------------------------------------
# 7. Visualizations for the report
# --------------------------------------------------

# Caterpillar plot:shows which corridors overpay/underpay with full uncertainty bands
corridor_means <- colMeans(post_gamma)
corridor_q05   <- apply(post_gamma, 2, quantile, 0.05)
corridor_q95   <- apply(post_gamma, 2, quantile, 0.95)
ord <- order(corridor_means)

par(mar = c(5, 12, 4, 2))  # wide left margin for corridor labels
plot(corridor_means[ord], 1:C, xlim = range(c(corridor_q05, corridor_q95)),
     pch = 19, yaxt = "n", xlab = "Posterior corridor effect (log-premium)",
     ylab = "", main = "Corridor Effects with 90% Credible Intervals")
segments(corridor_q05[ord], 1:C, corridor_q95[ord], 1:C)
axis(2, at = 1:C, labels = corridor_labels[ord], las = 1, cex.axis = 0.7)
abline(v = 0, lty = 2, col = "red")  # zero line: no premium

# Posterior histograms for a few interesting corridors
top_corridor <- ord[C]
bottom_corridor <- ord[1]

par(mfrow = c(1, 2))
hist(post_gamma[, top_corridor], breaks = 40, probability = TRUE,
     main = paste("Highest premium:", corridor_labels[top_corridor]),
     xlab = "Log-premium effect", col = "steelblue")
abline(v = 0, lty = 2, col = "red")

hist(post_gamma[, bottom_corridor], breaks = 40, probability = TRUE,
     main = paste("Lowest premium:", corridor_labels[bottom_corridor]),
     xlab = "Log-premium effect", col = "steelblue")
abline(v = 0, lty = 2, col = "red")
par(mfrow = c(1, 1))

# Position effects comparison
par(mar = c(5, 8, 4, 2))
position_means <- colMeans(post_delta)
position_q05   <- apply(post_delta, 2, quantile, 0.05)
position_q95   <- apply(post_delta, 2, quantile, 0.95)
ord_p <- order(position_means)

plot(position_means[ord_p], 1:P, xlim = range(c(position_q05, position_q95)),
     pch = 19, yaxt = "n", xlab = "Posterior position effect (log-premium)",
     ylab = "", main = "Position Effects with 90% Credible Intervals")
segments(position_q05[ord_p], 1:P, position_q95[ord_p], 1:P)
axis(2, at = 1:P, labels = position_labels[ord_p], las = 1, cex.axis = 0.8)
abline(v = 0, lty = 2, col = "red")



