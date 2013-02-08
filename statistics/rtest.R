#set.seed(123)
n_samples <- 5
alpha <- 0.05
power <- 0.95

x_ref <- rnorm(n_samples, -153, 3)
x_new <- rnorm(n_samples, -150, 3)
tst <- t.test(x_new, x_ref, 'g', var.equal=TRUE)

if (tst$p.value < alpha) {
  cat("x_new is better\n")
} else {
  cat("x_new is not better\n")
}

pwr <- power.t.test(n=NULL, 2, sd=2, sig.level=alpha, power=power, alternative="one.sided")
cat(sprintf("It would need %g (have: %g) samples to test with power %f\n",
            pwr$n, n_samples, power))



