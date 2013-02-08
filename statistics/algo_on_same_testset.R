n <- 10000
p_ref <- 0.0021
alpha <- 0.01
p_new <- 0.0001
iters <- 1000


rejects <- 0

for(i in 1:iters) {
  X_ref <- rbinom(n, 1, 1 - p_ref)
  X_new <- rbinom(n, 1, 1 - p_new)
  
#  pr_ref <- sum(X_ref) / n
#  pr_new <- sum(X_new) / n
  
  D <- X_ref - X_new
  delta <- mean(D)
  se <- sqrt(sum((D-delta)^2)) / n
  W <- delta / se
  
  z <- -qnorm(alpha / 2)
  
#  cat(sprintf("|W|: %g  z: %g\n", abs(W), z))
  if(abs(W) > z)
    rejects <- rejects + 1
}

cat(sprintf("Power for difference %g is %g\n",
            p_ref - p_new, rejects / iters))
