#set.seed(123)
n_samples <- 100
alpha <- 0.05
epsilon <- 1
x_ref <- rnorm(n_samples, -153, 3)
x_new <- rnorm(n_samples, -153, 3)

# check H_01: m_ref - m_new - epsilon >=  0
H1_tst <- t.test(x_ref - x_new - epsilon, NULL, 'l')

# check H_02: m_ref - m_new + epsilon <=  0
H2_tst <- t.test(x_ref - x_new + epsilon, NULL, 'g')
 
if (H1_tst$p.value < alpha && H2_tst$p.value < alpha) {
  cat("x_new is identical to x_ref +/- epsilon")
} else {
  cat("x_new is not identical to x_ref +/- epsilon")
}

 