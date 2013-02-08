n1 <- 5
n <- 10
alpha <- 0.05

phat <- n1 / n
z <- qnorm(alpha/2)
se <- -z * sqrt(phat*(1-phat) / n)

cat(sprintf("phat = %g +/- %g\n", phat, se))

ref <- binom.test(n1, n, conf.level=alpha)
cat(sprintf("R estimate: phat = %g +/- %g (%g)\n", 
            ref$estimate, ref$estimate - ref$conf.int[1], ref$conf.int[2] - ref$estimate))

