n <- 10000
p_ref <- 1 - 0.0021
alpha <- 0.01
ps <- 1 - seq(0.0001, 0.005, 0.0001)

z <- -qnorm(alpha / 2)

Ws <- c()
zs <- c()

for(i in 1:length(ps)) {
  p <- ps[i]
  Ws[i] <- abs((p_ref - p) * sqrt(n) / sqrt(p_ref * (1-p_ref) + p * (1-p)))
  zs[i] <- z
}

plot(ps*100, Ws, "l", xlab="p_new in %", ylab="W(p_new)")
lines(ps*100, zs, col="red")
abline(v=p_ref*100, col="blue")

sleft <- uniroot(function(p) abs((p_ref - p) * sqrt(n) / sqrt(p_ref * (1-p_ref) + p * (1-p))) - z,
                 c(0,p_ref))
sright <- uniroot(function(p) abs((p_ref - p) * sqrt(n) / sqrt(p_ref * (1-p_ref) + p * (1-p))) - z,
                  c(p_ref,1))

cat(sprintf("Significant measured difference for level %g is %g%% (for better) or %g%% (for worse)", 
            alpha,
            (p_ref - sleft$root)*100,
            (sright$root - p_ref)*100))

