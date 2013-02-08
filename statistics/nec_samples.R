alpha <- 0.05
power <- 0.95
sds <- seq(0.25, 2, 0.25)

for(j in 1:length(sds)) {
  deltas <- seq(0.1, 10, 0.05)
  nec_samples <- c()
  
  for(i in 1:length(deltas)) {
    nec_samples[i] <- NA  
    try({
      p <- power.t.test(n=NULL, delta=deltas[i], sd=sds[j], sig.level=alpha, 
                        power=power, alternative="one.sided")
      nec_samples[i] <- p$n
    }, TRUE)
  }
  
  if(j == 1)
    plot(deltas, nec_samples, "l", log="y", xlab="", ylab="")
  else
    lines(deltas, nec_samples)
}

legend("topright",y=NULL, c("sd:",sds[length(sds):1]))

title(main="Necessary samples for test H0: m <= 0 vs H1: m > 0",
      sub=sprintf(paste("P(H0 is true but test rejected it) = %g  ", 
                        "P(H1 is true and test accepted it) = %g"), 
                  alpha, power),      
      xlab="True value of m", ylab="Number of necessary samples to reject H0")        

