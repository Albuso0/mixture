source("gmmGM.R")


model.p = c(0.6,0.4)
model.x = c(1.6,-1.2)

n <- 5000
k <- 2
x <- sampleGM(n=n, p=model.p, x=model.x, sigma=1)

print(gmmGM(k, x, sigma=1))
print(gmmGM(k, x))
