source("gmmGM.R")


printGM <- function(GM) {
    l <- length(GM$coefficients)
    k <- as.integer( (l+1)/2 )
    x <- GM$coefficients[1:k]
    p <- GM$coefficients[(k+1):l]
    cat("p= ", p, 1-sum(p), "\n", sep=" ")
    cat("x= ", x, "\n", sep=" ")
    ## cat(x,p,1-sum(p),"\n",sep=" ")
}
model.p = c(0.6,0.4)
model.x = c(1.6,-1.2)
x <- sampleGM(n=5000, p=model.p, x=model.x, sigma=2)
estimate <- gmmGM(k=2, x, sigma=-1)
esti.p <- estimate[[1]]
esti.x <- estimate[[2]]
esti.sigma <- estimate[[3]]
print(w1(model.p, model.x, esti.p, esti.x))
quit()


## cat("Model: \n")
model.p = c(1.)
model.x = c(0.)
## cat("p= ", model.p, "\n", sep=" ")
## cat("x= ", model.x, "\n", sep=" ")


sample <- (1:10)*500
rep <- 20


cat("Estimate: \n")


for (j in 1:length(sample)){
    n <- sample[j]
    cat("n= ", n, "\n")
    ptm <- proc.time()
    for ( i in 1:rep) {
        u <- sample(x = model.x, n, replace = T, prob = model.p) 
        z <- rnorm(n, mean = 0, sd = 1)
        x <- u+z

        best = 1e10
        for (rd in 1:5){
            
            ## GM <- gmm(g,x,initial)
            ## GM <- gmm(g3,x,c(x1=-1,x2=0,x3=1,p1=1./3,p2=1./3))
            ## GM <- gmm(g,x,initial,optfct="nlminb",
            ##           lower=c(rep.int(-Inf, k), rep.int(0,k-1)),
            ##           upper=c(rep.int(Inf, k), rep.int(1,k-1)))

            init.p = rdirichlet(1, rep(1,k))
            init.x = runif(k, min = -1, max = 1)
            initial = c(init.x, init.p[1:k-1])
            GMcand <- gmm(g,x,initial,optfct="constrOptim",ui=ui,ci=ci)
            if (GMcand$objective < best) {
                GM <- GMcand
                best = GMcand$objective
            }
        }
        printGM(GM)
            
        ## print(gmm(g,x,c(p1=0.5,x1=0,x2=0)),type="twoStep")
        
        ## vcov(res)
        ## print(summary(GM))
        ## print(res)
    }
    time = proc.time() - ptm
    cat("Time:", time[1]," seconds\n", sep=" ")
}

