library(gmm)
library(MCMCpack)
source("functions.R")
## library(transport)


printGM <- function(GM) {
    l <- length(GM$coefficients)
    k <- as.integer( (l+1)/2 )
    x <- GM$coefficients[1:k]
    p <- GM$coefficients[(k+1):l]
    cat("p= ", p, 1-sum(p), "\n", sep=" ")
    cat("x= ", x, "\n", sep=" ")
    ## cat(x,p,1-sum(p),"\n",sep=" ")
}

models <- read.table("result2/Models.tab")
models[1] <- NULL

args <- commandArgs(TRUE)
model <- as.integer(args[1])

## model = 5

k <- 5

cat("Model: \n")
## model.p = rdirichlet(1, rep(1,k))
## model.x = runif(k, min = -1, max = 1)
## model.x = c(-5,0,5)
## model.p = c(1./3, 1./3, 1./3)
model.p = unlist(models[2*model-1,], use.names=FALSE)
model.x = unlist(models[2*model,], use.names=FALSE)
cat("p= ", model.p, "\n", sep=" ")
cat("x= ", model.x, "\n", sep=" ")


sample <- (1:10)*500
rep <- 20
g <- g5 ## gk: k-component


cat("Estimate: \n")
## initial = c(x1=-1,x2=0,x3=1,x4=3,x5=2,p1=1./5,p2=1./5,p3=1./5,p4=1./5) ## initial guess (x_1,..x_k,p1,...p_{k-1})
## c(x1=0,x2=0,x3=1,p1=0.3,p2=0.3)
## c(x1=0,x2=0,x3=1,x4=1,p1=0.25,p2=0.25,p3=0.25)


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
            initial = c(init.x, init.p[1:4])
            GMcand <- gmm(g,x,initial,optfct="constrOptim",
                      ui = rbind(c(0,0,0,0,0,1,0,0,0),
                                 c(0,0,0,0,0,-1,0,0,0),
                                 c(0,0,0,0,0,0,1,0,0),
                                 c(0,0,0,0,0,0,-1,0,0),
                                 c(0,0,0,0,0,0,0,1,0),
                                 c(0,0,0,0,0,0,0,-1,0),
                                 c(0,0,0,0,0,0,0,0,1),
                                 c(0,0,0,0,0,0,0,0,-1),
                                 c(0,0,0,0,0,-1,-1,-1,-1)),
                      ci = c(0,-1,0,-1,0,-1,0,-1,-1) )
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












## wcnt <- 0

## tryCatch(print(gmm(g,x,c(p1=0.5,x1=0,x2=0))), warning=function(w) {
##     wcnt <<- wcnt + 1
##     print(w)
## } )

## print(sprintf("Warnings: %d/1000 experiments", wcnt))


