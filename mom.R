library(gmm)
source("functions.R")
## library(transport)


printGM <- function(GM) {
    l <- length(GM$coefficients)
    k <- as.integer( (l+1)/2 )
    x <- GM$coefficients[1:k]
    p <- GM$coefficients[(k+1):l]
    cat(x,p,1-sum(p),"\n",sep=" ")
}


sample <- (1:100)*1000
rep <- 20

cat("Model: (x1,x2,...,p1,p2,...)\n")
model.x = c(-0,0,0)
model.p = c(1./3, 1./3, 1./3)
cat(model.x,model.p,"\n",sep=" ")


cat("Estimate: (x1,x2,...,p1,p2,...)\n")
g <- g3 ## gk: k-component
initial = c(x1=-0,x2=0,x3=0,p1=1./3,p2=1./3) ## initial guess
## c(x1=0,x2=0,x3=1,p1=0.3,p2=0.3)
## c(x1=0,x2=0,x3=1,x4=1,p1=0.25,p2=0.25,p3=0.25)


for (j in 1:length(sample)){
    n <- sample[j]
    cat("n= ", n, "\n")
    for ( i in 1:rep) {
        u <- sample(x = model.x, n, replace = T, prob = model.p) 
        z <- rnorm(n, mean = 0, sd = 1)
        x <- u+z

        GM <- gmm(g,x,initial)
        ## GM <- gmm(g3,x,c(x1=-1,x2=0,x3=1,p1=1./3,p2=1./3))
        printGM(GM)
        ## print(gmm(g,x,c(p1=0.5,x1=0,x2=0)),type="twoStep")

        ## vcov(res)
        ## summary(res)
        ## print(res)
    }

}




## wcnt <- 0

## tryCatch(print(gmm(g,x,c(p1=0.5,x1=0,x2=0))), warning=function(w) {
##     wcnt <<- wcnt + 1
##     print(w)
## } )

## print(sprintf("Warnings: %d/1000 experiments", wcnt))


