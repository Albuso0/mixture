library(gmm)



g <- function(u,x) {
    m1 <- u[1]*u[2]+(1-u[1])*u[3]-x
    m2 <- u[1]*u[2]^2+(1-u[1])*u[3]^2+1-x^2
    m3 <- u[1]*u[2]^3+(1-u[1])*u[3]^3+3*x-x^3
    f <- cbind(m1,m2,m3)
    return(f)
}

g1 <- function(tet,x)
{
    m1 <- (tet[1]-x)
    m2 <- (tet[2]^2 - (x - tet[1])^2)
    m3 <- x^3-tet[1]*(tet[1]^2+3*tet[2]^2)
    f <- cbind(m1,m2,m3)
    ## f <- cbind(m1,m2)
    return(f)
}

n <- 10000
wcnt <- 0
for ( i in 1:1000) {
    x <- rnorm(n, mean = 0, sd = 1)
    ## print(res <- gmm(g1,x,c(mu = 0, sig = 0)))
    print(gmm(g,x,c(p1=0.5,x1=0,x2=0)))
    tryCatch(print(gmm(g,x,c(p1=0.5,x1=0,x2=0))), warning=function(w) {
        wcnt <<- wcnt + 1
        print(w)
    } )

    ## vcov(res)
    ## summary(res)
    ## print(res)
}

print(sprintf("Warnings: %d/1000 experiments", wcnt))
