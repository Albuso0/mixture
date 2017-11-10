gmmGM <- function(k, x, sigma=-1) {
    ## k: number of components
    ## x: samples
    ## sigma: common standard deviation. sigma<0 means unknown. 
    ## Returns:
    ## return estimated model (p,x,sigma)

    ## parameters
    ## if simga<0 (unknown), param = (x_1,...,x_k, p_1,...,p_{k-1}, sigma)
    ## if simga>0, param = (x_1,...,x_k, p_1,...,p_{k-1})

    ## define moment condition: g
    ## define constraints ui, ci: ui %*% param - ci >= 0
    if (sigma<0) {
        L <- 2*k
        trfm <- hermite(L)
        g <- function(param, sample) {
            std <- param[L]
            atm <- param[1:k]
            wgt <- c(param[(k+1):(2*k-1)],1-sum(param[(k+1):(2*k-1)]))
            momU <- moment(wgt, atm, L)
            sPow <- moment(1, std, L)
            xPow <- momentAll(sample/std, L)
            momX <- (trfm[[1]]%*%xPow + trfm[[2]])*sPow
            return(t(momU-momX))
        }
        ui <- matrix(nrow=2*k,ncol=L)
        ci <- numeric(2*k)
        for(i in 1:(k-1)) {
            row <- numeric(L)
            row[k+i] <- 1
            ui[2*i-1,] <- row
            row[k+i] <- -1
            ui[2*i,] <- row
            ci[2*i] <- -1
        }
        ui[2*k-1,] = c(rep(0,k), rep(-1,k-1),0)
        ci[2*k-1] = -1
        ui[2*k,] = c(rep(0,L-1),1)
    }
    else {
        L <- 2*k-1
        trfm <- hermite(L)
        g <- function(param, sample) {
            std <- sigma
            atm <- param[1:k]
            wgt <- c(param[(k+1):(2*k-1)],1-sum(param[(k+1):(2*k-1)]))
            momU <- moment(wgt, atm, L)
            sPow <- moment(1, std, L)
            xPow <- momentAll(sample/std, L)
            momX <- (trfm[[1]]%*%xPow + trfm[[2]])*sPow
            return(t(momU-momX))
        }
        ui <- matrix(nrow=2*k-1,ncol=L)
        ci <- numeric(2*k-1)
        for(i in 1:(k-1)) {
            row <- numeric(L)
            row[k+i] <- 1
            ui[2*i-1,] <- row
            row[k+i] <- -1
            ui[2*i,] <- row
            ci[2*i] <- -1
        }
        ui[2*k-1,] = c(rep(0,k), rep(-1,k-1))
        ci[2*k-1] = -1
    }


    best = 1e10
    for (rd in 1:5){
        init.p = rdirichlet(1, rep(1,k))
        init.x = runif(k, min = -1, max = 1)
        if (sigma < 0){
            init.sigma = runif(1, min = 0.5, max = 1.5)
            initial = c(init.x, init.p[1:k-1], init.sigma)
        }
        else{
            initial = c(init.x, init.p[1:k-1])
        }
        GMcand <- gmm(g,x,initial,optfct="constrOptim",ui=ui,ci=ci)
        if (GMcand$objective < best) {
            GM <- GMcand
            best = GMcand$objective
        }
    }
    
    param <- GM$coefficients
    atm <- param[1:k]
    wgt <- c(param[(k+1):(2*k-1)],1-sum(param[(k+1):(2*k-1)]))
    if (sigma<0) {
        std <- param[L]
    }
    else {
        std <- sigma
    }
    return(list(wgt,atm,std))
}

momentAll <- function(x, L) {
    ## x: atoms of length n
    ## L: highest degree
    ## Return:
    ## matrix of dimension (L*n), including all moments of each x of degree 1 to L
    n = length(x)
    mat <- matrix(nrow = L, ncol = n)
    mat[1,] <- x
    for(i in 2:L) {
        mat[i,] = mat[(i-1),]*x
    }
    return(mat)
}

moment <- function(p, x, L) {
    ## p: weights
    ## x: atoms
    ## L: highest degree
    ## return: moments of (p,x) of degree 1 to L
    res = numeric(L)
    k = length(x)
    pow = rep(1, k)
    for(i in 1:L) {
        pow = pow*x
        res[i] = sum(p*pow)
    }
    return(res)
}


hermite <- function(L) {
    ## Hermite transform matrix
    ## Return: list(A,b)
    ## given x=(x,x^2,...,x^L), we have Ax+b=(H_1(x),...,H_L(x))
    length <- L+1
    mat <- matrix(nrow = length, ncol = length)
    if (length > 0) {
        prepre <- numeric(length)
        prepre[1] <- 1
        mat[1,] <- prepre
    }
    if (length > 1) {
        pre <- numeric(length)
        pre[2] <- 1
        mat[2,] <- pre
    }
    for (k in 3:length) {
        ## recursion: H_{n+1}(x) = x * H_n(x) - n * H_{n-1}(x)
        coeffs <- c(0, pre[1:(length-1)]) - prepre*(k-2)
        mat[k,] <- coeffs
        prepre <- pre
        pre <- coeffs
    }

    return(list(mat[2:length, 2:length], mat[2:length, 1]))
}
