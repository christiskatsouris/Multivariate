# Multivariate

# I. Likelihood-Based Inference: Multivariate CAPM Model

## Example 1

Consider the following model for the excess returns 

$$Z_{it} = \alpha_i + \beta_i Z_{mt} + \epsilon_{it},$$ 
such that it holds that  
$$E ( \epsilon_{it} ) = 0, \ \  Var( \epsilon_{it} ) = \sigma_{\epsilon_i}^2.$$ 

We estimate the above multivariate CAPM model asset by asset using Ordinary Least Squares methods.

```R

#IMPORT DATASET
mydata <- read.csv("DATASET_VALUE1.csv",header=T)

# CONSTRUCT Y MATRIX
rm_rf <- mydata$rm_rf
Y     <- as.matrix(mydata[,c("z1","z2","z3","z4", "z5" )])

Y<-data.frame(Y)

#DEFINE THE MULTIVARIATE REGRESSION
mvmod  <- lm(Y ~ rm_rf, data=mydata)
mvsum  <- summary(mvmod)
alphas <- coef(mvmod)[1,]
betas  <- coef(mvmod)[2,]

ybar <- colMeans(Y)
n    <- nrow(Y)
m    <- ncol(Y)
Ybar <- matrix(ybar, n, m, byrow=TRUE)

# ESTIMATING THE SUMS OF SQUARES AND CROSS-PRODUCTS
SSCP.T <- crossprod(Y - Ybar)
SSCP.R <- crossprod(mvmod$fitted.values - Ybar)
SSCP.E <- crossprod(Y - mvmod$fitted.values)

SSCP.T
SSCP.R + SSCP.E

# ESTIMATING ERROR COVARIANCE MATRIX 
n <- nrow(Y)
p <- nrow(coef(mvmod)) - 1
SSCP.E     <- crossprod(Y - mvmod$fitted.values)
SigmaHat   <- SSCP.E / (n - p - 1)
SigmaTilde <- SSCP.E / n

# ESTIMATE THE WALD STATISTIC
wald.test(b=alphas, Sigma = SigmaTilde, Terms=1:5 )
wald.test(b=alphas, Sigma = SigmaHat, Terms=1:5 )

# MANUAL ESTIMATION OF WALD STATISTIC
alphas
wald <-  alphas %*% inv(SigmaTilde) %*% (alphas)
wald <-  alphas %*% inv(SigmaHat) %*% (alphas)

coeftest(mvmod, vcov = vcovHAC(mvmod))

w <- 1/rm_rf
mvmod <- lm(Y ~ rm_rf, weights=w, data=mydata)


```

# II. Vector Autoregression Model: Estimation and Inference


# III. Pooling Cross-Section and Time-Series Data: Realized Volatility  

