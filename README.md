# Multivariate

Learning Objectives: The main focus of this teaching page is to present some key applications which consider multivariate time series. Although the stochastic properties of such time series (e.g., stationary versus nonstationary) requires different techniques when considering statistical estimation and inference, here we examine some tools from the statistical point of view in terms of the dimensions of the problem rather than the stochastic properties of the data.  

# I. Likelihood-Based Inference

## A. Multivariate CAPM Model

## Example 1

Consider the following model for the excess returns 

$$Z_{it} = \alpha_i + \beta_i Z_{mt} + \epsilon_{it},$$ 
such that it holds that  
$$E ( \epsilon_{it} ) = 0, \ \  Var( \epsilon_{it} ) = \sigma_{\epsilon_i}^2.$$ 

We estimate the above multivariate CAPM model asset by asset using Ordinary Least Squares methods.

```R

# IMPORT DATASET
mydata <- read.csv("DATASET.csv",header=T)

# CONSTRUCT Y MATRIX
rm_rf <- mydata$rm_rf
Y     <- as.matrix(mydata[,c("z1","z2","z3","z4", "z5" )])
Y     <- data.frame(Y)

# DEFINE THE MULTIVARIATE REGRESSION
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
wald <-  alphas %*% inv(SigmaTilde) %*% (alphas)
wald <-  alphas %*% inv(SigmaHat) %*% (alphas)

coeftest(mvmod, vcov = vcovHAC(mvmod))
w <- 1/rm_rf
mvmod <- lm(Y ~ rm_rf, weights=w, data=mydata)

```

### Task 1

Write your own code that provides estimates for the Unconstrained and Constrained Likelihood-function for the multivariate CAPM model. Using your coding procedure, obtain the log-likelihood statistic for testing the assumptions of the Black CAPM model.  

```R

log.likelihood.test <- (T)*( log(det(sigma_hat_star))- log(det(sigma_hat)) )
ts.plot(abs(keep.const.like), main="Constrained log-likelihood function", xlab="Interval Estimation on the grid[-2,2]")

```

### References

Gibbons, M. R., Ross, S. A., & Shanken, J. (1989). A test of the efficiency of a given portfolio. Econometrica: Journal of the Econometric Society, 1121-1152.


## B. Multivariate Linear Regression Model

Consider the gollowing multivariate linear regression model 

$$y_t = A x_t + u_t, \ \ \ t = 1,...,n.$$

where yt is a d-dimensional vector of dependent variables and xt is is p-dimensional vector of exogenous variables such that 

$$u_t = R u_{t-1} + \varepsilon_t, \ \ \ t = 1,...,n.$$

Then, the likelihood function is given by 

$$L = \left( 2 \pi \right)^{ - \frac{1}{2} (d-1)(n-1) } | \Omega_n |^{ \frac{ - (n-1) }{ 2 } } e^{ - \frac{1}{2} \sum_{t=2}^n \epsilon_t^{\prime} \Omega_n^{-1}  \epsilon_t  }.$$

# II. Pooling Cross-Section and Time-Series: Realized Volatility  

A commonly used approach for modelling cross-sectional time series data, such as Cross-Sectional Realized Volatility measures, which implies the presence of a large number of regressors, often much larger than the time series observations, is to use Lasso-type estimators.

## Example 2

Consider a Lasso estimation for the cross-sectional Realized Volatility measures of S&P500 with a dependent variable the Realized Volatility measures of a particular firm. 

```R

# Lasso estimation step
cv <- cv.glmnet(x,y,alpha=1,nfolds=10)
l  <- cv$lambda.min
alpha <- 1

fits <- glmnet( x, y, family="gaussian", alpha=alpha, nlambda=100)

res <- predict(fits, s=l, type="coefficients")
res <- as.matrix( res )


```



