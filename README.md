# Multivariate

Learning Objectives: The main focus of this teaching page is to present some key applications which consider multivariate time series. Although the stochastic properties of such time series (e.g., stationary versus nonstationary) requires different techniques when considering statistical estimation and inference, here we examine some tools from the statistical point of view in terms of the dimensions of the problem rather than the stochastic properties of the data.  

# I. Likelihood-Based Estimation and Inference

## [A1.] Multivariate CAPM Model

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

Write your own code in [R](https://www.r-project.org/), [Matlab](https://uk.mathworks.com/help/matlab/getting-started-with-matlab.html) or [Stata](https://www.stata.com/bookstore/getting-started-windows/) that provides estimates for the Unconstrained and Constrained Likelihood-function for the multivariate CAPM model. Using your coding procedure, obtain the log-likelihood statistic for testing the assumptions of the Black CAPM model.  

```R

log.likelihood.test <- (T)*( log(det(sigma_hat_star))- log(det(sigma_hat)) )
ts.plot(abs(keep.const.like), main="Constrained log-likelihood function", xlab="Interval Estimation on the grid[-2,2]")

```

### References

- Gibbons, M. R., Ross, S. A., & Shanken, J. (1989). A test of the efficiency of a given portfolio. Econometrica: Journal of the Econometric Society, 1121-1152.
- Sharpe, W. F. (1964). Capital asset prices: A theory of market equilibrium under conditions of risk. The journal of finance, 19(3), 425-442.

## [B1.] Multivariate Linear Regression Model

Consider the gollowing multivariate linear regression model 

$$y_t = A x_t + u_t, \ \ \ t = 1,...,n.$$

where yt is a d-dimensional vector of dependent variables and xt is is p-dimensional vector of exogenous variables such that 

$$u_t = R u_{t-1} + \varepsilon_t, \ \ \ t = 1,...,n.$$

Then, the likelihood function is given by 

$$L = \left( 2 \pi \right)^{ - \frac{1}{2} (d-1)(n-1) } | \Omega_n |^{ \frac{ - (n-1) }{ 2 } } exp \left(- \frac{1}{2} \sum \epsilon_t^{\prime} \Omega_n^{-1}  \epsilon_t \right) .$$


### Task 2

Consider an adding-up multivariate linear regression model and write your own code in [R](https://www.r-project.org/), [Matlab](https://uk.mathworks.com/help/matlab/getting-started-with-matlab.html) or [Stata](https://www.stata.com/bookstore/getting-started-windows/) to obtain the maximum likelihood estimation for testing appropriate linear restricitions on the parameters of the model. 


### References

- Berndt, E. R., & Savin, N. E. (1975). Estimation and hypothesis testing in singular equation systems with autoregressive disturbances. Econometrica: Journal of the Econometric Society, 937-957.
- Henshaw Jr, R. C. (1966). Testing single-equation least squares regression models for autocorrelated disturbances. Econometrica: Journal of the Econometric Society, 646-660.

# II. Regression-Based Estimation and Inference

## [A2.] Time-Series Regression with Autocorrelated or Autoregressive Errors 

Consider the simple linear regression with autocorrelated errors as below

$$y_t = \mu + \alpha t + u_t, \ \ \ \text{for} \ t =1,...,n.$$

Notice that the usual OLS estimators operate under the assumptuon that the ut's are uncorrelated. However, many financial time series data are found to have some autocorrelation structure, therefore the above model is more realistic.   

Consider the time-series regression model with autoregressive error structure as below

$$y_t = x_t^{\top} \beta + u_t, \ \ u_t = \rho u_{t-1} + v_t, \ \ \ \text{for} \ t =1,...,n.$$

Next, consider the multivariate version of the above model such that

$$Y_t = B X_t + U_t, \ \ U_t = R U_{t-1} + V_t, \ \ \ \text{for} \ t =1,...,n.$$

### Task 3

Using a time-series dataset of your choice and by fitting the above model obtain the estimate of the coefficient matrix B by writing code in [R](https://www.r-project.org/), [Matlab](https://uk.mathworks.com/help/matlab/getting-started-with-matlab.html) or [Stata](https://www.stata.com/bookstore/getting-started-windows/).

### References

- Lee, J., & Lund, R. (2004). Revisiting simple linear regression with autocorrelated errors. Biometrika, 91(1), 240-245.

## [B2]. Pooling Cross-Section and Time-Series (Lasso Estimation)  

A commonly used approach for modelling cross-sectional time series data, such as Cross-Sectional Realized Volatility measures, which implies the presence of a large number of regressors, often much larger than the time series observations, is to use Lasso-type estimators. In other words, when p>>n, then the covariates are linearly dependent and X is not a full rank, and therefore the use of Lasso regression (or variable selection methods) is necessary in order to avoid ill-posed estimation problems.

## Example 2

Consider a Lasso estimation for the cross-sectional Realized Volatility measures of S&P500 with a dependent variable the Realized Volatility measures of a particular firm. Then, the cross-section predictive regression model for an h-period forecast horizon is given by 

$$y_{i,t+h} = \beta_{i0} + \sum_{j=1}^N  \beta_{ij}^{\top} X_{j,t}  + \epsilon_{i, t+h}, \ \ \ \text{for} \ \ t = 1,...,n.$$ 

When the number of cross-sectional units N is large we assume a sparse structure for the parameter vector and for estimation purposes we consider a penalised estimation approach. Statistical inference under the sparcity scenario when the dimension is larger than the sample size is now an active and challenging field. Although, we focus on the multivariate time series perspective here, which implies the presence of a large cross-section of regressors, we can observe that the dimensionality of the problem can grow fast, and thus a traditional multivariate time series model is replaced with a Lasso-type regularization to handle these challenges.  

### Task 4 

Write your own code in [R](https://www.r-project.org/), [Matlab](https://uk.mathworks.com/help/matlab/getting-started-with-matlab.html) or [Stata](https://www.stata.com/bookstore/getting-started-windows/) to estimate the one-period ahead forecasts for each of the firm of the cross-section with covariates being the set of Realized Volatility measures of the other firms (that is, at lag 1 day, lag 1 week, and lag 1 month - in trading days), except of the ones for the i-th firm, using a Lasso shrinkage norm and an appropriate penalty function. Some indicative R code is given below. 

```R

# Example 2.1: Consider the HAR model which is suitable for modelling Realized Volatility measures

install.packages("HARModel")
library(HARModel)

data("SP500RM")
attach( data("SP500RM") )
SP500rv = SP500RM$RV

# Estimate the HAR model:
FitHAR   <- HARestimate(vRealizedMeasure = SP500rv, vLags = c(1,5,22))
SP500rv  <- SP500RM$RV
SP500bpv <- SP500RM$BPV
vJumpComponent <- SP500rv - SP500bpv
vJumpComponent <- ifelse(vJumpComponent>=0, vJumpComponent, 0)

# Estimate the HAR-J model:
FitHARJ <- HARestimate(vRealizedMeasure = SP500rv, vJumpComponent = vJumpComponent, vLags = c(1,5,22), vJumpLags = c(1,5,22), type = "HARJ")

```

In practise, the main difficulty of the data structure for commonly used Realized Volatility measures is that due to the high frequency of observations, the time indicator is given to hours, minitues and seconds which requires to convert the dataset into a different format.    

```R

# Input the dataset with the RVs of all firms

mydata  <- read.csv( "RV_ALL_FIRMS.csv", header = TRUE )
Z1      <- as.matrix( mydata$V1 )
time1   <- as.matrix( mydata$Date )
data1    <- cbind( time1, Z1 )
data1    <- as.data.frame( data1 )
new.data <- type.convert( data1 )

date <- as.Date( as.character(new.data$V1), "%d/%m/%Y")
RVs  <- as.numeric(new.data$V2)

firm1   <- xts(RVs, date)
Z2      <- as.matrix( mydata$V2 )
time1   <- as.matrix( mydata$Date )

data2    <- cbind( time1, Z2 )
data2    <- as.data.frame( data2 )
new.data <- type.convert( data2 )
date     <- as.Date( as.character(new.data$V1), "%d/%m/%Y")
RVs      <- as.numeric(new.data$V2)
firm2    <- xts(RVs, date)

# Then the dataset can be created as below

mydata <- read.csv( "RV_ALL_FIRMS.csv", header = TRUE )
time   <- as.matrix( mydata$Date )

mydata_series <- read.csv( "RV_ALL_FIRMS_new.csv", header = TRUE )
dataset       <- as.matrix( mydata_series )

mylist <- list()
for (i in 1:23 )
{
  Z <- as.numeric( dataset[ ,i] )
  data    <- cbind( time, Z )
  data    <- as.data.frame( data )
  
  new.data <- type.convert( data )
  date <- as.Date( as.character(new.data$V1), "%d/%m/%Y")
  
  RVs  <- as.numeric(new.data$Z)
  firm <- xts(RVs, date)
  
  mylist[[i]] <- firm
  firm <- 0
}  
  
```

Next, we consider the Lasso estimation based on the cross-section of realized volatility measures. 

```R

# Example: Lasso estimation 

cv <- cv.glmnet(x,y,alpha=1,nfolds=10)
l  <- cv$lambda.min
alpha <- 1

fits <- glmnet( x, y, family="gaussian", alpha=alpha, nlambda=100)
res  <- predict(fits, s=l, type="coefficients")
res  <- as.matrix( res )

```

### References

- Katsouris, C. (2021). Forecast Evaluation in Large Cross-Sections of Realized Volatility. arXiv preprint [arXiv:2112.04887](https://arxiv.org/abs/2112.04887).
- Gupta, S. (2012). A note on the asymptotic distribution of LASSO estimator for correlated data. Sankhya A, 74(1), 10-28.
- Masini, R. P., Medeiros, M. C., & Mendes, E. F. (2019). Regularized estimation of high‐dimensional vector autoregressions with weakly dependent innovations. Journal of Time Series Analysis.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B (Methodological), 58(1), 267-288.

## [C2]. Vector Autoregression Processes

Lastly, we return back to Vector Autoregression Processes, although not necessarily a high-dimensional VAR(p) process. In practise, we consider the case where the dimension of the time-series vector is less than the sample size n. 

Formally, a p-dimensional vector-valued stationary time series 
$$ X_t \equiv \left(  X_{1t},..., X_{pt} \right), \ \ \ \ \ \text{for} \ \ \ t=1,...,n$$ 

can be modelled using a VAR representation of lag $d$ with serially uncorrelated Gaussian errors, which takes the following form

$$X_t = A_1 X_{t-1} + ... + A_d X_{t-d} + e_t, \ \ \ e_t \sim \mathcal{N} ( \boldsymbol{0}, \boldsymbol{\Sigma}_e )$$

where A1,..., Ad are (p x p) matrices and et is a p-dimensional vector of possibly correlated innovation shocks. Therefore, the main objective in VAR models is to estimate the transition matrices A1,..., Ad, together with the order of the model $d$, based on time series realizations ( X0, X1,..., Xn ). Furthermore, the structure of the transition matrices provides insights into the complex temporal relationships amongst the p time series and the particular representation provides a way to apply forecasting techniques.   

