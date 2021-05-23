Predictive Modeling - Recipe Sodium
================
Bill Peterson
12/21/2018

-   [What machine learning model best predicts sodium in a recipe?](#what-machine-learning-model-best-predicts-sodium-in-a-recipe)
-   [The Data - Epicurious - Recipes with Rating and Nutrition](#the-data---epicurious---recipes-with-rating-and-nutrition)
-   [Descriptive Statistics](#descriptive-statistics)
-   [Correlation Matrix](#correlation-matrix)
    -   [OLS Linear Regression - centered, scaled, and with interaction effects](#ols-linear-regession---centered-scaled-and-with-interaction-effects)
    -   [OLS Linear Regression - centered, scaled, PCA transformation, and with interaction effects](#ols-linear-regession---centered-scaled-pca-transformation-and-with-interaction-effects)
    -   [Partial Least Squares (PLS) Regression](#partial-least-squares-pls-regression)
    -   [Multivariate Adaptive Regression Splines (MARS)](#multivariate-adaptive-regression-splines-mars)
    -   [Generalized Linear Model (GLM) via glmnet](#generalized-linear-model-glm-via-glmnet)
    -   [Random Forest](#random-forest)
    -   [XGBoost](#xgboost)
-   [Conclusion](#conclusion)

# What machine learning model best predicts sodium in a recipe?

I fit additive and tree based regression models using k-fold cross-validation, centering, scaling, and hyperparameter tuning via grid search. I compare model performance via out of sample RMSE.

I use ***Ordinary Least Squares (OLS)***, ***Partial Least Squares (PLS)***, ***Multivariate Adaptive Regression Splines (MARS)***, and ***Generalized Linear Model (GLM)*** regression models to see the effect of using minimally preprocessed data (centering, scaling), 10 fold cross validation, built in feature selection, and easily interpretable variables of importance. I also fit for comparison the same ***OLS with data also preprocessed by PCA***.

I then fit ***Random Forest*** and ***XGBoost*** models to see if nonparametric tree based models were better predicting (produced a lower RMSE) than the additive models.

# The Data - Epicurious - Recipes with Rating and Nutrition

The recipe data "Epicurious - Recipes with Rating and Nutrition" is available as CSV and JSON files from Kaggle (<https://www.kaggle.com/hugodarwood/epirecipes>). I choose the JSON file to extract additional data for each recipe: the number of directions, the recipe description (an optional field), the number of ingredients listed in the recipe, and the number of categories the recipe was associated with.

There are 20130 rows with 11 variables in the raw data file, 1834 of which were duplicates. Of the 18,296 nonduplicative rows, 3,827 rows had missing values for sodium, fat, calories or protien and were removed. A total of 14,469 units remain for the analysis.

``` r
library(jsonlite)
library(purrr)
library(tidyr)
library(dplyr)


epi_json <- fromJSON("~/Downloads/full_format_recipes.json", flatten = TRUE) 
epi_json <- distinct(epi_json)

num_directions <- as_vector(map(epi_json$directions, length))
num_ingredients <- as_vector(map(epi_json$ingredients, length))
num_categories <- as_vector(map(epi_json$categories, length))
desc_length <- as_vector(map(epi_json$desc, nchar))
desc_length <- desc_length %>% replace_na(0)
title <- as.vector(epi_json$title)
title_length <- as_vector(map(epi_json$title, nchar))
fat <- as.vector(epi_json$fat)
calories <- as.vector(epi_json$calories)
protein <- as.vector(epi_json$protein)
sodium <- as.vector(epi_json$sodium)
rating <- as.vector(epi_json$rating)

recipes <- data.frame(title_length, desc_length, num_directions, num_ingredients, num_categories, fat, calories, protein, sodium, rating, title, stringsAsFactors = FALSE)

# remove individual column vectors from R environment
rm(num_categories)
rm(num_directions)
rm(num_ingredients)
rm(desc_length)
rm(fat)
rm(protein)
rm(calories)  
rm(rating)
rm(sodium)
rm(title)
rm(title_length)


# remove recipes that do not have nutritional information: fat, calories, protein, sodium
recipes_clean <- recipes[rowSums(is.na(recipes)) == 0,]
```

While the CSV file had over 600 dummy variables for ingredients and categories, it was not easily matchable to the JSON file. I used regular expressions to extract terms from the recipe titles to create nine categorical clusters that I am interested in: meat, seafood, herbs, spices, cheese, desserts, nuts, alcoholic, and chocolate. Another 99 records were also removed because of the large outlier nutrition data values which exceeded fat &lt; 500, calories &lt; 10000, sodium &lt; 15000, or protein &lt; 400. 14,370 observations remain. The calories variable was removed for being a highly correlated variable (.9) in the hope of reducing the error.

``` r
library(stringr)
library(caret)
meat_vec <- c("chicken", "poultry", "turkey", "fowl", "quail", "beef", "bacon",
           "veal", "pork", "steak", "hamburger", "cheeseburger", "hot dog",
           "hotdog", "sausage", "loin", "cutlet", "filet mignon", "sirloin",
           "rib", "wing", "thigh", "leg", "drumstick", "snails", "escargot",
           "brisket", "buffalo", "duck", "lamb", "ham", "meatball", "meatloaf",
           "carne", "prosciutto", "pancetta", "rabbit", "goose", "meat", 
           "venison", "game")

seafood_vec <- c("salmon", "swordfish", "oyster", "mussel", "clam", "anchovy",
             "bass", "cod", "crab", "fish", "mahi", "halibut", "lobster",
             "crayfish", "shrimp", "octopus", "pescatarian", "sardine", 
             "scallop", "seafood", "shellfish", "tilapia", "trout", "tuna",
             "snapper", "squid")

herbs_vec <- c("parsley", "sage", "rosemary", "thyme", "dill", "basil", "cilantro",
           "fennel", "herb", "oregano", "tarragon", "chive")

spices_vec <- c("caraway", "cumin", "cardamom", "allspice", "mace", "clove", 
            "chile", "cayenne", "paprika", "cinnamon", "coriander", "curry",
            "nutmeg", "poppy", "saffron", "sesame")

cheese_vec <- c("feta", "colby", "cheddar", "monterey jack", "ricotta",
            "mozzarella", "halumi", "gouda", "gruyere", "brie", "camabert",
            "fontina", "parmesan" , "cheese", "marscarpone")

nuts_vec <- c("hazelnut","macadamia", "peanut", "cashew", "peanut", "pecan", 
          "walnut", "brazil nut", "tree nut", "pistacio", "pine nut", "nut")

desserts_vec <- c("ice cream", "gelato", "cookie", "cake", "crumble", "cobbler",
              "pie", "tart", "galette", "sorbet", "souffle", "meringue", 
              "macaron", "macaroon", "phyllo", "pastry", "sweets", "brownie",
              "blondie", "candy", "custard", "pudding", "milkshake", "sundae",
              "dessert", "marshmallow", "smore", "cannoli" )

alcoholic_vec <- c("campari", "gin", "vodka", "bourbon", "scotch" , "vermouth",
               "martini", "wine", "whiskey", "triple sec", "tequila", "mezcal",
               "spirit", "sherry", "sangria", "sake", "rum", "pernod", 
               "midori", "frangelico", "liqueur", "kirsch", "kahlua", 
               "cocktail", "grand marnier", "grappa", "fortified", 
               "creme de cacao", "cognac", "armagnac", "chambord", 
               "chartreuse", "champagne", "prosecco", "cava", "brandy", "beer",
               "aperitif", "amaretto", "digestif", "margarita")

chocolate_vec <- c("chocolate", "cocoa")

meat_DV <- str_detect(str_to_lower(recipes_clean$title), 
                       paste(meat_vec, collapse = "|"))
seafood_DV <- str_detect(str_to_lower(recipes_clean$title), 
                          paste(seafood_vec, collapse = "|"))
herbs_DV <- str_detect(str_to_lower(recipes_clean$title), 
                        paste(herbs_vec, collapse = "|"))
spices_DV <- str_detect(str_to_lower(recipes_clean$title), 
                         paste(spices_vec, collapse = "|"))
cheese_DV <- str_detect(str_to_lower(recipes_clean$title), 
                         paste(cheese_vec, collapse = "|"))
nuts_DV <- str_detect(str_to_lower(recipes_clean$title), 
                       paste(nuts_vec, collapse = "|"))
desserts_DV <- str_detect(str_to_lower(recipes_clean$title), 
                           paste(desserts_vec, collapse = "|"))
alcoholic_DV <- str_detect(str_to_lower(recipes_clean$title), 
                            paste(alcoholic_vec, collapse = "|"))
chocolate_DV <- str_detect(str_to_lower(recipes_clean$title), 
                            paste(chocolate_vec, collapse = "|"))

recipes_clean <- recipes_clean %>% 
                                mutate(meat_DV = as.factor(meat_DV),
                                       seafood_DV = as.factor(seafood_DV),
                                       herbs_DV = as.factor(herbs_DV),
                                       spices_DV = as.factor(spices_DV),
                                       cheese_DV = as.factor(cheese_DV),
                                       nuts_DV = as.factor(nuts_DV),
                                       desserts_DV = as.factor(desserts_DV),
                                       alcoholic_DV = as.factor(alcoholic_DV),
                                       chocolate_DV = as.factor(chocolate_DV))

                                         
recipes_clean <- recipes_clean %>% dplyr::filter(fat < 500, 
                                       calories < 10000, 
                                       sodium < 15000, 
                                       protein < 400)


## remove Highly Correlated Variables and Near Zero Predictors
highCorr <- findCorrelation(cor(recipes_clean[,1:10]), cutoff = .8)
recipes_clean <- recipes_clean[, - highCorr]

NZV_predictor <- nearZeroVar(recipes_clean) #removed chocolate recipes
recipes_clean <- recipes_clean[, -NZV_predictor]

# remove the column with the recipe title 
recipes_clean <- recipes_clean[, -10]

# remove individual column vectors from R environment
rm(alcoholic_DV)
rm(alcoholic_vec)
rm(cheese_DV)
rm(cheese_vec)
rm(chocolate_DV)
rm(chocolate_vec)
rm(herbs_DV)
rm(herbs_vec)
rm(desserts_DV)
rm(desserts_vec)
rm(meat_DV)
rm(meat_vec)
rm(nuts_DV)
rm(nuts_vec)
rm(seafood_DV)
rm(seafood_vec)
rm(spices_DV)
rm(spices_vec)
rm(recipes)
rm(epi_json)
```

# Descriptive Statistics

``` r
stargazer::stargazer(recipes_clean[,1:9], type = "text")
```

    ## 
    ## ======================================================================
    ## Statistic         N     Mean   St. Dev.  Min  Pctl(25) Pctl(75)  Max  
    ## ----------------------------------------------------------------------
    ## title_length    14,370 34.634   15.040    4      22       45     113  
    ## desc_length     14,370 112.175 140.762    0      0       166    1,738 
    ## num_directions  14,370  3.547   2.338     0      2        5       24  
    ## num_ingredients 14,370 10.250   4.626     1      7        13      51  
    ## num_categories  14,370 12.237   4.942     0      8        16      33  
    ## fat             14,370 27.526   36.753    0      8        34     495  
    ## protein         14,370 20.581   30.534    0      3        28     394  
    ## sodium          14,370 582.164 956.445    0      89      722    14,276
    ## rating          14,370  3.762   1.275   0.000  3.750    4.375   5.000 
    ## ----------------------------------------------------------------------

``` r
psych::describe(recipes_clean[,1:9])
```

    ##                 vars     n   mean     sd median trimmed    mad min   max range
    ## title_length       1 14370  34.63  15.04  33.00   33.78  16.31   4   113   109
    ## desc_length        2 14370 112.17 140.76  73.00   85.78 108.23   0  1738  1738
    ## num_directions     3 14370   3.55   2.34   3.00    3.23   1.48   0    24    24
    ## num_ingredients    4 14370  10.25   4.63  10.00    9.86   4.45   1    51    50
    ## num_categories     5 14370  12.24   4.94  11.00   12.01   5.93   0    33    33
    ## fat                6 14370  27.53  36.75  18.00   20.97  17.79   0   495   495
    ## protein            7 14370  20.58  30.53   9.00   14.45  10.38   0   394   394
    ## sodium             8 14370 582.16 956.44 302.00  402.71 379.55   0 14276 14276
    ## rating             9 14370   3.76   1.28   4.38    4.03   0.93   0     5     5
    ##                  skew kurtosis   se
    ## title_length     0.53    -0.04 0.13
    ## desc_length      2.17     7.58 1.17
    ## num_directions   1.50     3.92 0.02
    ## num_ingredients  0.97     1.78 0.04
    ## num_categories   0.41    -0.42 0.04
    ## fat              4.85    37.57 0.31
    ## protein          4.07    27.71 0.25
    ## sodium           5.97    56.49 7.98
    ## rating          -2.01     3.39 0.01

# Correlation Matrix

``` r
ggcorrplot::ggcorrplot(cor(sapply(recipes_clean[,1:9], as.numeric)), 
           hc.order = TRUE,
           type = "upper",
           lab = TRUE, 
           digits = 2
) 
```

![](Predictive_Modeling---Recipe_Sodium2_files/figure-markdown_github/unnamed-chunk-4-1.png)

The data was split into training and testing sets and will be modeled using 10-fold cross-validation unless otherwise specified.

``` r
set.seed(12345)

in_train_sodium <- createDataPartition(y = recipes_clean$sodium, p = 3 / 4, list = FALSE)
training_sodium <- recipes_clean[in_train_sodium, ]
testing_sodium <- recipes_clean[-in_train_sodium, ]

ctrl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
```

## OLS Linear Regression - centered, scaled, and with interaction effects

``` r
OLS_sodium_cs <- train(sodium ~ (.) ^ 2, 
             data = training_sodium, 
             method = "lm",
             trControl = ctrl, 
             preProcess = c("center", "scale" ))
```

    ## + Fold01: intercept=TRUE 
    ## - Fold01: intercept=TRUE 
    ## + Fold02: intercept=TRUE 
    ## - Fold02: intercept=TRUE 
    ## + Fold03: intercept=TRUE 
    ## - Fold03: intercept=TRUE 
    ## + Fold04: intercept=TRUE 
    ## - Fold04: intercept=TRUE 
    ## + Fold05: intercept=TRUE 
    ## - Fold05: intercept=TRUE 
    ## + Fold06: intercept=TRUE 
    ## - Fold06: intercept=TRUE 
    ## + Fold07: intercept=TRUE 
    ## - Fold07: intercept=TRUE 
    ## + Fold08: intercept=TRUE 
    ## - Fold08: intercept=TRUE 
    ## + Fold09: intercept=TRUE 
    ## - Fold09: intercept=TRUE 
    ## + Fold10: intercept=TRUE 
    ## - Fold10: intercept=TRUE 
    ## Aggregating results
    ## Fitting final model on full training set

``` r
y_hat_sodium_OLS_cs <- predict(OLS_sodium_cs, newdata = testing_sodium)

defaultSummary(data.frame(obs = testing_sodium$sodium, pred = y_hat_sodium_OLS_cs))
```

    ##        RMSE    Rsquared         MAE 
    ## 746.3521867   0.3508302 377.0229501

``` r
varImp(OLS_sodium_cs)
```

    ## lm variable importance
    ## 
    ##   only 20 most important variables shown (out of 136)
    ## 
    ##                                   Overall
    ## `protein:meat_DVTRUE`              100.00
    ## protein                             84.21
    ## `protein:herbs_DVTRUE`              82.51
    ## `protein:seafood_DVTRUE`            77.86
    ## `fat:herbs_DVTRUE`                  65.26
    ## fat                                 62.61
    ## `fat:cheese_DVTRUE`                 55.92
    ## `num_directions:protein`            55.85
    ## `num_ingredients:seafood_DVTRUE`    49.49
    ## `num_ingredients:fat`               44.99
    ## `fat:meat_DVTRUE`                   43.72
    ## `rating:seafood_DVTRUE`             43.39
    ## `num_directions:fat`                42.96
    ## `fat:seafood_DVTRUE`                42.08
    ## `title_length:fat`                  38.17
    ## `num_categories:fat`                32.70
    ## `desc_length:protein`               31.81
    ## `fat:nuts_DVTRUE`                   30.23
    ## `num_ingredients:desserts_DVTRUE`   29.81
    ## `title_length:desc_length`          29.23

OLS chooses the coefficient values in order to minimize the sum-of-squared residuals (SSR) or sum-of-squared errors (SSE). Residuals are difference between the predicted value and actual observed value. Under strong assumptions, OLS is the Best Linear Unbiased Estimator (BLUE), meaning that among all linear estimators that are unbiased, OLS has the smallest variance in the coefficient estimates from one sample of size *N* to the next. Using an unbiased estimator for the coefficients presumes the true coefficients in the population are of interest. If we do not care about the distribution of the coefficient estimates from one random sample of size *N* to the next, then it is quite possible to obtain a non-linear or biased estimator that yields better predictions in the testing data.

A simple OLS regression model, with data that was centered and scaled, yielded a root mean R squared (RMSE) of 746.3521867.

Interestingly ***fat*** and ***protein*** appear in individual and interaction terms for 16 of the top 20 variables of importance for the OLS model. As we will later see, the ***number of ingredients*** is also a recurring variable in later variables of importance. These variables are flagged as most important to the OLS model due to the size of the absolute value of the t-value for the variable. The t-value and the associated p-value measure the accuracy of the coefficient estimates. The larger t-statistic, the more precise the coefficient estimate is.

## OLS Linear Regression - centered, scaled, PCA transformation, and with interaction effects

``` r
OLS_sodium_pca <- train(sodium ~ (.) ^ 2, 
             data = training_sodium, 
             method = "lm",
             trControl = ctrl, 
             preProcess = c("center", "scale", "pca"))
```

    ## + Fold01: intercept=TRUE 
    ## - Fold01: intercept=TRUE 
    ## + Fold02: intercept=TRUE 
    ## - Fold02: intercept=TRUE 
    ## + Fold03: intercept=TRUE 
    ## - Fold03: intercept=TRUE 
    ## + Fold04: intercept=TRUE 
    ## - Fold04: intercept=TRUE 
    ## + Fold05: intercept=TRUE 
    ## - Fold05: intercept=TRUE 
    ## + Fold06: intercept=TRUE 
    ## - Fold06: intercept=TRUE 
    ## + Fold07: intercept=TRUE 
    ## - Fold07: intercept=TRUE 
    ## + Fold08: intercept=TRUE 
    ## - Fold08: intercept=TRUE 
    ## + Fold09: intercept=TRUE 
    ## - Fold09: intercept=TRUE 
    ## + Fold10: intercept=TRUE 
    ## - Fold10: intercept=TRUE 
    ## Aggregating results
    ## Fitting final model on full training set

``` r
y_hat_sodium_OLSpca <- predict(OLS_sodium_pca, newdata = testing_sodium)

defaultSummary(data.frame(obs = testing_sodium$sodium, pred = y_hat_sodium_OLSpca))
```

    ##        RMSE    Rsquared         MAE 
    ## 728.9543839   0.3782961 384.0642705

The previous OLS model with a Principal Components Analysis (PCA) transformation of the predictors yielded a RMSE of 728.9543839. PCA constructs latent variables or components that maximize the variance of the predictor space. The first principal component is orthogonal to the second principal component, and so on, however this transformation comes at the expensive of coefficient interpretability. The OLS model with PCA improved upon the predictive accuracy of the previous OLS model and reduced the RMSE by over 17.

## Partial Least Squares (PLS) Regression

Partial Least Squares is similar to Principal Components Regression but does a matrix decomposition including the outcome as well, making it a more supervised learning approach. PLS is generally suited to a wide data, small sample problems with many possibly correlated covariates or predictor variables and few units of observation. The PLS method was included to explore as the number of predictors is quite large as each variable is interacted on each other. Unlike PCA, PLS coefficients retain interpretability.

``` r
pls_grid <- data.frame(.ncomp = 1:100)
PLS_sodium_bl <- train(sodium ~ (.) ^ 2, 
             data = training_sodium, 
             method = "pls",
             trControl = ctrl, 
             tuneGrid = pls_grid, 
             preProcess = c("center", "scale"))
```

    ## + Fold01: ncomp=100 
    ## - Fold01: ncomp=100 
    ## + Fold02: ncomp=100 
    ## - Fold02: ncomp=100 
    ## + Fold03: ncomp=100 
    ## - Fold03: ncomp=100 
    ## + Fold04: ncomp=100 
    ## - Fold04: ncomp=100 
    ## + Fold05: ncomp=100 
    ## - Fold05: ncomp=100 
    ## + Fold06: ncomp=100 
    ## - Fold06: ncomp=100 
    ## + Fold07: ncomp=100 
    ## - Fold07: ncomp=100 
    ## + Fold08: ncomp=100 
    ## - Fold08: ncomp=100 
    ## + Fold09: ncomp=100 
    ## - Fold09: ncomp=100 
    ## + Fold10: ncomp=100 
    ## - Fold10: ncomp=100 
    ## Aggregating results
    ## Selecting tuning parameters
    ## Fitting ncomp = 21 on full training set

``` r
y_hat_sodiumpls <- predict(PLS_sodium_bl, newdata = testing_sodium)
defaultSummary(data.frame(obs = testing_sodium$sodium, pred = y_hat_sodiumpls))
```

    ##        RMSE    Rsquared         MAE 
    ## 745.2232463   0.3527855 377.3724629

``` r
varImp(PLS_sodium_bl)
```

    ## pls variable importance
    ## 
    ##   only 20 most important variables shown (out of 136)
    ## 
    ##                               Overall
    ## protein                        100.00
    ## `num_ingredients:protein`       97.13
    ## `protein:seafood_DVTRUE`        89.84
    ## `num_categories:protein`        86.77
    ## `protein:rating`                86.64
    ## `num_directions:protein`        83.72
    ## `title_length:protein`          75.78
    ## `protein:meat_DVTRUE`           65.17
    ## `fat:meat_DVTRUE`               56.02
    ## fat                             54.54
    ## `fat:protein`                   53.88
    ## `desc_length:protein`           52.26
    ## `title_length:fat`              51.70
    ## `num_ingredients:fat`           51.01
    ## `fat:rating`                    50.65
    ## `num_categories:fat`            46.77
    ## `num_directions:fat`            46.53
    ## num_ingredients                 43.08
    ## `num_ingredients:meat_DVTRUE`   40.96
    ## `num_directions:meat_DVTRUE`    40.14

I anticipated the PLS model to be similar to the OLS model with PCA. The PLS model yielded a RMSE of 745.2232463, which is marginally better than the original OLS model (RMSE: 746.3521867), however still larger than the OLS with PCA (RMSE: 728.9543839).

## Multivariate Adaptive Regression Splines (MARS)

The MARS model pieces multiple linear regression models to capture nonlinearity of polynomial regression by assessing knots or cutpoints much like step functions. The MARS model's hyperparameter tuning grid enables automatic feature selection by selecting the optimal number of interaction effects (degree) and the number of items to retain (nprune). Another beneift is the MARS model does not require preprocessing.

``` r
library(doMC)
```

    ## Loading required package: foreach

    ## 
    ## Attaching package: 'foreach'

    ## The following objects are masked from 'package:purrr':
    ## 
    ##     accumulate, when

    ## Loading required package: iterators

    ## Loading required package: parallel

``` r
registerDoMC(parallel::detectCores())

library(earth)
```

    ## Loading required package: Formula

    ## Loading required package: plotmo

    ## Loading required package: plotrix

    ## Loading required package: TeachingDemos

``` r
marsGrid <- expand.grid(.degree = 1:3, .nprune = 1:10)

MARS_sodium <- train(sodium ~ (.) ^ 2, 
               data = training_sodium, 
               method = "earth",
               trControl = ctrl, 
               tuneGrid = marsGrid)  
```

    ## Warning in nominalTrainWorkflow(x = x, y = y, wts = weights, info = trainInfo, :
    ## There were missing values in resampled performance measures.

    ## Aggregating results
    ## Selecting tuning parameters
    ## Fitting nprune = 7, degree = 1 on full training set

``` r
coef(MARS_sodium$finalModel)
```

    ##                    (Intercept) h(num_ingredients:protein-297) 
    ##                    877.1515821                      0.6526517 
    ##   h(protein:seafood_DVTRUE-75) h(num_categories:protein-2304) 
    ##                     21.7308953                     -0.5687538 
    ##      h(protein:meat_DVTRUE-13)                  h(protein-25) 
    ##                     -9.0351528                     10.9513816 
    ##                  h(25-protein) 
    ##                    -30.0254521

``` r
varImp(MARS_sodium)
```

    ## earth variable importance
    ## 
    ##                         Overall
    ## num_ingredients:protein  100.00
    ## protein                   58.43
    ## protein:seafood_DVTRUE    36.56
    ## protein:meat_DVTRUE       10.78
    ## num_categories:protein     0.00

``` r
defaultSummary(data.frame(obs = testing_sodium$sodium, pred = predict(MARS_sodium, newdata = testing_sodium)[,1]))
```

    ##        RMSE    Rsquared         MAE 
    ## 737.0283201   0.3670836 372.3374835

The best MARS model was obtained from a tuning grid of degree = 1 and nprune = 7, which yielded and RMSE of 737.0283201. The MARS model improved upon the RMSE of the base OLS and PLS models, however still fell short of the RMSE of the OLS with PCA.

## Generalized Linear Model (GLM) via glmnet

When a model overfits the data or there are collinearity issues, the OLS regression estimates may become inflated and increase the variance of the model. Regularization techniques introduce bias in the model to control the parameter estimates (and minimize SSE). By sacrificing some bias via the added penalty/regularization, the model may reduce the variance enough to make the overall mean squared error (MSE) lower than in an unbiased model.

Glmnet fits a generalized linear model via penalized maximum likelihood, using a mix of lasso (alpha = 1) and ridge (alpha = 0) regression, and the tuning parameter lambda which measures the strength of the regularization.

Ridge regression adds a penalty to the SSE that only adds parameters if there is a significant reduction in the SSE. Ridge regression shrinks parameters estimates to zero as the lambda penalty becomes large. Similar but different, lasso regression sets parameters to zero and removes them from the model.

``` r
glmGrid <- expand.grid( alpha = 0:1,
                        lambda = seq(0.0001, 0.1, length = 10))

glmnet_sodium <- train(sodium ~ (.) ^ 2, 
             data = training_sodium, 
             method = "glmnet",
             trControl = ctrl, 
             tuneGrid = glmGrid, 
             preProcess = c("center", "scale"))
```

    ## Aggregating results
    ## Selecting tuning parameters
    ## Fitting alpha = 0, lambda = 0.1 on full training set

``` r
y_hat_sodium_glmnet <- predict(glmnet_sodium, newdata = testing_sodium)
defaultSummary(data.frame(obs = testing_sodium$sodium, pred = y_hat_sodium_glmnet))
```

    ##        RMSE    Rsquared         MAE 
    ## 731.8066370   0.3732348 375.2005661

``` r
plot(glmnet_sodium)
```

![](Predictive_Modeling---Recipe_Sodium2_files/figure-markdown_github/glmnet-1.png)

``` r
varImp(glmnet_sodium)
```

    ## glmnet variable importance
    ## 
    ##   only 20 most important variables shown (out of 136)
    ## 
    ##                                 Overall
    ## protein                          100.00
    ## protein:meat_DVTRUE               75.62
    ## protein:seafood_DVTRUE            75.09
    ## fat                               67.17
    ## num_ingredients:protein           54.72
    ## num_directions:protein            49.47
    ## protein:herbs_DVTRUE              44.54
    ## fat:meat_DVTRUE                   41.02
    ## protein:rating                    37.13
    ## num_categories:protein            35.44
    ## rating:seafood_DVTRUE             34.00
    ## num_ingredients:seafood_DVTRUE    32.90
    ## title_length:fat                  31.00
    ## num_ingredients:fat               29.79
    ## num_ingredients                   28.88
    ## fat:herbs_DVTRUE                  28.39
    ## fat:cheese_DVTRUE                 26.11
    ## num_ingredients:desserts_DVTRUE   25.23
    ## title_length:desc_length          21.00
    ## fat:seafood_DVTRUE                19.90

The final GLM model is a ridge regression (alpha = 0) with lambda of 0.1, with a RMSE of 731.8066370. Grid search determined the best alpha and lambda values for the glmnet model.

## Random Forest

Decision trees algorithmically partition predictor variables into subsets with a focus minimize the variance in the response variable within subsets. Decision trees are non-parametric, do not assume any particular functional form.

The random forest model is an ensemble method of decisions trees using bagging or Bootstrap Aggregating. Bagging randomly samples subsets of the training data. With each random sample a decision tree is fit which seeks to minimize the variance in the response variable (the predicted variable) within subsets. For a random forest, the final predictions of all trees are then aggregated and averaged by subset. Rather than searching for best predictors, the random sampling builds a series of diverse single trees. These single trees are "weak learners" are simple models that do better than random chance. These weak learners when averaged together reduce the overall variance of the model at the expense of equal or greater bias.

``` r
model <- train(sodium ~ (.) ^ 2, 
               tuneLength = 5,
               data = training_sodium, 
               method = "ranger", 
               trControl = trainControl(method = "cv", 
                                        number = 5, 
                                        verboseIter = TRUE))
```

    ## Aggregating results
    ## Selecting tuning parameters
    ## Fitting mtry = 35, splitrule = extratrees, min.node.size = 5 on full training set
    ## Growing trees.. Progress: 98%. Estimated remaining time: 0 seconds.

``` r
model
```

    ## Random Forest 
    ## 
    ## 10779 samples
    ##    16 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 8624, 8623, 8623, 8623, 8623 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  splitrule   RMSE      Rsquared   MAE     
    ##     2   variance    800.7176  0.3267198  396.2571
    ##     2   extratrees  822.6173  0.3172119  423.3084
    ##    35   variance    790.1311  0.3322641  392.8759
    ##    35   extratrees  779.8470  0.3455708  387.2643
    ##    69   variance    798.6387  0.3218709  397.5189
    ##    69   extratrees  781.8079  0.3439654  389.7444
    ##   102   variance    804.7804  0.3154377  399.6534
    ##   102   extratrees  786.3644  0.3383985  392.8630
    ##   136   variance    807.2091  0.3126635  400.9450
    ##   136   extratrees  787.7211  0.3366762  393.7482
    ## 
    ## Tuning parameter 'min.node.size' was held constant at a value of 5
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were mtry = 35, splitrule = extratrees
    ##  and min.node.size = 5.

``` r
y_hat_sodium_RF_bt <- predict(model, newdata = testing_sodium)
defaultSummary(data.frame(obs = testing_sodium$sodium, pred = y_hat_sodium_RF_bt))
```

    ##        RMSE    Rsquared         MAE 
    ## 734.0469071   0.3749043 377.0089254

The best random forest model using 5-fold cross-validation yielded a RMSE of 734.0469071, with 5 as min.node.size or depth of the decision tree. Perhaps a finer tuned random forest, with a smaller min.node.size (closer to the XGBoost max depth parameter) might yield a lower RMSE.

## XGBoost

eXtreme Gradient Boosting (XGBoost) is a tree based ensemble model that boosts weak learners by interatively learning from previous models. Boosting starts by fitting an initial model, and then a subsequent model is built that focuses on accurately predicting the cases where the previous model performs poorly. This process is then repeated where each successive model attempts to correct for the shortcomings of the previous model. The of these two models is expected to be better than either model alone. Then you repeat this process of boosting many times. Each successive model attempts to correct for the shortcomings of the combined boosted ensemble of all previous models.

XGBoost improves upon the boosting model framework by focusing on minimzing the overall prediction error of successive models using gradient descent. Each new model is fit to the new residuals based on the gradient of the error with respect to the prediction.

``` r
nrounds <- 1000

tune_grid <- expand.grid(
  nrounds = seq(from = 200, to = nrounds, by = 50),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

tune_control <- trainControl(
  method = "cv", 
  number = 5, 
  verboseIter = FALSE, 
  allowParallel = FALSE # FALSE for reproducible results 
)

xgb_tune <- caret::train(
  sodium ~ (.) ^ 2, 
  data = training_sodium, 
  trControl = tune_control,
  tuneGrid = tune_grid,
  method = "xgbTree",
  verbose = TRUE
)
```

    ## [07:44:32] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:44:47] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:45:06] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:45:29] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:45:58] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:46:31] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:46:46] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:47:05] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:47:29] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:47:57] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:48:30] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:48:45] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:49:05] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:49:29] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:49:59] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:50:35] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:50:51] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:51:11] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:51:35] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:52:04] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:52:44] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:53:01] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:53:20] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:53:44] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:54:13] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:54:47] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:55:03] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:55:22] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:55:47] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:56:16] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:56:50] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:57:06] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:57:26] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:57:51] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:58:20] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:58:56] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:59:11] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:59:32] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [07:59:58] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:00:29] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:01:06] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:01:21] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:01:42] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:02:06] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:02:36] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:03:14] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:03:30] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:03:50] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:04:15] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:04:45] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:05:19] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:05:35] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:05:56] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:06:21] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:06:51] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:07:25] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:07:41] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:08:02] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:08:27] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:08:57] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:09:33] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:09:49] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:10:10] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:10:35] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:11:05] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:11:41] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:11:57] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:12:18] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:12:49] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:13:19] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:13:55] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:14:12] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:14:32] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:14:58] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:15:30] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:16:06] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:16:23] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:16:44] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:17:10] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:17:41] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:18:18] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:18:35] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:18:56] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:19:22] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:19:53] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:20:29] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:20:45] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:21:07] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:21:33] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:22:04] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:22:46] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:23:04] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:23:25] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:23:52] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:24:23] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:25:00] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:25:16] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:25:38] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:26:04] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:26:36] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.
    ## [08:27:13] WARNING: amalgamation/../src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.

``` r
xgb_tune$bestTune
```

    ##    nrounds max_depth  eta gamma colsample_bytree min_child_weight subsample
    ## 87     250         2 0.05     0                1                1         1

``` r
yhat_xbgbest <- predict(xgb_tune, newdata = testing_sodium)
defaultSummary(data.frame(obs = testing_sodium$sodium, pred = yhat_xbgbest))
```

    ##        RMSE    Rsquared         MAE 
    ## 729.7695761   0.3784768 373.4830721

The XGBoost model with 5-fold cross-validation yielded an RMSE of 729.7695761. This best XGBoost model used a tuning grid to determine the best max depth for the decision tree (2), and the number of decision trees (250) to fit per cross-validation fold.

# Conclusion

The base OLS regression model was the worst predicting model. This is no surprise as the OLS approach focuses on producing the least biased estimates at the expense of variance. Dimensionality reduction techiniques via OLS with PCA transformation of the regressors and the PLS model also improvded the base OLS model with mixed results. The OLS model with PCA transformation produced the best model overall with an RMSE of 728.9543839. This was somewhat surprising given the reputation of GLM and ensemble tree based models to produce accurate and robust models.

The greater flexibility allowed by the MARS model also improved upon the base OLS and PLS model but might have also been a limiting factor. The GLM model with penalization/regularization improved upon the previous predictions of the base OLS, PLS and MARS models by allowing for more biased models with lower variance. The non-parametic based models of Random Forests and XGBoost aggregated many iterations of decisions trees produce two of the more predictive models, with the XGBoost producing the second best predictive model with an RMSE of 729.7695761.

Interestingly, the variables of importance for when applicable indicated that protein and fat and to a lesser extent the number of ingredients, routinely appeared as some of the most influential variables to the models that predicited recipe sodium. This seems intuitive as sodium is a flavor enhancer, fats can help distribute sodium throughout foods and provide a mouth feel, and protein is critical to growth and repair in the human body. Presumably as recipes get more complicated with more ingredients, sodium will be more likely to appear and possibly in greater amounts.

The final models and RMSEs: model: RMSE

OLS (center & scale) - RMSE: 746.3521867

OLS (center, scale, & PCA) - RMSE: 728.9543839

PLS (center & scale) - RMSE: 745.2232463

MARS (center & scale) - RMSE: 737.0283201

glmnet (center & scale) - RMSE: 731.8066370

Random forest - RMSE: 734.0469071

XGBoost - RMSE: 729.7695761
