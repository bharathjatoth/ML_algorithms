# ML_algorithms
Linear Regression:</br>
  This will be used for predicting continuous Values. This uses both L1 and L2 Regularization's. </br>
Graph for the Linear Regression:</br>
  Step1 : Fit a line for a shop dataset with only one value i.e., predict the price(y) given the quantity(x)</br>
  Step2: Next we see to minimize the error by the squared error method. R2 score in the scikitlearn package. y=m*x+c we see the m value i.e., the coefficent and min it.</br>
  Step 3: we use Gradient Descent to minimize the coefficients of the features (x0*a0+x1*a1+x2*a1) (a1,a2 are the coefficients of the features). min coefficients.</br>
  Step 4: For those of more than one coefficent or more than one feature how to see the dependecy of the coefficient of the one (theta1 = 0.001, theta2 = 0.0004) here theta1 will decide the amount of importance to the model. We need to take modulus of the value</br>
  Step5: Select the right features among the listed out ones'. So we get a polynomial Regression of power 6 or 7 to fit the data points in the graph. </br>
  Step 6: This will increase the Varience(which fits training data correctly but model will not predict the test data correctly) and the bias will be decreased. This will be the case of Overfitting.</br>
  If we lower the number of features then we will get underfit. It will not fit the training data well (A small example to remember this will be: with High bias we will be taking decisions at ultra speed with out taking into other factors as considerations) So here Varience will be very low Varience(Inconsistence). This is called Underfit.</br>
  We need to choose parameters which is not overfit or underfit we need the fit in between.</br>
  Step 7: With this we have two options to decrease the polynomial Regression. Decrease Features(leads to Underfit) and add Regularization Parameter which will decrease the coefficient weights.</br>
  Ridge Regression(L2 Regression): Here the coefficent will add (J+(lambda*(theta**2))) Here J is representing Cost. This will reduce the weights of Coefficients</br>
  Lasso Regression(L1 Regression): Here the Coefficient will be not squared (J+ lambda*(theta)). This will be elemenating other parameters theta0,theta1. </br>
Logistic Regression:
 This is for Classification in binary. Example, What is the probability of a person picking a item, will it rain today or not?. This will be used for predicting outcome in binary (0 or 1).
 
## to do the visualisations in the tensorboard the command needed to be run is
tensorboard --logdir=fullpath(ex:C:\Users\bharathjatoth\need\log) --host=127.0.0.1
