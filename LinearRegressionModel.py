import math
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def regressorValues(_y_test, _pred, _regressor):

  mse = mean_squared_error(_y_test, _pred)
  rmse = math.sqrt(mse)

  mae = round(metrics.mean_absolute_error(_y_test, _pred), 2)
  mse = round(mse, 2)
  evr = round(metrics.explained_variance_score(_y_test, _pred), 2)
  rsquared = round(metrics.r2_score(_y_test, _pred), 2)

  print('RMSE: ', round(rmse, 2), 'R-squared: ', rsquared, 'Intercept:', round(_regressor.intercept_,2))
  for coef in _regressor.coef_:
    print('{0:.5f}'.format(coef))
  # print('Coefficients:', '{0:.10f}'.format(_regressor.coef_))
  # print(_regressor.coef_)

  con_mat = metrics.confusion_matrix(_y_test, _pred.round())
  axes = sns.heatmap(con_mat, square=True, annot=True, fmt='d', cbar=True, cmap=plt.cm.GnBu)

def polynomialLinearRegression(_X, _y):
  # print(_X.shape, _y.shape)
  poly = PolynomialFeatures(degree=2)
  poly_variables = poly.fit_transform(_X)

  X_train, X_test, y_train, y_test = train_test_split(poly_variables, _y, test_size=0.3, random_state=0)
  model = LinearRegression()
  model.fit(X_train, y_train)
  pred = model.predict(X_test)
  regressorValues(y_test, pred, model)

def linearRegressionModel(_X, _y):
  X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=0.3, random_state=0)
  # print(X_train.shape, y_train.shape)
  model = LinearRegression()
  model.fit(X_train, y_train)
  pred = model.predict(X_test)
  regressorValues(y_test, pred, model)