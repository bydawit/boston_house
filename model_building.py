import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

# Loads the Boston House Price Dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Build Random forest model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestRegressor()
model.fit(X, Y)

# Saving the model
import pickle
pickle.dump(model, open('boston_house_clf.pkl', 'wb'))

