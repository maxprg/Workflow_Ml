import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Charger les données d'entraînement
data = pd.read_csv('training_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Entraîner le modèle linéaire
model = LinearRegression()
model.fit(X, y)

# Sauvegarder le modèle
joblib.dump(model, 'linear_model.pkl')

# Sauvegarder les coefficients du modèle
with open('linear_model.txt', 'w') as f:
    f.write(f'Coefficients: {model.coef_}\nIntercept: {model.intercept_}\n')
