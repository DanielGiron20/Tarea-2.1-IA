#El archivo csv contiene 20 registros de personaonas que conozco en persona
#En el ejercicio relacionaremos la altura y el peso de manera basica sin tomar en cuenta el sexo de la persona
import pandas as pd
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dte = pd.read_csv('altura_peso.csv')

print(dte.head())
dt = dte.drop(columns=['sexo', 'edad'])
#Graficar los datos para ver la relación entre parametros
sb.scatterplot(x='altura', y='peso', data=dt)
plt.title('Relación entre altura y peso')
plt.xlabel('Altura (m)')
plt.ylabel('Peso (kg)')
plt.show()

X = dt[['altura']].values  
y = dt['peso'].values    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
modelo = LinearRegression()
modelo.fit(X_train, y_train)


altura = 1.63  #altura de mi hermana
prediccion_peso = modelo.predict([[altura]])
print(f"Para una altura de {altura} m, el peso estimado es {prediccion_peso[0]:.2f} kg")
print(f"Peso (pendiente): {modelo.coef_[0]:.2f}")
print(f"Sesgo (intercepto): {modelo.intercept_:.2f}")
