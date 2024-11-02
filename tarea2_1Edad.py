#TAREA 2.1 MODELO DE PREDICCION DE PESO
#NOTA : Esta es la misma prueba de prediccion de peso pero ahora tomando en cuenta la edad de la persona 
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dte = pd.read_csv('altura_peso.csv')
dt = dte.drop(columns=['sexo'])
print(dt)

plt.figure(figsize=(10,6))
sb.scatterplot(x="altura", y="peso", data=dt, hue="edad", palette="coolwarm")
plt.title("Relación entre Altura y Peso")
plt.xlabel("Altura (m)")
plt.ylabel("Peso (kg)")
plt.show()
print(dt)

X = dt[['altura', 'edad']]  
y = dt['peso']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)
altura_nueva = 1.80 #La altura de mi Padre 
edad_nueva = 56 #la edad de mi Padre
nueva_entrada = [[altura_nueva, edad_nueva]]
peso_estimado = modelo.predict(nueva_entrada)

print(f"El peso estimado para una altura de {altura_nueva} metros y edad {edad_nueva} es {peso_estimado[0]} kg.")

print(modelo.intercept_, modelo.coef_[0], modelo.coef_[1])
#EL MODELO PREDIJO 82.22 KG CON LOS DATOS DE MI PAPA ESA PREDICCION FUE EXACTA PUES MI PAPA PESA 180 LB
#EL MODELO PREDIJO 57.5KG EL PESO DE MI HERAMANA ES 54.5KG ESTUVO BASTANTE CERCA PERO MAS LEJOS QUE EL MODELO INICIAL
#EL MODELO PREDIJO 67.6 KG CON MIS DATOS, MI PESO ES 70KG ESTUVO CERCA PERO MAS LEJOS QUE EL MODELO INICIAL IGUALMENTE

#Conclusion: El modelo tomando en cuenta las edades es ligeramente mas impreciso con personas menores pero en edades mayores
#mostro ser mas preciso pues acerto casi de lleno las edades de mi padre,madre,tia y otro conocido de 51 anios
