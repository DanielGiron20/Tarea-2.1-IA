#TAREA 2.1 MODELO DE PREDICCION DE PESO
#NOTA : AUNQUE LA TAREA DICE ALTURA Y PESO, INVESTIGANDO VI QUE EL IMC(indice de masa corporal) ES DISTINTO PARA HOMBRE Y MUJERES POR LO QUE DECIDI ANADIRLE MAS 
#PRECISION AL EJERCICIO TOMANDO EN CUENTA TAMBIEN EL SEXO, EL ARCHIVO CSV RECOPILA LA ALTURA, EL PESO Y EL SEXO DE 20 PERSONAS QUE CONOZCO EN PERSONA
#PARA MEDIR LA PRECISION UTILIZARE A MI HERMANA QUIEN ESTA EN SU PESO IDEAL SEGUN SU ESTATURA YA QUE MIDE 1.63 Y PESA 55KG

import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dte = pd.read_csv('altura_peso.csv')
dt = dte.drop(columns=['edad'])
datos = pd.get_dummies(dt, columns=['sexo'], drop_first=True)
plt.figure(figsize=(10,6))
sb.scatterplot(x="altura", y="peso", data=dt, hue="sexo", palette="coolwarm")
plt.title("Relación entre Altura y Peso")
plt.xlabel("Altura (m)")
plt.ylabel("Peso (kg)")
plt.show()
print(datos)

X = datos[['altura', 'sexo_M']]  
y = datos['peso']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)
altura_nueva = 1.63 #La altra de mi hermana  
sexo_nuevo = 0 #femenina
nueva_entrada = [[altura_nueva, sexo_nuevo]]
peso_estimado = modelo.predict(nueva_entrada)

print(f"El peso estimado para una altura de {altura_nueva} metros y sexo {'masculino' if sexo_nuevo == 1 else 'femenino'} es {peso_estimado[0]} kg.")
print(modelo.intercept_, modelo.coef_[0], modelo.coef_[1])
#EL MODELO PREDIJO CASI EXACTAMENTE EL PESO DE MI HERMANA EL CUAL ES 54.5KG EL MODELO ARROJO 55.5KG 
#Nota: algo que puedo notar es que el peso estimado en caso de otros familiares que tienen sobrpeso el valor del peso es menor
#ya que si revisa el archivo vera que puse el peso de personas que estan en su peso ideal por lo que sesga el modelo a predecir siempre
#valores de una persona delgada podria decirse

#Conclusion: El modelo tomando en cuenta el sexo fue mas exacto que el modelo basico que solo toma altura

