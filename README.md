# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Read the dataset from the CSV file and extract the input feature (R&D Spend) and output variable (Profit). Apply feature scaling to the input data.

 2.Initialize parameters such as weight w, bias b, learning rate α, and number of iterations (epochs).

 3.Perform Gradient Descent:

  Predict output using the equation ŷ = w·x + b

  Compute the Mean Squared Error (loss)

  Calculate gradients dw and db

  Update parameters w and b

 4.Repeat the process for the given number of iterations and display the final regression line and loss curve.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: 
RegisterNumber:  
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/acer/Downloads/50_Startups.csv")
x = data["R&D Spend"].values
y = data["Profit"].values

x_mean = np.mean(x)
x_std = np.std(x)
x = (x - x_mean) / x_std

w = 0.0
b = 0.0
alpha = 0.01
epochs = 100
n = len(x)


losses = []

for _ in range(epochs):
    y_hat = w * x + b
    loss = np.mean((y_hat - y)**2)
    losses.append(loss)


dw = (2/n) * np.sum((y_hat - y) * x)
 db = (2/n) * np.sum(y_hat - y)


 w -= alpha * dw
b -= alpha * db


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

plt.subplot(1, 2, 2)
plt.scatter(x, y, label="Data")
plt.plot(x, w * x + b, label="Regression Line")
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression using Gradient Descent")
plt.legend()

plt.tight_layout()
plt.show()

print("Final Weight (w):", w)
print("Final Bias (b):", b)


```

## Output:
<img width="1039" height="426" alt="Screenshot 2026-01-30 201025" src="https://github.com/user-attachments/assets/18ca0f94-8134-407f-9274-3e982cdcb8d9" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
