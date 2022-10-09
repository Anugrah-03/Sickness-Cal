import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import pickle

data = pd.read_csv ("diabetes.csv")

df = pd.DataFrame (data)

X = pd.DataFrame (data , columns = ["Pregnancies" , "Glucose" , "BloodPressure" , "SkinThickness" , "Insulin" , "BMI"
                                   , "DiabetesPedigreeFunction" , "Age"]) # Features
y = data.Outcome # Target variables
print(y)
# Training = 75 , Testing = 25

X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.25 , random_state = 0)

logreg = LogisticRegression ()

logreg.fit (X_train , y_train)                     # Fitting a  model      # Predicted probabilities from test features

pickle.dump(logreg,open("diabetes.pkl","wb"))
