import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from joblib import dump

#load the dataset

df = pd.read_csv("https://raw.githubusercontent.com/siddiquiamir/ML-MODEL-DEPLOYMENT-USING-FLASK/main/iris.csv")

print(df.head())


# select features and target
X = df[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']]
y = df['Class']
# Split the data into train and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=123)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

# Instantiate the model
classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)

# Make a pickle file for our model
with open('model.pkl', 'wb') as file:
    pickle.dump((classifier, sc), file)