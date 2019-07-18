import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import requests
import json
from sklearn import preprocessing

dataset =pd.read_csv('indian_liver_patient_dataset.csv')
dataset = pd.DataFrame(dataset)
dataset['gender'] = pd.Categorical(dataset['gender'])
dfDummies = pd.get_dummies(dataset['gender'],prefix='category')
print(dataset.describe())

df = pd.concat([dataset, dfDummies], axis=1)

df.drop(["gender"], axis = 1, inplace = True)

#print("before preprocess",df['class'])
#x = df.values #returns a numpy array
#min_max_scaler = preprocessing.MinMaxScaler()
#df=pd.DataFrame(min_max_scaler.fit_transform(x), columns=df.columns, index=df.index)
print("newdf",list(df.columns.values) )
from sklearn.model_selection import train_test_split
X=df[['age', 'TB', 'DB', 'alkphos', 'sgpt', 'sgot', 'TP', 'ALB', 'A_G']]
y=df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print("predict class",y)
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=1000)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=6, min_weight_fraction_leaf=0.0,
            n_estimators=1000, n_jobs=2, oob_score=False, random_state=0,
            verbose=0, warm_start=False)
y_pred=clf.predict(X_test)
print("Y-pred",y_pred)
print('goooooooooooo',X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

feature_imp = pd.Series(clf.feature_importances_,index=X.columns.values).sort_values(ascending=False)
print(feature_imp)

#important fetures
import matplotlib.pyplot as plt
import seaborn as sns

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

# Saving model to disk
pickle.dump(clf, open('liver_model_1.pkl','wb'))

# Loading model to compare the results
#model = pickle.load( open('livermodel.pkl','rb'))
