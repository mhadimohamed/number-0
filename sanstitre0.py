#load libraries
import numpy as np
from sklearn.pipeline import make_pipeline
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
#load my dataset
titanic=sns.load_dataset("titanic")
#split-out dataset validain
Y=titanic['survived']
X=titanic.drop("survived",axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=10000)
#data transform
from sklearn.compose import make_column_transformer
transformer= make_column_transformer((StandardScaler(),["age","fare"]))
transformer.fit_transform(X)
num_features=["pclass","fare","age"]
num_pipeline=make_pipeline(SimpleImputer(),StandardScaler())
catg_features=["sex","deck","alone"]
catg_pipeline=make_pipeline(SimpleImputer(strategy="most_frequent"),OneHotEncoder())
pre_pross=make_column_transformer((num_pipeline,num_features),(catg_pipeline,catg_features))
#check algorithm
                #MODEL number:1_SGDClassifier
model=make_pipeline(pre_pross,SGDClassifier())
model.fit(X_train,Y_train)
predictions = model.predict(X_test)
kfold = StratifiedKFold(n_splits=5, random_state=1000, shuffle=True)
print("results of:SGD is")
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))                                                               
               #MODEL number:2_KNeighborsClassifier
model=make_pipeline(pre_pross, KNeighborsClassifier())
model.fit(X_train,Y_train)
predictions = model.predict(X_test)
kfold = StratifiedKFold(n_splits=5, random_state=1000, shuffle=True)
print("results of:KNC is")
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
               #MODEL number:3_RandomForestClassifier
model=make_pipeline(pre_pross,RandomForestClassifier())
model.fit(X_train,Y_train)
predictions = model.predict(X_test)
kfold = StratifiedKFold(n_splits=5, random_state=1000, shuffle=True)
print("result of:RFC is")
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
               #MODEL number:4_DecisionTreeClassifier
model=make_pipeline(pre_pross,DecisionTreeClassifier())
model.fit(X_train,Y_train)
predictions = model.predict(X_test)
kfold = StratifiedKFold(n_splits=5, random_state=1000, shuffle=True)
print("results of:DTC is")
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))                                    
                #MODEL number:5_GaussianNB
model=make_pipeline(pre_pross,GaussianNB())
model.fit(X_train,Y_train)
predictions = model.predict(X_test)
kfold = StratifiedKFold(n_splits=5, random_state=1000, shuffle=True)
print("results of:GNB is")
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
               
               

                

                  
                              
                                 
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  
                                  