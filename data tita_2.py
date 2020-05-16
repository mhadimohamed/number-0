##load libraries
from matplotlib import pyplot
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#load my dataset
titanic=sns.load_dataset("titanic")
Y=titanic['survived']
X=titanic.drop("survived",axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1000)

#data transform
from sklearn.compose import make_column_transformer
transformer= make_column_transformer((StandardScaler(),["age","fare"]))
transformer.fit_transform(X)
num_features=["pclass","fare","age"]
num_pipeline=make_pipeline(SimpleImputer(),StandardScaler())
catg_features=["sex","deck","alone"]
catg_pipeline=make_pipeline(SimpleImputer(strategy="most_frequent"),OneHotEncoder())
pre_pross=make_column_transformer((num_pipeline,num_features),(catg_pipeline,catg_features))

# Spot Check Algorithms
models = []
models.append(('LR',make_pipeline(pre_pross,LogisticRegression(solver='liblinear', multi_class='ovr'))))
models.append(('LDA',make_pipeline(pre_pross,LinearDiscriminantAnalysis())))
models.append(('KNN',make_pipeline(pre_pross,KNeighborsClassifier())))
models.append(('DTC',make_pipeline(pre_pross,DecisionTreeClassifier())))
models.append(('NB',make_pipeline(pre_pross,GaussianNB())))
models.append(('SVM',make_pipeline(pre_pross,SVC(gamma='auto'))))
for model in models:
    model.fit(X_train,Y_train)
    predictions = model.predict(X_test)
    
#evaluate the best model
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1000, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
for model in models:
   print(accuracy_score(Y_test, predictions))
   print(confusion_matrix(Y_test, predictions))
   print(classification_report(Y_test, predictions))

"""We now have 6 models and accuracy estimations for each. We need to
compare the models to each other and select the most accurate.
Running the example above, we get the following raw results:"""
#Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()













