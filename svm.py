from sklearn import svm
#from sklearn.metrics import accuracy score
X=[[0,0],[1,1]]
Y=[0,1]
classifier=svm.SVC()
print(classifier.fit(X,Y))
print(classifier.predict([[2.,2.]]))


