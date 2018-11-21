
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = preprocess()

features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 

clf = svm.SVC(kernel="linear")
t0 = time()
clf.fit(features_train, labels_train)


t1 = time()
prediction = clf.predict(features_test)
print "Prediction time:", round(time()-t1, 3), "s"

print accuracy_score(prediction, labels_test)

