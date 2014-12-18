print(__doc__)

import csv
import numpy as np
import pylab as pl
import random
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from pylab import *
from sklearn import cross_validation
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

#Base d'entrainement reduite prise j'ai un algo qui genere permutation en O(nln(n)) mais le data est trop grande pour qu'il marche. pour le moment je prends une plus petite base

taille_data_train=2000000

with open("bio_train.csv","rb") as f:
	X=[]
	y=[]
	i=0
	for row in csv.reader(f):
		if i<taille_data_train+1:	
			res=True
			j=0
			label=row[0]
			row_1=row[1:1170]
			while j<=len(row_1)-1:
				if row_1[j]<>'':
					j=j+1
				else:
					res=False
					j=j+1
			if res==True:
				X.append(row_1)
				
				y.append(label)
				i=i+1
			else:
				i=i+1
		else:
			break	

X=X[1:]
y=y[1:]
for i in xrange(0,len(X)):
	y[i]=int(y[i])
	for j in xrange(0,len(X[i])):
			X[i][j]=float(X[i][j])

		

#splitting
X=preprocessing.scale(X)
(X_train, X_test, y_train, y_test) = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)

#Classifieur

C=532797.894587
gamma=1.77827941004e-09
clf=SVC(C=C,gamma=gamma,kernel='rbf')
classifier = clf
y_score = classifier.fit(X_train, y_train).decision_function(X_test)




# ROC
fpr, tpr, thresholds = roc_curve(y_test,y_score)
roc_auc = auc(fpr, tpr)
print "Area under the ROC curve : %f" % roc_auc

# Plot ROC 
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic example')
pl.legend(loc="lower right")
pl.show()