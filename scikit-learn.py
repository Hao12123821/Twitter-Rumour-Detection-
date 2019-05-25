import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time

# Load dataset
dataset = pd.read_csv('dataset.csv')
x = dataset.iloc[: , 1:-1]
y = dataset.iloc[: , -1]

# Split dataset to training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state=1)

# Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 1.Logistic regression
start = time.time()
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(x_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(x_test, y_test)))

pred = logreg.predict(x_test)
logistic_reg_conf_matrix = confusion_matrix(y_test, pred)
print(classification_report(y_test, pred))
end = time.time()
print('run time:{:.3f}s\n\n'.format(end - start))


# 2.Decision tree
start = time.time()
clf = DecisionTreeClassifier().fit(x_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(x_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(x_test, y_test)))

pred = clf.predict(x_test)
decision_tree_conf_matrix = confusion_matrix(y_test, pred)
print(classification_report(y_test, pred))
end = time.time()
print('run time:{:.3f}s\n\n'.format(end - start))

# 3.K nearest neighbor
start = time.time()
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(x_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(x_test, y_test)))

pred = knn.predict(x_test)
k_neighbors_conf_matrix = confusion_matrix(y_test, pred)
print(classification_report(y_test, pred))
end = time.time()
print('run time:{:.3f}s\n\n'.format(end - start))

# 4. Linear Discriminant Analysis
start = time.time()
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(x_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
     .format(lda.score(x_test, y_test)))

pred = lda.predict(x_test)
linear_disc_conf_matrix = confusion_matrix(y_test, pred)
print(classification_report(y_test, pred))
end = time.time()
print('run time:{:.3f}s\n\n'.format(end - start))

# 5.Gaussian naive bayes
start = time.time()
gnb = GaussianNB()
gnb.fit(x_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(x_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
     .format(gnb.score(x_test, y_test)))

pred = gnb.predict(x_test)
gaussian_bayes_conf_matrix = confusion_matrix(y_test, pred)
print(classification_report(y_test, pred))
end = time.time()
print('run time:{:.3f}s\n\n'.format(end - start))

# 6.Support vector machine
start = time.time()
svm = SVC()
svm.fit(x_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(x_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(x_test, y_test)))

pred = svm.predict(x_test)
svm_conf_matrix = confusion_matrix(y_test, pred)
print(classification_report(y_test, pred))
end = time.time()
print('run time:{:.3f}s\n\n'.format(end - start))


# Ignore the warning as they are all 0
dumb_conf_matrix = confusion_matrix(y, [0 for ii in y.tolist()]); 

# Visualize the confusion matrices of various ML models
conf_matrix = {
                1: {
                    'matrix': logistic_reg_conf_matrix,
                    'title': 'Logistic Regression',
                   },
                2: {
                    'matrix': decision_tree_conf_matrix,
                    'title': 'Decision Tree',
                   },
                3: {
                    'matrix': k_neighbors_conf_matrix,
                    'title': 'K Nearest Neighbors',
                   },
                4: {
                    'matrix': linear_disc_conf_matrix,
                    'title': 'Linear Discriminant Analysis',
                   },
                5: {
                    'matrix': gaussian_bayes_conf_matrix,
                    'title': 'Gaussian Naive Bayes',
                   },
                6: {
                    'matrix': svm_conf_matrix,
                    'title': 'Support Vector Machine',
                   }
}
                
fix, ax = plt.subplots(figsize=(16, 12))
plt.suptitle('Confusion Matrix of Various Classifiers')
for ii, values in conf_matrix.items():
    matrix = values['matrix']
    title = values['title']
    plt.subplot(3, 2, ii) # starts from 1 to 6
    plt.title(title);
    sns.heatmap(matrix, annot=True,  fmt='')
