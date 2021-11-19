from matplotlib.pyplot import yticks
from lib import *

import pandas
import seaborn as sns

from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# Read Datasets
digits = 10
y_train, X_train = create_tables("../datasets/train.txt")
y_test, X_test = create_tables("../datasets/test.txt")

# Take random samples to visualize them
index = []  # list of indexes
y_index = []  # list of numbers
for i in range (digits):
  w = np.where(y_train == i)[0][0]  # find index
  index.append(w) # append it to list
  y_index.append(y_train[w])

print('Sequence of numbers: {}\n'.format(y_index))

plot_digits_samples([], digits, X_train, y_train)

X_avg = np.array([digit_mean(X_train, y_train, i) for i in range(digits)])
X_var = np.array([digit_variance(X_train, y_train, i) for i in range(digits)])   # transform lists to ndarrays

# Plot the Digits using stats
print("     Digits using AVERAGE:\n")
plot_digits_samples([i for i in range(digits)], digits, X_avg)

print("     Digits using VARIANCE:\n")
plot_digits_samples([i for i in range(digits)], digits, X_var)

# Call Euclidean Classifier and check the accuracy score
eucl = EuclideanDistanceClassifier()
eucl.fit(X_train, y_train)
accuracy = eucl.score(X_test, y_test)

print("The success rate of our classifier is : {} %".format(accuracy))

# Apply Cross-Validation for the Euclidean-Classifier
folds = 5
scores = cross_val_score(EuclideanDistanceClassifier(), X_train, y_train, cv=KFold(n_splits=folds), scoring="accuracy") # calculate 5-fold-cross-validation
print("CV accuracy = %f +-%f" % (np.mean(scores), np.std(scores)))  # print accuracy
print("CV error = %f +-%f" % (1. - np.mean(scores), np.std(scores)))  # print error

# Reduce dimension to plot the desicion surfaces on 2D-planar
pca=PCA(n_components=2)   # minimize dimensionality of features to 2
pca.fit(X_train)
X_new = pca.transform(X_train)

labels = [0,1,2,3,4,5,6,7,8,9]
eucl_new = EuclideanDistanceClassifier()
eucl_new.fit(X_new, y_train)
plot_clf(eucl_new, X_new, y_train, labels)

# Plot the learning Curve of the Euclidean-Classifier
train_sizes, train_scores, test_scores = learning_curve(EuclideanDistanceClassifier(), X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(.80, .89))
plt.show()

# Custom Naive-Bayes Classifier
gNB = CustomNBClassifier()
gNB.fit(X_train,y_train)
print("Custom GaussNB: ", gNB.score(X_test, y_test))

columns_labels = ['Class','A Priori']
row_labels  = ['Digit 0','Digit 1','Digit 2','Digit 3','Digit 4','Digit 5','Digit 6','Digit 7','Digit 8','Digit 9']
values = zip(row_labels, gNB.pC)

cm = sns.light_palette("pink", as_cmap=True)

df1 = pandas.DataFrame(values, columns=columns_labels)
pandas.options.display.float_format = "{:.3f}".format

df1.style.set_caption("A Priori Probability")\
    .background_gradient(cmap=cm)


gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("Sci-kit Learn: ", gnb.score(X_test, y_test))

# Evaluation of 'sklearn_NB'
scores = cross_val_score(sklearn.naive_bayes.GaussianNB(), X_train, y_train, cv=KFold(n_splits=folds), scoring="accuracy")  # calculate  k-fold-cross-validation 
print("CV accuracy of 'sklearn_NB' classifier = %f +-%f" % (np.mean(scores), np.std(scores)))  # print accuracy
print("CV error of 'sklearn_NB' classifier = %f +-%f" % (1. - np.mean(scores), np.std(scores)))  # print error

plot_clf(sklearn.naive_bayes.GaussianNB().fit(X_new, y_train), X_new, y_train, labels)

train_sizes, train_scores, test_scores = learning_curve(sklearn.naive_bayes.GaussianNB(), X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(.6, 0.85))
plt.show()

# Evaluation of 'custom_NB'
scores = cross_val_score(CustomNBClassifier(), X_train, y_train, cv=KFold(n_splits=folds), scoring="accuracy")  # calculate  k-fold-cross-validation 
print("CV accuracy of 'custom_NB' classifier = %f +-%f" % (np.mean(scores), np.std(scores)))  # print accuracy
print("CV error of 'custom_NB' classifier = %f +-%f" % (1. - np.mean(scores), np.std(scores)))  # print error

plot_clf(CustomNBClassifier().fit(X_new, y_train), X_new, y_train, labels)

train_sizes, train_scores, test_scores = learning_curve(CustomNBClassifier(), X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(.8, 0.87))
plt.show()

# Variance == 1 (we explain the results in review)
nb = CustomNBClassifier(use_unit_variance=True)
nb.fit(X_train,y_train)
print("Custom GaussNB(std=1): ", nb.score(X_test, y_test))

# Voting Technic

clf1 = EuclideanDistanceClassifier()
clf2 = sklearn.naive_bayes.GaussianNB()
clf3 = KNeighborsClassifier(n_neighbors=5)
clf_V1 = VotingClassifier(estimators=[('eucl', clf1), ('gnb', clf2), ('knn', clf3)], voting='hard')

acc = clf_V1.fit(X_train,y_train).score(X_test,y_test)*100
print("The success rate of the Voting classifier is : {} %".format(acc))

clf1 = EuclideanDistanceClassifier()
clf2 = sklearn.svm.SVC(kernel='linear')
clf3 = KNeighborsClassifier(n_neighbors=5)
clf_V2 = VotingClassifier(estimators=[('eucl', clf1), ('svm', clf2), ('knn', clf3)], voting='hard')

acc = clf_V2.fit(X_train,y_train).score(X_test,y_test)*100
print("The success rate of the Voting classifier is : {} %".format(acc))

# Evaluation of Voting Classifier

scores = evaluate_voting_classifier(X_train, y_train)
print("CV accuracy of Voting classifier = %f +-%f" % (np.mean(scores), np.std(scores)))  # print accuracy
print("CV error of Voting classifier = %f +-%f" % (1. - np.mean(scores), np.std(scores)))  # print error

clf1 = EuclideanDistanceClassifier()
clf2 = sklearn.naive_bayes.GaussianNB()
clf3 = KNeighborsClassifier(n_neighbors=5)
clf_V1 = VotingClassifier(estimators=[('eucl', clf1), ('gnb', clf2), ('knn', clf3)], voting='hard')
plot_clf(clf_V1.fit(X_new, y_train), X_new, y_train, labels)

clf1 = EuclideanDistanceClassifier()
clf2 = sklearn.naive_bayes.GaussianNB()
clf3 = KNeighborsClassifier(n_neighbors=5)
clf_V1_new = VotingClassifier(estimators=[('eucl', clf1), ('gnb', clf2), ('knn', clf3)], voting='hard')
train_sizes, train_scores, test_scores = learning_curve(clf_V1_new, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(.82, .95))
plt.show()

# Bagging Technic

clf_B1 = BaggingClassifier(base_estimator=EuclideanDistanceClassifier(), n_estimators=10, random_state=0)
acc = clf_B1.fit(X_train,y_train).score(X_test,y_test)*100
print("The success rate of the Bagging classifier is : {} %".format(acc))

clf_B2 = BaggingClassifier(base_estimator=CustomNBClassifier(), n_estimators=10, random_state=0)
acc = clf_B2.fit(X_train,y_train).score(X_test,y_test)*100
print("The success rate of the Bagging classifier is : {} %".format(acc))

# Evaluation of Bagging Classifier

scores = evaluate_bagging_classifier(X_train, y_train)
print("CV accuracy of Bagging classifier = %f +-%f" % (np.mean(scores), np.std(scores)))  # print accuracy
print("CV error of Bagging classifier = %f +-%f" % (1. - np.mean(scores), np.std(scores)))  # print error

clf_B1 = BaggingClassifier(base_estimator=EuclideanDistanceClassifier(), n_estimators=10, random_state=0)
plot_clf(clf_B1.fit(X_new, y_train), X_new, y_train, labels)

clf_B1_new = BaggingClassifier(base_estimator=EuclideanDistanceClassifier(), n_estimators=10, random_state=0)
train_sizes, train_scores, test_scores = learning_curve(clf_B1_new, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(.8, 0.885))
plt.show()

# MLP-Classifier

le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train) # update label in train set
y_test = le.transform(y_test) # update label in test set
n_classes = le.classes_.size # the number of the total labels
hidden_layer = 128
n_features = X_train.shape[1]
layers = [] #1-hidden layer
EPOCHS = 30 # more epochs means more training on the given data. IS this good ?? 60
BATCH_SZ = 128
ETA = 1e-2 
weight_decay = 1e-7

kwargs = {
    'BATCH_SZ': BATCH_SZ,
    'EPOCHS': EPOCHS,
    'layers': layers,
    'n_features': n_features,
    'hidden_layer': hidden_layer,
    'n_classes': n_classes,
    'ETA': ETA,
    'weight_decay': weight_decay
}

DNN = PytorchNNModel(**kwargs)
DNN.fit(X_train, y_train)
score = DNN.score(X_test, y_test)

print(f"Accuracy Score on Test:{score}")

# Evaluation of MLP

scores = evaluate_nn_classifier(X_train, y_train)
print("CV accuracy of 'PytorchNNModel' classifier = %f +-%f" % (np.mean(scores), np.std(scores)))  # print accuracy
print("CV error of 'PytorchNNModel' classifier = %f +-%f" % (1. - np.mean(scores), np.std(scores)))  # print error