#MiniProjectPath3
import numpy as np
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
#import models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import KernelPCA
import copy


rng = np.random.RandomState(1)
digits = datasets.load_digits()
images = digits.images
labels = digits.target

#Get our training data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.6, shuffle=False)

def dataset_searcher(number_list,images=images,labels=labels):
  #insert code that when given a list of integers, will find the labels and images
  #and put them all in numpy arrary (at the same time, as training and testing data)

  images_np = np.array(images)
  labels_np = np.array(labels)

  mask = np.isin(labels_np, number_list)

  images_nparray = images_np[mask]
  labels_nparray = labels_np[mask]

  return images_nparray, labels_nparray

def print_numbers(images,labels):
  #insert code that when given images and labels (of numpy arrays)
  #the code will plot the images and their labels in the title. 
  for i in range(len(images)):
        plt.imshow(images[i], cmap=plt.cm.gray_r, interpolation="nearest")
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
        plt.show()

class_numbers = [2,0,8,7,5]
#Part 1
class_number_images , class_number_labels = dataset_searcher(class_numbers)
#Part 2
print_numbers(class_number_images[:5] , class_number_labels )


model_1 = GaussianNB()

#however, before we fit the model we need to change the 8x8 image data into 1 dimension
# so instead of having the Xtrain data beign of shape 718 (718 images) by 8 by 8
# the new shape would be 718 by 64
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

#Now we can fit the model
model_1.fit(X_train_reshaped, y_train)
#Part 3 Calculate model1_results using model_1.predict()
model1_results = model_1.predict(X_test_reshaped)


def OverallAccuracy(results, actual_values):
  #Calculate the overall accuracy of the model (out of the predicted labels, how many were correct?)
  
  results = np.array(results)
  actual_values = np.array(actual_values)
  correct = np.sum(results == actual_values)
  Accuracy = correct / len(actual_values)
  return Accuracy


# Part 4
Model1_Overall_Accuracy = OverallAccuracy(model1_results, y_test)
print("\nThe overall results of the Gaussian model is " + str(Model1_Overall_Accuracy))


#Part 5
allnumbers = [0,1,2,3,4,5,6,7,8,9]
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers)

allnumbers_images_reshaped = allnumbers_images.reshape(allnumbers_images.shape[0], -1)

model1_allnumbers_pred = model_1.predict(allnumbers_images_reshaped)

print("Example GaussianNB predictions on digits 0-9:", model1_allnumbers_pred[:10])

#Part 6
#Repeat for K Nearest Neighbors
model_2 = KNeighborsClassifier(n_neighbors=10)
model_2.fit(X_train_reshaped, y_train)
model2_results = model_2.predict(X_test_reshaped)
Model2_Overall_Accuracy = OverallAccuracy(model2_results, y_test)
print("\nThe overall accuracy of the KNN model (clean training) is:", Model2_Overall_Accuracy)

model2_allnumbers_pred = model_2.predict(allnumbers_images_reshaped)
print("Example KNN predictions on digits 0-9:", model2_allnumbers_pred[:10])

#Repeat for the MLP Classifier
model_3 = MLPClassifier(random_state=0)
model_3.fit(X_train_reshaped, y_train)
model3_results = model_3.predict(X_test_reshaped)
Model3_Overall_Accuracy = OverallAccuracy(model3_results, y_test)
print("\nThe overall accuracy of the MLP model (clean training) is:", Model3_Overall_Accuracy)

model3_allnumbers_pred = model_3.predict(allnumbers_images_reshaped)
print("Example MLP predictions on digits 0-9:", model3_allnumbers_pred[:10])


#Part 8
#Poisoning
# Code for generating poison data. There is nothing to change here.
noise_scale = 10.0
poison = rng.normal(scale=noise_scale, size=X_train.shape)

X_train_poison = X_train + poison

X_train_poison_reshaped = X_train_poison.reshape(X_train_poison.shape[0], -1)
#Part 9-11
#Determine the 3 models performance but with the poisoned training data X_train_poison and y_train instead of X_train and y_train
model_1_poison = GaussianNB()
model_1_poison.fit(X_train_poison_reshaped, y_train)
model1_poison_results = model_1_poison.predict(X_test_reshaped)
Model1_Poison_Accuracy = OverallAccuracy(model1_poison_results, y_test)
print("\nGaussianNB accuracy with poisoned training data:", Model1_Poison_Accuracy)

model1_poison_allnumbers_pred = model_1_poison.predict(allnumbers_images_reshaped)
print("Example GaussianNB (poisoned) predictions on digits 0-9:", model1_poison_allnumbers_pred[:10])

model_2_poison = KNeighborsClassifier(n_neighbors=10)
model_2_poison.fit(X_train_poison_reshaped, y_train)
model2_poison_results = model_2_poison.predict(X_test_reshaped)
Model2_Poison_Accuracy = OverallAccuracy(model2_poison_results, y_test)
print("\nKNN accuracy with poisoned training data:", Model2_Poison_Accuracy)

model2_poison_allnumbers_pred = model_2_poison.predict( allnumbers_images_reshaped)
print("Example KNN (poisoned) predictions on digits 0-9:", model2_poison_allnumbers_pred[:10])

model_3_poison = MLPClassifier(random_state=0)
model_3_poison.fit(X_train_poison_reshaped, y_train)
model3_poison_results = model_3_poison.predict(X_test_reshaped)
Model3_Poison_Accuracy = OverallAccuracy(model3_poison_results, y_test)
print("\nMLP accuracy with poisoned training data:", Model3_Poison_Accuracy)

model3_poison_allnumbers_pred = model_3_poison.predict(allnumbers_images_reshaped)
print("Example MLP (poisoned) predictions on digits 0-9:", model3_poison_allnumbers_pred[:10])
#Part 12-13
# Denoise the poisoned training data, X_train_poison. 
# hint --> Suggest using KernelPCA method from sklearn library, for denoising the data. 
# When fitting the KernelPCA method, the input image of size 8x8 should be reshaped into 1 dimension
# So instead of using the X_train_poison data of shape 718 (718 images) by 8 by 8, the new shape would be 718 by 64

kpca = KernelPCA(n_components=32, kernel="linear", fit_inverse_transform=True)
X_train_poison_kpca = kpca.fit_transform(X_train_poison_reshaped)
X_train_denoised = kpca.inverse_transform(X_train_poison_kpca)

#Part 14-15
#Determine the 3 models performance but with the denoised training data, X_train_denoised and y_train instead of X_train_poison and y_train
#Explain how the model performances changed after the denoising process.
# GaussianNB, denoised training
model_1_denoised = GaussianNB()
model_1_denoised.fit(X_train_denoised, y_train)
model1_denoised_results = model_1_denoised.predict(X_test_reshaped)
Model1_Denoised_Accuracy = OverallAccuracy(model1_denoised_results, y_test)
print("\nGaussianNB accuracy with denoised training data:", Model1_Denoised_Accuracy)

model1_denoised_allnumbers_pred = model_1_denoised.predict(allnumbers_images_reshaped)
print("Example GaussianNB (denoised) predictions on digits 0-9:", model1_denoised_allnumbers_pred[:10])

model_2_denoised = KNeighborsClassifier(n_neighbors=10)
model_2_denoised.fit(X_train_denoised, y_train)
model2_denoised_results = model_2_denoised.predict(X_test_reshaped)
Model2_Denoised_Accuracy = OverallAccuracy(model2_denoised_results, y_test)
print("\nKNN accuracy with denoised training data:", Model2_Denoised_Accuracy)

model2_denoised_allnumbers_pred = model_2_denoised.predict( allnumbers_images_reshaped)
print("Example KNN (denoised) predictions on digits 0-9:", model2_denoised_allnumbers_pred[:10])

model_3_denoised = MLPClassifier(random_state=0)
model_3_denoised.max_iter = 1000
model_3_denoised.fit(X_train_denoised, y_train)
model3_denoised_results = model_3_denoised.predict(X_test_reshaped)
Model3_Denoised_Accuracy = OverallAccuracy(model3_denoised_results, y_test)
print("\nMLP accuracy with denoised training data:", Model3_Denoised_Accuracy)

model3_denoised_allnumbers_pred = model_3_denoised.predict(allnumbers_images_reshaped)
print("Example MLP (denoised) predictions on digits 0-9:", model3_denoised_allnumbers_pred[:10])
