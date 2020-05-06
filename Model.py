import pickle
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout, Embedding
from keras.models import load_model

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt

from util import map_to_X, map_to_Y


## Process data

with open("training.txt", "rb") as f1:
	train_keywords = pickle.load(f1)

with open("validation.txt", "rb") as f2:
	validate_keywords = pickle.load(f2)

with open("test.txt", "rb") as f3:
	test_keywords = pickle.load(f3)


X_train = map_to_X(train_keywords)
y_train = map_to_Y(train_keywords)

X_validate = map_to_X(validate_keywords)
y_validate = map_to_Y(validate_keywords)

X_test = map_to_X(test_keywords)
y_test = map_to_Y(test_keywords)


## Train model

# Create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(300, embedding_vecor_length, input_length=30))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_validate, y_validate), epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_validate, y_validate, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
# Save the model
model.save("model-repetitive-5.h5")



### Plot ROC graph

# Load model
# model = load_model("model-repetitive-5.h5")

# Prepare y values
y_test_predictions = model.predict(X_test)
y_validate_predictions = model.predict(X_validate)

# Compute plot data
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_predictions)
fpr_validate, tpr_validate, thresholds_validate = roc_curve(y_validate, y_validate_predictions)

# Compute auc values
auc_test = auc(fpr_test, tpr_test)
auc_validate = auc(fpr_validate, tpr_validate)

# Plot test data
plt.plot(fpr_test, tpr_test, marker='.', label="Logistic ROC = " + str(auc_test))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title("ROC curve for test data")
plt.show()

# Plot validation data
plt.plot(fpr_test, tpr_test, marker='.', label="Logistic ROC = " + str(auc_validate))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title("ROC curve for validation data")
plt.show()



