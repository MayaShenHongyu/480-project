import pickle

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt

from util import map_to_X, map_to_Y



with open("training.txt", "rb") as f1:
    train_keywords = pickle.load(f1)

with open("validation.txt", "rb") as f2:
	test_keywords = pickle.load(f2)



X_train = map_to_X(train_keywords)
y_train = map_to_Y(train_keywords, train_keywords)

X_test = map_to_X(test_keywords)
y_test = map_to_Y(test_keywords, test_keywords)

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(300, embedding_vecor_length, input_length=30))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

with open("model-repetitive.obj", "wb") as f1:
    pickle.dump(model, f1)


### Plot ROC graph

# with open("model-repetitive.obj", "rb") as f4:
# 	model = pickle.load(f4)


# y_predictions = model.predict(X_test)

# fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_predictions)
# auc_keras = auc(fpr_keras, tpr_keras)
# plt.plot(fpr_keras, tpr_keras, marker='.', label="Logistic ROC = " + str(auc_keras))
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
# plt.show()

