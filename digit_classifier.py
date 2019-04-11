import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
ids = np.arange(1,28001)

y = train['label']
X = train.drop(['label'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

# trying different classifiers with default params
rfc = RandomForestClassifier(verbose=True)
mlp = MLPClassifier(verbose=True)
mnb = MultinomialNB()

classifiers = {"Random Forest": rfc, "Multi Layer Perceptron": mlp,
               "Multinomial Bayes": mnb}

for name, model in classifiers.items():
    print("Training ", name)
    # if name != "Support Vector":
    model.fit(X_train, y_train)
        # print("%s Accuracy : {:.3lf}".format(name, model.score(X_test, y_test)))
    print("%s Accuracy : %f" %(name, model.score(X_test, y_test)))



def write_to_csv(predictions):
    results = pd.DataFrame(columns=["ImageId", "Label"])
    results['ImageId'] = ids
    results['Label'] = predictions
    results.to_csv("output.csv", index=False)

# write_to_csv(predictions)

