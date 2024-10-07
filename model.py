import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score


# read data
data = pd.read_csv("data.csv")

# fast feature engineering
oe = OrdinalEncoder()

obj_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# divide to input and output and split it to train and test
X, y = data.drop("HeartDisease", axis=1), data["HeartDisease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train[obj_columns] = oe.fit_transform(X_train[obj_columns])


# modeling
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

X_test[obj_columns] = oe.transform(X_test[obj_columns])


# testing
train_predicted, test_predicted = model.predict(X_train), model.predict(X_test)
train_accuracy, train_f1 = accuracy_score(y_train, train_predicted), \
                           f1_score(y_train, train_predicted)
test_accuracy, test_f1 = accuracy_score(y_test, test_predicted), \
                           f1_score(y_test, test_predicted)

print("Train accuracy -->", train_accuracy)
print("Train f1 -->", train_f1)
print("Test accuracy -->", test_accuracy)
print("Test f1 -->", test_f1)



# save encoder and model to use it in the API
if __name__ == "__main__":
    import pickle


    with open('ordinal_encoder.pkl', 'wb') as file:
        pickle.dump(oe, file)

    # Save the model to a file
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)