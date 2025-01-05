import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

X_train_cv = joblib.load('pickles/X_train_cv.pkl')
X_test_cv = joblib.load('pickles/X_test_cv.pkl')
y_train = joblib.load('pickles/y_train.pkl')
y_test = joblib.load('pickles/y_test.pkl')

model = SVC()

results = []

print(f"\nTraining and evaluating {model}")
try:
    model.fit(X_train_cv, y_train)
    y_pred = model.predict(X_test_cv)
    acc = accuracy_score(y_test, y_pred)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for SVM')
    plt.show()


    results.append({
        'Model': model,
        'Accuracy': acc * 100,
    })

    joblib.dump(model, 'pickles/svm_model.pkl')

except Exception as e:
    print(f"Error training {model} model: {e}")

results_df = pd.DataFrame(results)

print("\n",results_df)