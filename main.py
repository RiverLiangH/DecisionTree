from src.model_evaluation import *
from src.decision_tree import DecisionTreeClassifier
from src.utils import data_handler, load_data
import logging
import pickle
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Configure Logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler("decision_tree.log"),
                              logging.StreamHandler()])

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load train and test data
    logging.info('Loading Dataset...')
    train_data = load_data('data/adult.data')
    test_data = load_data('data/adult.test')
    logging.info('Load Succ')

    logging.info('Handling Data...')
    X_train, y_train, means, std_devs, X_test, y_test = data_handler(train_data, test_data)
    logging.info('Preprocess Data Done')

    # Instance Model
    logging.info('Building Decision Tree...')
    tree = DecisionTreeClassifier(max_depth=50)

    # Train Model
    tree.fit(X_train, y_train)
    logging.info('Decision Tree Constructed')

    with open('decision_tree_model.pkl', 'wb') as model_file:
        pickle.dump(tree, model_file)
        logging.info('Decision Tree Model Saved to decision_tree_model.pkl')

    # Predict
    logging.info('Predicting...')
    y_pred = tree.predict(X_test)

    y_pred = [pred.replace('.', '').strip() for pred in y_pred]
    y_test = [test.replace('.', '').strip() for test in y_test]

    # # Evaluation Part
    # for i in range(3):
    #     print(f"第 {i + 1} 行: {y_pred[i]}")
    #
    # print("-----------------------------------")
    # for i in range(3):
    #     print(f"第 {i + 1} 行: {y_test[i]}")

    # Output samples of prediction and ground truth.
    print("Sample of true labels:", y_test[:10])  # 显示前10个真实标签
    print("Sample of predicted labels:", y_pred[:10])  # 显示前10个预测标签

    # Output statistics of classification
    unique, counts = np.unique(y_test, return_counts=True)
    print("True labels distribution:", dict(zip(unique, counts)))

    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    print("Predicted labels distribution:", dict(zip(unique_pred, counts_pred)))

    print("Type of y_test:", type(y_test[0]), "Type of y_pred:", type(y_pred[0]))

    # Compute and output Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.5f}")

    # Compute and output Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Compute and output Precision, Recall and F1-Score
    precision, recall, f1 = precision_recall_f1(y_test, y_pred)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

