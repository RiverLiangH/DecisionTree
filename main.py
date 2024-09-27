from src.model_evaluation import *
from src.decision_tree import DecisionTreeClassifier
from src.utils import data_handler, load_data
import logging
import pickle

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
    tree = DecisionTreeClassifier(max_depth=15)

    # Train Model
    tree.fit(X_train, y_train)
    logging.info('Decision Tree Constructed')

    with open('decision_tree_model.pkl', 'wb') as model_file:
        pickle.dump(tree, model_file)
        logging.info('Decision Tree Model Saved to decision_tree_model.pkl')

    # Predict
    logging.info('Predicting...')
    y_pred = tree.predict(X_test)

    # Evaluation Part
    print_hi('Evaluation')

    # Compute and output Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # # Compute and output Confusion Matrix
    # conf_matrix = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix:")
    # print(conf_matrix)

    # Compute and output Precision, Recall and F1-Score
    precision, recall, f1 = precision_recall_f1(y_test, y_pred)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

