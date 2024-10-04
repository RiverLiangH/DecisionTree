# Decision Tree Project

This repository contains the implementation of a decision tree algorithm using the ID3 algorithm to perform a classification task on the Census Income dataset. The goal of this project is to predict whether an individual's income exceeds $50K or is less than or equal to $50K based on a range of demographic and employment attributes.

## Project Structure

```plaintext
decision_tree_project/
│
├── data/
│   ├── adult.data         # Training dataset
│   └── adult.name         # Testing dataset
│
├── src/
│   ├── decision_tree.py   # Implementation of the decision tree algorithm
│   ├── utils.py           # Utility functions (e.g., data loading, evaluation)
│   └── model_evaluation.py # Code for model performance evaluation
│
├── notebooks/             # Jupyter notebooks for analysis (if applicable)
│
├── requirements.txt       # List of dependencies (optional)
├── README.md              # Project documentation
└── main.py                # Script to run the model training and evaluation

```
## How to Run the Project

1. Clone the repository
```bash
git clone https://github.com/YourUsername/DecisionTreeProject.git
cd decison-tree-project 
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run the model: Execute the main script to train and evaluate the decision tree model
```bash
python main.py
```
## Data Description
The Census Income dataset is used for training and testing the model. It includes various demographic and employment-related features, with the target variable being the income category (>50K or <=50K).

- Training Dataset: `data/adult.data`
- Testing Dataset: `data/adult.name`

For more information on the dataset, please refer to the UCI Machine Learning Repository.

## Evaluation Metrics

The performance of the model is evaluated using the following metrics:

- Accuracy: The overall correctness of the model.
- Precision: The fraction of relevant instances among the retrieved instances.
- Recall: The fraction of relevant instances that have been retrieved.
- F1 Score: The harmonic mean of precision and recall.
  
The results of the model can be found in the output of the main.py script.

## Model Performance

| Metric                        | Value                                                             |
|-------------------------------|-------------------------------------------------------------------|
| Sample of True Labels         | `['<=50K', '<=50K', '>50K', '>50K', '<=50K', '<=50K', '<=50K', '>50K', '<=50K', '<=50K']` |
| Sample of Predicted Labels    | `['<=50K', '<=50K', '<=50K', '<=50K', '<=50K', '<=50K', '<=50K', '>50K', '<=50K', '<=50K']` |
| True Labels Distribution       | `{'<=50K': 12435, '>50K': 3846}`                                  |
| Predicted Labels Distribution  | `{'<=50K': 12603, '>50K': 3678}`                                  |
| Correct Predictions            | `13,205`                                                          |
| Total Predictions              | `16,281`                                                          |
| Accuracy                      | `0.81107`                                                         |
| Confusion Matrix              | `[[10,981, 1,454], [1,622, 2,224]]`                              |
| Precision                     | `[0.8713, 0.6047]`                                               |
| Recall                        | `[0.8831, 0.5783]`                                               |
| F1 Score                     | `[0.8771, 0.5912]`                                              |

