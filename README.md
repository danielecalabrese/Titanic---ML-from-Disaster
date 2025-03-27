Let's write some simple scripts just to train our skill in ML.

# Reference
Good train for this dataset:
https://www.kaggle.com/competitions/titanic

# Explanations
- 1-tutorial.py:
    https://www.kaggle.com/code/alexisbcook/titanic-tutorial/notebook

    ### Understanding the Data
    We took into consideration the file format (csv) associated with the dataset.
    ### Importing the data
    We used the pandas library to load the given data in our notebook.
    ### Analyzing the data
    We used the head command of the pandas library to analyze the attributes of the training and the testing data.
    ### Pattern Identification
    We performed very preliminary EDA to understand the percentage of male/female survivors.
    We used loc command of pandas to access rows/columns by label.
    ### ML Model design
    We used Random Forest algorithm to predict the survival of a person.
    ### Notes:
    Train/Test Split Data: The train-test split technique is used to estimate the performance of machine learning   algorithms when they are used to make predictions on data not used to train the model. Scikit-Learn is the go-to  library to perform train_test_split.

    #### Random Forest Algorithm: 
    A random forest is an ensemble-based machine learning technique that's used to solve regression and classification problems. It utilizes ensemble learning, which is a technique that combines many classifiers to provide solutions to complex problems. A random forest algorithm consists of many decision trees.