Let's write some simple scripts just to train our skill in ML.

# Reference
Script in this folder use the following dataset:
https://www.kaggle.com/competitions/titanic

# Explanations

- 3-DataScienceSolution.py (Titanic Data Science Solutions): https://www.kaggle.com/code/startupsci/titanic-data-science-solutions/notebook

    ### Note:
    - Some instruction are not correct.
    
    ### Aquiring and show data
    As usual. Here we also show info of our dataset with df.info() and df.describe(...)

    ### Analyze by pivoting features
    We can quickly analyze our feature correlations by pivoting features against each other.
    We can see that the suvival field is correlated with Pclass, Sex, SibSp (sibling or spouse) and Parch (parents or child).

    ### Analyze by visualizing data
    Let us start by understanding correlations between numerical features and our solution goal (Survived).




- 2-2-TF-DF.py (Titanic competition w/ TensorFlow Decision Forests): https://www.kaggle.com/code/gusthema/titanic-competition-w-tensorflow-decision-forests/comments

    ### Note:
    On March 2025, Tensorflow package is not compatible with python 3.13 (https://github.com/tensorflow/tensorflow/issues/78774#issuecomment-2498148533). So this project is written with Google Colab.

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