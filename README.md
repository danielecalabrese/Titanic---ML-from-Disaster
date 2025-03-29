
<p>Kif Kroker: "Captain, may I have a word with you?"<br>
Captain Zapp Brannigan:"No. Kif Kroker". <br>
Kif Kroker: "It's an emergency, sir". <br>
Captain Zapp Brannigan: "Come back when it's a catastrophe". <br>
<em>[a huge rumbling is heard]</em> <br>
Captain Zapp Brannigan: "Oh, very well".</p>


Ok, the nerd reference has been done, now let's write some simple scripts just to train our skill in ML :)

# Reference
Script in this folder use the following dataset:
https://www.kaggle.com/competitions/titanic

# Explanations

### 4-NLP-tutorial.py: [Knowledge Graph & NLP Tutorial](https://www.kaggle.com/code/pavansanagapati/knowledge-graph-nlp-tutorial-bert-spacy-nltk)
TBD



## 3-TitanicDataScienceSolution.py: [Titanic Data Science Solutions](https://www.kaggle.com/code/startupsci/titanic-data-science-solutions/notebook)

### Note:
- Some instruction are not correct.
### Aquiring and show data
As usual. Here we also show info of our dataset with df.info() and df.describe(...)
### Analyze by pivoting features
We can quickly analyze our feature correlations by pivoting features against each other.
We can see that the suvival field is correlated with Pclass, Sex, SibSp (sibling or spouse) and Parch (parents or child).
### Analyze by visualizing data
Let us start by understanding correlations between numerical features and our solution goal (Survived).
### Model Predicting and solve
TBD


## 2-TF-DF.py: [Titanic competition w/ TensorFlow Decision Forests](https://www.kaggle.com/code/gusthema/titanic-competition-w-tensorflow-decision-forests/comments)

### Note:
On March 2025, Tensorflow package is not compatible with python 3.13 (https://github.com/tensorflow/tensorflow/issues/78774#issuecomment-2498148533). So this project is written with Google Colab.

## 1-Tutorial.py: [General Tutorial](https://www.kaggle.com/code/alexisbcook/titanic-tutorial/notebook)

1- Understanding the Data: we took into consideration the file format (csv) associated with the dataset.

2- Importing the data: we used the pandas library to load the given data in our notebook.

3- Analyzing the data: we used the head command of the pandas library to analyze the attributes of thetraining and the testing data.

4- Pattern Identification: we performed very preliminary EDA to understand the percentage of male/femalesurvivors. We used loc command of pandas to access rows/columns by label.

5- ML Model design: we used Random Forest algorithm to predict the survival of a person.

### Notes:
Train/Test Split Data: The train-test split technique is used to estimate theperformance of machine learning   algorithms when they are used to makepredictions on data not used to train the model. Scikit-Learn is the go-to library to perform train_test_split.
### Random Forest Algorithm: 
A random forest is an ensemble-based machine learning technique that's used tosolve regression and classification problems. It utilizes ensemble learning,which is a technique that combines many classifiers to provide solutions tocomplex problems. A random forest algorithm consists of many decision trees.