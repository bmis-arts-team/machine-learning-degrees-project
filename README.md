# Heart-Disease-Prediction Web Application

## A Web Application to predict Heart-Disease using SVM


### 1.Project Requirements or Dependencies
* Jupyter Notebook Python (to create, train and test the model)
* Pip install flask (For Front-end)

### 2. Load Dataset
Heart-Disease-Prediction Original Data Set

Attribute Information:
1.	Sample code number: ID number
2.	Clump Thickness:1-10
3.	Uniformity of Cell size:1-10
4.	Uniformity of Cell shape:1-10
5.	Marginal Adhesion:1-10
6.	Single Epithelial Cell Size:1-10
7.	Bare Nuclei:1-10
8.	Bland Chromatin:1-10
9.	Normal Nucleoli:1-10
10.	Mitoses:1-10
11.	Class: (2 for Benign, 4 for Malignant)

### 3.Build and Train the model using SVM

Using SVM (Support Vector Machines) we build and train a model using human cell records, and classify cells to predict whether the samples are Effected or Not-Affected.

### 4.Flask Creation

1.	Heart-Disease-Prediction.ipynb — This contains code for the machine learning model to predict heart disease based on the class.
2.	app.py — This contains Flask APIs that receives cells details through GUI or API calls, computes the predicted value based on our model and returns it
3.	templates & static  — This folders contains the HTML template and CSS styling to allow user to enter cells details and displays the predicted output.

### 5.Backend creation using model.pkl file

Use this pretrained model and connect it with our Flask application.
Use this for prediction for model and to show the output

### 6. Adding form to flask app
 
### 7.Integrating web application with machine learning backend.

### 8. Env Packages Reqired Versions

1. Python=3.11.2
2. Flask==2.3.2
3. Werkzeug==2.3.6
4. Pandas==2.0.3
5. Numpy==1.24.3
6. Sklearn==1.3.0
7. Matplotlib==3.7.2
8. Keras==2.13.1
