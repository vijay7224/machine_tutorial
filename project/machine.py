import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
from PIL import Image
import joblib

st.sidebar.title("MACHINE LEARNING")
st.balloons()
x=["REGRESSION","Classification"]
option=st.sidebar.selectbox("chose the machine learning model",x)
if option=="REGRESSION":
    y=["LinearRegression","polynomialRegression","LassoRegression","RidgeRegression"]
    options=st.sidebar.selectbox("chose the Regression model",y)
    if options=="LinearRegression":
        st.header("LINEAR REGRESSION ")
        st.subheader("Linear regression is a machine learning algorithm that uses a linear equation to predict the value of a variable based on another variable. It's a supervised machine learning algorithm that uses labeled datasets to learn and make predictions. ")
        st.subheader("LINE EQUATION IN LINEAR REGRESSION IS")
        st.subheader("Y=MX + C")
        

        st.markdown(" #### x is Independent Variable, Plotted along X-axis")
        st.write("y is Dependent Variable, Plotted along Y-axis")
        st.write("The slope of the regression line is “M”, and the intercept value of regression line is “c”(the value of y when x = 0).")
        #img=Image.open("xy.jpg")
        st.image("C:\\Users\\payal\\Desktop\\1.png",width=700)
        code= '''# DATA IS INDIVIDE INTO TRAIN AND TEST
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
        # IMPORT LINEAR REGRESSION MODEL
        from sklearn.linear_model import LinearRegression
        f=LinearRegression()
        # DATA FIT INTO MODEL
        f.fit(x_train,y_train)
        d=f.predict(x)'''
        st.code(code)
        st.subheader("COLLAGE PLACEMENT DATASET")
        data=pd.read_csv("C:\\Users\\payal\\Desktop\\place.csv")

        st.write(data)
        st.write("size of data is ",data.shape)
        st.header("LINEAR REGRESSION MACHINE LEARNING MODEL")
        cgpa=st.number_input("CGPA",placeholder="enter the cgpa")
        model = joblib.load("linear")
        j=model.predict([[cgpa]])
        if st.button("SUBMIT"):
          st.header("MODEL PREDICTION IS ")
        

          if j<1:
            st.subheader("NOT PLACEMENT")
          else:
            st.subheader("YOU ARE PLACED")
    elif options=="polynomialRegression":
      st.title("Plynomial Regression")
      st.subheader("Polynomial regression is a machine learning technique that models the relationship between variables using a polynomial function. It's an extension of linear regression that's used to model non-linear relationships. ")
      st.subheader("How it works")
      k='''\n 
      Polynomial regression fits a polynomial function to the data, using higher-degree functions of the independent variable, such as squares and cubes \n
      It's used when there is no linear correlation between the variables \n
      It's a supervised learning method that's used to make predictions 
       '''
      st.write(k)
      st.image("C:\\Users\\payal\\Desktop\\2.png")
      g='''
      # Fitting Polynomial Regression to the dataset
      from sklearn.preprocessing import PolynomialFeatures
 
      poly = PolynomialFeatures(degree=4)
      X_poly = poly.fit_transform(X)
 
      poly.fit(X_poly, y)
      lin2 = LinearRegression()
      lin2.fit(X_poly, y)'''
      st.code(g)
      st.subheader("Polynomial regression types ")
      f="""\n 
      Linear, if the degree is 1 \n
      Quadratic, if the degree is 2 \n
      Cubic, if the degree is 3"""
      st.write(f)
    elif options=="LassoRegression":
        st.title("Lasso Regression") 
        st.subheader("Lasso regression is a machine learning technique that helps build accurate models by reducing overfitting and selecting relevant variables. It's also known as Least Absolute Shrinkage and Selection Operator (LASSO) or L1 regularization. ")  
        st.subheader("How does Lasso regression work? ")
        st.image("C:\\Users\\payal\\Desktop\\download.png",width=600)
        l=''' \n
         1 Adds a penalty term to the linear regression model\n
         2 Penalizes the model based on the absolute values of the coefficients\n
         3 Shrinks some coefficients to zero, effectively removing unimportant variables
       '''
        st.write(l)
        st.header("STUDENT NUMBER SUBJECT WICE")
        data=pd.read_csv("C:\\Users\\payal\\Desktop\\a.csv")
        st.write(data)
        p='''
        #DATA IS LOAD
        data=pd.read_csv("C:\\Users\\payal\\Desktop\\a.csv")
        print(data)
        
        x=data[["number_courses","time_study"]]
        y=data[["Marks"]] 
        # DATA IS SPLIT TRAIN AND TEST PART
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

        #RASSO MODEL IS LOAD
        from sklearn.linear_model import Lasso
        ab=Lasso()
        #DATA IS FIT IN MODEL
        ab.fit(x_train,y_train)
        v=ab.score(x_test,y_test)
        print(v)
        joblib.dump(ab, "lasso")'''
        st.code(p)
        losso = joblib.load("lasso")
        st.title("LASSO REGRESSION MODEL PREDICTION")
        a=st.number_input("NUMBER OF SUBJECT",placeholder="ENTER THE NUMBER OF SUBJECT")
        b=st.number_input("STUDY HOURES",placeholder="ENTER THE STUDY HOURES")
        if st.button("SUBMIT"):
           d=losso.predict([[a,b]])
           st.subheader("LASSON MODEL PREDICT IS YOU ARE")
           if d>26 :
              st.subheader("PASS")
           else:
              st.subheader("NOT PASS")
    else: 
       st.title("RIDGE REGRESSION") 
       st.subheader("Ridge regression, also known as  L2 regularization, is a technique used in linear regression to address the problem of multicollinearity among predictor variables. Multicollinearity occurs when independent variables in a regression model are highly correlated, which can lead to unreliable and unstable estimates of regression coefficients.")         
       st.image("C:\\Users\\payal\\Desktop\\3.png",width=600) 
       st.header("STUDENT NUMBER SUBJECT WICE")
       data=pd.read_csv("C:\\Users\\payal\\Desktop\\a.csv")
       st.write(data)
       p='''
        #DATA IS LOAD
        data=pd.read_csv("C:\\Users\\payal\\Desktop\\a.csv")
        print(data)
        
        x=data[["number_courses","time_study"]]
        y=data[["Marks"]] 
        # DATA IS SPLIT TRAIN AND TEST PART
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

        #RIDGE MODEL IS LOAD
        from sklearn.linear_model import Ridge
        ab=Ridge()
        #DATA IS FIT IN MODEL
        ab.fit(x_train,y_train)
        v=ab.score(x_test,y_test)
        print(v)
        joblib.dump(ab, "ridge")'''
       st.code(p)
       ridge= joblib.load("ridge")
       st.title("RIDGE REGRESSION MODEL PREDICTION")
       a=st.number_input("NUMBER OF SUBJECT",placeholder="ENTER THE NUMBER OF SUBJECT")
       b=st.number_input("STUDY HOURES",placeholder="ENTER THE STUDY HOURES")
       if st.button("SUBMIT"):
           d=ridge.predict([[a,b]])
           st.subheader("RIDGE MODEL PREDICT IS YOU ARE")
           if d>26 :
              st.subheader("PASS")
           else:
              st.subheader("NOT PASS")                 
else:
    z=["SVM","Decision tree","LogisticRression","Naive Bayes"]
    options=st.sidebar.selectbox("chose the classification model",z)
    if options=="SVM":
       st.header("SUPPORT VECTOR MACHINRE")
       st.subheader("A support vector machine (SVM) is a supervised machine learning algorithm that classifies data by finding a hyperplane that separates data points of different classes. ")
       st.image("C:\\Users\\payal\\Desktop\\Capture.jpg")
    
       st.subheader("How does SVM work?")
       s=""" \n 
       1 SVM analyzes data to find the best line or hyperplane that separates data points of different classes \n
       2 It maximizes the margin, which is the distance between the hyperplane and the closest data points of each category \n 
       3 SVM uses a technique called the kernel trick to transform data into a higher-dimensional space, where it is easier to find a boundary """
       
       st.write(s) 
       st.subheader("When is SVM used?" )
       g="""\n
       SVMs are particularly good at solving binary classification problems\n
       They can be used for classification or regression problems\n
       They can be used to detect outliers\n
       They can be used to understand and recognize handwriting\n
       They can be used to classify facial expressions""" 
       st.write(g)
       st.subheader("Advantages of SVM ")
       k='''\n 
       1--They perform well on smaller or complex datasets with minimal tuning \n
       2---They are effective in high dimensional spaces \n
       3---They generalize well to new data and make accurate classification predictions'''
       st.write(k)
       st.subheader("IRIS DATASETS")
       dataset=sea.load_dataset("iris")
       st.write(dataset)
       st.write("size of dataset is",dataset.shape)
       code="""x=dataset.iloc[:,[0,1,2,3]]
       x.ndim
       y=dataset[["species"]]
       #train and test data is split using train_test_split function
      from sklearn.model_selection import train_test_split
      x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
      #import SUPPORT VECTOR CLASSIFIER ALGORITHM
        from sklearn.svm import SVC
       dc=SVC()
       #FIT THE DATA INTO MODEL
       dc.fit(x_train,y_train)
       l=dc.predict([[5.1,3.5,1.4,0.2]])
       print(l)
       joblib.dump(dc, "decision")"""
       st.code(code,language="python")

       load = joblib.load("decision")
       st.header("SUPPORT VECTOR MACHINE MODEL PREDICTION")
       
       p=st.number_input("SEPAL_LENGTH",placeholder="ENTER THE SEPAL_LENGTH ")
       q=st.number_input("SEPAL_width",placeholder="ENTER THE SEPAL_WIDTH")
       r=st.number_input("PETAL_LENGTH",placeholder="ENTER THE PETAL_LENGTH ")
       s=st.number_input("PETAL_WIDTH",placeholder="ENTER THE PETAL_WIDTH ")
       h=load.predict([[p,q,r,s]])
       if st.button("SUBMIT"):
          st.subheader("folower is ")
          st.subheader(h)
    elif options =="Decision tree":
       st.title("DECISION TREE ALGORITHM")
       st.subheader("Decision tree is a simple diagram that shows different choices and their possible results helping you make decisions easily. This article is all about what decision trees are, how they work, their advantages and disadvantages and their applications.")
       st.image("C:\\Users\\payal\\Desktop\\Dc.webp")
       st.subheader("Understanding Decision Tree")
       z='''  \n 
           1-Root Node is the starting point that represents the entire dataset .\n
           2-Branches: These are the lines that connect nodes. It shows the flow from one decision to another.\n
           3-Internal Nodes are Points where decisions are made based on the input features.\n
           4-Leaf Nodes: These are the terminal nodes at the end of branches that represent final outcomes or predictions'''
       st.write(z)
       c=sea.load_dataset("diamonds")
       st.header("DIAMONDS DATASETS")
       st.write(c)