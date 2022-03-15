# TO RUN THE APP:
#	* open terminal
#	* navigate to streamlit file location
#	* to install required libraries, run: "pip install -r requirements.txt"
#	* use command: "streamlit run myfirstapp2.py"


import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings('ignore')
#import plotly.express as px 
    # to show plot in streamlit


header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

with header:
    st.title('House Price Prediction')
    st.write('      So you are moving to Ames City for whatever reason, and are looking for a house to buy. Or maybe you are an real estate investor who like to get better insight in housing market in this city. So we are here to help you in predicting your dream house price in Ames city, Iowa, USA')
    
    
with dataset:
    st.header('How do we predict the house price?')
    st.write("We use a sophisticated method by collecting datas from 1460 houses located in Ames City, and analize the most affecting features to the the house sale prices. We study about 80 features of the houses from the house size, its contruction material, year it's build, until the neighbourhood environtment. Below is the sample of our data;")  #we use st.write instead of st.text because the text too long and read friendly in streamlit.
    df = pd.read_csv('train.csv')
    df_test= pd.read_csv('test.csv')
    st.write(df.head())
    
 
 
    
    
    
with features:
    st.header ('Why are some houses more expensive than the others?')
    st.write('From our data, lets see the most affecting features that contribute to the house sale price... ')
    st.write(' See the correlation value below, from 1 - 0, showing from the most contributing                                                              features to the house price, to the least contributing features.' )
    corr_y = df.corr()
    corr_y['SalePrice'].sort_values(ascending=False).abs()[1:]
    st.write(corr_y)
    st.write('From the table above, we can see the house overall quality plays the most important role in                                                  pricing a house, followed by the the size of living area and the size of the garage. ')
   



 #deleting all columns except the 5 columns that affect the house price the most, or maybe we can just pick that 5 columns and assign them with new string, that will be easier.  
with modelTraining:
    st.header('5 major factors')
    st.write(' Just so we can get a better understanding of that 5 factors, here we show a sample of a few houses and its most important features, and its price. ')
    
df = df[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF', 'SalePrice']]
st.write(df.head())

st.subheader("Let's see how much your house would cost  ") 
st.header("←")
st.header("")
st.header("")
st.header("")
st.header("")
st.header("")
st.header("")

#input the numbers sidebar
st.sidebar.title('Want to know your house price? Tell us its features:')
st.sidebar.text('')
OverallQual = st.sidebar.slider('House overall quality', 0, 10, 5)
GrLivArea =  st.sidebar.slider('Ground living area in square feet',int(df.GrLivArea.min()),int(df.GrLivArea.max()),int(df.GrLivArea.mean()) )
GarageCars = st.sidebar.slider('For the garage size, how many cars you want to fit in?',int(df.GarageCars.min()),int(df.GarageCars.max()) )
GarageArea =   st.sidebar.slider('Garage size in square feet',int(df.GarageArea.min()),int(df.GarageArea.max()),int(df.GarageArea.mean()) )
TotalBsmtSF =   st.sidebar.slider('Basement size in square feet',int(df.TotalBsmtSF.min()),int(df.TotalBsmtSF.max()),int(df.TotalBsmtSF.mean()) )




#splitting the data
X = df.drop('SalePrice', axis = 1)
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=42)

#modelling step
model=LinearRegression()

#fitting and predict  model
model.fit(X_train, y_train)
model.predict(X_test)
errors = np.sqrt(mean_squared_error(y_test,model.predict(X_test))) 
#predictions = model.predict([[OverallQual,GrLivArea,GarageCars,TotalBsmtSF]])[0]
#Lol, when i run the model.predict code, i got an error.... matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 5 is different from 4), 
# Just realised I left out one variable which is GarageArea. So lets add it into the script..
predictions = model.predict([[OverallQual,GrLivArea,GarageCars,TotalBsmtSF,GarageArea]])[0]

#Lets see our prediction
if st.sidebar.button          ("Show my house price"):
    st.sidebar.header("Fair price for your house is = {} USD ".format(int(predictions)))
    st.sidebar.subheader("Similar house are sold in your state with price ranged  {}  USD  to  {}  USD ".format(int(predictions-errors),int(predictions+errors) ))
               
st.write("Made by Ahmad Nafi'")    
st.write("Source of data: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview")
st.write("Credit to tutors and various coders that help me finishing this webapp.")
st.write('#IlOVEMOHE #everybodycanfly')





    



                                
                                
                                



    
    



