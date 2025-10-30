#911callsdatacapstone.py
# Import necessary libraries for data analysis and visualization
# These libraries include numpy for numerical operations, pandas for data manipulation,
# matplotlib for plotting, and seaborn for statistical data visualization.
# Make sure to have these libraries installed in your Python environment.   
# You can install them using pip if you haven't done so:
# pip install numpy pandas matplotlib seaborn   
#pip install -r requirements.txt
# Now, let's import the libraries.
# Importing libraries
# ----------------------------------------
# ----------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# ----------------------------------------
# ---------------------------------------
# Load the dataset
df=pd.read_csv('911.csv')
# ----------------------------------------
# ----------------------------------------
############################################################ Exploratory Data Analysis (EDA)#################################################################

'''# Display the first few rows of the dataset to understand its structure
print(df.head())
# Display the column names to understand the features available in the dataset
print(df.columns)
# Display the information about the dataset including data types and non-null counts
print(df.info())
# Display the statistical summary of the dataset to understand the distribution of numerical features
print(df.describe())
# Check for missing values in the dataset to ensure data quality
print(df.isnull().sum())
# Check the number of unique values in each column to understand categorical features
print(df.nunique())
# Display the data types of each column to understand the nature of the data
print(df.dtypes)
# Display the shape of the dataset to understand its dimensions
print(df.shape)
# Display the index of the dataset to understand its structure
print(df.index)
# Display the column names of the dataset to understand its features
print(df.columns)
# Display the values of the dataset to understand its content
print(df.values)'''

print(df.columns)
####################################################################################################################################################################

################################# Display the correlation matrix to understand relationships between numerical features     ##################################################      
print(df.select_dtypes(include=['number']).corr())
# Visualize the correlation matrix using a heatmap for better interpretation
plt.figure(figsize=(16, 9))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm')
plt.title('1.Correlation Matrix Heatmap')    
plt.show() 
print(df.columns)
######################################################## Additional exploratory data analysis can be performed as needed##################################################
############################################# visualizing the distribution of calls by types        #########################################################   
plt.figure(figsize=(16, 9))     
sns.countplot(data=df, x='title', order=df['title'].value_counts().index[:10])
plt.title('2.Distribution of Call Types')         
plt.xticks(rotation=90)
plt.show()
print(df.columns)
#######################################################Visualize the number of calls by location#########################################################
plt.figure(figsize=(16, 9)) 
sns.countplot(data=df, x='twp', order=df['twp'].value_counts().index[:10])
plt.title('3.Top 10 Townships by Number of Calls')    
plt.xticks(rotation=90)
plt.show()
print(df.columns)
print(df.columns)
####################################################### Visualize the number of calls over time #########################################################
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df.set_index('timeStamp', inplace=True)
plt.figure(figsize=(16, 9))
df.resample('D').size().plot()
plt.title('4.Number of Calls Over Time')  
plt.xlabel('Date')
plt.ylabel('Number of Calls')
plt.show()
df.reset_index(inplace=True)#important to reset index after resampling
print(df.columns)
################################### Extracting the reason for the call from the 'title' column      #########################################################

x=df['title'].iloc[0]
print(x.split(':')[0])

df['Reason']=df['title'].apply(lambda title:title.split(':')[0])    
print(df['Reason'].head())

# Visualize the distribution of calls by reason
plt.figure(figsize=(16, 9))
sns.countplot(data=df, x='Reason', order=df['Reason'].value_counts().index)
plt.title('5.Distribution of Calls by Reason')
plt.show()

print(df.columns)
########################### Analyzing calls by month and day of the week and visualize       #########################################################
df['Month']=df['timeStamp'].dt.month  
df['Day of Week'] = df['timeStamp'].dt.day_name()
df['Hour'] = df['timeStamp'].dt.hour
print(df['Day of Week'].head())

plt.figure(figsize=(16, 9))
df['Reason']=df['title'].apply(lambda title:title.split(':')[0])
sns.countplot(data=df, x='Day of Week',hue='Reason', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('6.Number of Calls by Day of the Week')
plt.show()
print(df.columns)
##############################Visualize the number of calls by month                #########################################################
plt.figure(figsize=(16, 9))
sns.countplot(data=df, x='Month', order=range(1, 13))
plt.title('7.Number of Calls by Month')
plt.show()    
print(df.columns)
#################################### Creating a 'Date' column for daily analysis    #########################################################
df['Date'] = df['timeStamp'].dt.date
print(df['Date'].head())
# Visualize the number of calls over time on a daily basis  
plt.figure(figsize=(16, 9))
df.groupby('Date').size().plot()    
plt.title('8.Number of Calls Over Time (Daily)')
plt.xlabel('Date')
plt.ylabel('Number of Calls')
plt.show()
print(df.columns)   
###########################linear fit on the number of calls per month  #########################################################
by_month = df.groupby('Month').size()   
by_month.plot.line()
plt.title('9.Linear Fit on Number of Calls per Month')    
plt.xlabel('Month')
plt.ylabel('Number of Calls')
plt.show()
print(df.columns)
################################# lineplot of calls for each reason over time   ######################################################### 
# Count number of calls per month and reason
calls_per_month = df.groupby(['Month', 'Reason']).size().reset_index(name='Count')


sns.lmplot(x='Month', y='Count', data=calls_per_month, hue='Reason', aspect=2, height=6)
plt.title('10.Number of 911 Calls per Month by Reason')
plt.xlabel('Month')
plt.ylabel('Number of Calls')
plt.show()
print(df.columns)
######################## sns heatmap USING Unstack method based on hours and calls  #########################################################
dayHour = df.groupby(by=['Day of Week', 'Hour']).count()['Reason'].unstack()
plt.figure(figsize=(12, 6))
sns.heatmap(dayHour, cmap='viridis')    
plt.title('11.Heatmap of Calls by Day of Week and Hour')
plt.show()
print(df.columns)

# -----------------------------------------#
#                LOKESH CHANDRA TATA       #
# ---------------------------------------- #      
