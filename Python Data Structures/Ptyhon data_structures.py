#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: singhmnprt01
"""

'''

Python data structures:-
    Array (NumPy)  - Python does not have built-in support for Arrays, but Python Lists or numpy can be used instead
    Daatframe (pandas)
    Lists
    Dictionaries
    
'''
##############################################################################
################################## Lists #####################################
##############################################################################

'''
List is a collection which is ordered and changeable. Allows duplicate members.
In Python lists are written with square brackets.

'''

mylist = ['venue','creta','xuv']

## access any item
mylist[2]
mylist[-1]
mylist[0:2] '''The search will start at index 0 (included) and end at index 2 (not included)'''
mylist[:3]
mylist[0] = "selto"

for x in mylist:
    print(x)
    
if "venue" in mylist:
    print("yes")    

len(mylist)    

mylist.append("venue")

mylist.insert(2,"brezza") ''' shifts the list by 1 and insert the value at 2 '''

mylist.remove("brezza")
del mylist[0]
del mylist
mylist.clear()

mylist2 =  mylist.copy() ''' copy a list to another list  '''

list3 = mylist + mylist2 ''' add/join two or more lists '''

mylist = mylist + mylist2 
mylist.extend(mylist2)


mylist.sort() ''' sort the list '''

mylist.reverse() ''' reverse the list '''


##############################################################################
############################### Dictionaries #################################
##############################################################################

'''
A dictionary is a collection which is unordered, changeable and indexed. 
In Python dictionaries are written with curly brackets, and they have keys and values.

A Python dictionary is used to collect key-value pairs. 
You may need Python directories to store unstructured documents. 
On the other hand, Pandas data frames are a two-dimensional table that can be used to store tabular data.

'''
mydisct = {"Name":"Manpreet",
           "Class" : "General",
           "Gender":"Male"}

## Creating a dataframe from a dictinary of lists !

## Lists
list_country = ["India","USA","China","Russia"]
list_GDP = [2,22,16,4]
list_flag =["tricolor","Stars","Star","No Idea"]

## Disctionary
cont_dict = {"Country":list_country,
             "GDP":list_GDP,
             "flag":list_flag}

import pandas as pd

## DataFrame
cont_df = pd.DataFrame(cont_dict)


##############################################################################
################################## NumPy #####################################
##############################################################################

import numpy as np
pip install sklearn

print(np.__version__)
# 1.18.1


# create a numpy 1D array  from 10 to 15
myarray = np.arange(10,16)
myarray


#Create a 2*2 numpy array of all True’s
myarray = np.full((2,2),True, dtype=bool)


#Extract all odd numbers from arr
myarray = np.arange(1,21)
myarray_odd = myarray[myarray % 2 ==1 ]
myarray_odd


#Replace all odd numbers in arr with -1
myarray = np.arange(1,21)
myarray[myarray % 2 ==1 ] = -1 
myarray


#Replace all odd numbers in arr with -1 without changing arr
myarray = np.arange(1,21)
outmyarray_new = np.where(myarray%2==1,-1,myarray)
outmyarray_new
myarray


# Convert a 1D array to a 2D array with 4 rows
myarray = np.arange(1,21)
myarray_2d = myarray.reshape(4,-1) # 2 here is number of rows and -1 let the function decide the columns
myarray_2d


##Stack arrays a and b vertically
myarray1 = np.arange(10).reshape(2,-1)
myarray2 = np.repeat(1,10).reshape(2,-1)

#mthod 1
myarray_stacked = np.concatenate([myarray1,myarray2], axis=0)

#method2

myarray_stacked = np.vstack([myarray1,myarray2])


##Stack arrays a and b horizontally
#method 1
myarray_stacked = np.concatenate([myarray1,myarray2], axis =1 )

#method 2
myarray_stacked = np.hstack([myarray1,myarray2])


#How to get the common items between two python numpy arrays?
a = np.array([1,2,3,4,5,6,7,8,9])
b = np.array([5,6,7,8,5,10,11,12,9])
np.intersect1d(a,b)


# How to get the positions where elements of two arrays match?
np.where([a==b])



#Get all items between 5 and 10 from a
a = np.array([2, 6, 1, 9, 10, 3, 27])
#method 1
output = np.where((a<=10)&(a>=5))
a[output]

#method 2
a[(a<=10)&(a>=5)]


# swap two columns/rows in a 2d array
a = np.arange(9).reshape(3,3) # by default the order of column is 0,1,2 and rows are 0,1,2

a[:,[1,0,2]] # swap 0 with 1 at column level

a[[1,0,2],:] # swap 0 with 1 at row level



# reverse a rows/columns 2d array 
a= np.arange(9).reshape(3,3)
a[::-1]  # rows 
a[:,::-1]  # columns



# reverse a 1d array  -- similar to row reverse
a = np.arange(1,15)
a[::-1]


# Write a NumPy program to create a null vector of size 10 and update sixth value to 11
a = np.zeros(9).reshape(3,3)
a = np.zeros(9)
a[5]= 11
a


################# Linear Algebra with NumPy #######################

# multiplication of 2 matrixes:-
a = np.array([1,2,3,4]).reshape(2,2)
b = np.array([5,6,7,8]).reshape(2,2)
np.dot(a,b)
np.dot(a,b.T)


# Compute the outer product of two given vectors
a = np.array([1,2,3,4]).reshape(2,2)
b = np.array([5,6,7,8]).reshape(2,2)

a*b # normal product
np.outer(a,b) # outer product


#Write a NumPy program to find a matrix or vector norm.
a = np.array([1,2,3,4]).reshape(2,2)
np.shape(a)
norm = np.linalg.norm(a, axis=1, ord=2, keepdims=True)
b = a/norm

## or ##

from sklearn import preprocessing
c = preprocessing.normalize(a,norm='l2')

##----- confusion of axis=1 and axis =0 :-
##https://stackoverflow.com/questions/17079279/how-is-axis-indexed-in-numpys-array


# find inverse of a matrix
a = np.array([1,2,3,4,5,6,7,8,9]).reshape(3,3)
np.linalg.inv(a)


################# Random Number with NumPy ####################
a = np.random.randn(1,5) 
#or
a = np.random.randn(5)


# Write a NumPy program to generate six random integers between 5 and 100.
a = np.random.randint(low=5,high=100, size=6)

#Write a NumPy program to create a 3x3x3 array of radnom values
a= np.random.rand(3,3,3)


# find minimum and maximum value in an array
a = np.random.randn(3,3)
a.min()
a.max()

#sort an array (row wise)
a.sort()
a

# Write a NumPy program to create random vector of size 15 and replace the maximum value by -1.

a = np.random.randn(15)
a[a.argmax()] = -1


# Get the 2 largest values of an array
a = np.random.randn(10)
a[np.argsort(a)[-2::]]



##############################################################################
################################## pandas ####################################
##############################################################################
'''
The primary two components of pandas are the Series and DataFrame.

A Series is essentially a column, and a DataFrame is a multi-dimensional table made up of a collection of Series.
'''

import pandas as pd

#### Structure of a dataframe
'''
    pandas.DataFrame( data, index, columns, dtype, copy)
'''

#### create a dataframe from scratch
# Lists
list_country = ["India","USA","China","Russia"]
list_GDP = [2,22,16,4]
list_flag =["tricolor","Stars","Star","No Idea"]

# Dictionary
cont_dict = {"Country":list_country,
             "GDP":list_GDP,
             "flag":list_flag}

# DataFrame
cont_df = pd.DataFrame(cont_dict, index = ["row1","row2","row3","row4"])

type(cont_df["Country"]) ## a series  -- pandas.core.series.Series
type(cont_df[["Country"]])    ## a DataFrame -- pandas.core.frame.DataFrame 


#### Data Loading from CSV:-
data_loaded  = pd.read_csv("Default_Credit_Card.csv")
data_loaded  = pd.read_csv("Default_Credit_Card.csv", index_col=0  ''' this will male column 0 in csv file as index of the dataframe '''

                           
#### Various Operations on dataframes 

# get top 5 / bottom 5 records
data_loaded.head(5)
data_loaded.tail(5) 

data_loaded.shape                          

data_loaded.describe()


# check number of unique records
data_loaded.index.unique()
data_loaded.index.nunique()


# get column names
data_loaded.columns
my_column_list = list(data_loaded.columns)


# get data column wise, row wise, index wise
data_loaded.loc[[0]] ## this will give where index value is 0
cont_df.loc[["row1"]]
cont_df.loc[["row1","row4"]]

data_loaded.iloc[[0]] ## this will give you the 1st row value, irrespective of its index value
cont_df.iloc[[0,3]]
cont_df.iloc[0:3]


# play with data types
data_loaded.dtypes
data_loaded.LIMIT_BAL.dtypes

data_loaded.LIMIT_BAL = data_loaded.LIMIT_BAL.astype(int) '''LIMIT_BAL column datatype changed to int'''

data_loaded['default.payment.next.month'].value_counts() ''' count catgeories of values of a column in dataframe'''

# rename column(s)
data_loaded.rename(columns = {'default.payment.next.month':'target'},inplace=True)

data_loaded.rename(columns = lambda x : x[0:5]), inplace = True) '''change the name of all the columns to first 5 leters of their originla column word  '''


# sorting 
data_loaded =data_loaded.sort_values('target', ascending = True)


# sort the index values / rearrange the index values if the data has been sorted, merged etc.
data_loaded.index = range(len(data_loaded))


# Drop row(s) or column(s)
data_loaded = data_loaded.drop(['SEX','ID'], axis=1)


# get data insights after grouping the dataframe based on a column
data_loaded.groupby('target').MARRIAGE.min()

data_loaded.groupby('target').LIMIT_BAL.agg(['count','min','max','mean'])


# data filter
data_loaded[data_loaded.PAY_AMT1 > 23000];

data_loaded[data_loaded.PAY_AMT1 > 23000]['LIMIT_BAL'] ''' limit bal of customers whose pay_amt1 is more than 23000 '''

data_loaded[data_loaded.PAY_AMT1 > 23000][['LIMIT_BAL','MARRIAGE']] ''' limit bal and marriage status of customers whose pay_amt1 is more than 23000 '''

data_loaded.loc[data_loaded.index.isin([1,2,45,34]),:]

data_loaded.loc[:, data_loaded.columns  != 'target']


# missing values 
data_loaded.isnull()
data_loaded.isnull().sum()
data_loaded.isnull().any()
data_loaded.isnull().values.any()
data_loaded.isnull().values.any().sum()


# merge vs concat 

'''
concat() simply stacks multiple DataFrame together either vertically, or stitches horizontally after aligning on index
merge() first aligns two DataFrame' selected common column(s) or index, and then pick up the remaining columns from the aligned rows of each DataFrame.

'''

df1= pd.DataFrame({'Key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})

df2 = pd.DataFrame({'Key': ['a','b','d'],'data2':range(3)})



## Concat
    # by rows
pd.concat([df1,df2],axis=0)

    # by columns
pd.concat([df1,df2],axis=1)


## merge ''' merge is equivalent of join in sql '''
pd.merge(df1, df2, on = 'Key')                            

pd.merge(df1, df2, on = 'Key', how='inner') 
pd.merge(df1, df2, on = 'Key', how='left') 
pd.merge(df1, df2, on = 'Key', how='right') 
pd.merge(df1, df2, on = 'Key', how='outer') 
