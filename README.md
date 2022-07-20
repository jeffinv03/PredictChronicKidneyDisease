
# Predict Status of Chronic Kidney Disease using Machine Learning and Data Science - Jeffin Varghese (UT Austin B.S. Computer Science)

I decided to build a model to predict status of Chronic Kidney Disease using Machine Learning and Data Science,
After creating a project that could detect E coli based on DNA strands, I thought it would be fitting to take this
experience and build my own model that from the ground up. Over the course of several weeks, and learning Jupyter,
choosing which imports to utilize, and how to preprocess and clean my data, this is the final documentation for my 
project. 

The data was retreived from the public CDC records for Kidney Disease for the Southwestern region of the USA
in 2021. The model will be cross trained with the first quarter of 2022 data as soon as it comes out.

My model was based on papers by [Andrew S Levey](https://www.sciencedirect.com/science/article/pii/S0140673611601785/pdfft?casa_token=8NT6bA1DvQ4AAAAA:emGdR3Ly_DHv_U_dqoJpcbMHFOwE6mc3mzroS-zN88Cr-5_L6h2UMPg-pKv6WUzTOc5Og9kc6uk&md5=694b788cffe32a2b85c6719618e373c9&pid=1-s2.0-S0140673611601785-main.pdf) and the [Centers for Disease Control and Prevention](https://fluoridealert.org/studytracker/38332/)

By Jeffin Varghese

SECTION 1: Perform Data-Preprocessing and Prepate your data for Analysis and Modelling our purpose.

```python
import pandas as pd # used for data extraction
import numpy as np # used for numerical tasks on data
import matplotlib.pyplot as plt # data visualization
import seaborn as sns

df = pd.read_csv(r'/Users/jeffinvarghese/Downloads/Machine Learning/kidney_disease.csv')
df.head()

```
Initial Data: 

<img width="691" alt="Screen Shot 2022-07-19 at 3 42 37 PM" src="https://user-images.githubusercontent.com/97994153/179845828-a2063443-ef0e-4248-a3b7-b75d92f59429.png">


```python

# We can see the data headers are ambiugous (we know bp means blood pressure but what if our reader doesnt?)
# So let's read the data description text file:
columns = pd.read_csv('/Users/jeffinvarghese/Downloads/Machine Learning/data_description.txt', sep='-')
columns = columns.reset_index()

columns.columns = ['cols', 'abb_col_names']
columns

# Now replace the headers 
df.columns=columns['abb_col_names'].values
df.head()

```
We get better data specification: 

<img width="977" alt="Screen Shot 2022-07-19 at 3 51 34 PM" src="https://user-images.githubusercontent.com/97994153/179846080-b8ae9967-e1c4-4534-93f6-c9b1daf99f27.png">

We need to ensure our data is all in the same format as well. We can notice that some of the data, such as white blood cell count, packed cell volume, are stored as object data types, when clearly, looking at the data, they are stored as ints. So we need to convert using a function: 

```python
def convert_dtype(df,feature):
    df[feature] = pd.to_numeric(df[feature], errors='coerce')

features=['packed cell volume', 'white blood cell count', 'red blood cell count']

for feature in features:
    convert_dtype(df,feature)

```
Updated Data Types: (not some are still in object format but that's because they aren't able to be in raw numerical form)

<img width="391" alt="Screen Shot 2022-07-19 at 3 56 06 PM" src="https://user-images.githubusercontent.com/97994153/179846833-ab322ff8-519b-4ce9-b41a-d94b83f3ad73.png">


Finally, the the ID column is redundant because we could just use the index of each column:

```python
df.drop('id', axis=1,inplace=True)
```
SECTION 2: Apply Data Cleaning Techniques

First, let's seperate each column of data into two categories: category (non-numerical) and numerical.

```python
def extract_cat_num(df):
    
    cat_col = [col for col in df.columns if df[col].dtype == 'object']
    num_col = [col for col in df.columns if df[col].dtype != 'object']
    return cat_col,num_col

cat_col,num_col=extract_cat_num(df)

```

For each non-numerical category, figure out the unique values that could be associated with each data point.

```python
for col in cat_col:
    print('{} has {} values '.format(col,df[col].unique()))
    print('\n')
```

We get: 

<img width="649" alt="Screen Shot 2022-07-19 at 4 27 10 PM" src="https://user-images.githubusercontent.com/97994153/179851579-19c7855e-795b-46a7-9200-6a57c74ab5b5.png">


Now we are seeing some 'dirty' data. Notice in the diabetes mellitus category that our data could be: yes, no, \tno , \tyes, or nan. \tno and \tyes are clearly supposed to be yes or no, so we need to clean that up. We could see these features in several other categories. 

```python
df['diabetes mellitus'].replace(to_replace={'\tno':'no','\tyes':'yes'}, inplace=True)

df['coronary artery disease'].replace(to_replace={'\tno':'no','\tyes':'yes'}, inplace=True)

df['class'] = df['class'].replace(to_replace='ckd\t',value='ckd')

# Now double check that our data is clean
for col in cat_col:
    print('{} has {} values '.format(col,df[col].unique()))
    print('\n')
    
```

<img width="555" alt="Screen Shot 2022-07-19 at 4 30 06 PM" src="https://user-images.githubusercontent.com/97994153/179851940-183297ee-eb67-491e-8914-ff7d9f5308fa.png">

CLEAN DATA!!!

SECTION 3: Analyzing distribution of each numerical column and checking label distribution of categorial data

Let's start by visualizing the distribution of each numerical category.

```python
plt.figure(figsize=(30,20))

for i,feature in enumerate(num_col):
    plt.subplot(5, 3, i+1)
    df[feature].hist()
    plt.title(feature)
```
This gives us:

<img width="932" alt="Screen Shot 2022-07-19 at 4 49 51 PM" src="https://user-images.githubusercontent.com/97994153/179854695-e0f1ab4c-dbce-4bed-aafd-3057d0a6ce21.png">


Now that our data is more viewable, let's analyze it. There are several high-postive outliers (we'll deal with this later). For example in the blood pressure graph we see some high-level outliers that may skew our numerical analysis. 

Now let's do label distribution of categorical data:

```python


plt.figure(figsize=(14, 14))

for i,feature in enumerate(cat_col):
    plt.subplot(4, 3, i+1)
    sns.countplot(df[feature])
    
```

<img width="861" alt="Screen Shot 2022-07-19 at 4 51 38 PM" src="https://user-images.githubusercontent.com/97994153/179854887-b83e2c63-118c-4971-a8eb-4ab23a827356.png">

Section 4: Check how columns correlate, and it's impact on Chronic Kidney Disease (class)

```python
# Let's get our raw correlation data
df.corr()
```
<img width="996" alt="Screen Shot 2022-07-19 at 5 22 50 PM" src="https://user-images.githubusercontent.com/97994153/179859010-2e498dbc-3f53-4364-b7a8-07f1c6c04a43.png">

This raw data is hard to read and unless you're a statistician, you can't draw too many strong conclusions from it. So let's convert it to a heatmap:

```python

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
```

<img width="796" alt="Screen Shot 2022-07-19 at 5 23 46 PM" src="https://user-images.githubusercontent.com/97994153/179859016-149e0d0d-90ee-417d-ad1c-5ac295b98fd7.png">

Looking at our heatmap, let's draw some conclusions. The specific gravity column and red blood cell count gives us .58
which is pretty good correlation. Similarly, the specific gravity and hemoglobin has a .6 correlation. This means if 
the specific gravity column increases, there is a 60% chance the hemoglobin also increases.We can also draw some negative correaltions from this data set. For example blood urea and red blood cell count gives 
us -.58. 

Let's group based on the basis of red blood count

```python
df.groupby(['red blood cells', 'class'])['red blood cell count'].agg(['count', 'mean','median','min', 'max'])
```

<img width="491" alt="Screen Shot 2022-07-19 at 5 25 14 PM" src="https://user-images.githubusercontent.com/97994153/179859023-e9f3ffff-7e42-459f-ab4f-c20782a0a053.png">

This table can be interpreted like this: There are 25 people with an abnormal red blood cell count who have Chronic Kidney Disease. The mean red blood cell 
count of those indivuals is 3.8 million cells per microlitre (cells/mcL). Similarly there are 134 people without a 
Chronic Kidney Disease and their average rbc is much higher (5.4 cells/mcL). 


Let's use plotly's data visualization libraries to see this correlation illustrated.
```python
px.violin(df,y='red blood cell count', x='class', color='class')
```

<img width="948" alt="Screen Shot 2022-07-19 at 5 26 58 PM" src="https://user-images.githubusercontent.com/97994153/179859033-f2c8ccf4-4088-4252-91a6-3df7d7d4505b.png">

This handy violin data visualization tool allows us to extract that indivuals with Chronic Kidney Disease have, 
on average,  a much higher red blood cell count then those without CKD. Using this information, we can start to get 
ideas on what our model should focus on and the degree of inclusion.


SECTION 5: Automate our Analysis
    
Based on [this](https://www.banglajol.info/index.php/CMOSHMCJ/article/view/15508/10998) paper by Prof. S. Khanam we see there exists a relationship between hemoglobin and packed cell volume, so lets 
investigate the relationship on that as well. But first let's automate functions for the violin graph and a scatter plot.

```python
# Let's make a function to automate this violin graph:

def violin(col):
    fig = px.violin(df,y=col,x='class',color='class', box=True)
    return fig.show()
    
    
# Let's make another function to automate the scatter plot as well:

def scatters():
    fig = px.scatter(df, x=col1,y=col2, color='class')
    return fig.show
    
    
def kde_plot(feature):
    grid = sns.FacetGrid(df, hue='class',aspect=2)
    grid.map(sns.kdeplot, feature)
    grid.add_legend()

```
SECTION 6: Dealing with missing values

If we don't deal with missing values, our model will incorrectly portray our data and our conclusions. Similarly if we wish to expand our model to include more data in the future, we cannot ensure that data is perfect. We must clean our data and adapt our model to account for this.

```python
df.isna().sum()
```
<img width="345" alt="Screen Shot 2022-07-19 at 7 00 51 PM" src="https://user-images.githubusercontent.com/97994153/179868223-1076ecd1-430b-4182-83b5-3a2c0220fcd5.png">

This is not an insignificant amount of missing data. Particularly with the red blood/white blood cell count, which is critical for our model for Chronic Kidney Disease. The solution was to fill missing values with some value. However, we can't use median/mean/std. dev because if a large amount of our data is missing (over 30%) then our machine learning model, based on normal distribution, will be impacted negatively, so we make a function that fills missing data with a random value that already exists in the data set to keep ratios the same.



```python
def Random_value_Imputation(feature):
    random_sample = df[feature].dropna().sample(df[feature].isnull().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(),feature]= random_sample
        
def impute_mode(feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)

```

```python
# Now let's check missing values in categorical and numerical features and fix it using our newly created function

for col in num_col:
    Random_value_Imputation(col)
    
for col in cat_col:
    impute_mode(col)
```

<img width="478" alt="Screen Shot 2022-07-19 at 7 03 21 PM" src="https://user-images.githubusercontent.com/97994153/179868242-cceeda0c-5d8f-4f2e-b1fd-be49fc804c26.png">

We should also Apply feature encoding technique onto our data. The reason we need this is because whenever our data passes into our ML model, it won't understand non-numerical data. We need to convert these features into numerical categories. Let's see how many different categories we will 
need by running a loop that can check this:

```python
for col in cat_col:
    print('{} has {} categories'.format(col,df[col].nunique()))
```

We get in return:

<img width="337" alt="Screen Shot 2022-07-20 at 9 03 19 AM" src="https://user-images.githubusercontent.com/97994153/180002104-b9a8405e-ba81-42e7-8ab2-a9841f022fa6.png">

Moving onto the Label Encoding, for most data sets we can just assume binary values. For example normal = 0, abnormal = 1

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cat_col:
    df[col] = le.fit_transform(df[col])
    
# Now there is no more categorical data left in our data
df.head()

<img width="1500" alt="Screen Shot 2022-07-20 at 9 05 06 AM" src="https://user-images.githubusercontent.com/97994153/180002453-1b0f29f6-b90a-4f8e-9e16-4dd2751cdc22.png">

```
SECTION 8: Selecting Best Features (using suitable Feature Importance Techniques). 

We need to narrow our 24 features to 10 of the most important features that impact Chronic Kidney Disease. We can use the sklearn import to help us determine that.

```python
from sklearn.feature_selection import SelectKBest

# internally this class will check if probbality values are less than 0.25:

from sklearn.feature_selection import chi2

ind_col= [col for col in df.columns if col!='class']
dep_col = 'class'

# Create our indpendent and dependent variables
X = df[ind_col]
Y = df[dep_col]

X.head() #Independent Variables

#Select the best features depending on probablity values
ordered_ranks_features = SelectKBest(score_func=chi2,k=20) 
ordered_feature = ordered_ranks_features.fit(X,Y)

ordered_feature
# Let's get the ranking:
ordered_feature.scores_
```
<img width="675" alt="Screen Shot 2022-07-20 at 9 09 07 AM" src="https://user-images.githubusercontent.com/97994153/180003400-281ee0b8-4cde-4ede-8c07-542110ea73fc.png">

Let's make this data more user-friendly/readable by converting it into a data frame

```python

datascores = pd.DataFrame(ordered_feature.scores_, columns = ['Score'])
dfcols = pd.DataFrame(X.columns)
features_rank = pd.concat([dfcols, datascores], axis = 1) 
features_rank.columns = ['Features', 'Score'] #Organize Table
features_rank.nlargest(10, 'Score') # We just want the top 10 features
```
<img width="397" alt="Screen Shot 2022-07-20 at 9 10 29 AM" src="https://user-images.githubusercontent.com/97994153/180003652-20aa7663-d1c8-4de6-8c1c-b4df886b2600.png">

So the 10 features we'll build our model on is: ['white blood cell count', 'blood glucose random', 'blood urea',
       'packed cell volume', 'serum creatinine', 'albumin', 'haemoglobin',
       'age', 'sugar', 'ypertension']

SECTION 9: Build a cross validated model and predict accuracy

Now that we have cleaned our data and narrowed down to 10 best features, we can start to train our model. We can use the sklearn model import to do this heavy lifting for us.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_new,Y,random_state=0, test_size=0.25)
# Print how many dimensions of data
print(X_train.shape)
print(X_test.shape)
Y_train.value_counts()
```
Checking whether our data is imbalanced or not is very important. Our program returned 0 imbalanced situations compared to 188. And to double check we ran it again and it returned 1 imbalanced sitaution compared to 112. These are very promising ratios! Now we know it's not imbalanced, we can pass it into a ML model.

```python

!pip install xgboost
from xgboost import XGBClassifier

# Now we can cross train our model
XGBClassifier()


params = {
    'learning_rate':[.05, .20, .25],
    'max-depth':[5, 8, 10],
    'min_child_weight':[1,3,5,7],
    'gamma':[0.0, 0.1, 0.2, 0.4],
    'colsample_bytree':[0.3,0.4,0.7]
}

from sklearn.model_selection import RandomizedSearchCV

classifier = XGBClassifier()
random_search = RandomizedSearchCV(classifier, param_distributions = params,n_iter=5,scoring='roc_auc', n_jobs=-1, cv=5, verbose=3)

random_search.fit(X_train,Y_train)

random_search.best_estimator_

```
These are the parameters our program returned and so we'll pass it in to test our model:

<img width="739" alt="Screen Shot 2022-07-20 at 9 18 49 AM" src="https://user-images.githubusercontent.com/97994153/180005574-283a748c-f6dc-4d94-b153-ff32248bb3b6.png">

```python
# To get the best parameters for our model:
random_search.best_params_

# Now initialize our model
classifier = XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.3,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, gamma=0.0, gpu_id=-1, grow_policy='depthwise',
              importance_type=None, interaction_constraints='',
              learning_rate=0.05, max_depth=10, max_bin=256,
              max_cat_to_onehot=4, max_delta_step=0, max_leaves=0,
              min_child_weight=3, monotone_constraints='()',
              n_estimators=100, n_jobs=0, num_parallel_tree=1, predictor='auto',
              random_state=0, reg_alpha=0)
              
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

confusion_matrix(Y_test, Y_pred)
accuracy_score(Y_test, Y_pred)

```

And finally after a couple minutes of data processing, we got back:

<img width="391" alt="Screen Shot 2022-07-20 at 9 21 38 AM" src="https://user-images.githubusercontent.com/97994153/180006321-24f25455-7966-40fd-b8c3-ac6c3bd695ab.png">


99% accuracy!!! We predicted around 95%-97% accuracy for our model, so this was amazing. And now we've succesfully built our model from the ground up and it's able to input even more patient data to continue to fine tune our model. 













    
    






























