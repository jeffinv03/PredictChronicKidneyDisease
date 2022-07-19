Project Name: Predict Status of Chronic Kidney Disease using Machine Learning and Data Science

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





    
    






























