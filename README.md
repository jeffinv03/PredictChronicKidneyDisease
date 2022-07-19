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
























