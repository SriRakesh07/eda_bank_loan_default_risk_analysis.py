#!/usr/bin/env python
# coding: utf-8

# ![](https://apps.tsn.go.tz/public/uploads/fd2b657ba7ea5c49c5473dc452481cb0.png)

# <h3 >   
#       <font color = black >
#             <span style='font-family:Georgia'>
#             Introduction:
#             </span>   
#         </font>    
# </h3>
# <p>
#     <span style='font-family:Georgia'>
#     This case study aims to give an idea of applying EDA in a real business scenario. In this case study, we will develop a basic understanding of risk analytics in banking and financial services and understand how data is used to minimise the risk of losing money while lending to customers.
#     </span>
# </p>   
# <hr>
# <h3>
#     <font color = black >
#         <span style='font-family:Georgia'>
#             Business Understanding:
#             </span>   
#         </font>    
# </h3>
# <p>
#     <span style='font-family:Georgia'>
#     The loan providing companies find it hard to give loans to the people due to their insufficient or non-existent credit history. Because of that, some consumers use it as their advantage by becoming a defaulter. Suppose you work for a consumer finance company which specialises in lending various types of loans to urban customers. You have to use EDA to analyse the patterns present in the data. This will ensure that the applicants capable of repaying the loan are not rejected.<br>
#         When the company receives a loan application, the company has to decide for loan approval based on the applicant’s profile. Two types of risks are associated with the bank’s decision:
#     </span>
# </p>
# <ul>
#     <span style='font-family:Georgia'>
#         <li>If the applicant is likely to repay the loan, then not approving the loan results in a loss of business to the company</li>
#         <li>If the applicant is not likely to repay the loan, i.e. he/she is likely to default, then approving the loan may lead to a financial loss for the company.</li>
#     </span>
# </ul>
#     
# <p><span style='font-family:Georgia'>The data given below contains the information about the loan application at the time of applying for the loan. It contains two types of scenarios:</span></p>
# <ul>
#     <span style='font-family:Georgia'> 
#         <li><b>The client with payment difficulties:</b> he/she had late payment more than X days on at least one of the first Y instalments of the loan in our sample</li>
#         <li><b>All other cases:</b> All other cases when the payment is paid on time</li>
#     </span>
# </ul>
#     
# <p><span style='font-family:Georgia'>When a client applies for a loan, there are four types of decisions that could be taken by the client/company):</span></p>
# 
# <ol>
#     <span style='font-family:Georgia'>
#         <li><b>Approved:</b> The Company has approved loan Application</li>
#         <li><b>Cancelled:</b> The client cancelled the application sometime during approval. Either the client changed her/his mind about the loan or in some cases due to a higher risk of the client he received worse pricing which he did not want.</li>
#         <li><b>Refused:</b> The company had rejected the loan (because the client does not meet their requirements etc.)</li>
#         <li><b>Unused offer:</b>  Loan has been cancelled by the client but on different stages of the process.</li>
#     </span>
# </ol>
# <hr>
# <h3>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#             Business Objective:
#             </span>   
#         </font>    
# </h3>
# <p>
#     <span style='font-family:Georgia'>
#         This case study aims to identify patterns which indicate if a client has difficulty paying their installments which may be used for taking actions such as denying the loan, reducing the amount of loan, lending (to risky applicants) at a higher interest rate, etc. This will ensure that the consumers capable of repaying the loan are not rejected. Identification of such applicants using EDA is the aim of this case study.<br>
#         In other words, the company wants to understand the driving factors (or driver variables) behind loan default, i.e. the variables which are strong indicators of default.  The company can utilise this knowledge for its portfolio and risk assessment.
#     </span>
# </p>

# <h3 name='libraries'>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#              Import Python Libraries:
#             </span>   
#         </font>    
# </h3>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')

# setting up plot style 
style.use('seaborn-poster')
style.use('fivethirtyeight')


# <h3 name='libraries'>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#              Supress Warnings:
#             </span>   
#         </font>    
# </h3>

# In[2]:


import warnings
warnings.filterwarnings('ignore')


# <h3 name='libraries'>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#              Adjusting Jupyer Views:
#             </span>   
#         </font>    
# </h3>

# In[3]:


pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 600)
pd.set_option('display.width', 1200)
pd.set_option('display.expand_frame_repr', False)


# <a id="import"></a>
# <h2>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#              Read & Understand the data
#             </span>   
#         </font>    
# </h2>

# <a id="input"></a>
# <h3 name='libraries'>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#              Importing the input files
#             </span>   
#         </font>    
# </h3>

# In[4]:


import os
for dirname, _, filenames in os.walk('/user/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[5]:


New_DF = pd.read_csv('/Users/srirakeshnagasai/Downloads/application_data.csv')
previousDF = pd.read_csv('/Users/srirakeshnagasai/Downloads/previous_application.csv')
New_DF.head()


# In[6]:


New_DF.head()


# <a id="inspect"></a>
# <h3 name='libraries'>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#              Inspect Data Frames
#             </span>   
#         </font>    
# </h3>

# In[7]:


# Database dimension
print("Database dimension - New_DF     :",New_DF.shape)
print("Database dimension - previousDF :",previousDF.shape)

#Database size
print("Database size - New_DF          :",New_DF.size)
print("Database size - previousDF      :",previousDF.size)


# In[8]:


# Database column types
New_DF.info(verbose=True)


# In[9]:


previousDF.info(verbose=True)


# In[10]:


# Checking the numeric variables of the dataframes
New_DF.describe()


# In[11]:


previousDF.describe()


# <a id="clean"></a>
# <h2>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#             Data Cleaning & Manipulation
#             </span>   
#         </font>    
# </h2>

# <a id="null"></a>
# <h3>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#              Null Value Calculation
#             </span>   
#         </font>    
# </h3>

# <h4>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#              applicationDF Missing values
#             </span>   
#         </font>    
# </h4>

# In[12]:


# % null value in each column
round(New_DF.isnull().sum() / New_DF.shape[0] * 100.00,2)


# In[13]:


null_New_DF = pd.DataFrame((New_DF.isnull().sum())*100/New_DF.shape[0]).reset_index()
null_New_DF.columns = ['Column Name', 'Null Values Percentage']
fig = plt.figure(figsize=(18,9))
ax = sns.pointplot(x="Column Name",y="Null Values Percentage",data=null_New_DF,color='green')
plt.xticks(rotation =90,fontsize =8)
ax.axhline(35, ls='--',color='red')
plt.title("Percentage of Missing values in application data")
plt.ylabel("Null Values PERCENTAGE")
plt.xlabel("COLUMNS")
plt.show()


# In[14]:


# more than or equal to 35% empty rows columns
nullcol_35_New_DF = null_New_DF[null_New_DF["Null Values Percentage"]>=35]
nullcol_35_New_DF


# In[15]:


# How many columns have more than or euqal to 35% null values ?
len(nullcol_35_New_DF)


# <h4>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#              previousDF Missing Values
#             </span>   
#         </font>    
# </h4>

# In[16]:


# checking the null value % of each column in previousDF dataframe
round(previousDF.isnull().sum() / previousDF.shape[0] * 100.00,2)


# In[17]:


null_previousDF = pd.DataFrame((previousDF.isnull().sum())*100/previousDF.shape[0]).reset_index()
null_previousDF.columns = ['Column Name', 'Null Values Percentage']
fig = plt.figure(figsize=(18,6))
ax = sns.pointplot(x="Column Name",y="Null Values Percentage",data=null_previousDF,color ='green')
plt.xticks(rotation =90,fontsize =8)
ax.axhline(35, ls='--',color='red')
plt.title("Percentage of Missing values in previousDF data")
plt.ylabel("Null Values PERCENTAGE")
plt.xlabel("COLUMNS")
plt.show()


# In[18]:


# more than or equal to 35% empty rows columns
nullcol_35_previous = null_previousDF[null_previousDF["Null Values Percentage"]>=35]
nullcol_35_previous


# In[19]:


# How many columns have more than or euqal to 35% null values ?
len(nullcol_35_previous)


# <a id="clean1"></a>
# <h3>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#             Analyze & Delete Unnecessary Columns in applicationDF
#             </span>   
#         </font>    
# </h3>

# <h4>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#              EXT_SOURCE_X
#             </span>   
#         </font>    
# </h4>

# In[20]:


# Checking correlation of EXT_SOURCE_X columns vs TARGET column
Source = New_DF[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3","TARGET"]]
source_corr = Source.corr()
ax = sns.heatmap(source_corr,
            xticklabels=source_corr.columns,
            yticklabels=source_corr.columns,
            annot = True,
            cmap ="RdYlGn")


# In[21]:


# create a list of columns that needs to be dropped including the columns with >40% null values
Unwanted_New_DF = nullcol_35_New_DF["Column Name"].tolist()+ ['EXT_SOURCE_2','EXT_SOURCE_3'] 
# as EXT_SOURCE_1 column is already included in nullcol_35_application 
len(Unwanted_New_DF)


# <h4>   
#       <font color = darkgreen >
#             <span style='font-family:Georgia'>
#             4.2.2 Flag Document
#             </span>   
#         </font>    
# </h4>

# In[22]:


# Checking the relevance of Flag_Document and whether it has any relation with loan repayment status
col_Doc = [ 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6','FLAG_DOCUMENT_7', 
           'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
           'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
           'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
df_flag = New_DF[col_Doc+["TARGET"]]

length = len(col_Doc)

df_flag["TARGET"] = df_flag["TARGET"].replace({1:"Defaulter",0:"Repayer"})

fig = plt.figure(figsize=(22,25))

for i,j in itertools.zip_longest(col_Doc,range(length)):
    plt.subplot(5,4,j+1)
    ax = sns.countplot(df_flag[i],hue=df_flag["TARGET"],palette=["r","g"])
    plt.yticks(fontsize=8)
    plt.xlabel("")
    plt.ylabel("")
    plt.title(i)


# In[23]:


# Including the flag documents for dropping the Document columns
col_Doc.remove('FLAG_DOCUMENT_3') 
Unwanted_New_DF = Unwanted_New_DF + col_Doc
len(Unwanted_New_DF)


# <h4>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#              Contact Parameters
#             </span>   
#         </font>    
# </h4>

# In[24]:


# checking is there is any correlation between mobile phone, work phone etc, email, Family members and Region rating
contact_col = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
       'FLAG_PHONE', 'FLAG_EMAIL','TARGET']
Contact_corr = New_DF[contact_col].corr()
fig = plt.figure(figsize=(8,8))
ax = sns.heatmap(Contact_corr,
            xticklabels=Contact_corr.columns,
            yticklabels=Contact_corr.columns,
            annot = True,
            cmap ="RdYlGn",
            linewidth=1)


# In[25]:


# including the 6 FLAG columns to be deleted
contact_col.remove('TARGET') 
Unwanted_New_DF = Unwanted_New_DF + contact_col
len(Unwanted_New_DF)


# In[26]:


# Dropping the unnecessary columns from applicationDF
New_DF.drop(labels=Unwanted_New_DF,axis=1,inplace=True)


# In[27]:


# Inspecting the dataframe after removal of unnecessary columns
New_DF.shape


# In[28]:


# inspecting the column types after removal of unnecessary columns
New_DF.info()


# <a id="clean2"></a>
# <h3>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#             Analyze & Delete Unnecessary Columns in previousDF
#             </span>   
#         </font>    
# </h3>

# In[29]:


# Getting the 11 columns which has more than 35% unknown
Unwanted_previous = nullcol_35_previous["Column Name"].tolist()
Unwanted_previous


# In[30]:


# Listing down columns which are not needed
Unnecessary_previous = ['WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START',
                        'FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY']


# In[31]:


Unwanted_previous = Unwanted_previous + Unnecessary_previous
len(Unwanted_previous)


# In[32]:


# Dropping the unnecessary columns from previous
previousDF.drop(labels=Unwanted_previous,axis=1,inplace=True)
# Inspecting the dataframe after removal of unnecessary columns
previousDF.shape


# In[33]:


# inspecting the column types after after removal of unnecessary columns
previousDF.info()


# <a id="stdval"></a>
# <h3>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#               Standardize Values
#             </span>   
#         </font>    
# </h3>

# In[34]:


# Converting Negative days to positive days

date_col = ['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH']

for col in date_col:
    New_DF[col] = abs(New_DF[col])


# In[35]:


# Binning Numerical Columns to create a categorical column

# Creating bins for income amount
New_DF['AMT_INCOME_TOTAL']=New_DF['AMT_INCOME_TOTAL']/100000

bins = [0,1,2,3,4,5,6,7,8,9,10,11]
slot = ['0-100K','100K-200K', '200k-300k','300k-400k','400k-500k','500k-600k','600k-700k','700k-800k','800k-900k','900k-1M', '1M Above']

New_DF['AMT_INCOME_RANGE']=pd.cut(New_DF['AMT_INCOME_TOTAL'],bins,labels=slot)


# In[36]:


New_DF['AMT_INCOME_RANGE'].value_counts(normalize=True)*100


# In[37]:


# Creating bins for Credit amount
New_DF['AMT_CREDIT']=New_DF['AMT_CREDIT']/100000

bins = [0,1,2,3,4,5,6,7,8,9,10,100]
slots = ['0-100K','100K-200K', '200k-300k','300k-400k','400k-500k','500k-600k','600k-700k','700k-800k',
       '800k-900k','900k-1M', '1M Above']

New_DF['AMT_CREDIT_RANGE']=pd.cut(New_DF['AMT_CREDIT'],bins=bins,labels=slots)


# In[38]:


#checking the binning of data and % of data in each category
New_DF['AMT_CREDIT_RANGE'].value_counts(normalize=True)*100


# In[39]:


# Creating bins for Age
New_DF['AGE'] = New_DF['DAYS_BIRTH'] // 365
bins = [0,20,30,40,50,100]
slots = ['0-20','20-30','30-40','40-50','50 above']

New_DF['AGE_GROUP']=pd.cut(New_DF['AGE'],bins=bins,labels=slots)


# In[40]:


#checking the binning of data and % of data in each category
New_DF['AGE_GROUP'].value_counts(normalize=True)*100


# In[41]:


# Creating bins for Employement Time
New_DF['YEARS_EMPLOYED'] = New_DF['DAYS_EMPLOYED'] // 365
bins = [0,5,10,20,30,40,50,60,150]
slots = ['0-5','5-10','10-20','20-30','30-40','40-50','50-60','60 above']

New_DF['EMPLOYMENT_YEAR']=pd.cut(New_DF['YEARS_EMPLOYED'],bins=bins,labels=slots)


# In[42]:


#checking the binning of data and % of data in each category
New_DF['EMPLOYMENT_YEAR'].value_counts(normalize=True)*100


# In[43]:


#Checking the number of unique values each column possess to identify categorical columns
New_DF.nunique().sort_values()


# <a id="dconv"></a>
# <h3>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#               Data Type Conversion
#             </span>   
#         </font>    
# </h3>

# In[44]:


# inspecting the column types if they are in correct data type using the above result.
New_DF.info()


# In[45]:


#Conversion of Object and Numerical columns to Categorical Columns
categorical_columns = ['NAME_CONTRACT_TYPE','CODE_GENDER','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
                       'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START',
                       'ORGANIZATION_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','LIVE_CITY_NOT_WORK_CITY',
                       'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','REG_REGION_NOT_WORK_REGION',
                       'LIVE_REGION_NOT_WORK_REGION','REGION_RATING_CLIENT','WEEKDAY_APPR_PROCESS_START',
                       'REGION_RATING_CLIENT_W_CITY'
                      ]
for col in categorical_columns:
    New_DF[col] =pd.Categorical(New_DF[col])


# In[46]:


# inspecting the column types if the above conversion is reflected
New_DF.info()


# <h4>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#               Standardize Values for previousDF
#             </span>   
#         </font>    
# </h4>

# In[47]:


#Checking the number of unique values each column possess to identify categorical columns
previousDF.nunique().sort_values() 


# In[48]:


# inspecting the column types if the above conversion is reflected
previousDF.info()


# In[49]:


#Converting negative days to positive days 
previousDF['DAYS_DECISION'] = abs(previousDF['DAYS_DECISION'])


# In[50]:


#age group calculation e.g. 388 will be grouped as 300-400
previousDF['DAYS_DECISION_GROUP'] = (previousDF['DAYS_DECISION']-(previousDF['DAYS_DECISION'] % 400)).astype(str)+'-'+ ((previousDF['DAYS_DECISION'] - (previousDF['DAYS_DECISION'] % 400)) + (previousDF['DAYS_DECISION'] % 400) + (400 - (previousDF['DAYS_DECISION'] % 400))).astype(str)


# In[51]:


previousDF['DAYS_DECISION_GROUP'].value_counts(normalize=True)*100


# In[52]:


#Converting Categorical columns from Object to categorical 
Catgorical_col_p = ['NAME_CASH_LOAN_PURPOSE','NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE',
                    'CODE_REJECT_REASON','NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY','NAME_PORTFOLIO',
                   'NAME_PRODUCT_TYPE','CHANNEL_TYPE','NAME_SELLER_INDUSTRY','NAME_YIELD_GROUP','PRODUCT_COMBINATION',
                    'NAME_CONTRACT_TYPE','DAYS_DECISION_GROUP']

for col in Catgorical_col_p:
    previousDF[col] =pd.Categorical(previousDF[col])


# In[53]:


# inspecting the column types after conversion
previousDF.info()


# <a id="impute"></a>
# <h3>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#              Null Value Data Imputation
#             </span>   
#         </font>    
# </h3>

# <h4>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#               Imputing Null Values in applicationDF
#             </span>   
#         </font>    
# </h4>

# In[54]:


# checking the null value % of each column in applicationDF dataframe
round(New_DF.isnull().sum() / New_DF.shape[0] * 100.00,2)


# <p>
#     <span style='font-family:Georgia'>
#            Impute categorical variable 'NAME_TYPE_SUITE' which has lower null percentage(0.42%) with the most frequent category using mode()[0]:
#     </span>
# </p>
# 
#     

# In[55]:


New_DF['NAME_TYPE_SUITE'].describe()


# In[56]:


New_DF['NAME_TYPE_SUITE'].fillna((New_DF['NAME_TYPE_SUITE'].mode()[0]),inplace = True)


# In[57]:



New_DF['OCCUPATION_TYPE'] = New_DF['OCCUPATION_TYPE'].cat.add_categories('Unknown')
New_DF['OCCUPATION_TYPE'].fillna('Unknown', inplace =True) 


# In[58]:


New_DF[['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY',
               'AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
               'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']].describe()


# In[59]:


amount = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
         'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']

for col in amount:
    New_DF[col].fillna(New_DF[col].median(),inplace = True)


# In[60]:


# checking the null value % of each column in previousDF dataframe
round(New_DF.isnull().sum() / previousDF.shape[0] * 100.00,2)


# > <h4>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#               Imputing Null Values in previousDF
#             </span>   
#         </font>    
# </h4>

# In[61]:


# checking the null value % of each column in previousDF dataframe
round(previousDF.isnull().sum() / previousDF.shape[0] * 100.00,2)


# In[62]:


plt.figure(figsize=(6,6))
sns.kdeplot(previousDF['AMT_ANNUITY'])
plt.show()


# In[63]:


previousDF['AMT_ANNUITY'].fillna(previousDF['AMT_ANNUITY'].median(),inplace = True)


# In[64]:


plt.figure(figsize=(6,6))
sns.kdeplot(previousDF['AMT_GOODS_PRICE'][pd.notnull(previousDF['AMT_GOODS_PRICE'])])
plt.show()


# <p>
#     <span style='font-family:Georgia'>
#            There are several peaks along the distribution. Let's impute using the mode, mean and median and see if the distribution is still about the same.
#     </span>
# </p>

# In[65]:


statsDF = pd.DataFrame() # new dataframe with columns imputed with mode, median and mean
statsDF['AMT_GOODS_PRICE_mode'] = previousDF['AMT_GOODS_PRICE'].fillna(previousDF['AMT_GOODS_PRICE'].mode()[0])
statsDF['AMT_GOODS_PRICE_median'] = previousDF['AMT_GOODS_PRICE'].fillna(previousDF['AMT_GOODS_PRICE'].median())
statsDF['AMT_GOODS_PRICE_mean'] = previousDF['AMT_GOODS_PRICE'].fillna(previousDF['AMT_GOODS_PRICE'].mean())

cols = ['AMT_GOODS_PRICE_mode', 'AMT_GOODS_PRICE_median','AMT_GOODS_PRICE_mean']

plt.figure(figsize=(18,10))
plt.suptitle('Distribution of Original data vs imputed data')
plt.subplot(221)
sns.distplot(previousDF['AMT_GOODS_PRICE'][pd.notnull(previousDF['AMT_GOODS_PRICE'])]);
for i in enumerate(cols): 
    plt.subplot(2,2,i[0]+2)
    sns.distplot(statsDF[i[1]])


# <div class="alert alert-block alert-info">
#     <span style='font-family:Georgia'>
#         <b>Insight: </b> <br>The original distribution is closer with the distribution of data imputed with mode in this case
#     </span>    
# </div>

# In[66]:


previousDF['AMT_GOODS_PRICE'].fillna(previousDF['AMT_GOODS_PRICE'].mode()[0], inplace=True)


# <p>
#     <span style='font-family:Georgia'>
#            Impute CNT_PAYMENT with 0 as the NAME_CONTRACT_STATUS for these indicate that most of these loans were not started:
#     </span>
# </p>

# In[67]:


previousDF.loc[previousDF['CNT_PAYMENT'].isnull(),'NAME_CONTRACT_STATUS'].value_counts()


# In[68]:


previousDF['CNT_PAYMENT'].fillna(0,inplace = True)


# In[69]:


# checking the null value % of each column in previousDF dataframe
round(previousDF.isnull().sum() / previousDF.shape[0] * 100.00,2)


# <a id="outlier"></a>
# <h3>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#              Identifying the outliers
#             </span>   
#         </font>    
# </h3>

# <p>
#     <span style='font-family:Georgia'>
#            Finding outlier information in applicationDF
#     </span>
# </p>

# In[70]:


plt.figure(figsize=(22,10))

app_outlier_col_1 = ['AMT_ANNUITY','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_GOODS_PRICE','DAYS_EMPLOYED']
app_outlier_col_2 = ['CNT_CHILDREN','DAYS_BIRTH']
for i in enumerate(app_outlier_col_1):
    plt.subplot(2,4,i[0]+1)
    sns.boxplot(y=New_DF[i[1]])
    plt.title(i[1])
    plt.ylabel("")

for i in enumerate(app_outlier_col_2):
    plt.subplot(2,4,i[0]+6)
    sns.boxplot(y=New_DF[i[1]])
    plt.title(i[1])
    plt.ylabel("")


# In[71]:


New_DF[['AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'DAYS_BIRTH','CNT_CHILDREN','DAYS_EMPLOYED']].describe()


# <p>
#     <span style='font-family:Georgia'>
#            Finding outlier information in previousDF
#     </span>
# </p>

# In[72]:


plt.figure(figsize=(22,8))

prev_outlier_col_1 = ['AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','AMT_GOODS_PRICE','SELLERPLACE_AREA']
prev_outlier_col_2 = ['SK_ID_CURR','DAYS_DECISION','CNT_PAYMENT']
for i in enumerate(prev_outlier_col_1):
    plt.subplot(2,4,i[0]+1)
    sns.boxplot(y=previousDF[i[1]])
    plt.title(i[1])
    plt.ylabel("")

for i in enumerate(prev_outlier_col_2):
    plt.subplot(2,4,i[0]+6)
    sns.boxplot(y=previousDF[i[1]])
    plt.title(i[1])
    plt.ylabel("") 


# In[73]:


previousDF[['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'SELLERPLACE_AREA','CNT_PAYMENT','DAYS_DECISION']].describe()


# <a id="analysis"></a>
# <h2>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#             Data Analysis
#             </span>   
#         </font>    
# </h2>

# <a id="imbalance"></a>
# <h3>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#               Imbalance Analysis
#             </span>   
#         </font>    
# </h3>

# In[74]:


Imbalance = New_DF["TARGET"].value_counts().reset_index()

plt.figure(figsize=(10,4))
x= ['Repayer','Defaulter']
sns.barplot(x,"TARGET",data = Imbalance,palette= ['g','r'])
plt.xlabel("Loan Repayment Status")
plt.ylabel("Count of Repayers & Defaulters")
plt.title("Imbalance Plotting")
plt.show()


# In[75]:


count_0 = Imbalance.iloc[0]["TARGET"]
count_1 = Imbalance.iloc[1]["TARGET"]
count_0_perc = round(count_0/(count_0+count_1)*100,2)
count_1_perc = round(count_1/(count_0+count_1)*100,2)

print('Ratios of imbalance in percentage with respect to Repayer and Defaulter datas are: %.2f and %.2f'%(count_0_perc,count_1_perc))
print('Ratios of imbalance in relative with respect to Repayer and Defaulter datas is %.2f : 1 (approx)'%(count_0/count_1))


# <a id="oltfunc"></a>
# <h3>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#               Plotting Functions
#             </span>   
#         </font>    
# </h3>

# <p>
#     <span style='font-family:Georgia'>
#            Following are the common functions customized to perform uniform anaysis that is called for all plots:
#     </span>
# </p>

# In[76]:


# function for plotting repetitive countplots in univariate categorical analysis on applicationDF
# This function will create two subplots: 
# 1. Count plot of categorical column w.r.t TARGET; 
# 2. Percentage of defaulters within column

def univariate_categorical(feature,ylog=False,label_rotation=False,horizontal_layout=True):
    temp = New_DF[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = New_DF[[feature, 'TARGET']].groupby([feature],as_index=False).mean()
    cat_perc["TARGET"] = cat_perc["TARGET"]*100
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20,24))
        
    # 1. Subplot 1: Count plot of categorical column
    # sns.set_palette("Set2")
    s = sns.countplot(ax=ax1, 
                    x = feature, 
                    data=New_DF,
                    hue ="TARGET",
                    order=cat_perc[feature],
                    palette=['g','r'])
    
    # Define common styling
    ax1.set_title(feature, fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'}) 
    ax1.legend(['Repayer','Defaulter'])
    
    # If the plot is not readable, use the log scale.
    if ylog:
        ax1.set_yscale('log')
        ax1.set_ylabel("Count (log)",fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})   
    
    
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    # 2. Subplot 2: Percentage of defaulters within the categorical column
    s = sns.barplot(ax=ax2, 
                    x = feature, 
                    y='TARGET', 
                    order=cat_perc[feature], 
                    data=cat_perc,
                    palette='Set2')
    
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.ylabel('Percent of Defaulters [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_title(feature + " Defaulter %", fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 

    plt.show();


# In[77]:


# function for plotting repetitive countplots in bivariate categorical analysis

def bivariate_bar(x,y,df,hue,figsize):
    
    plt.figure(figsize=figsize)
    sns.barplot(x=x,
                  y=y,
                  data=df, 
                  hue=hue, 
                  palette =['g','r'])     
        
    # Defining aesthetics of Labels and Title of the plot using style dictionaries
    plt.xlabel(x,fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})    
    plt.ylabel(y,fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})    
    plt.title(col, fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 
    plt.xticks(rotation=90, ha='right')
    plt.legend(labels = ['Repayer','Defaulter'])
    plt.show()


# In[78]:


# function for plotting repetitive rel plots in bivaritae numerical analysis on applicationDF

def bivariate_rel(x,y,data, hue, kind, palette, legend,figsize):
    
    plt.figure(figsize=figsize)
    sns.relplot(x=x, 
                y=y, 
                data=New_DF, 
                hue="TARGET",
                kind=kind,
                palette = ['g','r'],
                legend = False)
    plt.legend(['Repayer','Defaulter'])
    plt.xticks(rotation=90, ha='right')
    plt.show()


# In[79]:


#function for plotting repetitive countplots in univariate categorical analysis on the merged df

def univariate_merged(col,df,hue,palette,ylog,figsize):
    plt.figure(figsize=figsize)
    ax=sns.countplot(x=col, 
                  data=df,
                  hue= hue,
                  palette= palette,
                  order=df[col].value_counts().index)
    

    if ylog:
        plt.yscale('log')
        plt.ylabel("Count (log)",fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})     
    else:
        plt.ylabel("Count",fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})       

    plt.title(col , fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 
    plt.legend(loc = "upper right")
    plt.xticks(rotation=90, ha='right')
    
    plt.show()


# In[80]:


# Function to plot point plots on merged dataframe

def merged_pointplot(x,y):
    plt.figure(figsize=(8,4))
    sns.pointplot(x=x, 
                  y=y, 
                  hue="TARGET", 
                  data=loan_process_df,
                  palette =['g','r'])
   # plt.legend(['Repayer','Defaulter'])


# <a id="catvar"></a>
# <h3>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#              Categorical Variables Analysis
#             </span>   
#         </font>    
# </h3>

# <h4>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#               Segmented Univariate Analysis
#             </span>   
#         </font>    
# </h4>

# In[81]:


# Checking the contract type based on loan repayment status
univariate_categorical('NAME_CONTRACT_TYPE',True)


# In[82]:


# Checking the type of Gender on loan repayment status
univariate_categorical('CODE_GENDER')


# In[83]:


# Checking if owning a car is related to loan repayment status
univariate_categorical('FLAG_OWN_CAR')


# In[84]:


# Checking if owning a realty is related to loan repayment status
univariate_categorical('FLAG_OWN_REALTY')


# In[85]:


# Analyzing Housing Type based on loan repayment status
univariate_categorical("NAME_HOUSING_TYPE",True,True,True)


# In[86]:


# Analyzing Family status based on loan repayment status
univariate_categorical("NAME_FAMILY_STATUS",False,True,True)


# In[87]:


# Analyzing Education Type based on loan repayment status
univariate_categorical("NAME_EDUCATION_TYPE",True,True,True)


# In[88]:


# Analyzing Income Type based on loan repayment status
univariate_categorical("NAME_INCOME_TYPE",True,True,False)


# In[89]:


# Analyzing Region rating where applicant lives based on loan repayment status
univariate_categorical("REGION_RATING_CLIENT",False,False,True)


# In[90]:


# Analyzing Occupation Type where applicant lives based on loan repayment status
univariate_categorical("OCCUPATION_TYPE",False,True,False)


# In[91]:


# Checking Loan repayment status based on Organization type
univariate_categorical("ORGANIZATION_TYPE",True,True,False)


# In[92]:


# Analyzing Flag_Doc_3 submission status based on loan repayment status
univariate_categorical("FLAG_DOCUMENT_3",False,False,True)


# In[93]:


# Analyzing Age Group based on loan repayment status
univariate_categorical("AGE_GROUP",False,False,True)


# In[94]:


# Analyzing Employment_Year based on loan repayment status
univariate_categorical("EMPLOYMENT_YEAR",False,False,True)


# In[95]:


# Analyzing Amount_Credit based on loan repayment status
univariate_categorical("AMT_CREDIT_RANGE",False,False,False)


# In[96]:


# Analyzing Amount_Income Range based on loan repayment status
univariate_categorical("AMT_INCOME_RANGE",False,False,False)


# In[97]:


# Analyzing Number of children based on loan repayment status
univariate_categorical("CNT_CHILDREN",True)


# In[98]:


# Analyzing Number of family members based on loan repayment status
univariate_categorical("CNT_FAM_MEMBERS",True, False, False)


# <h4>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#               Categorical Bi/Multivariate Analysis
#             </span>   
#         </font>    
# </h4>

# In[99]:


New_DF.groupby('NAME_INCOME_TYPE')['AMT_INCOME_TOTAL'].describe()


# In[100]:


# Income type vs Income Amount Range
bivariate_bar("NAME_INCOME_TYPE","AMT_INCOME_TOTAL",New_DF,"TARGET",(18,10))


# <a id="numvar"></a>
# <h3>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#              Numeric Variables Analysis
#             </span>   
#         </font>    
# </h3>

# <h4>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#               Bifurcating the applicationDF dataframe based on Target value 0 and 1 for correlation and other analysis
#             </span>   
#         </font>    
# </h4>

# In[101]:


New_DF.columns


# In[102]:


# Bifurcating the New_DF dataframe based on Target value 0 and 1 for correlation and other analysis
cols_for_correlation = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
                        'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 
                        'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                        'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
                        'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
                        'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START',
                        'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 
                        'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE',
                        'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_3', 
                        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
                        'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']


Repayer_df = New_DF.loc[New_DF['TARGET']==0, cols_for_correlation] # Repayers
Defaulter_df = New_DF.loc[New_DF['TARGET']==1, cols_for_correlation] # Defaulters


# <h4>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#               Correlation between numeric variable
#             </span>   
#         </font>    
# </h4>

# In[103]:


# Getting the top 10 correlation for the Repayers data
corr_repayer = Repayer_df.corr()
corr_repayer = corr_repayer.where(np.triu(np.ones(corr_repayer.shape),k=1).astype(np.bool))
corr_df_repayer = corr_repayer.unstack().reset_index()
corr_df_repayer.columns =['VAR1','VAR2','Correlation']
corr_df_repayer.dropna(subset = ["Correlation"], inplace = True)
corr_df_repayer["Correlation"]=corr_df_repayer["Correlation"].abs() 
corr_df_repayer.sort_values(by='Correlation', ascending=False, inplace=True) 
corr_df_repayer.head(10)


# In[104]:


fig = plt.figure(figsize=(12,12))
ax = sns.heatmap(Repayer_df.corr(), cmap="RdYlGn",annot=False,linewidth =1)


# In[105]:


# Getting the top 10 correlation for the Defaulter data
corr_Defaulter = Defaulter_df.corr()
corr_Defaulter = corr_Defaulter.where(np.triu(np.ones(corr_Defaulter.shape),k=1).astype(np.bool))
corr_df_Defaulter = corr_Defaulter.unstack().reset_index()
corr_df_Defaulter.columns =['VAR1','VAR2','Correlation']
corr_df_Defaulter.dropna(subset = ["Correlation"], inplace = True)
corr_df_Defaulter["Correlation"]=corr_df_Defaulter["Correlation"].abs()
corr_df_Defaulter.sort_values(by='Correlation', ascending=False, inplace=True)
corr_df_Defaulter.head(10)


# In[106]:


fig = plt.figure(figsize=(12,12))
ax = sns.heatmap(Defaulter_df.corr(), cmap="RdYlGn",annot=False,linewidth =1)


# <h4>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#               Numerical Univariate Analysis
#             </span>   
#         </font>    
# </h4>

# In[107]:


# Plotting the numerical columns related to amount as distribution plot to see density
amount = New_DF[[ 'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY', 'AMT_GOODS_PRICE']]

fig = plt.figure(figsize=(16,12))

for i in enumerate(amount):
    plt.subplot(2,2,i[0]+1)
    sns.distplot(Defaulter_df[i[1]], hist=False, color='r',label ="Defaulter")
    sns.distplot(Repayer_df[i[1]], hist=False, color='g', label ="Repayer")
    plt.title(i[1], fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 
    
plt.legend()

plt.show() 


# <h4>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#               Numerical Bivariate Analysis
#             </span>   
#         </font>    
# </h4>

# In[108]:


# Checking the relationship between Goods price and credit and comparing with loan repayment staus
bivariate_rel('AMT_GOODS_PRICE','AMT_CREDIT',New_DF,"TARGET", "line", ['g','r'], False,(15,6))


# In[109]:


# Plotting pairplot between amount variable to draw reference against loan repayment status
amount = New_DF[[ 'AMT_INCOME_TOTAL','AMT_CREDIT',
                         'AMT_ANNUITY', 'AMT_GOODS_PRICE','TARGET']]
amount = amount[(amount["AMT_GOODS_PRICE"].notnull()) & (amount["AMT_ANNUITY"].notnull())]
ax= sns.pairplot(amount,hue="TARGET",palette=["g","r"])
ax.fig.legend(labels=['Repayer','Defaulter'])
plt.show()


# <a id="merge"></a>
# <h2>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#             Merged Dataframes Analysis
#             </span>   
#         </font>    
# </h2>

# In[110]:


#merge both the dataframe on SK_ID_CURR with Inner Joins
loan_process_df = pd.merge(New_DF, previousDF, how='inner', on='SK_ID_CURR')
loan_process_df.head()


# In[111]:


#Checking the details of the merged dataframe
loan_process_df.shape


# In[112]:


# Checking the element count of the dataframe
loan_process_df.size


# In[113]:


# checking the columns and column types of the dataframe
loan_process_df.info()


# In[114]:


# Checking merged dataframe numerical columns statistics
loan_process_df.describe()


# In[115]:


# Bifurcating the applicationDF dataframe based on Target value 0 and 1 for correlation and other analysis

L0 = loan_process_df[loan_process_df['TARGET']==0] # Repayers
L1 = loan_process_df[loan_process_df['TARGET']==1] # Defaulters


# <p>
#     <span style='font-family:Georgia'>
#         <b> Plotting Contract Status vs purpose of the loan: </b>
#     </span>
# </p>

# In[116]:


univariate_merged("NAME_CASH_LOAN_PURPOSE",L0,"NAME_CONTRACT_STATUS",["#548235","#FF0000","#0070C0","#FFFF00"],True,(18,7))

univariate_merged("NAME_CASH_LOAN_PURPOSE",L1,"NAME_CONTRACT_STATUS",["#548235","#FF0000","#0070C0","#FFFF00"],True,(18,7))


# In[117]:


# Checking the Contract Status based on loan repayment status and whether there is any business loss or financial loss
univariate_merged("NAME_CONTRACT_STATUS",loan_process_df,"TARGET",['g','r'],False,(12,8))
g = loan_process_df.groupby("NAME_CONTRACT_STATUS")["TARGET"]
df1 = pd.concat([g.value_counts(),round(g.value_counts(normalize=True).mul(100),2)],axis=1, keys=('Counts','Percentage'))
df1['Percentage'] = df1['Percentage'].astype(str) +"%" # adding percentage symbol in the results for understanding
print (df1)


# In[118]:


# plotting the relationship between income total and contact status
merged_pointplot("NAME_CONTRACT_STATUS",'AMT_INCOME_TOTAL')


# In[119]:


# plotting the relationship between people who defaulted in last 60 days being in client's social circle and contact status
merged_pointplot("NAME_CONTRACT_STATUS",'DEF_60_CNT_SOCIAL_CIRCLE')


# <a id="conclusion"></a>
# <h2>   
#       <font color = black >
#             <span style='font-family:Georgia'>
#             Conclusions
#             </span>   
#         </font>    
# </h2>

# <p>
#     <span style='font-family:Georgia'>
#         After analysing the datasets, there are few attributes of a client with which the bank would be able to identify if they will repay the loan or not. The analysis is consised as below with the contributing factors and categorization:
#     </span>
# </p>

# <div class="alert alert-block alert-success">
#     <span style='font-family:Georgia'>
#         <b>Decisive Factor whether an applicant will be Repayer: </b> 
#         <ol>
#             <li>NAME_EDUCATION_TYPE: Academic degree has less defaults. </li>
#             <li>NAME_INCOME_TYPE: Student and Businessmen have no defaults.</li>
#             <li>REGION_RATING_CLIENT: RATING 1 is safer.</li>
#             <li>ORGANIZATION_TYPE: Clients with Trade Type 4 and 5 and Industry type 8 have defaulted less than 3%</li>
#             <li>DAYS_BIRTH: People above age of 50 have low probability of defaulting</li>
#             <li>DAYS_EMPLOYED: Clients with 40+ year experience having less than 1% default rate</li>
#             <li>AMT_INCOME_TOTAL:Applicant with Income more than 700,000 are less likely to default</li>
#             <li>NAME_CASH_LOAN_PURPOSE: Loans bought for Hobby, Buying garage are being repayed mostly.</li>
#             <li>CNT_CHILDREN: People with zero to two children tend to repay the loans.</li>
#         </ol>
#     </span>    
# </div>

# <div class="alert alert-block alert-danger">
#     <span style='font-family:Georgia'>
#         <b>Decisive Factor whether an applicant will be Defaulter: </b> 
#         <ol>
#             <li>CODE_GENDER: Men are at relatively higher default rate</li>
#             <li>NAME_FAMILY_STATUS : People who have civil marriage or who are single default a lot. </li>
#             <li>NAME_EDUCATION_TYPE: People with Lower Secondary & Secondary education</li>
#             <li>NAME_INCOME_TYPE: Clients who are either at Maternity leave OR Unemployed default a lot.</li>
#             <li>REGION_RATING_CLIENT: People who live in Rating 3 has highest defaults.</li>
#             <li>OCCUPATION_TYPE: Avoid Low-skill Laborers, Drivers and Waiters/barmen staff, Security staff, Laborers and Cooking staff as the default rate is huge.</li>
#             <li>ORGANIZATION_TYPE: Organizations with highest percent of loans not repaid are Transport: type 3 (16%), Industry: type 13 (13.5%), Industry: type 8 (12.5%) and Restaurant (less than 12%). Self-employed people have relative high defaulting rate, and thus should be avoided to be approved for loan or provide loan with higher interest rate to mitigate the risk of defaulting.</li>
#             <li>DAYS_BIRTH: Avoid young people who are in age group of 20-40 as they have higher probability of defaulting</li>
#             <li>DAYS_EMPLOYED: People who have less than 5 years of employment have high default rate.</li>
#             <li>CNT_CHILDREN & CNT_FAM_MEMBERS: Client who have children equal to or more than 9 default 100% and hence their applications are to be rejected.</li>
#             <li>AMT_GOODS_PRICE: When the credit amount goes beyond 3M, there is an increase in defaulters.</li>
#         </ol>
#     </span>    
# </div>

# <div class="alert alert-block alert-warning">
#     <span style='font-family:Georgia'>
#         <p>The following attributes indicate that people from these category tend to default but then due to the number of people and the amount of loan, the bank could provide loan with higher interest to mitigate any default risk thus preventing business loss:  </p> 
#         <ol>
#             <li>NAME_HOUSING_TYPE: High number of loan applications are from the category of people who live in Rented apartments & living with parents and hence offering the loan would mitigate the loss if any of those default.</li>
#             <li>AMT_CREDIT: People who get loan for 300-600k tend to default more than others and hence having higher interest specifically for this credit range would be ideal.</li>
#             <li>AMT_INCOME: Since 90% of the applications have Income total less than 300,000 and they have high probability of defaulting, they could be offered loan with higher interest compared to other income category.</li>
#             <li>CNT_CHILDREN & CNT_FAM_MEMBERS: Clients who have 4 to 8 children has a very high default rate and hence higher interest should be imposed on their loans.</li>
#             <li>NAME_CASH_LOAN_PURPOSE: Loan taken for the purpose of Repairs seems to have highest default rate. A very high number applications have been rejected by bank or refused by client in previous applications as well which has purpose as repair or other. This shows that purpose repair is taken as high risk by bank and either they are rejected, or bank offers very high loan interest rate which is not feasible by the clients, thus they refuse the loan. The same approach could be followed in future as well.</li>
#         </ol>
#     </span>    
# </div>

# In[ ]:





# In[ ]:




