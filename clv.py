#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
sns.set(style="ticks")
from datetime import date

pd.set_option('display.max_columns', 100)
np.random.seed(42)

# Customer data
customers = pd.read_csv("olist_customers_dataset.csv")

# Orders data
orders = pd.read_csv("olist_orders_dataset.csv")

# Payments data
payments = pd.read_csv("olist_order_payments_dataset.csv")

# Items data
items = pd.read_csv("olist_order_items_dataset.csv")

# creating master dataframe
df1 = payments.merge(items, on='order_id')
df2 = df1.merge(orders, on='order_id')
data = df2.merge(customers, on='customer_id')

# creating a new dataframe with only the customer details necessary for calculating metrics

df = data.loc[:,['order_id','customer_unique_id','customer_id','order_purchase_timestamp','order_status','payment_value']]
df.columns = ['OrderID','CustomerUniqueID','CustomerID','PurchaseTime','OrderStatus','PaymentValue']
# df.head()

#converting the PurchaseTime Dtype to datetime
df['PurchaseTime'] = pd.to_datetime(df['PurchaseTime'] , format='%Y-%m-%d %H:%M:%S')

# displaying missing value counts and corresponding percentage against total observations
missing_values = df.isnull().sum().sort_values(ascending = False)
percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
pd.concat([missing_values, percentage], axis=1, keys=['Values', 'Percentage']).transpose()

#No nan values

# dropping the duplicate entries (which might have been created due to the merged payment details)
print('Shape before duplicates: ',df.shape)
df.drop_duplicates(inplace=True)
print('Shape after duplicates: ',df.shape)

month = df.PurchaseTime.apply(lambda x: x.month).astype(str)
month = month.apply(lambda x: '0' + x if len(x) == 1 else x)

year =  df.PurchaseTime.apply(lambda x: x.year)

df['YearMonth'] = year.astype(str) + '-' + month.astype(str)

# excluding incomplete 2016 and 2018 data 
df = df.query("YearMonth != '2016-12' and YearMonth != '2016-10' and YearMonth != '2016-09' and YearMonth != '2018-09'")
# df.head()

#calculate Revenue for each row and create a new dataframe with YearMonth - Revenue columns
df_revenue = df.groupby(['YearMonth'])['PaymentValue'].sum().reset_index()
df_revenue.columns = ['YearMonth','Revenue']
# df_revenue

# sns.set(rc={'figure.figsize':(20,8)})
# sns.lineplot(data=df_revenue, x="YearMonth", y="Revenue", marker='o')
# plt.title('Monthly Revenue')
# plt.show()

df_revenue['MonthlyGrowth'] = df_revenue['Revenue'].pct_change()
# df_revenue.head()

sns.lineplot(data=df_revenue, x="YearMonth", y="MonthlyGrowth", marker='o')
sns.lineplot(data=df_revenue, x="YearMonth", y=np.zeros(df_revenue.shape[0]))
plt.title('Monthly Growth Rate')
# plt.show()

#creating monthly active customers dataframe by counting unique Customer IDs
df_monthly_active = df.groupby('YearMonth')['CustomerUniqueID'].nunique().reset_index()
df_monthly_active.columns = ['YearMonth','UniqueCustomer']


# sns.barplot(x='YearMonth', y='UniqueCustomer', data=df_monthly_active)
# plt.ylabel('Count of New Customers')
# plt.xlabel('Year-Month')
# plt.show()

#creating monthly active customers dataframe by counting unique Customer IDs
df_monthly_sales = df.groupby('YearMonth')['OrderStatus'].count().reset_index()
df_monthly_sales.columns = ['YearMonth','OrderCount']

sns.barplot(x='YearMonth', y='OrderCount', data=df_monthly_sales)
plt.ylabel('Order Count')
plt.xlabel('Year-Month')
# plt.show()

# create a new dataframe for average revenue by taking the mean of it
df_monthly_order_avg = df.groupby('YearMonth')['PaymentValue'].mean().reset_index()
df_monthly_order_avg.columns = ['YearMonth','AvgRevenuePerOrder']

sns.barplot(x='YearMonth', y='AvgRevenuePerOrder', data=df_monthly_order_avg)
plt.ylabel('Average Revenue per Order')
plt.xlabel('Year-Month')
# plt.show()

#create a dataframe contaning CustomerID and first purchase date
df_min_purchase = df.groupby('CustomerUniqueID').PurchaseTime.min().reset_index()
df_min_purchase.columns = ['CustomerUniqueID','MinPurchaseDate']
df_min_purchase['MinPurchaseDate'] = df_min_purchase['MinPurchaseDate'].map(lambda date: 100*date.year + date.month)

df_min_purchase.head()

#merge first purchase date column to our main dataframe df
df = pd.merge(df, df_min_purchase, on='CustomerUniqueID')
df.head()

#create a column called User Type and assign Existing if User's First Purchase Year Month before the selected Year Month
df['UserType'] = 'New'

#creating a new col of YearMonth with type int for comparison 
df['YearMonthInt'] = df['PurchaseTime'].map(lambda date: 100*date.year + date.month)

df.loc[df['YearMonthInt']>df['MinPurchaseDate'],'UserType'] = 'Existing'

#calculate the Revenue per month for each user type
df_user_type_revenue = df.groupby(['YearMonth','UserType'])['PaymentValue'].sum().reset_index()

# df_user_type_revenue

fig, ax = plt.subplots(figsize=(15, 6))
sns.set(palette='muted', color_codes=True)
ax = sns.lineplot(x='YearMonth', y='PaymentValue', data=df_user_type_revenue.query("UserType == 'New'"), label='New')
ax = sns.lineplot(x='YearMonth', y='PaymentValue', data=df_user_type_revenue.query("UserType == 'Existing'"), label='Existing')
ax.set_title('Existing vs New Customer Comparison')
ax.tick_params(axis='x', labelrotation=90)
# plt.show()

#create a dataframe that shows new user ratio - we also need to drop NA values (as first month new user  is 0)
df_user_ratio = df.query("UserType == 'New'").groupby(['YearMonth'])['CustomerUniqueID'].nunique()/df.query("UserType == 'Existing'").groupby(['YearMonth'])['CustomerUniqueID'].nunique() 
df_user_ratio = df_user_ratio.reset_index()

#dropping nan values that resulted from first and last month
df_user_ratio = df_user_ratio.dropna()
df_user_ratio.columns = ['YearMonth','NewCustomerRatio']

#print the dafaframe
# df_user_ratio

fig, ax = plt.subplots(figsize=(12, 6))
sns.set(palette='muted', color_codes=True, style='whitegrid')
sns.barplot(x='YearMonth', y='NewCustomerRatio', data=df_user_ratio)
ax.tick_params(axis='x', labelrotation=90)
# plt.show()

#Monthly Retention Rate = Retained Customers From Prev. Month/Active Customers Total (using crosstab)

#identifying active users by looking at their revenue per month
df_user_purchase = df.groupby(['CustomerUniqueID','YearMonthInt'])['PaymentValue'].sum().reset_index()
df_user_purchase.head()

#creating retention matrix with crosstab using df_user_purchase
df_retention = pd.crosstab(df_user_purchase['CustomerUniqueID'], df_user_purchase['YearMonthInt']).reset_index()
df_retention.head()

#creating an array of dictionary which keeps Retained & Total User count for each month
months = df_retention.columns[2:]
retention_array = []
for i in range(len(months)-1):
    retention_data = {}
    selected_month = months[i+1]
    prev_month = months[i]
    retention_data['YearMonth'] = int(selected_month)
    retention_data['TotalUserCount'] = df_retention[selected_month].sum()
    retention_data['RetainedUserCount'] = df_retention[(df_retention[selected_month]>0) & (df_retention[prev_month]>0)][selected_month].sum()
    retention_array.append(retention_data)
    
#convert the array to dataframe and calculate Retention Rate
df_retention = pd.DataFrame(retention_array)
df_retention['RetentionRate'] = df_retention['RetainedUserCount']/df_retention['TotalUserCount']

# df_retention

fig, ax = plt.subplots(figsize=(20, 6))
sns.set(palette='muted', color_codes=True, style='whitegrid')
sns.lineplot(x=df_retention.YearMonth.astype(str), y=df_retention.RetentionRate, marker='o')
ax.tick_params(axis='x', labelrotation=90)
# plt.show()

#create our retention table again with crosstab() - we need to change the column names for using them in .query() function

df_retention = pd.crosstab(df_user_purchase['CustomerUniqueID'], df_user_purchase['YearMonthInt']).reset_index()
new_column_names = [ 'm_' + str(column) for column in df_retention.columns]
df_retention.columns = new_column_names
df_retention.head()

#create the array of Retained users for each cohort monthly
retention_array = []
for i in range(len(months)):
    retention_data = {}
    selected_month = months[i]
    prev_months = months[:i]
    next_months = months[i+1:]
    for prev_month in prev_months:
        retention_data[prev_month] = np.nan
        
    total_user_count =  retention_data['TotalUserCount'] = df_retention['m_' + str(selected_month)].sum()
    retention_data[selected_month] = 1 
    
    query = "{} > 0".format('m_' + str(selected_month))
    

    for next_month in next_months:
        query = query + " and {} > 0".format(str('m_' + str(next_month)))
        retention_data[next_month] = np.round(df_retention.query(query)['m_' + str(next_month)].sum()/total_user_count,2)
    retention_array.append(retention_data)

    #create the array of Retained users for each cohort monthly
retention_array = []
for i in range(len(months)):
    retention_data = {}
    selected_month = months[i]
    prev_months = months[:i]
    next_months = months[i+1:]
    for prev_month in prev_months:
        retention_data[prev_month] = np.nan
        
    total_user_count =  retention_data['TotalUserCount'] = df_retention['m_' + str(selected_month)].sum()
    retention_data[selected_month] = 1 
    
    query = "{} > 0".format('m_' + str(selected_month))
    

    for next_month in next_months:
        query = query + " and {} > 0".format(str('m_' + str(next_month)))
        retention_data[next_month] = np.round(df_retention.query(query)['m_' + str(next_month)].sum()/total_user_count,2)
    retention_array.append(retention_data)
    
df_retention = pd.DataFrame(retention_array)
df_retention.index = months

#showing new cohort based retention table
# df_retention

count = df["CustomerUniqueID"].value_counts().value_counts()
# count

#creates a generic user dataframe to keep CustomerID and new segmentation scores
df_user = pd.DataFrame(df['CustomerUniqueID'])
df_user.columns = ['customer_unique_id']

#gets the max purchase date for each customer and create a dataframe with it
df_max_purchase = df.groupby('CustomerUniqueID').PurchaseTime.max().reset_index()
df_max_purchase.columns = ['customer_unique_id', 'MaxPurchaseDate']

#we take our observation point as the max purchase date in our dataset
df_max_purchase['Recency'] = (df_max_purchase['MaxPurchaseDate'].max() - df_max_purchase['MaxPurchaseDate']).dt.days

#merge this dataframe to our new user dataframe
df_user = pd.merge(df_user, df_max_purchase[['customer_unique_id','Recency']], on='customer_unique_id')

df_user.head()

# plotting the distribution of the continous feature set
sns.set(palette='muted', color_codes=True, style='white')
fig, ax = plt.subplots(figsize=(12, 6))
sns.despine(left=True)
sns.distplot(df_user['Recency'], bins=30)
# plt.show()

from sklearn.cluster import KMeans

sse={}
df_recency = df_user[['Recency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_recency)
    df_recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_
    
plt.figure(figsize=(10, 5))
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
# plt.show()

#building 5 clusters for recency and adding it to dataframe
kmeans = KMeans(n_clusters=5)
kmeans.fit(df_user[['Recency']])
df_user['RecencyCluster'] = kmeans.predict(df_user[['Recency']])

#function for ordering cluster numbers
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

df_user = order_cluster('RecencyCluster', 'Recency',df_user,False)

#displaying the details of each cluster
df_user.groupby('RecencyCluster')['Recency'].describe()



#get order counts for each user and create a dataframe with it
df_frequency = df.groupby('CustomerUniqueID').PurchaseTime.count().reset_index()
df_frequency.columns = ['customer_unique_id','Frequency']

#add this data to our main dataframe
df_user = pd.merge(df_user, df_frequency, on='customer_unique_id')




# getting summary statistics of the recency table
df_user.Frequency.describe()

#k-means
kmeans = KMeans(n_clusters=5)
kmeans.fit(df_user[['Frequency']])
df_user['FrequencyCluster'] = kmeans.predict(df_user[['Frequency']])

#order the frequency cluster
df_user = order_cluster('FrequencyCluster', 'Frequency',df_user,True)

#see details of each cluster
df_user.groupby('FrequencyCluster')['Frequency'].describe()

#calculate revenue for each customer
df_revenue = df.groupby('CustomerUniqueID').PaymentValue.sum().reset_index()
df_revenue.columns = ['customer_unique_id','payment_value']

#merge it with our main dataframe
df_user = pd.merge(df_user, df_revenue, on='customer_unique_id')


sse={}
df_revenue = df_user[['payment_value']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_revenue)
    df_revenue["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_
    
plt.figure(figsize=(10, 5))
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
# plt.show()

#apply clustering
kmeans = KMeans(n_clusters=6)
kmeans.fit(df_user[['payment_value']])
df_user['RevenueCluster'] = kmeans.predict(df_user[['payment_value']])


#order the cluster numbers
df_user = order_cluster('RevenueCluster', 'payment_value',df_user,True)

#show details of the dataframe
df_user.groupby('RevenueCluster')['payment_value'].describe()

#######

# Preparing to separate df into 4 and 6 months
clv_df = df[['CustomerUniqueID','PurchaseTime','PaymentValue']]
clv_df['PurchaseTime'] = pd.to_datetime(clv_df['PurchaseTime']).dt.date
clv_df.columns = ['customer_unique_id','PurchaseTime','Revenue']
# clv_df

# Time range of customer purchase
maxdate = clv_df['PurchaseTime'].max()
mindate = clv_df['PurchaseTime'].min()
print(f"Time range of the transactions are : {mindate} to {maxdate}")

# Create 4 month and 6 month dataframes
df_4m = clv_df[(clv_df.PurchaseTime < date(2018,3,1)) & (clv_df.PurchaseTime >= date(2017,11,1))].reset_index(drop=True)
df_6m = clv_df[(clv_df.PurchaseTime >= date(2018,3,1)) & (clv_df.PurchaseTime < date(2018,9,1))].reset_index(drop=True)

# Check and dropping duplicates
df_4m = df_4m.drop_duplicates(keep='first')
df_4m.duplicated().any()

# Calculating revenue for 4 months data 
df_user_4m = df_4m.groupby('customer_unique_id')['Revenue'].sum().reset_index()


# Check number of clusters
sse={}
df_revenue02 = df_user_4m[['Revenue']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_revenue02)
    df_revenue02["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_
    
plt.figure(figsize=(10, 5))
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
# plt.show()

# Creating 3 clusters
km = KMeans(n_clusters=3)
km.fit(df_user_4m[['Revenue']])
df_user_4m['RevenueCluster_4m'] = km.predict(df_user_4m[['Revenue']])

# Ordering cluster method
df_user_4m = order_cluster('RevenueCluster_4m', 'Revenue',df_user_4m,True)

# Calculating recency score for 4 months data
max_purchase = df_4m.groupby('customer_unique_id').PurchaseTime.max().reset_index()
max_purchase.columns = ['customer_unique_id','MaxPurchaseDate']
max_purchase['Recency_4m'] = (max_purchase['MaxPurchaseDate'].max() - max_purchase['MaxPurchaseDate']).dt.days
df_user_4m = pd.merge(df_user_4m, max_purchase[['customer_unique_id','Recency_4m']], on='customer_unique_id')


# Check number of clusters using Elbow Method to get the optimal cluster number for optimal inertia
sse={}
df_recency02 = df_user_4m[['Recency_4m']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_recency02)
    df_recency02["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_
    
plt.figure(figsize=(10, 5))
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
# plt.show()

# Creating 3 clusters
km = KMeans(n_clusters=3)
km.fit(df_user_4m[['Recency_4m']])
df_user_4m['RecencyCluster_4m'] = km.predict(df_user_4m[['Recency_4m']])

# Ordering cluster method
df_user_4m = order_cluster('RecencyCluster_4m', 'Recency_4m',df_user_4m,False)
df_user_4m.head()

# Calcuate frequency score for 4 months data
df_frequency02 = df_4m.groupby('customer_unique_id').PurchaseTime.count().reset_index()
df_frequency02.columns = ['customer_unique_id','Frequency_4m']
df_user_4m = pd.merge(df_user_4m, df_frequency02, on='customer_unique_id')


# Check number of clusters
sse={}
df_frequency02 = df_user_4m[['Frequency_4m']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_frequency02)
    df_frequency02["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_
    
plt.figure(figsize=(10, 5))
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
# plt.show()

# Creating 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_user_4m[['Frequency_4m']])
df_user_4m['FrequencyCluster_4m'] = kmeans.predict(df_user_4m[['Frequency_4m']])

# Ordering cluster method
df_user_4m = order_cluster('FrequencyCluster_4m', 'Frequency_4m',df_user_4m,True)
# df_user_4m

# Dropping recency
# df_user_4m = df_user_4m.drop(['Recency_4m','RecencyCluster_4m'], axis=1)
# df_user_4m

# Calculate overall score and use mean() to see details
df_user_4m['OverallScore'] = df_user_4m['RecencyCluster_4m'] + df_user_4m['FrequencyCluster_4m'] + df_user_4m['RevenueCluster_4m']
df_user_4m.groupby('OverallScore')['Recency_4m','Frequency_4m','Revenue'].mean()

# Check each cluster count to determine the segment
df_user_4m.groupby('OverallScore')['Revenue'].count()

# Overall scoring
df_user_4m['Segment'] = 'Low-Value'
df_user_4m.loc[df_user_4m['OverallScore'] > 2,'Segment'] = 'Mid-Value' 
df_user_4m.loc[df_user_4m['OverallScore'] > 4,'Segment'] = 'High-Value' 

# 6 months LTV for each customer which we are going to use for training our model
df_6m = df_6m.drop_duplicates(keep='first')
df_6m['customer_unique_id'].value_counts()

# Calculating Revenue for 6 months data
# Using Revenue as direct CLV 
df_user_6m = df_6m.groupby('customer_unique_id')['Revenue'].sum().reset_index()
# df_user_6m

# Creating new df with 4 months and 6 months df
df_merge02 = pd.merge(df_user_4m, df_user_6m, on='customer_unique_id', how='left', suffixes=['_4m', '_6m'])

# treating nan values with replacing into 0
df_merge02 = df_merge02.fillna(0)

# Mean for each segment
df_merge02.groupby('Segment')['Revenue_6m'].mean()

# Count customer who made a purchase in both 4 and 6 months period
df_merge02['Revenue_6m'].value_counts()

df_merge02.duplicated().sum()

# Check number of clusters
sse={}
df_revenue_6m = df_merge02[['Revenue_6m']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_revenue_6m)
    df_revenue_6m["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_
    
plt.figure(figsize=(10, 5))
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
# plt.show()

#creating 2 clusters
km = KMeans(n_clusters=2)
km.fit(df_merge02[['Revenue_6m']])
df_merge02['CLVCluster'] = km.predict(df_merge02[['Revenue_6m']])


# Ordering cluster number based on CLV
df_merge02 = order_cluster('CLVCluster', 'Revenue_6m', df_merge02, True)

# Check details of the clusters
df_merge02.groupby('CLVCluster')['Revenue_6m'].describe()

# Making a copy of the df - with outlier
df_cluster = df_merge02.copy()
df_cluster.head()

# Drop customerID column
df_cluster = df_cluster.iloc[:, 1:]

# Convert categorical variable into dummy/indicator variables
df_cluster = pd.get_dummies(df_cluster)

# Correlation matrix
corr_matrix = df_cluster.corr()
corr_matrix['CLVCluster'].sort_values(ascending=False)

# Feature scaling using Robust Scaler
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df_cluster))
scaled_df.columns = df_cluster.columns
# scaled_df

# # Feature scaling using Standard Scaler
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaled_df = pd.DataFrame(scaler.fit_transform(df_cluster))
# scaled_df.columns = df_cluster.columns
# scaled_df
# # Gives an error: Unknown label type: 'continuous'

# Create x (feature set), y (target set) - with outlier
x = scaled_df.drop(['CLVCluster','Revenue_6m'],axis=1)
y = scaled_df['CLVCluster']


# Split training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

print('Train Before : ',Counter(y_train))
print('Test Before : ',Counter(y_test),'\n')

# Initiating random undersampler
under = RandomUnderSampler()
# Resampling x, y
x_train_under, y_train_under = under.fit_resample(x_train, y_train)
x_test_under, y_test_under = under.fit_resample(x_test, y_test)

print('Train After : ',Counter(y_train_under))
print('Test After : ',Counter(y_test_under))

from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# # Creating object of the models
xg = XGBClassifier(random_state=1, use_label_encoder=False)
bc = BaggingClassifier() 
ada = AdaBoostClassifier()
gbc = GradientBoostingClassifier()
rf = RandomForestClassifier(n_estimators= 5, criterion="entropy")
dt = DecisionTreeClassifier()
lg = LogisticRegression()
svc = SVC()
nb = GaussianNB()

# List of models
#models_2 = [rf, dt]
models_2 = [xg, bc, ada, gbc, rf, dt, lg, svc]
# models_2

## Classification Report - undersample
for i in models_2:

  # Model fit to train data
  i.fit(x_train_under, y_train_under)
  
  # Predicting on test
  pred_test2 = i.predict(x_test_under)
  acc_test2 = accuracy_score(y_test_under, pred_test2)

  # Predicting on train
  pred_train2 = i.predict(x_train_under)
  acc_train2 = accuracy_score(y_train_under, pred_train2)

  ##classification report
  report_under = classification_report(y_test_under, pred_test2)
  print(f"{i} Classification report on under sampled data\n\n",report_under)


  print(f"Test accuracy for {i} is: {acc_test2}")
  print(f"Train accuracy for {i} is: {acc_train2}\n\n")

# streamlit part

# import module
import streamlit as st
 
# Title
st.title("CLV Prediction")

form = st.form(key="my_form")

with form:
    cols = st.columns((1,1))
    recency = cols[0].text_input("Days since last purchase (recency):")
    frequency = cols[1].text_input("Total number of purchases (frequency):")
    revenue = st.text_area("Total spending (revenue):")
    submitted = st.form_submit_button(label="Submit")

if submitted:
    # create user input dataframe
    my_data = [['dummy_id',float(recency),float(revenue),float(frequency)]]
    my_df = pd.DataFrame(my_data, columns = ['customer_unique_id', 'Recency', 'Revenue_4m','Frequency'])

    # compute revenue cluster
    my_user_df = my_df.groupby('customer_unique_id')['Revenue_4m'].sum().reset_index()
    km = KMeans(n_clusters = 3)
    km.fit(df_user_4m[['Revenue']])
    my_user_df['RevenueCluster_4m'] = km.predict(my_user_df[['Revenue_4m']])

    # compute recency cluster
    km = KMeans(n_clusters = 3)
    km.fit(df_user_4m[['Recency_4m']])
    my_user_df['Recency_4m'] = my_df['Recency']
    my_user_df['RecencyCluster_4m'] = km.predict(my_df[['Recency']])

    # compute frequency cluster
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(df_user_4m[['Frequency_4m']])
    my_user_df['Frequency_4m'] = my_df['Frequency']
    my_user_df['FrequencyCluster_4m'] = kmeans.predict(my_df[['Frequency']])

    # compute overall score
    my_user_df['OverallScore'] = my_user_df['RecencyCluster_4m'] + my_user_df['FrequencyCluster_4m'] + my_user_df['RevenueCluster_4m']

    # add new columns to dataframe
    my_user_df['Segment'] = 'Low-Value'
    my_user_df.loc[my_user_df['OverallScore'] > 2,'Segment'] = 'Mid-Value'
    my_user_df.loc[my_user_df['OverallScore'] > 4,'Segment'] = 'High-Value'

    # scaling dataframe
    # Convert categorical variable into dummy/indicator variables
    my_user_df['Revenue_6m'] = 0
    my_user_df['CLVCluster'] = 0
    my_user_df['Segment_High-Value'] = 0
    my_user_df['Segment_Low-Value'] = 0
    my_user_df['Segment_Mid-Value'] = 0

    if my_user_df.loc[[0]].values[0][8] == 'High-Value':
        my_user_df['Segment_High-Value'] = 1
    elif my_user_df.loc[[0]].values[0][8] == 'Low-Value':
        my_user_df['Segment_Low-Value'] = 1
    elif my_user_df.loc[[0]].values[0][8] == 'Mid-Value':
        my_user_df['Segment_Mid-Value'] = 1

    my_user_df.drop(columns=['customer_unique_id','Segment'])

    df_cluster2 = df_cluster.append(my_user_df.drop(columns=['customer_unique_id','Segment']), ignore_index=True)
    scaled_df2 = pd.DataFrame(scaler.fit_transform(df_cluster2))
    scaled_df2.columns = df_cluster2.columns

    my_user_input_df = scaled_df2[26411:]

    my_x = my_user_input_df.drop(columns=['CLVCluster','Revenue_6m'])

    st.success("Done! Printing Results.")
    st.title("Cluster [1] = High Value Customer, Cluster [0] = Low Value Customer")
    st.title(ada.predict(my_x))