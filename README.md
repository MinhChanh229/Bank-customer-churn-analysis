# Bank-customer-churn-analysis

## Overview:
- Back ground:

Banks today are facing an increasing customer churn rate. Since the cost of acquiring new customers is often much higher than the cost of retaining existing ones, building an early warning system for customer churn can help banks reduce losses and optimize long-term profitability.

- Goal of the Project:

Based on the data, descriptive nalytics to understand the behavior of customers who have chunred  we will find out what causes customers to leave and then come up with appropriate strategies to help the bank.

- Source:

Kaggle: https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn/data

Customer-Churn-Records.csv(837.42 kB)

18 columns and 10.000 rows

- Project Objective:

The primary objective of this notebook is to ETL and build machine learning model to predict combine with SQL to analysis more about outliers.

Buid dashboard to track customer churn rate.
  
## 👣 The First Steps
### 📥 Data import

First, let's import the needed libraries: Pandas, Matplotlib & Seaborn.
```code
# import pandas, matplotlip, seaborn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### Extract data:

Then load data from csv file.
```code
# load data form csv
df = pd.read_csv('/content/Customer-Churn-Records.csv')
df.head()
```
 <img width="1749" height="235" alt="image" src="https://github.com/user-attachments/assets/9c48a027-f54c-4315-9282-cea3cbcd14b6" />

```code
df.shape()
```
(10000, 18)

```code
# check random
df.sample(10)
```
<img width="1766" height="416" alt="image" src="https://github.com/user-attachments/assets/9e579618-b1ff-4d1d-b3b3-6b0a82dd1846" />

 Great now we can see basic information about this file.

### Transform data:
- Data exploration:

Show the columns of the dataframe and their types.
  ```code
  df.info()
  ```
  <img width="484" height="508" alt="image" src="https://github.com/user-attachments/assets/d0dba80d-cdc7-4d2d-9f8e-e3e03a9738b3" />

Prevailing 4 objects type (surname, geography, gender, card type)
  
- Check missing value:
```code
print(df.isnull().sum())
```
<img width="266" height="395" alt="image" src="https://github.com/user-attachments/assets/552c50c1-0b5d-4584-9c00-1d3828edc097" />

  No data missing and no duplicate.
  When there are no missing values in the dataset, it usually indicates:
  1. Good data collection process
  All fields are fully entered, minimizing errors or omissions.

  2. Higher data quality
  Reduced risk of bias due to missing information.
  Analysis and modeling become more reliable.

  3. Less complex preprocessing required
  No need for imputation techniques or record deletion.

  4. Mandatory data entry by the system
  The system may be designed to require all fields to be filled in before saving the data.

- Fix data type:
```code
df['Surname'] = df['Surname'].astype('string')
df['Geography'] = df['Geography'].astype('string')
df['Gender'] = df['Gender'].astype('string')
df['Card Type'] = df['Card Type'].astype('string')
# check again
df.info()
```
<img width="498" height="504" alt="image" src="https://github.com/user-attachments/assets/2d06ed0a-63ca-4e6f-b454-7da7cc0b3694" />

  Convert data all object type to string type.

- Basic descriptive statistics:

Show a descriptive statistics of the numeric columns.
```code
df.describe()
```
<img width="1762" height="347" alt="image" src="https://github.com/user-attachments/assets/1b5bb0f3-9e45-49c7-891d-79b09d05e8c5" />

  Age: ranges from 18 to 92 years old, with a median of 37
  
  Balance: many customers have a balance of 0 (median = 97,198.54, but 25th percentile = 0).
  
  NumOfProducts: most customers use 1–2 products.
  
  HasCrCard and IsActiveMember are binary variables (0 or 1).
  
  Satisfaction Score: ranges from 1 to 5, with an average of around 3.
  
  Point Earned: ranges from 119 to 1,000, with a relatively wide distribution.

- Customer obvious analysis:

Boxplots to compare distributions by churn: Age, Balance, EstimatedSalary, CreditScore
```code
# Boxplots to compare distributions by churn
for col in ['Age', 'Balance', 'EstimatedSalary', 'CreditScore']:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Exited', y=col, data=df)
    plt.title(f'{col} vs. Exited')
    plt.show()
```
<img width="602" height="459" alt="image" src="https://github.com/user-attachments/assets/8667d363-5105-4868-a45a-8308f76528d6" />
<img width="639" height="448" alt="image" src="https://github.com/user-attachments/assets/6dc5bb98-e9f0-4ac2-8750-8f747e475445" />
<img width="694" height="481" alt="image" src="https://github.com/user-attachments/assets/707423b0-ac96-451f-a318-7de66c88bb27" />
<img width="653" height="436" alt="image" src="https://github.com/user-attachments/assets/9c877282-2401-4bc0-9f63-0fb1246d6692" />

### Remark:


Violin to see more about Balance vs Churn









