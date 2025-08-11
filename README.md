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
The primary objective of this notebook is to EDA and build machine learning model to predict.
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
<img width="1736" height="216" alt="image" src="https://github.com/user-attachments/assets/5616a894-8c2a-4bab-9b6e-b3b61dad643d" />

```code
df.shape
```
(10000, 18)
 Great now we can see basic information about this file

```code
# check random
df.sample(10)
```
<img width="1756" height="417" alt="image" src="https://github.com/user-attachments/assets/34fcc444-43a7-4c99-a449-9a9d14201f42" />

### Transform data:
- Data exploration:
  Show the columns of the dataframe and their types
```code
df.info()
```
<img width="446" height="506" alt="image" src="https://github.com/user-attachments/assets/d8c60199-3836-4760-8487-5f8fb6b6b562" />

- Check missing value:
  ```code
  print(df.isnull().sum())
  ```
<img width="219" height="391" alt="image" src="https://github.com/user-attachments/assets/57e7539e-6cd5-4fc0-bd46-07930cacd2aa" />

- Fix data type:
```code
df['Surname'] = df['Surname'].astype('string')
df['Geography'] = df['Geography'].astype('string')
df['Gender'] = df['Gender'].astype('string')
df['Card Type'] = df['Card Type'].astype('string')
# Check again
df.info()
```
<img width="432" height="502" alt="image" src="https://github.com/user-attachments/assets/06e60a2d-2d7e-4b2c-9151-0966683336bf" />

- Basic descriptive statistics:
  Show a descriptive statistics of the numeric columns
```code
df.describe()
```
<img width="1751" height="336" alt="image" src="https://github.com/user-attachments/assets/6bf2240e-2481-4177-b444-cce287bb9977" />

- Customer obvious analysis:
```code
# Boxplots to compare distributions by churn
for col in ['Age', 'Balance', 'EstimatedSalary', 'CreditScore']:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Exited', y=col, data=df)
    plt.title(f'{col} vs. Exited')
    plt.show()
```
<img width="585" height="441" alt="image" src="https://github.com/user-attachments/assets/b3a2eb5c-aa57-489c-9be5-c79fea361682" />
<img width="638" height="438" alt="image" src="https://github.com/user-attachments/assets/5024b3ec-5d0b-4394-a699-c058b91c7b12" />
<img width="647" height="448" alt="image" src="https://github.com/user-attachments/assets/fa3d1dd2-d211-460c-90e3-a2a45c689802" />
<img width="613" height="439" alt="image" src="https://github.com/user-attachments/assets/04aacc44-6f26-4751-8572-e050d6aa61a4" />

```code
# Violin to see more about Balance vs Churn
sns.violinplot(x='Exited', y='Balance', data=df)
plt.title('Balance vs Churn')
plt.xlabel('Exited')
plt.ylabel('Balance')
plt.show()
```
<img width="667" height="520" alt="image" src="https://github.com/user-attachments/assets/2e85591a-3fae-408e-8e6d-28ce201ce99b" />








