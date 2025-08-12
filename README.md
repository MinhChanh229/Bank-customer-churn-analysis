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
(10000, 18)
 Great now we can see basic information about this file.
 Check randoom 10 rows.

### Transform data:
- Data exploration:
  Show the columns of the dataframe and their types, prevail 4 objects type (surname, geography, gender, card type)
  
- Check missing value:
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
  Convert data all object type to string type.

- Basic descriptive statistics:
  Show a descriptive statistics of the numeric columns
  Age: ranges from 18 to 92 years old, with a median of 37
  Balance: many customers have a balance of 0 (median = 97,198.54, but 25th percentile = 0).
  NumOfProducts: most customers use 1–2 products.
  HasCrCard and IsActiveMember are binary variables (0 or 1).
  Satisfaction Score: ranges from 1 to 5, with an average of around 3.
  Point Earned: ranges from 119 to 1,000, with a relatively wide distribution.

- Customer obvious analysis:
  Boxplots to compare distributions by churn: Age, Balance, EstimatedSalary, CreditScore
  Violin to see more about Balance vs Churn









