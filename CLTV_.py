from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
import pymysql

import pandas as pd

## Connecting to the Database ##
conn = pymysql.connect(host='34.88.156.118',port=int(3306),user='group_07',passwd='#######',db='#######')


##### 6-month CLTV prediction for 2010-2011 UK customers  #####
pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)
retail_mysql_df = pd.read_sql_query("select * from online_retail_2010_2011", conn)
retail_mysql_df


#### Dataset information ####
retail_mysql_df.shape
retail_mysql_df.head()
retail_mysql_df.info()

### Copy of Dataset  ####
df = retail_mysql_df.copy()
df


## Selecting UK Customers ##
df["Country"].value_counts()
df = df[df["Country"] == "United Kingdom"]

## Descriptive Statistics ##
df.describe().T


### Data Preparation ####
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

### Extracting Outlier Values ####
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")


## Discarding Missing Values ###
df.dropna(inplace=True)
## TotalPrice değişkeninin oluşturulması ####
df["TotalPrice"] = df["Quantity"] * df["Price"]

## Today's Date ##

today_date = dt.datetime(2011,12,11)


cltv_df = df.groupby('CustomerID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days], 
                                                        'Invoice': lambda num: num.nunique(), 'TotalPrice': lambda TotalPrice: TotalPrice.sum()})


cltv_df.columns = cltv_df.columns.droplevel(0)


#### Definition of recency, T ,frequency, monetary #####
cltv_df.columns = ["recency", "T", "frequency", "monetary"]
cltv_df

#### (total monetary / number of transactions)  ###

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

### frequency > 1   ###

cltv_df = cltv_df[(cltv_df["frequency"] > 1 )]
cltv_df
## Since it is weekly data, we convert recency and T to weekly ###

cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T"] / 7
cltv_df

##### BG/NBD Model #####

bgf = BetaGeoFitter (penalizer_coef = 0.001)

bgf.fit(cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

## Expected Avarage Profit ##
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'],cltv_df["monetary"])


#### CLTV analysis consisting of different time period s#####
# Calculation of CLTV with 6-month BG-NBD and GG model #

cltv_df["cltv_6_month"] = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  
                                   freq="W", 
                                   discount_rate=0.01)
# Calculation of CLTV with 1-month BG-NBD and GG model #
cltv_df["cltv_1_month"] = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1, 
                                   freq="W",  
                                   discount_rate=0.01)
# Calculation of CLTV with 12-month BG-NBD and GG model #
cltv_df["cltv_12_month"] = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,  
                                   freq="W",  
                                   discount_rate=0.01)

cltv_df.head(10)
cltv_df["expected_purc_1_week"] = bgf.predict(1,cltv_df['frequency'], cltv_df['recency'],cltv_df['T'])
cltv_df["expected_purc_1_month"] = bgf.predict(4, cltv_df['frequency'],cltv_df['recency'], cltv_df['T'])
cltv_df["expected_average_profit_clv"] = ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary'])
cltv_df.columns
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_df[["cltv_6_month"]])
cltv_df["scaled_clv"] = scaler.transform(cltv_df[["cltv_6_month"]])

## Segmentation and Action Recommendations##

cltv_df["segment"] = pd.qcut(cltv_df["cltv_6_month"], 4, labels=["D", "C", "B", "A"])
cltv_df.sort_values(by="cltv_6_month", ascending=False).head()

cltv_df.groupby("segment").agg(
    {"count", "mean", "sum"})
cltv_df.drop(["cltv_1_month","cltv_12_month","cltv_6_month"], axis=1, inplace=True)
cltv_df

conn.close()
conn = pymysql.connect(host='34.88.156.118',port=int(3306),user='group_07',passwd='###',db='#####')
##cltv_df["CustomerID"] = cltv_df["CustomerID"].astype(int)
pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)

## Export results to database

cltv_df.to_sql(name="bahar_onel", con=conn, if_exists='replace', index=False)










