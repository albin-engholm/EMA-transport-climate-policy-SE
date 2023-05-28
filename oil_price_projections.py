# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:48:45 2023

@author: aengholm
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

oil_df=pd.read_csv("Oil_2017.csv",sep=";")
oil_price_2017=oil_df[oil_df["Year"]==2017]["Price"].values
oil_price_2012=oil_df[oil_df["Year"]==2012]["Price"].values
oil_df["Relative change to 2017"]=(oil_df["Price"]-oil_price_2017)/oil_price_2017
oil_df["Relative change to 2012"]=(oil_df["Price"]-oil_price_2012)/oil_price_2012
sns.lineplot(data=oil_df,x="Year",y="Price",hue="Projection")
plt.legend(title="")
plt.ylabel("Crude oil priceaen [2017 USD per BOE] ")
#plt.title("Historical and projected oil prices in 2017 USD/BOE")
plt.figure()
oil_p_df=oil_df[(oil_df["Projection"]!="Historical")]
sns.barplot(data=oil_p_df,x="Projection",y="Relative change to 2017")