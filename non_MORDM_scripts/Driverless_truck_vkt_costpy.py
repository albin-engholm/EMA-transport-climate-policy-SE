# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:12:33 2023

@author: aengholm
"""

#Linear decrease of DL truck costs as function of penetration rate

cost_km_MD=100
cost_reduction_AV_final=0.5
cost_reduction_AV_initial=0.8

utilization_increase_AV_initial=0.2
utilization_increase_AV_final=0.5
AV_utilization=[]

AV_VKT_share=[]
MD_VKT_share=[]
import numpy as np
penetration=np.linspace(0,1,11)
cost_AV=[]
cost_average=[]
count=0
for i in penetration:

    AV_utilization.append(1+utilization_increase_AV_initial+(utilization_increase_AV_final-utilization_increase_AV_initial)*i)
    print(i)
    AV_VKT_share.append(i*AV_utilization[count]/((1-i)+i*AV_utilization[count]))
    MD_VKT_share.append(1-AV_VKT_share[count])
    cost_AV.append(cost_km_MD*cost_reduction_AV_initial-(cost_reduction_AV_initial-cost_reduction_AV_final)*AV_VKT_share[count]*cost_km_MD)
    cost_average.append(cost_km_MD*MD_VKT_share[count]+cost_AV[count]*(AV_VKT_share[count]))
    count=count+1

VKT_share_check=[AV_VKT_share[i]+MD_VKT_share[i] for i in range(len(AV_VKT_share))]    
import matplotlib.pyplot as plt
plt.plot(penetration,cost_average)
plt.plot(penetration,cost_AV)
plt.legend(["Average road freight cost","Driverless truck cost","Driverelss trucks VKT share"])
print("AV cost at 50% penetration: ",cost_AV[4])
print("Average cost at 50% penetration: ",cost_average[4])
print("AV VKT share at 50% penetration: ",AV_VKT_share[4])
plt.figure()
plt.plot(penetration,AV_VKT_share)
plt.plot(penetration,penetration)