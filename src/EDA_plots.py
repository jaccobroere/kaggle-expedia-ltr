import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log10

data = pd.read_csv("training_set_VU_DM.csv")


def SelectandTransformData(data):
    f = lambda x : log10(x) if x>0 else x
    df = data.filter(items=['srch_id','price_usd', 'booking_bool','click_bool'])
    df['price_usd'] = df["price_usd"].apply(f)
    return df



def PlotMissing(data):
    x = round(100*data.isnull().mean(axis=0))
    x = x.sort_values()
    feat = list(x.index)
    perc_NA = x.values
    perc_full = 100 - perc_NA
    
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 6)
    
    ax.tick_params(width=2, labelsize = 8, length = 6)
    for axis in ['bottom','left','right','top']:
        if axis == 'bottom' or axis == 'left':
            ax.spines[axis].set_linewidth(2)
        else:
            ax.spines[axis].set_visible(False)

    ax.set_xticks(range(len(feat)))
    ax.set_xticklabels(feat, rotation = 45, ha = 'right')
    ax.set_yticklabels(['0 %','20 %','40 %','60 %','80 %','100 %'], fontsize=15)
    ax.bar(np.arange(0,54), perc_full, label = "% Observed")
    ax.bar(np.arange(0,54), perc_NA, bottom = perc_full, color = 'grey', label = "% Missing")
    ax.legend(frameon=True, fontsize = 18)
    fig.tight_layout()




def Plotlog10priceusdbookingbool(df):
    bins = [[-1,1.29],[1.29,1.49],[1.49,1.69],[1.69, 1.9],[1.9,2.09],[2.09,2.29],[2.29,2.5],[2.5,2.7],[2.7,7.29]]
    ajax = [np.mean(df.iloc[np.where((df["price_usd"] > b[0]) & (df["price_usd"] < b[1]))]["booking_bool"])*100 for b in bins]
    fig, ax = plt.subplots()
    
    ax.tick_params(width=2, labelsize = 8, length = 6, axis ='y')
    for axis in ['bottom','left','right','top']:
        if  axis == 'left':
            ax.spines[axis].set_linewidth(2)
        else:
            ax.spines[axis].set_visible(False)
    ax.set_title("log10(price_usd)")
    ax.set_ylim([0,5])
    ax.set_xlabel("book_bool")
    ax.set_ylabel("Percentage of Hotels being Booked")
    ax.set_yticks([0,1,2,3,4,5])
    ax.set_xticks([])
    ax.set_yticklabels(['0 %','1 %','2 %','3 %','4 %','5 %'], fontsize=15)
    ax.bar(range(len(ajax)),ajax)
    ax.legend(frameon=True, fontsize = 18)
    fig.tight_layout()


    
def Plotlog10priceusdclickbool(df):
    bins = [[-1,1.29],[1.29,1.49],[1.49,1.69],[1.69, 1.9],[1.9,2.09],[2.09,2.29],[2.29,2.5],[2.5,2.7],[2.7,7.29]]
    fey = [np.mean(df.iloc[np.where((df["price_usd"] > b[0]) & (df["price_usd"] < b[1]))]["click_bool"])*100 for b in bins]
    fig, ax = plt.subplots()
    
    ax.tick_params(width=2, labelsize = 8, length = 6, axis ='y')
    for axis in ['bottom','left','right','top']:
        if  axis == 'left':
            ax.spines[axis].set_linewidth(2)
        else:
            ax.spines[axis].set_visible(False)
    ax.set_title("log10(price_usd)")
    ax.set_ylim([0,5])
    ax.set_xlabel("click_bool")
    ax.set_ylabel("Percentage of Hotels being Booked")
    ax.set_yticks([0,1,2,3,4,5])
    ax.set_xticks([])
    ax.set_yticklabels(['0 %','1 %','2 %','3 %','4 %','5 %'], fontsize=15)
    ax.bar(range(len(fey)),fey)
    ax.legend(frameon=True, fontsize = 18)
    fig.tight_layout()
    



def Plotlog10priceusdbookingboolNormalized(df_norm):
    bins = [[-5.3,-2.8],[-2.8,-2.3],[-2.3,-1.8],[-1.8, -1.3],[-1.3,-0.8],[-0.8,-0.3],[-0.3,0.2],[0.2,0.7],[0.7,1.2],[1.2,1.7],[1.7,2.2],[2.2,5.16]]
    az = [np.mean(df_norm.iloc[np.where((df_norm["price_usd"] > b[0]) & (df_norm["price_usd"] < b[1]))]["booking_bool"])*100 for b in bins]
    fig, ax = plt.subplots()
    
    ax.tick_params(width=2, labelsize = 8, length = 6, axis ='y')
    for axis in ['bottom','left','right','top']:
        if  axis == 'left':
            ax.spines[axis].set_linewidth(2)
        else:
            ax.spines[axis].set_visible(False)
    ax.set_title("Normalized log10(price_usd) w.r.t. srch_id")
    ax.set_ylim([0,8])
    ax.set_xlabel("book_bool")
    ax.set_ylabel("Percentage of Hotels being Booked")
    ax.set_yticks([0,2,4,6,8])
    ax.set_xticks([])
    ax.set_yticklabels(['0 %','2 %','4 %','6 %','8 %'], fontsize=15)
    ax.bar(range(len(az)),az)
    ax.legend(frameon=True, fontsize = 18)
    fig.tight_layout()
    
    
def Plotlog10priceusdclickboolNormalized(df_norm):
    bins = [[-5.3,-2.8],[-2.8,-2.3],[-2.3,-1.8],[-1.8, -1.3],[-1.3,-0.8],[-0.8,-0.3],[-0.3,0.2],[0.2,0.7],[0.7,1.2],[1.2,1.7],[1.7,2.2],[2.2,5.16]]
    az = [np.mean(df_norm.iloc[np.where((df_norm["price_usd"] > b[0]) & (df_norm["price_usd"] < b[1]))]["click_bool"])*100 for b in bins]
    fig, ax = plt.subplots()
    
    ax.tick_params(width=2, labelsize = 8, length = 6, axis ='y')
    for axis in ['bottom','left','right','top']:
        if  axis == 'left':
            ax.spines[axis].set_linewidth(2)
        else:
            ax.spines[axis].set_visible(False)
    ax.set_title("Normalized log10(price_usd) w.r.t. srch_id")
    ax.set_ylim([0,8])
    ax.set_xlabel("click_bool")
    ax.set_ylabel("Percentage of Hotels being Clicked")
    ax.set_yticks([0,2,4,6,8])
    ax.set_xticks([])
    ax.set_yticklabels(['0 %','2 %','4 %','6 %','8 %'], fontsize=15)
    ax.bar(range(len(az)),az)
    ax.legend(frameon=True, fontsize = 18)
    fig.tight_layout()    
    
    
    
def main():
    PlotMissing(data)
    df = SelectandTransformData(data)
    Plotlog10priceusdbookingbool(df)
    Plotlog10priceusdclickbool(df)
    
    df_norm = df.copy()
    df_norm["price_usd"] = df_norm.groupby('srch_id')["price_usd"].transform(lambda x: ((x - x.mean())/x.std()))
    
    Plotlog10priceusdbookingboolNormalized(df_norm)
    Plotlog10priceusdclickboolNormalized(df_norm)

if __name__ == "__main__":
    main()




