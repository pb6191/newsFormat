# %% load modules

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import ttest_ind


pd.set_option(
    "display.max_rows",
    8,
    "display.max_columns",
    None,
    "display.width",
    None,
    "display.expand_frame_repr",
    True,
    "display.max_colwidth",
    None,
)

np.set_printoptions(
    edgeitems=5,
    linewidth=233,
    precision=4,
    sign=" ",
    suppress=True,
    threshold=50,
    formatter=None,
)

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#%%

df1 = pd.read_csv("../data/raw/googleNews_June 18, 2023_14.25_num.csv")

df = df1.drop(labels=[0,1], axis=0)

#%%

df.to_csv("../data/clean/clean_data.csv", index=False)
df = pd.read_csv("../data/clean/clean_data.csv")
df.dtypes

df[["condition"]].value_counts()
df["dynamic"] = df["condition"].str.contains("dynamic")
df["anon"] = df["condition"].str.contains("Anon")

df[["dynamic"]].value_counts()
df[["anon"]].value_counts()

df2 = df.groupby(['condition']).mean()
df2 = df2.drop(df2.columns[0:10], axis=1)

ttest_ind(df[(df.condition == "dynamicAnon")]['clickImages'], df[(df.condition == "staticGoogle")]['clickImages'])
ttest_ind(df[(df.condition == "dynamicAnon")]['clickImages'], df[(df.condition == "dynamicGoogle")]['clickImages'])

ttest_ind(df[(df.condition == "dynamicAnon")]['seeImages'], df[(df.condition == "staticGoogle")]['seeImages'])

ttest_ind(df[(df.condition == "dynamicAnon")]['annoyImages'], df[(df.condition == "staticAnon")]['annoyImages'])
ttest_ind(df[(df.condition == "dynamicAnon")]['annoyImages'], df[(df.condition == "dynamicGoogle")]['annoyImages'])
ttest_ind(df[(df.condition == "dynamicAnon")]['annoyImages'], df[(df.condition == "staticGoogle")]['annoyImages'])

ttest_ind(df[(df.condition == "staticGoogle")]['speakerClick'], df[(df.condition == "staticAnon")]['speakerClick'])
ttest_ind(df[(df.condition == "dynamicAnon")]['speakerClick'], df[(df.condition == "staticAnon")]['speakerClick'])



df3 = df.groupby(['dynamic']).mean()
df3 = df3.drop(df3.columns[0:10], axis=1)

ttest_ind(df[(df.dynamic == False)]['annoyImages'], df[(df.dynamic == True)]['annoyImages'])



df4 = df.groupby(['anon']).mean()
df4 = df4.drop(df4.columns[0:10], axis=1)

ttest_ind(df[(df.anon == False)]['clickImages'], df[(df.anon == True)]['clickImages'])
ttest_ind(df[(df.anon == False)]['speakerClick'], df[(df.anon == True)]['speakerClick'])



df2 = df2.reset_index()

df2 = df2.replace({'condition': {"dynamicAnon": "Dynamic Videos on\nAnonymous Website", "dynamicGoogle": "Dynamic Videos on\nGoogle News", "staticAnon": "Static Images on\nAnonymous Website", "staticGoogle": "Static Images on\nGoogle News"}})

#%%



fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=False, sharex=False, figsize=(13, 18))

# Create a list of colors for the boxplots based on the number of features you have
boxplots_colors = ['yellowgreen', 'olivedrab', 'yellowgreen']



dflist = df2[["annoyImages", "clickImages", "speakerClick", "condition"]].T.values.tolist()

bp1 = ax1.bar(df2["condition"], df2["annoyImages"], color ='darkgreen', width = 0.4)
ax1.set_title("\n\nLikelihood of annoyance from videos/images next to the headlines\n")


bp2 = ax2.bar(df2["condition"], df2["clickImages"], color ='olivedrab', width = 0.4)
ax2.set_title("\n\nLikelihood of clicking on videos/images next to the headlines\n")


bp3 = ax3.bar(df2["condition"], df2["speakerClick"], color ='yellowgreen', width = 0.4)
ax3.set_title("\n\nLikelihood of clicking on a speaker icon on videos/images to listen to the news headline\n")

ax1.set_ylim([1, 4])
ax1.set_yticks([1, 2, 3, 4])
ax1.set_yticklabels(["Extremely     \nUnlikely     ", "Very     \nUnlikely     ", "Unlikely     ", "Likely     "])

ax2.set_ylim([1, 4])
ax2.set_yticks([1, 2, 3, 4])
ax2.set_yticklabels(["Extremely     \nUnlikely     ", "Very     \nUnlikely     ", "Unlikely     ", "Likely     "])

ax3.set_ylim([1, 3])
ax3.set_yticks([1, 2, 3])
ax3.set_yticklabels(["Extremely     \nUnlikely     ", "Very     \nUnlikely     ", "Unlikely     "])


fig.tight_layout()
fig.savefig("../figures/plot.png")
plt.show()





#%%



