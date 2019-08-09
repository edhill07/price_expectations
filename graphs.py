import matplotlib.pyplot as plt
import pandas as pd

from cleaners import cleanDataFrame, opinionCleaner

df = pd.read_csv('2018H2.csv')
cleanDataFrame(df,'ANOVA')

# Graphs of distributions

fig, axs = plt.subplots(1,2, sharey=True)

filters = ['very negative','very positive about the result']
titles = ['Very negative about the result','Very positive about the result']

counts = [df['brexit'].value_counts()[f] for f in filters]
points = [df[df['brexit'] == f].priexp1.value_counts().sort_index() * 100 / counts[i] for i, f in enumerate(filters)]

for i, x in enumerate(axs):
  x.plot(points[i], 'o', color = 'k')
  x.errorbar(points[i].index, points[i], xerr=0.5, linestyle='',linewidth=0.5, color = '0.75')
  
  x.set_title(titles[i])
  x.set_xlabel('Expected price change (%)')

  x.spines["top"].set_visible(False)     
  x.spines["right"].set_visible(False)

axs[0].set_ylabel('Percentage of households')

plt.show()

#Graphs of trends

variables = ['pripas1', 'brexit', 'age_grp', 'hhfinpas1']
for x in variables:
  if x != 'age_grp':
    df[x] = opinionCleaner(df[x]) #To get orderings right
means = [df.groupby(x)['priexp1'].mean() for x in variables]
stds = [df.groupby(x)['priexp1'].std() for x in variables]

fig, axs = plt.subplots(2,2)

for i, row in enumerate(axs):
  for j, x in enumerate(row):
    graph_number = i*2 + j
    x.plot(means[graph_number], 'o', color = 'k')
    x.spines["top"].set_visible(False)     
    x.spines["right"].set_visible(False)
    x.errorbar(means[graph_number].index, means[graph_number], yerr=2*stds[graph_number], linestyle='',linewidth=0.5, color = '0.75')
    x.set_title(variables[graph_number], y=0.97)

age = axs[1][0]
for tick in age.xaxis.get_major_ticks():
  tick.label.set_fontsize('x-small') 
  tick.label.set_rotation('vertical')

axs[0][1].set_xticklabels(['', '-ve', '', '',  '','+ve'])
axs[1][1].set_xticklabels(['', '-ve', '', '',  '','+ve'])

plt.show()