import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols, wls
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.svm import SVC

from cleaners import cleanDataFrame

AT_LEAST_THIS_MANY_OBS = 4000 # Only includes columns which have more non-nulls
METHOD = 'SVM' #'ANOVA', 'DT' or 'SVM'
ANOVA_LEVELS = 3 # Number of columns to add when improving in ANOVA method
INCLUDE_PRIPAS1 = False # whether to allow pripas1 in the fit
TRAINING_SET = 4000 # size of training set for DT/SVM
INPUT_FILE = '2018H2.csv' # input file

# Gets the columns which most improve the adjusted R^2 score
def getBestColumns(df, columns, patsy_string_so_far, for_method, includePripas1):
  best_columns = []
  for x in columns:
    if df[x].nunique() > 20 and for_method == 'ANOVA':
      continue
    # Remove future-looking columns
    if x in (['subsid', 'weight', 'priexp1'] + ([] if includePripas1 else ['pripas1'])):
      continue
    if 'priexp' in x or 'genecon' in x or x == 'downchance':
      continue
    if 'exp' in x or 'brexit_' in x:
      continue
    if for_method in ['DT', 'SVM'] and 'age_grp' == x:
      continue
    formula = 'df.priexp1 ~ ' + patsy_string_so_far + 'C(' + x + ')'
    try:
      lm = wls(formula, df, weights = df.weight).fit()
      if lm.nobs > AT_LEAST_THIS_MANY_OBS:
        best_columns.append([lm.rsquared_adj,x, lm.params])
    except:
      pass #don't handle

  best_columns.sort(reverse = True)
  return best_columns[:(5 if for_method == 'SVM' else 20)]

# BEGIN CODE

df = pd.read_csv(INPUT_FILE)
df = cleanDataFrame(df, METHOD)

columns = [x for x in df.columns][1:]
columns = [x for x in columns if df[x].count() > AT_LEAST_THIS_MANY_OBS]

if METHOD == 'ANOVA':
  print ('Ran ANOVA {0} levels and {1}'.format(ANOVA_LEVELS, 'including pripas1' if INCLUDE_PRIPAS1 else 'without pripas1'))
  patsy_string_so_far = ''
  for lk in range(ANOVA_LEVELS):
    best_columns = getBestColumns(df, columns, patsy_string_so_far, METHOD, INCLUDE_PRIPAS1)
    patsy_string_so_far += 'C(' + best_columns[0][1] + ') +'
  formula = 'df.priexp1 ~ ' + patsy_string_so_far[:-1]
  lm = wls(formula, df, weights = df.weight).fit()
  print (lm.summary())
  table = sm.stats.anova_lm(lm)
  print (table)

elif METHOD in ['DT','SVM']:
  print ('Ran {0} {1}'.format(METHOD,'including pripas1' if INCLUDE_PRIPAS1 else 'without pripas1'))
  best_columns = [ x[1] for x in getBestColumns(df, columns, '', METHOD, INCLUDE_PRIPAS1)]
  df_tree = df[best_columns + ['priexp1','weight']]
  df_tree = df_tree.dropna(subset = ['priexp1'])
  df_tree = df_tree.replace(to_replace = np.nan, value = -3)
  if METHOD == 'DT':
    clf = tree.DecisionTreeClassifier(max_depth=6)
    clf = clf.fit(df_tree[best_columns][:TRAINING_SET], df_tree['priexp1'][:TRAINING_SET])
    df_tree['predicted'] = clf.predict(df_tree[best_columns])
    df_test = df_tree[['predicted', 'priexp1', 'weight']][TRAINING_SET:]
    features = list(zip(clf.feature_importances_, best_columns))
  elif METHOD == 'SVM':
    svm = SVC(gamma='auto', kernel = 'linear')
    svm.fit(df_tree[best_columns][:TRAINING_SET], df_tree['priexp1'][:TRAINING_SET])
    df_tree['predicted'] = svm.predict(df_tree[best_columns])
    df_test = df_tree[['predicted', 'priexp1', 'weight']][TRAINING_SET:]
    coef = svm.coef_.tolist()
    max_coef = [ max([abs(coef[c][i]) for c in range(len(coef))]) for i in range(len(best_columns))]
    features = list(zip(max_coef, best_columns))
  features.sort(reverse = True)
  print ('Features')
  print (features)

  sum_overall=0
  sums_by_class = [0] * 8
  total_by_class = [0] * 8
  for i in range(len(df_test)):
    clas = int(df_test.iloc[i]['priexp1'])
    total_by_class[clas] += 1
    if clas == df_test.iloc[i]['predicted']:
      sum_overall +=1
      sums_by_class[clas] += 1
  
  total_possible = len(df_test)
  random_chance = sum([total_by_class[i] * total_by_class[i] for i in range(8)]) / (total_possible * total_possible)
  print ('Number correct', sum_overall / total_possible)
  print ('Random chance', random_chance)