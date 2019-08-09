# Functions to clean columns and make categories numerical where required
import numpy as np

USE_FOR_DONT_KNOW = np.nan

def toInt(x, divide):
  try:
    return int(int(x) / divide) * divide
  except ValueError:
    return np.nan
  return np.nan

def opinionToCategory(x):
  if not isinstance(x, str):
    return x
  xl = x.lower()
  direction = None
  if 'same' in xl or 'ffect' in xl or 'no ' in xl or xl == "don't know":
    direction = 0
  elif 'better' in xl or 'positive' in xl or 'increase' in xl or xl == 'yes':
    direction = 1
  elif 'worse' in xl or 'negative' in xl or 'decrease' in xl or xl == 'no':
    direction = -1
  magnitude = None
  if 'much' in xl or 'lot' in xl or 'very' in xl:
    magnitude = 1
  elif 'little' in xl or 'somewhat' in xl:
    magnitude = 0
  if direction == None:
    return np.nan
  return direction + (magnitude * direction if direction != 0 and magnitude != None else 0)

def opinionCleaner(col):
  return col.apply(lambda x : opinionToCategory(x))

priexp1map = {
  "Go up by 2% but less than 3%" : 2.5,
  "Go up by 1% but less than 2%": 1.5,
  "Don't know" : USE_FOR_DONT_KNOW,
  "Go up by 1% or less": 0.5,
  "Go up by 5% or more": 5.5,
  "Go up by 3% but less than 4%": 3.5,
  "Not change": 0.0,
  "Go up by 4% but less than 5%": 4.5,
  "Go down": -0.5,

  "go up by 2% but less than 3%" : 2.5,
  "go up by 1% but less than 2%": 1.5,
  "don't know" : USE_FOR_DONT_KNOW,
  "go up by 1% or less": 0.5,
  "go up by 5% or more": 5.5,
  "go up by 3% but less than 4%": 3.5,
  "not change": 0.0,
  "go up by 4% but less than 5%": 4.5,
  "go down": -0.5,

  "gone up by 2% but less than 3%" : 2.5,
  "gone up by 1% but less than 2%": 1.5,
  "don't know" : USE_FOR_DONT_KNOW,
  "gone up by 1% or less": 0.5,
  "gone up by 5% or more": 5.5,
  "gone up by 3% but less than 4%": 3.5,
  "not changed": 0.0,
  "gone up by 4% but less than 5%": 4.5,
  "gone down": -0.5,

  "Gone up by 2% but less than 3%" : 2.5,
  "Gone up by 1% but less than 2%": 1.5,
  "Don't know" : USE_FOR_DONT_KNOW,
  "Gone up by 1% or less": 0.5,
  "Up by 1% or less": 0.5,
  "Gone up by 5% or more": 5.5,
  "Gone up by 3% but less than 4%": 3.5,
  "Not changed": 0.0,
  "Gone up by 4% but less than 5%": 4.5,
  "Gone down": -0.5,
}

def toIntegers(x):
  return {
        -0.5: 0,
        0.0: 1,
        0.5: 2,
        1.5: 3,
        2.5: 4,
        3.5: 5,
        4.5: 6,
        5.5: 7
  }.get(x, x)

def cleanDataFrame(df, for_method):
  if not for_method in ['ANOVA', 'DT', 'SVM']:
    raise ValueError(for_method + 'must be one of ANOVA, DT, SVM')
  # Map priexp1 and pripas1
  df['priexp1'] = df['priexp1'].map(priexp1map)
  df['pripas1'] = df['pripas1'].map(priexp1map)

  # Convert to classes if needed
  if for_method in ['DT', 'SVM']:
    df['priexp1'] = df['priexp1'].apply(lambda x : toIntegers(x))
    df['pripas1'] = df['pripas1'].apply(lambda x : toIntegers(x))

  # Bin multi-valued numeric columns
  df['nadult'] = df['nadult'].apply(lambda x : toInt(x,1))
  df['otheradults'] = df['otheradults'].apply(lambda x : toInt(x,1))
  df['xpus_m'] = df['xpus_m'].apply(lambda x : toInt(x,1000))
  df['ustot_m'] = df['ustot_m'].apply(lambda x : toInt(x,5000))
  df['fihhyr2_m'] = df['fihhyr2_m'].apply(lambda x : toInt(x,1000))
  df['saveamount_m'] = df['saveamount_m'].apply(lambda x : toInt(x,2000))
  df['nvesttot_m'] = df['nvesttot_m'].apply(lambda x : toInt(x,50000))
  df['hr_sav_sp'] = df['hr_sav_sp'].apply(lambda x : toInt(x,200))

  # Get rid of nulls
  df = df.replace(to_replace = 'Missing', value = np.nan)
  df = df.replace(to_replace = 'Not applicable', value = np.nan)
  df = df.replace(to_replace = 'missing', value = np.nan)
  df = df.replace(to_replace = 'not applicable', value = np.nan)
  df = df.replace(to_replace = ' not applicable', value = np.nan)
  df = df.replace(to_replace = 'not main reason', value = np.nan)
  df = df.replace(to_replace = 'refused', value = np.nan)
  df = df.replace(to_replace = -9, value = np.nan)

  # Clean opinions to integers
  if for_method in ['DT','SVM']:
    for k in df.columns:
      if df[k].dtype == 'object':
        if not k in ['jbstat2','region', 'tenure', 'age_grp', 'incomequestion', 'sex']:
          df[k] = opinionCleaner(df[k])
        else:
          df[k] = df[k].astype('category').cat.codes
  
  return df