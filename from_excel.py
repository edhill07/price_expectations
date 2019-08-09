import pandas as pd
import sys

if len(sys.argv) != 2:
  raise SystemExit("Wrong number of arguments : provide the path to the xlsx file")
else :
  for sheet in ['2017H2','2018H2']:
    df = pd.read_excel(sys.argv[1], sheet_name = sheet)
    df.to_csv(sheet + '.csv')