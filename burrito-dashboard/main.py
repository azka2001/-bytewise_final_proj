import pandas as pd
import numpy as np
import atoti as tt
import re
from sklearn.preprocessing import MinMaxScaler

ATOTI_HIDE_EULA_MESSAGE = True
ATOTI_DISABLE_TELEMETRY = True

df = pd.read_csv('burritos_01022018.csv')
#print(df.to_string())
#print(df.head)
#print(df.describe)
rm_par = df.columns = [re.sub("([\(\[]).*?([\)\]])", "", x).strip() for x in df.columns]
rm_col = df.columns = [x.replace(':','_').strip() for x in df.columns]
#print(rm_par)
#print(rm_col)

#print(df.isnull().any())
#print(df.isnull().sum()/df.shape[0])

#session = tt.create_session(config={'user_content_storage': "./content", "port": 9000})
session = tt.Session()

scaler = MinMaxScaler()
burrito_vars_norm = df.loc[:, ['Circum', 'Volume', 'Length', 'Mass', 'Cost']]
#print(burrito_vars_norm.isnull().sum())
bnorms = scaler.fit_transform(burrito_vars_norm)*10

df[['Circum_norm', 'Volume_norm', 'Length_norm', 'Mass_norm', 'Cost_norm']] = bnorms
#print(df.columns)

burrito_variables = pd.melt(df.reset_index(), id_vars=['Location', 'Burrito'],
                            value_vars=['Circum_norm', 'Volume_norm', 'Length_norm', 'Mass_norm', 'Cost_norm'])
#print(burrito_variables)

burrito_table = session.read_pandas(df, table_name='burritos')
#print(burrito_table.head())

cube = session.create_cube(burrito_table)
h = cube.hierarchies
l = cube.levels
m = cube.measures

m['five']=5
#print[m['five']]

session.visualize('exploration 1')
