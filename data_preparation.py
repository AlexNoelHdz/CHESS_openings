import pandas as pd
org_data_path = "../CHESS/data/df_3.csv"
df_3 = pd.read_csv(org_data_path)
# Eliminar empate ya que no abona al objetivo
df_3 = df_3[df_3['winner'] != 'Draw']
print(df_3.info())

# Codificación con Label Encoder
from sklearn.preprocessing import LabelEncoder
# Codificando todas las variables categóricas consideradas en valores numéricos
le1 = LabelEncoder() #rated
le2 = LabelEncoder() #winner
le3 = LabelEncoder() #time_increment
le4 = LabelEncoder() #opening_code
le5 = LabelEncoder() #opening_fullname
le6 = LabelEncoder() #opening_shortname
le7 = LabelEncoder() #opening_variation
le8 = LabelEncoder() #moves_fen
le9 = LabelEncoder() #current_turn

# Apply label encoding to each categorical column
df_3['rated_cod'] = le1.fit_transform(df_3['rated']) # Ordinal
df_3['winner_cod'] = le2.fit_transform(df_3['winner']) # No ordinal, pero solo tres clases
df_3['current_turn_cod'] = le9.fit_transform(df_3['current_turn']) # Ordinal
df_3['time_increment_cod'] = le3.fit_transform(df_3['time_increment']) # Ordinal
df_3['opening_code_cod'] = le4.fit_transform(df_3['opening_code']) # No ordinal, muchas categorias
df_3['opening_fullname_cod'] = le5.fit_transform(df_3['opening_fullname']) # No ordinal, muchas categorias
df_3['opening_shortname_cod'] = le6.fit_transform(df_3['opening_shortname']) # No ordinal, muchas categorias
df_3['opening_variation_cod'] = le7.fit_transform(df_3['opening_variation']) # No ordinal, muchas categorias
df_3['moves_fen_cod'] = le8.fit_transform(df_3['moves_fen']) # No ordinal, muchas categorias

# Visualización de clases para el target
pares_unicos = df_3[['winner', 'winner_cod']].drop_duplicates()
print(pares_unicos)

# Guardar el CSV 
df_3.to_csv("../CHESS/data/df_3_cod.csv", index=False) # Data frame con features basadas en ventaja posicional del turno