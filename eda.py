import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

org_data_path = "../CHESS/data/chess_games.csv"
chess_data = pd.read_csv(org_data_path)

# Tratamiento de valores faltantes
print(chess_data.isnull().sum())

# Calculo de valores faltantes en columna opening response
total_entries = 20058
missing_opening_response = 18851

percentage_missing = (missing_opening_response / total_entries) * 100
print(percentage_missing)

# Eliminación de columna opening response
df_1 = chess_data.drop(['opening_response'], axis=1)

#rellenando la columna Opening_variation los blanks con "traditional opening"
df_1['opening_variation'] = chess_data['opening_variation'].fillna('traditional opening')
df_1.info()

# Descripción de las columnas numéricas del DF
chess_data.describe()

'''
Ahora que se tiene un dataframe sin valores faltantes. Es posible proceder con los siguientes analisis: 
- Distribución de las variables numéricas
- Exploración de las variables categóricas
Para dividir las calificaciones de los jugadores en las categorías de "Beginner", "Intermediate", "Advanced", y "Master" se determinará mediante cuartiles.

Primero, se determinan los valores de estos cuartiles para las calificaciones de los jugadores de blanco y negro.

Luego, se crea una función que asigna cada calificación a una de estas categorías según los cuartiles calculados.
'''

import numpy as np
from scipy import stats
combined_ratings = pd.concat([chess_data['white_rating'], chess_data['black_rating']])
combined_ratings
print("Promedio de Ratings: %.3f" % np.mean(combined_ratings))
print("Mediana de Ratings: %.3f" % np.median(combined_ratings))
print("Desviación estandar de elos: %.3f" % np.std(combined_ratings))
overall_quartiles = combined_ratings.quantile([0.25, 0.5, 0.75]).values
print("50%c de los datos entre %.3f y %.3f" % ('%', overall_quartiles[0], overall_quartiles[2]))
print("Skew de la distribución: %.3f" % (stats.skew(combined_ratings))) 
print(overall_quartiles)

'''
Conversión a variable categórica de las coluimnas white_rating y black_rating

Clasificaciones:
- Begginer: Abajo de percentil 25%
- Intermediate: Entre 25% y 50%
- Advanced: Entre 50% y 75%
- Master: Arriba de 75%

Por tanto:
- Percentil 25%: 1394
- Percentil 50%: 1564
- Percentil 75%: 1788

Con esta información se crean dos categorías: white_category y black_category
'''
def categorize_rating(rating, overall_quartiles):
    if rating <= overall_quartiles[0]:
        return 'Beginner'
    elif rating <= overall_quartiles[1]:
        return 'Intermediate'
    elif rating <= overall_quartiles[2]:
        return 'Advanced'
    else:
        return 'Master'
    
chess_data['white_category'] = chess_data['white_rating'].apply(lambda x: categorize_rating(x, overall_quartiles))
chess_data['black_category'] = chess_data['black_rating'].apply(lambda x: categorize_rating(x, overall_quartiles))


# Simplificación de la variable categórica time_increment
# Definimos cadenas válidas para cada categoría
bullet_times = {str(i): 'Bullet' for i in range(0, 2)}
blitz_times = {str(i): 'Blitz' for i in range(3, 5)}
rapid_times = {str(i): 'Rapid' for i in range(6, 30)}
classical_times = {str(i): 'Classical' for i in range(30, 60)}
personalizado_times = {str(i): 'Personalizado' for i in range(60, 181)}

# Unimos todos los diccionarios en uno solo
all_times = {**bullet_times, **blitz_times, **rapid_times, **classical_times, **personalizado_times}

def categorize_time_base(time_increment):
    # Separar el tiempo base y el incremento
    time_base, _ = time_increment.split('+')
    # Devolver la categoría correspondiente basándonos solo en el tiempo base
    return all_times.get(time_base, 'Personalizado')

# Aplicar la función para crear una nueva columna 'category'
chess_data['time_category'] = chess_data['time_increment'].apply(categorize_time_base)

# Verificar los resultados
chess_data[['time_increment', 'time_category']].head()

# Visualización de columnas categóricas
categorical_columns = ['victory_status', 'winner', 'time_category',
       'moves', 'opening_code', 'opening_shortname', 'opening_variation', 'white_category',
       'black_category']

# Función para trazar las 5 categorías principales de una columna dada
def plot_top_categories(data, column, ax, title, font_size=20):
    top_categories = data[column].value_counts().head(5)
    top_categories.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title(title, fontsize=font_size + 5)
    # ax.set_xlabel(column, fontsize=font_size)
    ax.set_ylabel('Frequency', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.tick_params(axis='both', which='minor', labelsize=font_size - 2)

# Crear la figura y los ejes
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(30, 40))
fig.tight_layout(pad=5.0)

# Iterar a través de cada columna y eje, trazando las categorías principales
for ax, column in zip(axes.flatten(), categorical_columns):
    if column in chess_data:
        plot_top_categories(chess_data, column, ax, f'Top 5 {column.capitalize()}')

# Ajustar la figura para mostrar todos los gráficos claramente
plt.tight_layout()
plt.show()


# Visualización de distribución de variables numericas 
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
fig.tight_layout(pad=5.0)

# Lista de columnas numéricas a graficar, excluyendo 'game_id'
numeric_columns = ['turns', 'white_rating', 'black_rating', 'opening_moves']

# Graficar histogramas para cada columna numérica
for ax, column in zip(axes.flatten(), numeric_columns):
    chess_data[column].hist(bins=30, ax=ax, color='cadetblue')
    ax.set_title(f'Distribución columna {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frecuencia')

# Ajustar la disposición para evitar la superposición de etiquetas
plt.tight_layout()
plt.show()

# Sskewness and kurtosis for the specified columns
skewness = chess_data[['turns', 'opening_moves']].skew().rename('Skewness')
kurtosis = chess_data[['turns', 'opening_moves']].kurtosis().rename('Kurtosis')

# Combining the results into a single DataFrame
skew_kurtosis_stats = pd.concat([skewness, kurtosis], axis=1)
skew_kurtosis_stats


# Boxplot for the 'opening_moves' column
plt.figure(figsize=(8, 6))
plt.boxplot(chess_data['opening_moves'], vert=False)
plt.title('Boxplot of Opening Moves')
plt.xlabel('Number of Opening Moves')
plt.show()

# Distribución de nivel de jugadores vs Frecuencia de juegos
# Distribution of player ratings
plt.figure(figsize=(10, 5))
sns.histplot(chess_data['white_rating'], color='blue', bins=30, kde=True, label='Nivel Blancas')
sns.histplot(chess_data['black_rating'], color='red', bins=30, kde=True, label='Nivel Negras')
plt.axvline(x=overall_quartiles[0], linestyle='--', markersize=12)
plt.axvline(x=overall_quartiles[2], linestyle='--', markersize=12)
plt.title('Distribución de nivel de jugadores')
plt.xlabel('Nivel')
plt.ylabel('Frecuencia')
plt.legend()

plt.tight_layout()
plt.show()

# Aperturas más comunes según nivel y su frecuencia de juego
# Filter data for each category
beginner_openings = chess_data[chess_data['white_category'] == 'Beginner']['opening_fullname'].value_counts().head(5)
intermediate_openings = chess_data[chess_data['white_category'] == 'Intermediate']['opening_fullname'].value_counts().head(5)
advanced_openings = chess_data[chess_data['white_category'] == 'Advanced']['opening_fullname'].value_counts().head(5)
master_openings = chess_data[chess_data['white_category'] == 'Master']['opening_fullname'].value_counts().head(5)

# Set up the visualization layout
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

sns.barplot(ax=axes[0, 0], x=beginner_openings.values, y=beginner_openings.index, palette='Pastel1')
axes[0, 0].set_title('Top 5 Aperturas nivel principiante')
axes[0, 0].set_xlabel('Frecuencia')
axes[0, 0].set_ylabel('Apertura')

sns.barplot(ax=axes[0, 1], x=intermediate_openings.values, y=intermediate_openings.index, palette='Pastel2')
axes[0, 1].set_title('Top 5 Aperturas nivel intermedio')
axes[0, 1].set_xlabel('Frequency')
axes[0, 1].set_ylabel('Opening')

sns.barplot(ax=axes[1, 0], x=advanced_openings.values, y=advanced_openings.index, palette='Set3')
axes[1, 0].set_title('Top 5 Aperturas nivel avanzado')
axes[1, 0].set_xlabel('Frequency')
axes[1, 0].set_ylabel('Opening')

sns.barplot(ax=axes[1, 1], x=master_openings.values, y=master_openings.index, palette='Set1')
axes[1, 1].set_title('Top 5 Aperturas nivel Maestro')
axes[1, 1].set_xlabel('Frequency')
axes[1, 1].set_ylabel('Opening')

plt.tight_layout()
plt.show()


# Distribución de color y estatus de victoria de los ganadores
sns.set_theme(style="whitegrid")

victory_counts = chess_data['victory_status'].value_counts()
winner_counts = chess_data['winner'].value_counts()
victory_percentages = (victory_counts / victory_counts.sum()) * 100
winner_percentages = (winner_counts / winner_counts.sum()) * 100

# Set up the figure and axes for updated bar charts with numbers and percentages
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart for winners with count and percentage labels
ax[0].bar(winner_counts.index, winner_counts.values, color='orange')
ax[0].set_title('Distribución de color de los ganadores')
ax[0].set_xlabel('Ganador')
ax[0].set_ylabel('Número de partidas')
for i, v in enumerate(winner_counts):
    ax[0].text(i, v + 50, f"{v} ({winner_percentages[i]:.1f}%)", color='black', ha='center')

# Bar chart for victory statuses with count and percentage labels
ax[1].bar(victory_counts.index, victory_counts.values, color='purple')
ax[1].set_title('Distribución del estatus de la victoria')
ax[1].set_xlabel('Estatus de la victoria')
ax[1].set_ylabel('Número de juegos')
for i, v in enumerate(victory_counts):
    ax[1].text(i, v + 50, f"{v} ({victory_percentages[i]:.1f}%)", color='black', ha='center')



plt.tight_layout()
plt.show()


# Frecuancia del top 10 de aperturas de maestros
master_openings = chess_data[chess_data['white_category'] == 'Master']['opening_fullname'].value_counts().head(10)


# Win rates for these openings
win_rates = chess_data[chess_data['opening_fullname'].isin(master_openings.index)].groupby(['opening_fullname', 'winner']).size().unstack(fill_value=0)
win_rates['total'] = win_rates.sum(axis=1)
win_rates['white_win_rate'] = win_rates['White'] / win_rates['total']
win_rates['black_win_rate'] = win_rates['Black'] / win_rates['total']
win_rates['draw_rate'] = win_rates['Draw'] / win_rates['total']

# Sorting by total games to see the most popular openings in terms of games played
win_rates_sorted = win_rates.sort_values(by='total', ascending=False)

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
win_rates_sorted[['white_win_rate', 'black_win_rate', 'draw_rate']].plot(kind='barh', ax=ax, color=['purple', 'black', 'gray'])


ax.set_title('Frecuencia del top 10 de aperturas de maestros')
ax.set_xlabel('Frecuencia')
ax.set_ylabel('Apertura')
ax.legend(title='Jugador')
plt.xticks(rotation=90)
plt.show()

# Guardar el dataframe
chess_data.to_csv("../CHESS/data/df_1.csv", index=False)