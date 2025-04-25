import pandas as pd
import os

def load_mos_file(filepath):
    # Extension du fichier
    ext = os.path.splitext(filepath)[1].lower()

    # Lire le fichier selon l'extension
    if ext in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath)
    elif ext == '.csv':
        df = pd.read_csv(filepath, sep=None, engine='python')  # détecte automatiquement le séparateur
    else:
        raise ValueError("Format de fichier non supporté.")

    # Suppression des colonnes vides
    df = df.dropna(axis=1, how='all')

    # Suppression des lignes vides
    df = df.dropna(how='all')

    # Heuristique : on suppose que la première colonne contient le nom de l'objet
    name_col = df.columns[0]

    # Heuristique pour la colonne MOS : on cherche la première colonne numérique (float/int)
    mos_col = None
    for col in df.columns[1:]:
        if pd.api.types.is_numeric_dtype(df[col]):
            mos_col = col
            break

    if mos_col is None:
        raise ValueError("Aucune colonne de type numérique trouvée pour les MOS.")

    # Retourne un DataFrame propre
    mos_df = df[[name_col, mos_col]].copy()
    mos_df.columns = ['name', 'mos']

    return mos_df

# csvfile = load_mos_file('D:/These/Vscode/BDD/TSMD/TSMD_MOS/TSMD_MOS.xlsx')
# # Saving the DataFrame to a CSV file
# csvfile.to_csv('D:/These/Vscode/BDD/TSMD/TSMD_MOS/TSMD_MOS.csv', index=False, sep=',')
