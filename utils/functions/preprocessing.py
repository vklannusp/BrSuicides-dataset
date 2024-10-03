import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

# ----------------------------------------------------------------------------------------
# ---------------------- GENERAL FUNCTIONS AND VARIABLES ---------------------------------

initial_df_dtypes = {
    'ESTADO': 'object',
    'ANO': 'int64',
    'NATURAL': 'object',
    'DTNASC': 'object',
    'IDADE': 'object',
    'SEXO': 'object',
    'RACACOR': 'object',
    'ESTCIV': 'object',
    'ESC': 'object',
    'OCUP': 'object',
    'CODMUNRES': 'int64',
    'LOCOCOR': 'object',
    'CODMUNOCOR': 'int64',
    'CAUSABAS': 'object',
    'ESC2010': 'object',
    'ESCFALAGR1': 'object'
}

initial_parse_dates = ['DTOBITO']

late_df_dtypes = {
    'ESTADO': 'object',
    'ANO': 'int64',
}

late_parse_dates = ['DTOBITO', 'DTNASC']


def generate_path(path_to_file):
    current_dir = os.path.dirname(__file__)
    full_path = os.path.abspath(os.path.join(current_dir, path_to_file))
    return full_path


# If there is a need to save each state .csv separately
def save_cleaned_df_sep_by_years(df, year_col, folder_location):
    os.makedirs(folder_location, exist_ok=True)
    unique_years = df[year_col].unique()
    
    for year in unique_years:
        year_df = df[df[year_col] == year]
        if not year_df.empty:
            file_path = os.path.join(folder_location, f'{year}.csv')
            year_df.to_csv(file_path, index=False)
            print(f"Saved {file_path}")


# Attempts to convert a float to an int if the float is a whole number.
def float_to_int(x):
    if pd.isna(x):
        return x  # Keep NaN as is
    elif isinstance(x, float) and x.is_integer():
        # Converts floats that are whole numbers
        # to integers and then strings?????? it works though
        return str(int(x))  
    else:
        # Returns the original value
        return x


def check_missing_keys(df, c_dict):
    # Get value counts for each column in the df
    value_counts_dict = {col: df[col].value_counts().to_dict() for col in df.columns if col in c_dict}
    # To check later if the cols were filtered out
    missing_keys_columns = []
    # Checks for keys in the dataset that don't exist in the conversion dictionary
    for col, counts in value_counts_dict.items():
        conversion_keys = c_dict[col].keys()
        missing_keys = [key for key in counts.keys() if key not in conversion_keys and key != '' and key != np.nan]
        if missing_keys:
            missing_keys_columns.append(col)
            print(f"Column '{col}' nominal values missing: {missing_keys}")
    return missing_keys_columns

# ----------------------------------------------------------------------------------------
# ---------------------- DATES COLUMNS FUNCTIONS AND VARIABLES ---------------------------

def preprocess_date_column(df, date_col, remove_invalid=True):
    n_of_missing = 0
    n_of_missing += df[date_col].apply(lambda x: 1 if str(x).strip() == '' or x == 0 else 0).sum()
    # Convert the 'date' column to datetime and coerce errors
    print(f'Column: {date_col}')
    print(f'Unstandardized date example:\n{df[date_col][0:2]}')

    df[date_col] = df[date_col].apply(lambda x: str(x)[0:-6] + "-" + str(x)[-6:-4] + "-" + str(x)[-4:] if x != '' else np.nan)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', format="%d-%m-%Y")
    print(f'\nStandardized date example:\n{df[date_col][0:2]}\n')
    
    invalid_rows = df[df[[date_col]].isna().any(axis=1)]
    n_of_invalid = len(invalid_rows)

    n_of_preprocessed, n_of_removed = 0, 0
    if remove_invalid:
        df = df.dropna(subset=[date_col])
        n_of_valid = len(df) - n_of_missing - n_of_invalid
        print(f'Nº of removed {date_col} invalid rows: {len(invalid_rows)}\n')
        print(f'Example of removed {date_col} invalid rows:\n{invalid_rows[date_col].head(5)}\n')
        n_of_removed += n_of_invalid
        n_of_preprocessed += n_of_valid - n_of_removed
    else:
        n_of_valid = len(df) - n_of_missing - n_of_invalid
        print(f'Nº of not removed {date_col} invalid rows: {len(invalid_rows)}\n')
        n_of_preprocessed += n_of_invalid

    return df, [date_col, n_of_missing, n_of_invalid, n_of_valid, n_of_preprocessed, n_of_removed]

# ----------------------------------------------------------------------------------------
# ---------------------- NUMERIC COLUMNS FUNCTIONS AND VARIABLES -------------------------

def clean_numeric_columns(df, num_col):
    print(f'Handling {num_col} column...')
    # Initialize counters
    n_of_missing, n_of_invalid, n_of_valid = 0, 0, 0

    # Function to convert and count values
    def convert_and_count(x):
        nonlocal n_of_missing, n_of_invalid, n_of_valid
        if pd.isna(x):
            n_of_missing += 1
            return float('nan')
        try:
            num_value = pd.to_numeric(x, errors='raise')
            if float(num_value).is_integer():
                n_of_valid += 1
                return int(num_value)
            else:
                n_of_valid += 1
                return int(np.floor(num_value))
        except (ValueError, TypeError):
            n_of_invalid += 1
            return 0

    df[num_col] = df[num_col].apply(convert_and_count)
    return df, [num_col, n_of_missing, n_of_invalid, n_of_valid, 0, 0]

# ----------------------------------------------------------------------------------------
# ---------------------- CATEGORICAL COLUMNS FUNCTIONS AND VARIABLES ---------------------

def clean_categorical_values_column(df, current_col, column_conversion_dict):
    print(f'Handling {current_col} column...')
    # Convert values using the provided column_conversion_dict
    df[current_col] = df[current_col].map(column_conversion_dict).fillna(df[current_col])

    # Count value occurrences before cleaning
    df_value_counts = df[current_col].value_counts()
    print(df_value_counts, '\n')

    n_of_missing = df[current_col].isna().sum()
    nan_value = 'ignorado'
    n_of_invalid = df[current_col].str.lower().eq(nan_value).sum()
    n_of_valid = df[current_col].shape[0] - (n_of_missing + n_of_invalid)

    # Print updated counts after cleaning
    df_value_counts2 = df[current_col].value_counts()
    print(df_value_counts2, '\n')
    return df, [current_col, n_of_missing, n_of_invalid, n_of_valid, 0, 0]

# ----------------------------------------------------------------------------------------
# ---------------------- AGE AND BIRTH DATE FUNCTIONS AND VARIABLES ----------------------

def clean_age_values_column(df, age_col, birth_col, death_col):
    print(f'Handling {age_col} column...')

    # A very specific float to determine if the age is invalid
    invalid_age_value = 1.919191
    n_of_missing, n_of_invalid, n_of_valid = 0, 0, 0

    def calculate_age(row):
        nonlocal n_of_missing, n_of_invalid, n_of_valid
        age_str = str(row[age_col])
        # First step, check if the value is a valid number
        if age_str and age_str != 'nan':
            # Gets the first subgroup (that determines the scale of time, hours, mins etc.)
            first_digit = int(age_str[0])
            # Gets the second subgroup (that holds the value)
            value = int(age_str[1:]) if len(age_str) > 1 else None
            # Since it's suicide data, we only care about year units (4 or 5)
            if first_digit in [4]:
                if value >= 5:
                    n_of_valid += 1
                    return value
                # According to the sources, suicide only counts for victims 
                # equal to or older than 5 years
                else:
                    n_of_invalid += 1
                    return invalid_age_value
            # If it's 5, means if higher than 100 years, so we sum the value
            elif first_digit in [5]:
                n_of_valid += 1
                return (100 + value)
        # Second step, try to extract valid ages with date subtraction
        birth_date = pd.to_datetime(row[birth_col], errors='coerce')
        death_date = row[death_col]
        # If both are not null
        if pd.notna(birth_date) and pd.notna(death_date):
            # Get the age difference in years between death and birth dates
            age_diff = (death_date - birth_date).days // 365
            # If it's more or equal than 5
            if age_diff >= 5:
                print(f'IDADE: {first_digit}, {value}, trying to convert dates: , {death_date} - {birth_date} -> {age_diff}\n')
                n_of_valid += 1
                return age_diff
            else:
                n_of_invalid += 1
                return invalid_age_value
        # Third step, check if the value is missing or invalid
        if age_str == 'nan':
            n_of_missing += 1
            return np.nan
        else:
            n_of_invalid += 1
            return invalid_age_value

    # Apply the function to the column and transform values
    df[age_col] = df.apply(calculate_age, axis=1)
    
    # Print summary information
    print(f"Nº of age missing values: {n_of_missing}")
    print(f"Nº of age invalid values (age < 5): {n_of_invalid}")
    print(f"Nº of age valid values (age >= 5): {n_of_valid}")
    
    return df, [age_col, n_of_missing, n_of_invalid, n_of_valid, 0, 0]

# ----------------------------------------------------------------------------------------
# ---------------------- OCUPATION FUNCTIONS AND VARIABLES -------------------------------

def check_ocupation_code(code):
    if pd.isna(code):
        return code
    try:
        ocup_float = float(code)
        if ocup_float <= 0:
            return 0
        return int(np.floor(ocup_float))
    except ValueError:
        return 0

def clean_ocupation_column(df, ocup_col, ocup_conversion_dict=None):
    print(f'Handling {ocup_col} column...')
    # From where the CBO list was downloaded from:
    # http://www.mtecbo.gov.br/cbosite/pages/downloads.jsf
    
    CBO2002 = pd.read_csv(generate_path('../infos/cbo/CBO.csv'), sep=',')
    CBO2002_dict = CBO2002.sort_values(by="CODIGO").set_index("CODIGO").to_dict()
    CBO2002_dict['OCUP'] = CBO2002_dict.pop('OCUPACAO')
    # CBO94 conversion table:
    # http://www.joviccontabilidade.com.br/assets/tabelacbo-completa.pdf
    # CBO94 Database:
    # http://www.mtecbo.gov.br/cbosite/pages/tabua/BaseDados_CBO94.jsf

    # I'm not gonna do this for now
    # CBO94_dict = {
    #     "OCUP94": {
    #         # This has to be created because current CBO 
    #         # works only from 2002 onwards
    #         "61200": "Agricultor familiar generalizado"
    #     }
    if ocup_conversion_dict == None:
        ocup_conversion_dict = CBO2002_dict['OCUP']
    print(ocup_conversion_dict)

    n_of_missing, n_of_invalid, n_of_valid = 0, 0, 0
    if ocup_col and ocup_conversion_dict:
        df_ocupation = df.copy()
        # Apply the check_ocupation_code function first
        df_ocupation[ocup_col] = df_ocupation[ocup_col].apply(lambda x: check_ocupation_code(x))
        def convert_and_count(x):
            nonlocal n_of_missing, n_of_invalid, n_of_valid
            if pd.isna(x):
                n_of_missing += 1
                return float('nan')
            elif x in ocup_conversion_dict:
                n_of_valid += 1
                return ocup_conversion_dict.get(x)
            else:
                n_of_invalid += 1
                return int(x)

        # Apply the conversion, converting empty strings to NaN and preserving 0 and unconverted values
        df_ocupation[ocup_col] = df_ocupation[ocup_col].apply(convert_and_count)
        print(f"Ocupation codes for {ocup_col} replaced using a dict for ocupation codes.")
        return df_ocupation, [ocup_col, n_of_missing, n_of_invalid, n_of_valid, 0, 0]
    else:
        print('No column was altered because a column and dict must be given.')
        return df, _

# ----------------------------------------------------------------------------------------
# ---------------------- CITY CODES FUNCTIONS AND VARIABLES ------------------------------

def read_city_codes_csv_with_version(url_or_path):
    pd_version = pd.__version__
    if pd_version >= '1.3':
        return pd.read_csv(url_or_path, on_bad_lines='skip', sep=";", encoding="latin-1", usecols=["IBGE", "Município"])
    else:
        return pd.read_csv(url_or_path, error_bad_lines=False, sep=";", encoding="latin-1", usecols=["IBGE", "Município"])
    

def generate_municipality_dict(newer_dict, old_dict=None):
    combined_dict = newer_dict.copy()
    if old_dict:
        print('Combining the dictionaries:')
        combined_dict.update(old_dict)
    for key, value in newer_dict.items():
        truncated_key = key[:-1]
        if truncated_key not in combined_dict:
            combined_dict[truncated_key] = value
    print('Dictionaries combined.')
    return combined_dict


def get_municipality_codes_dict(show_results=False):
    import pkg_resources
    import subprocess
    # Package checking list
    packages_to_check = [
        {
        'name': 'xlrd',
        'version':'2.0.1'
        },
        ]
    # check if necessary packages are installed, and install if not
    for package in packages_to_check:
        print(package)
        try:
            pkg_resources.get_distribution(package['name'])
            print(f"{package['name']} is installed.")
        except pkg_resources.DistributionNotFound:
            print(f"{package['name']} is not installed. Installing:")
            # If you're using a .ipynb, use the default %pip install, otherwise use subprocess
            # %pip install {package['name']}=={package['version']}
            subprocess.check_call(["pip", "install", f"{package['name']}=={package['version']}"])

    # http://blog.mds.gov.br/redesuas/lista-de-municipios-brasileiros/
    # site do governo nunca tem assinatura digital
    url_municipality_ibge = "http://blog.mds.gov.br/redesuas/wp-content/uploads/2018/06/Lista_Munic%C3%ADpios_com_IBGE_Brasil_Versao_CSV.csv"

    municipality_ibge_path = generate_path("../infos/codmunres/Lista_Municipios_com_IBGE_Brasil_Versao_CSV.csv")
    municipality_dtb_path_new = generate_path("../infos/codmunres/RELATORIO_DTB_BRASIL_MUNICIPIO.xls")

    # Ensure the local directory exists
    os.makedirs(os.path.dirname(municipality_ibge_path), exist_ok=True)

    if not os.path.isfile(municipality_ibge_path):
        try:
            mun_codes = read_city_codes_csv_with_version(url_municipality_ibge)
            mun_codes.to_csv(municipality_ibge_path, sep=";", encoding="latin-1", index=False)
        except Exception as e:
            print(f"Error downloading the file: {e}")
            raise
    else:
        mun_codes = read_city_codes_csv_with_version(municipality_ibge_path)
    mun_codes = mun_codes.set_index("IBGE").to_dict()["Município"]
    mun_codes_new = pd.read_excel(municipality_dtb_path_new)
    # Select the columns from the 8th row onwards
    mun_codes_new = mun_codes_new.iloc[7:]
    rightmost_columns = mun_codes_new.iloc[1:, -2:]
    mun_codes_new = pd.Series(rightmost_columns.iloc[:, 1].values, index=rightmost_columns.iloc[:, 0]).to_dict()
    mun_codes_dict = generate_municipality_dict(mun_codes_new, mun_codes)

    if show_results:
        print([key_value for i, key_value in enumerate(mun_codes.items()) if i < 4])
        print([key_value for i, key_value in enumerate(mun_codes_new.items()) if i < 4])
        print([key_value for i, key_value in enumerate(mun_codes_dict.items()) if i < 4])
    return mun_codes_dict


def clean_municipality_code_column(df, mun_col, mun_codes_dict=None):
    print(f'Handling {mun_col} column...')
    mun_df = df.copy()

    if mun_codes_dict == None:
        mun_codes_dict = get_municipality_codes_dict()

    n_of_missing, n_of_invalid, n_of_valid = 0, 0, 0
    if mun_col and mun_codes_dict:
        def convert_and_count(x):
            nonlocal n_of_missing, n_of_invalid, n_of_valid
            if pd.isna(x):
                n_of_missing += 1
                return float('nan')
            elif str(x) in mun_codes_dict:
                n_of_valid += 1
                return mun_codes_dict.get(str(x))
            else:
                n_of_invalid += 1
                return x

        mun_df[mun_col] = mun_df[mun_col].apply(convert_and_count)
        print(f"Municipality codes for {mun_col} replaced using a dict for city codes.")

        return mun_df, [mun_col, n_of_missing, n_of_invalid, n_of_valid, 0, 0]
    else:
        print('No column was altered because a column and dict must be given.')
        return df, _