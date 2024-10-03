import pandas as pd

attributes_values_info_en = [
    'Attribute',
    'Missing', 
    'Nulls/Invalid', 
    'Valid', 
    'Preprocessed', 
    'Removed'
    ]

attributes_values_info_br = [
    'Atributo',
    'Faltantes', 
    'Nulos/Inválidos', 
    'Válidos', 
    'Tratados', 
    'Removidos'
    ]


def get_values_table_columns(lang='BR'):
    if lang == "BR":
        return attributes_values_info_br
    else:
        return attributes_values_info_en


def generate_attributes_values_table(dir_to_save, lang='BR'):
    # Create an empty DataFrame with the specified column names
    df = pd.DataFrame(columns=get_values_table_columns())
    # Save the DataFrame to a CSV file
    csv_file_path = f"{dir_to_save}/attributes_values_{lang}.csv"
    df.to_csv(csv_file_path, index=False)
    return csv_file_path


def append_to_attributes_values_table(table_dir, data_dict, lang='BR', return_df=False):
    # Read the existing CSV into a DataFrame
    df = pd.read_csv(f"{table_dir}/attributes_values_{lang}.csv", index_col=False)
    # Check if the column name already exists in the DataFrame
    if data_dict[0] in df[df.columns[0]].values:
        print('------------------------')
        print(f"Column '{data_dict[0]}' already exists in the table.")
        if return_df:
            return df 
        return
    # Create a new DataFrame from the provided dictionary
    new_row = pd.DataFrame([data_dict], columns=get_values_table_columns())
    # Append the new row to the DataFrame
    df = pd.concat([df, new_row], ignore_index=True)
    # Save the updated DataFrame back to the CSV
    df.to_csv(f"{table_dir}/attributes_values_{lang}.csv", index=False)
    if return_df:
        return df
    return


def update_attribute_values(table_dir, new_data_list, lang='BR', return_df=False):
    # Read the existing CSV into a DataFrame
    df = pd.read_csv(f"{table_dir}/attributes_values_{lang}.csv", index_col=False)

    first_col_name = df.columns[0]
    col_name = new_data_list[0]

    # Check if the column name already exists in the DataFrame
    if col_name in df[first_col_name].values:
        row_index = df.loc[df[first_col_name] == col_name].index[0]
        for i, new_value in enumerate(new_data_list):
            if not pd.isna(new_value):
                df.iloc[row_index, i] = new_value
        print(f"Updated values for '{col_name}' in the datatable.")
    else:
        print(f"{col_name} doesn't exist in the values datatable. Use the append function to add it.")

    # Save the updated DataFrame back to the CSV
    df.to_csv(f"{table_dir}/attributes_values_{lang}.csv", index=False)

    # Return the DataFrame if return_df is True
    if return_df:
        return df
    return


def convert_values_table_to_percentages(values_table, dataframe_lens):
    # dataframe_lens should be dataframe_lens = {'original': value, 'current': value}
    values_table_percent = values_table.copy()

    # List of columns to be converted to percentages (excluding attribute names column)
    '''
        cleaning_cols refer to the initial step of cleaning, that needs to use the original
        length of the dataframe, while preprocessing_cols requires the final length, since
        rows were removed
    '''
    cleaning_cols = get_values_table_columns()[1:4]
    preprocessed_col = get_values_table_columns()[-2:-1]
    removed_col = get_values_table_columns()[-1:]

    print(f"Cols: {cleaning_cols, preprocessed_col, removed_col}")

    for idx, row in values_table_percent.iterrows():
        for col in values_table_percent.columns[1:]:
            if col in cleaning_cols:
                # Apply the min with 'original' dataframe length and convert to percentage
                values_table_percent.at[idx, col] = min(row[col], dataframe_lens['original'])
                values_table_percent.at[idx, col] = (values_table_percent.at[idx, col] / dataframe_lens['original']) * 100
            # Check if the column belongs to preprocessed columns
            elif col in preprocessed_col:
                # Apply the min with 'current' dataframe length - removed values and convert to percentage
                removed_value = row[removed_col[0]] if removed_col[0] in row else 0
                values_table_percent.at[idx, col] = min(row[col], dataframe_lens['current'] - removed_value)
                values_table_percent.at[idx, col] = (values_table_percent.at[idx, col] / (dataframe_lens['current'])) * 100
            elif col in removed_col:
                # Apply the min with 'current' dataframe length - removed values and convert to percentage
                removed_value = row[removed_col[0]] if removed_col[0] in row else 0
                values_table_percent.at[idx, col] = min(row[col], dataframe_lens['current'])
                values_table_percent.at[idx, col] = (values_table_percent.at[idx, col] / (dataframe_lens['current'])) * 100
            # Format the result to 2 decimal places
            values_table_percent.at[idx, col] = '{:.2f}'.format(values_table_percent.at[idx, col])

    return values_table_percent


def get_attributes_values_table(table_dir, percentage=False, dataframe_lens=None, lang='BR'):
    # dataframe_lens should be dataframe_lens = {'original': value, 'current': value}
    if percentage == False:
        return pd.read_csv(f"{table_dir}/attributes_values_{lang}.csv")
    else:
        attributes_values_table = pd.read_csv(f"{table_dir}/attributes_values_{lang}.csv")
        return convert_values_table_to_percentages(attributes_values_table, (dataframe_lens))