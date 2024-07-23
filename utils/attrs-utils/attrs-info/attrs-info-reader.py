import os
import PyPDF2
import re
import pandas as pd

def get_abs_path(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, file_name)

# Path to the PDF file
pdf_path = get_abs_path('SIM_geral_estrutura_atributos.pdf')
print(pdf_path)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_n in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_n]
            text += page.extract_text()
    return text

# Extract text from the PDF
pdf_text = extract_text_from_pdf(pdf_path)

regex = r'\d{1,3} ([A-Z0-9_]+(?: [A-Z0-9_]+){0,2}) ([A-Z] ?\d{1,2}) ((?:(?!\d{4} \([A-Z] ?\d{1,2}\)).|\n)+?)? ?(\d{4}) (\([A-Z] ?\d{1,2}\))\n?'
attribute_pattern = re.compile(regex)

# Find all matches in the PDF text
matches = attribute_pattern.findall(pdf_text)

# Extracted attributes
attributes = []

for match in matches:
    # print(match,'\n')
    name = match[0].replace(' ', '').strip()
    description = match[2].replace('\n', ' ').strip() if match[1].strip() else "Sem descricao."
    year_added = int(match[3])
    attributes.append({'name': name, 'description': description, 'year_added': year_added})

df = pd.DataFrame(attributes)
# print(df)

# Filter attributes by year_added (less than 1996)
filtered_attributes = [attr for attr in attributes if attr['year_added'] <= 1996]
df_filtered = pd.DataFrame(filtered_attributes)
print(df_filtered)

# Save the DataFrame to a CSV file
df.to_csv(get_abs_path('datasus-attrs-info.csv'), index=False)
# Save the filtered DataFrame to a CSV file
df_filtered.to_csv(get_abs_path('datasus-attrs-info_filtered.csv'), index=False)
