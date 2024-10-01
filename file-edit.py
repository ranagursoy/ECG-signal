import pandas as pd
from docx import Document
import unicodedata

# Türkçe karakterleri normalize etmek ve büyük/küçük harf farkını ortadan kaldırmak için fonksiyon
def normalize_string(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8').lower()
    return s

# Word dosyasındaki isimleri okuma ve isim-soyisim olarak ayırma
def read_names_from_word(doc_path):
    doc = Document(doc_path)
    word_names = []
    for para in doc.paragraphs:
        if para.text.strip():
            lines = para.text.strip().splitlines()
            for line in lines:
                name_parts = [part.strip() for part in line.split(",")]
                if len(name_parts) == 2:
                    full_name = name_parts[1] + " " + name_parts[0]
                    word_names.append(full_name)
                else:
                    word_names.append(line.strip())
    
    return word_names

# Excel dosyasını okuma ve isimleri ^ işaretinden ayırma
def read_excel_data(excel_path):
    df = pd.read_excel(excel_path)
    df[['Soyad', 'Ad']] = df['HastaAdSoyad'].str.split('^', expand=True)
    df['FullName'] = df['Ad'] + " " + df['Soyad']
    return df

# İsim eşlemesi yapma
def match_names(word_names, excel_df):
    matched_names = []
    unmatched_names = []
    excel_df['NormalizedName'] = excel_df['FullName'].apply(normalize_string)

    for word_name in word_names:
        normalized_word_name = normalize_string(word_name)
        matched = excel_df[excel_df['NormalizedName'].str.contains(normalized_word_name, case=False, na=False, regex=False)]
        if not matched.empty:
            matched_names.append((word_name, matched['FullName'].values[0]))
        else:
            unmatched_names.append(word_name)

    return matched_names, unmatched_names

# Eşleşen isimleri kaydetme
def save_matched_to_excel(matched_names, output_path):
    if matched_names:
        matched_df = pd.DataFrame(matched_names, columns=['Word Name', 'Excel Name'])
        matched_df.to_excel(output_path, index=False)

# Eşleşmeyen isimleri kaydetme
def save_unmatched_to_txt(unmatched_names, txt_path):
    if unmatched_names:
        with open(txt_path, 'w', encoding='utf-8') as f:
            for name in unmatched_names:
                f.write(name + '\n')

# Ana fonksiyon
def main():
    word_doc_path = 'ecg-hastaisim.docx'
    excel_file_path = '20240909-092201 - Kopya.xlsx'
    matched_excel_path = 'matched_names2.xlsx'
    unmatched_txt_path = 'unmatched_names2.txt'

    word_names = read_names_from_word(word_doc_path)
    excel_df = read_excel_data(excel_file_path)

    matched_names, unmatched_names = match_names(word_names, excel_df)

    save_matched_to_excel(matched_names, matched_excel_path)
    save_unmatched_to_txt(unmatched_names, unmatched_txt_path)

if __name__ == "__main__":
    main()
