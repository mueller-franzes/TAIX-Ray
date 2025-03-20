from pathlib import Path 
import pandas as pd 
import re
import hashlib
import numpy as np

def hash_id(value):
    return hashlib.sha256(str(value).encode()).hexdigest()

def extract_value_after_key(text: str, key: str) -> str:
    pattern = rf"{re.escape(key)}\s*:\s*([\S\s]+?)(?:\n|$)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None

def extract_label_heart(text):
    if text is None:
        return None
    pattern = (
        r'\b('
        r'normal(?:es|e|er|em|en)?|'  # Matches "normal", "normales", "normale" etc.
        r'grenzwertig(?:es|e|er|em|en)?|'  # Matches "grenzwertig", "grenzwertiges", etc.
        r'massiv vergrößert(?:es|e|er|em|en)?|'  # Matches "massiv vergrößert", "massiv vergrößertes", etc. 
        r'vergrößert(?:es|e|er|em|en)?|'  # Matches "vergrößert", etc. (warning also matches "deutlich vergrößert")
        r'nicht beurteilbar(?:es|e|er|em|en)?'  # Matches "nicht beurteilbar", "nicht beurteilbares"
        r')\b'
    )
    match = re.search(pattern, text)
    to_num = {'nicht beurteilbar': -1, 'normal': 0, 'grenzwertig': 1, 'vergrößert': 2, 'massiv vergrößert': 3}
    # to_num = {'normal': 1, 'grenzwertig': 2, 'vergrößert': 3, 'massiv vergrößert': 4, 'nicht beurteilbar': 5} # old 
    if match:
        canonical = re.sub(r'(es|e|er|em|en)$', '', match.group(0)).strip()
        return to_num[canonical]
    return text

def extract_label(text):
    if text is None:
        return None
    match = re.search(r'kein|\(\+\)|\+{1,3}', text)
    to_num = {'kein': 0, '(+)': 1,  '+': 2, '++': 3, '+++': 4}
    # to_num = {'kein': 1, '(+)': 5,  '+': 2, '++': 3, '+++': 4} # old labeling  
    if match:
        return to_num[match.group(0)]
    return text

def extract_location(text):
    # Try to extract "OF", "MF", "UF", "basal", "OF/MF", "OF/UF", "MF/UF" first, if not possible return the input
    if text is None:
        return None
    match = re.search(r'OF/MF|OF/UF|MF/UF|OF|MF|UF|basal', text) # WARNING: order matters 
    to_num = {'OF': 0, 'MF': 1, 'UF': 2, 'basal': 3, 'OF/MF': 4, 'OF/UF': 5, 'MF/UF': 6}
    if match:
        return to_num[match.group(0)]
    return text

def extract_re_li(text, return_label=True):
    if text is None:
        return None, None
    # Match "re <word(s)> li <word(s)>" or "re <word(s)>; <word(s)>" or "<word(s)>; li <word(s)>" or "<word(s)>; <word(s)>" 
    # pattern = r"^(?:re\s*)?(.*?)\s*(?:;\s*li|li|;)\s*(.*)$"
    # Match "re <word(s)>; li <word(s)>"
    pattern = r"re\s*(.*?);\s*li\s*(.*)"

    match = re.search(pattern, text)
    if match:
        # Extract words after "re" and "li"
        re_part = match.group(1).strip()
        li_part = match.group(2).strip()
        
        if return_label:
            # Try to extract "+"-label, if not possible return the input
            re_label = extract_label(re_part) 
            li_label = extract_label(li_part) 
            return re_label, li_label

        else:
            # Try to extract the location, if not possible return the input
            re_location = extract_location(re_part) 
            li_location = extract_location(li_part) 
            return re_location, li_location
        
    return None, None 

if __name__ == "__main__":
    # Setting 
    path_root = Path('/ocean_storage/data/UKA/UKA_Thorax/download')
    path_metadata = path_root/'metadata'

    path_root_out = Path('/ocean_storage/data/UKA/UKA_Thorax/public_export')
    path_out_metadata = path_root_out / 'metadata'
    path_out_metadata.mkdir(parents=True, exist_ok=True)

    # translation = {
    #     'Herzgröße': 'cardiomegaly', 
    #     'Stauung': 'congestion', 
    #     'Pleuraerguss': 'pleural_effusion', 
    #     'Infiltrate': 'pneumonic_infiltrates',
    #     'Bel.-störungen': 'atelectasis', 
    # }
    # german_labels = list(translation.keys())
    # single_side_labels = [ 'cardiomegaly', 'congestion']
    # double_side_labels = [ 'pleural_effusion', 'pneumonic_infiltrates', 'atelectasis']
    # double_side_labels_lr = [ f'{label}_{side}' for label in double_side_labels for side in ['right', 'left']]


    translation = {
        'Herzgröße': 'HeartSize', 
        'Stauung': 'PulmonaryCongestion', 
        'Pleuraerguss': 'PleuralEffusion', 
        'Infiltrate': 'PulmonaryOpacities', 
        'Bel.-störungen': 'Atelectasis', 
    }
    german_labels = list(translation.keys())
    single_side_labels = [ 'HeartSize', 'PulmonaryCongestion']
    double_side_labels = [ 'PleuralEffusion', 'PulmonaryOpacities', 'Atelectasis']
    double_side_labels_lr = [ f'{label}_{side}' for label in double_side_labels for side in ['Right', 'Left']]


    # Load data
    df = pd.read_excel(path_metadata/'reports.xlsx')
    print("Original data", len(df))

    # Remvoe rows with missing values
    df = df.dropna(subset=[ 'Untersuchungsnummer', 'Geburtsdatum', 'Untersuchungsdatum', 'Befundarzt', 'Dokumentinhalt']).reset_index(drop=True)
    print("After removing missing values", len(df))

    # Remove duplicates
    df = df.drop_duplicates(subset=['Untersuchungsnummer']).reset_index(drop=True)
    print("After removing duplicates", len(df))

    # Cleanup 
    df['Dokumentinhalt'] = df['Dokumentinhalt'].replace('\r','\n', regex=True)
    df['Untersuchungsnummer'] = df['Untersuchungsnummer'].str.replace('-', '0')
    df['Untersuchungsdatum'] = pd.to_datetime(df['Untersuchungsdatum'], format='%Y-%m-%d')
    df['Geburtsdatum'] = pd.to_datetime(df['Geburtsdatum'], format='%Y-%m-%d')
    df['Age'] = (df['Untersuchungsdatum'] - df['Geburtsdatum']).dt.days

    # Blur exact exaimation date 
    random_offsets = np.random.randint(-100, 101, size=len(df))
    df['StudyDate'] =  df['Untersuchungsdatum'] +  pd.to_timedelta(random_offsets, unit="D")

    # Anonyimzed 
    df['PhysicianID'] = df['Befundarzt'].apply(hash_id)

    # Remove unreasonable age
    # df = df[df['Age']>=0] # Age should be not negative 
    # df = df[df['Age']<365*150] # Age should be smaller than 150 years (max Age 110)
    # print("After removing unresonable age", len(df), "Age range", df['Age'].min()/365, df['Age'].max()/365)


    # --------------------- Load mapping ( AccessionNumber -> PseudoAccessionNumber) ---------------------
    df_mapping = pd.read_csv(path_metadata/'pseudo_table.csv')
    df_mapping = df_mapping.drop_duplicates('AccessionNumber').reset_index(drop=True)
    df_mapping['PseudoPatientID'] = df_mapping['PatientID'].apply(hash_id) # Fix BUG 
    df_mapping = df_mapping[['AccessionNumber', 'PseudoAccessionNumber', 'PseudoPatientID']]
    df = pd.merge(df_mapping , df, left_on='AccessionNumber', right_on='Untersuchungsnummer', how='inner')
    print("After merging with mapping file", len(df))
    # df_mapping[df_mapping['AccessionNumber'].duplicated(keep=False)].sort_values('AccessionNumber')[['AccessionNumber', 'PseudoAccessionNumber']]

    
    # Remove patients with multiple or unkown genders 
    df = df.dropna(subset=['Geschlecht'])
    df = df[df['Geschlecht'].isin(['M', 'W'])] # Removes "U"
    genders_per_patient = df.groupby('PseudoPatientID')['Geschlecht'].nunique()
    multi_gender = genders_per_patient[genders_per_patient>1].index
    df = df[~df['PseudoPatientID'].isin(multi_gender)]
    df['Sex'] = df['Geschlecht'].map({'M':'M', 'W':'F'})
    print("After removing patients with multiple or unkown genders", len(df))

    # ----------------------- Merge with image data ----------------------
    df_images = pd.read_csv(path_out_metadata/'metadata.csv', usecols=['PatientName', 'PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'Filename'])
    df_images['PseudoAccessionNumber'] = df_images['PatientName'].str.split('_', n=1).str[1]
    assert len(df_images) == len(df_images['PseudoAccessionNumber'].unique()), "Exams with multiple images"
    df = pd.merge(df, df_images, on='PseudoAccessionNumber', how='inner')
    print("After merging with image data", len(df))
    df['PatientID'] = df['PseudoPatientID'] # TODO: Remove after fix - should be the same. 

    # ------------------------ Extract values ---------------------
    # Extract values after labels
    for key_value in german_labels:
        df[key_value] = df['Dokumentinhalt'].apply(lambda x: extract_value_after_key(x, key_value))
    
    # Workaround: Merge Pleuraerguss and Pleuraerguß
    df['Pleuraerguss'] = df['Pleuraerguss'].fillna(
        df['Dokumentinhalt'].apply(lambda x: extract_value_after_key(x, 'Pleuraerguß'))
    )

    # Translate
    df = df.rename(columns=translation)

    # ------------------------ Extract labels ---------------------
    # Single value labels
    for key_value in single_side_labels:
        if key_value == 'HeartSize': 
            df[key_value] = df[key_value].apply(extract_label_heart)
        else:
            df[key_value] = df[key_value].apply(extract_label)
    
    # Extract labels for left and right separately
    for key_value in double_side_labels:
        df[f'{key_value}_Right'], df[f'{key_value}_Left'] = zip(*df[key_value].apply(extract_re_li))


    # ------------------------- Cleanup ---------------------------
    for key_value in single_side_labels:
        df.loc[~df[key_value].astype(str).str.isnumeric(), key_value] = -2

    for key_value in double_side_labels_lr:
        df.loc[~df[key_value].astype(str).str.isnumeric(), key_value] = -2
    
    # Drop cases with no labels 
    df = df[~(df[single_side_labels + double_side_labels_lr].eq(-2).any(axis=1))]
    print("After removing cases with no labels", len(df))

    # ------------------------- Save --------------------------------
    
    df_annonymized = df[['PatientID', 'PhysicianID', 'StudyDate', 'Age', 'Sex', *single_side_labels, *double_side_labels_lr]]
    df_annonymized.insert(0, 'UID', df['PseudoAccessionNumber'])
    df_annonymized.to_csv(path_out_metadata/'annotation.csv', index=False)



    # # -------------------------- Verify the extracted labels with old extracted labels ----------------------
    # df_old = pd.read_csv(path_metadata/'old_backup/labels_extracted_old.csv')
    # df_old['image_id'] = df_old['image_id'].astype(str)

    # print("Old labels", len(df_old))
    # print("New labels", len(df))
    # df_merge = pd.merge(df, df_old, left_on='Untersuchungsnummer', right_on='image_id', how='inner', suffixes=('_new', '_old'))
    # print("Merge ", len(df_merge))

    # path_out = Path.cwd() / 'error'
    # path_out.mkdir(parents=True, exist_ok=True)
    # for label in single_side_labels:
    #     error = df_merge[df_merge[f'{label}_old'] != df_merge[f'{label}_new']]
    #     print(f'{label}', len(error))
    #     error[['Untersuchungsnummer', f'{label}_old', f'{label}_new', 'Dokumentinhalt']].to_excel(path_out/f'error_{label}.xlsx', index=False)

    # for label in double_side_labels:
    #     for side in ['right', 'left']:
    #         error = df_merge[df_merge[f'{label}_{side}_old'] != df_merge[f'{label}_{side}_new']]
    #         print(f'{label}_{side}', len(error))
    #         error[['Untersuchungsnummer', f'{label}_{side}_old', f'{label}_{side}_new', 'Dokumentinhalt']].to_excel(path_out/f'error_{label}_{side}.xlsx', index=False)