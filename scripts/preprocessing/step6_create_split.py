from pathlib import Path 
import pandas as pd 

from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, StratifiedKFold, KFold

def get(df, col):
    if col is None:
        return None
    return df[col]

def create_split(df, label_col=None, group_col=None):
    df = df.reset_index(drop=True)

    # Select appropriate split method
    if (label_col is None) and (group_col is None):
        Method = KFold 
    elif (label_col is not None) and (group_col is None):
        Method = StratifiedKFold
    elif (label_col is None) and (group_col is not None):
        Method = GroupKFold
    else:
        Method = StratifiedGroupKFold

    y = df[label_col] if label_col else None
    groups = df[group_col] if group_col else None

    outer_splitter = Method(n_splits=5, shuffle=True, random_state=0)

    # Store copies of the DataFrame for each fold
    split_dfs = []
    for fold_i, (train_val_idx, test_idx) in enumerate(outer_splitter.split(df, y, groups)):
        df_split = df.copy()  # Create a new copy for each fold
        df_split["Fold"] = fold_i
        df_split["Split"] = None

        df_split.loc[test_idx, "Split"] = "test"

        df_trainval = df.iloc[train_val_idx]
        y_trainval = df_trainval[label_col] if label_col else None
        groups_trainval = df_trainval[group_col] if group_col else None

        inner_splitter = Method(n_splits=5, shuffle=True, random_state=0)
        train_idx, val_idx = next(inner_splitter.split(df_trainval, y_trainval, groups_trainval))

        df_split.loc[df_trainval.index[train_idx], "Split"] = "train"
        df_split.loc[df_trainval.index[val_idx], "Split"] = "val"

        split_dfs.append(df_split)  # Store the modified copy

    return pd.concat(split_dfs, ignore_index=True)  # Return all splits combined


if __name__ == "__main__":
    path_root = Path('/ocean_storage/data/UKA/UKA_Thorax')/'public_export'
    path_root_metadata = path_root/'metadata'

    df = pd.read_csv(path_root_metadata/'annotation.csv', dtype={'Patient ID':str})

    print("Patients", df['PatientID'].nunique())
    # print("Studies", df['StudyInstanceUID'].nunique())
    # print("Series", df['SeriesInstanceUID'].nunique())
    print("Total", len(df))

    for pathology in df.columns[6:]:
        for grade, count in df[pathology].value_counts().sort_index().items():
            print(f"{pathology} Grade {grade}: {count}")
        print("-----------------")

    # df = df[['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID']]
    df = df[['UID', 'PatientID']]

    df_splits = create_split(df, group_col='PatientID')
    df_splits = df_splits.drop(columns='PatientID')
    df_splits.to_csv(path_root_metadata/'split.csv', index=False)


        
    