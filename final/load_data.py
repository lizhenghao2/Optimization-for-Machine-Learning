# load the data from the csv file and return it as a pandas dataframe
import os
import pandas as pd

def load_patient_info(file_path="archive/PatientInfo.csv"):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_csv(file_path)

    # Save deceased Korean patients
    deceased = data[
        (data['state'] == 'deceased') &
        (data['country'] == 'Korea')
    ].copy()

    deceased.to_csv("deceased_patients.csv", index=False)

    # Keep only Korean patients
    data = data[data['country'] == 'Korea'].copy()

    print("Predictors: sex, age, province, infection_case")

    # Keep relevant columns
    data = data[
        ['sex', 'age', 'province', 'infection_case', 'confirmed_date', 'state']
    ].copy()

    # Remove rows with missing sex or age
    data = data.dropna(subset=['sex', 'age'])

    # Save cleaned data
    data.to_csv("cleaned_patient_info.csv", index=False)

    return data


def preprocess_features(df):
    X = df[['sex', 'age', 'province', 'infection_case', 'confirmed_date']].copy()
    y = df['state'].copy()

    print("\nFeature distribution:")
    print(X['sex'].value_counts())
    print(X['age'].value_counts())
    print(X['province'].value_counts())
    print(X['infection_case'].value_counts())

    print("\nState distribution in each class:")
    print(y.value_counts())

    # Merge 90s and 100s into 90s+
    X.loc[:, 'age'] = X['age'].replace({
        '90s': '90s+',
        '100s': '90s+'
    })

    mapping = {
        # Nursing home
        'Bonghwa Pureun Nursing Home': 'Nursing home',
        'Gyeongsan Seorin Nursing Home': 'Nursing home',
        'Gyeongsan Jeil Silver Town': 'Nursing home',

        # Hospital
        'Cheongdo Daenam Hospital': 'Hospital',
        "Eunpyeong St. Mary's Hospital": 'Hospital',

        # Religious gathering
        'Shincheonji Church': 'Relious gathering',
        'Onchun Church': 'Religious gathering',
        'Dongan Church': 'Religious gathering',
        'Geochang Church': 'Religious gathering',
        'SMR Newly Planted Churches Group': 'Religious gathering',
        'River of Grace Community Church': 'Religious gathering',
        'Biblical Language study meeting': 'Religious gathering',
        'Pilgrimage to Israel': 'Religious gathering',

        # Call center
        'Guro-gu Call Center': 'Call center',

        # Community center / shelter / apartment
        'Milal Shelter': 'Community center, shelter and apartment',
        'Orange Town': 'Community center, shelter and apartment',
        'Seongdong-gu APT': 'Community center, shelter and apartment',
        'Gyeongsan Cham Joeun Community Center':
            'Community center, shelter and apartment',

        # Gym facility
        'gym facility in Cheonan': 'Gym facility',
        'gym facility in Sejong': 'Gym facility',

        # Overseas inflow
        'overseas inflow': 'Overseas inflow',

        # Contact with patients
        'contact with patient': 'Contact with patients'
    }

    X.loc[:, 'infection_case'] = (
        X['infection_case']
        .map(mapping)
        .fillna('Others')
    )

    print("\nFeature distribution after merging:")
    print(X['infection_case'].value_counts())
    
    # map the date to year-month
    X['confirmed_date'] = pd.to_datetime(
        X['confirmed_date'],
        errors='coerce'
    )

    X['confirmed_date'] = (
        X['confirmed_date']
        .dt.to_period('M')
        .astype(str)
    )
    print("\nConfirmed date distribution after mapping to year-month:")
    print(X['confirmed_date'].value_counts())
    

    # released and isolated = 0; deceased = 1
    y_mapping = {
        'released': 0,
        'isolated': 0,
        'deceased': 1
    }

    y = y.map(y_mapping)

    # check for unknown labels
    if y.isna().any():
        print("Unknown labels found:")
        print(df['state'][y.isna()].unique())

    y = y.astype(int)

    # One-hot encoding
    X = pd.get_dummies(
        X,
        columns=['sex', 'age', 'province', 'infection_case', 'confirmed_date'],
        drop_first=False
    )
    

    return X, y


def main():
    try:
        df = load_patient_info()
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns from archive/PatientInfo.csv")
        print(df.head().to_string(index=False))
        # summarize state
        print("\nState distribution:")
        print(df['state'].value_counts())
        X, y = preprocess_features(df)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()