import pandas as pd

def get_data():
    # The URL of your Google Sheets document's published CSV link
    url = 'https://docs.google.com/spreadsheets/d/1r6mS8WzukVhHnVFOEGmQtwL2VmSqkGXQy1H2Al6IO2o/gviz/tq?tqx=out:csv'

    # Read the CSV data
    df = pd.read_csv(url)
    df = df[df['Gabriel'] == 'Yes']
    df.drop(['Gabriel', 'Dat', 'Proposed Level', 'Reason', 'Note'], axis=1, inplace=True)
    print(df)
    return df

