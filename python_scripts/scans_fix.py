import os
import pandas as pd
from flask import jsonify

def analyse_scans():
    scans_path = os.path.join(os.path.dirname(__file__), '../scans.csv')
    scans = pd.read_csv(scans_path)
    scans['combined_length'] = scans[['Level_0', 'Level_1', 'Level_2']].apply(lambda row: len(''.join(row.values.astype(str))), axis=1)
    scans = scans.sort_values('combined_length')
    print(scans.head())


def delete_app(app):
    scans_path = os.path.join(os.path.dirname(__file__), '../scans.csv')
    scans = pd.read_csv(scans_path)
    scans = scans[scans['App'] != app]
    scans.to_csv(scans_path, index=False)

analyse_scans()
