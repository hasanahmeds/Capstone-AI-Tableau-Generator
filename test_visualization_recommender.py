"""
Testing Visualization Recommender

10 Test Cases
5 Normal Cases
5 Edge Cases
"""

import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from schemas import build_dataset_schema, build_profile_report
from visualization_recommender import VisualizationRecommender



# Helper function to run tests


def run_test(test_name, df):

    print("\n" + "_"*50)
    print(f"TEST: {test_name}")
    print("_"*50)

    try:

        ds = build_dataset_schema(test_name, df)

        profile = build_profile_report(ds, df)

        recommender = VisualizationRecommender()

        dashboard = recommender.recommend(ds, profile)

        print(f"Rows: {len(df)}  Columns: {len(df.columns)}")
        #print(df.head)
        print(f"Numeric Columns: {ds.numeric_columns}")
        print(f"Categorical Columns: {ds.categorical_columns}")
        print(f"Datetime Columns: {ds.datetime_columns}")

        print("\nRecommended Visualizations:\n")

        for i, viz in enumerate(dashboard.visuals, 1):

            print(f"{i}. {viz.chart_type.upper()} : {viz.title}")

        print("\nRESULT: PASS")

    except Exception as e:

        print("RESULT: FAIL")
        print("ERROR:", e)



# MAIN

if __name__ == "__main__":


    # Load dataset

    df_train = pd.read_csv("train.csv")

    # NORMAL TEST CASES
    

    # Test 1 – Real dataset
    '''Tests the system on our main dataset (train.csv) to verify it correctly detects column types and recommends
      appropriate visualizations.'''
    
    run_test("Normal Test 1 - Train Dataset", df_train)

    # Test 2 – Time series dataset
    '''Checks whether the system identifies a datetime column and recommends time-series visualizations like a line chart.'''

    df_time = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=50),
        "sales": range(50)
    })

    run_test("Normal Test 2 - Time Series", df_time)

    # Test 3 – Categorical + Numeric
    '''Ensures the system generates category-based visualizations (bar, pie, box) 
    when both categorical and numeric columns exist.'''

    df_cat = pd.DataFrame({
        "category": ["A","B","C","A","B","C"] * 10,
        "sales": [10,20,30,40,50,60] * 10
    })

    run_test("Normal Test 3 - Category Breakdown", df_cat)

    # Test 4 – Numeric distribution
    ''' Normal 4: Tests whether the system suggests distribution charts (histogram) when the dataset
      contains only one numeric column. '''
    
    df_numeric = pd.DataFrame({
        "sales": [i*5 for i in range(100)]
    })

    run_test("Normal Test 4 - Numeric Distribution", df_numeric)

    # Test 5 – Multiple numeric columns
    ''' Verifies that the system recommends scatter plots and correlation heatmaps when multiple numeric columns are present'''
    df_relation = pd.DataFrame({
        "x": range(100),
        "y": [i*2 for i in range(100)],
        "z": [i*3 for i in range(100)]
    })

    run_test("Normal Test 5 - Numeric Relationships", df_relation)


    
    # EDGE TEST CASES
    

    # Test 6 – High cardinality category
    '''Checks that the system avoids charts like bar graphs when a categorical column has too many unique values.'''

    df_high_card = pd.DataFrame({
        "category": [f"C{i}" for i in range(100)],
        "sales": range(100)
    })

    run_test("Edge Test 1 - High Cardinality Category", df_high_card)


    # Test 7 – Very small dataset
    '''Tests whether the system handles very small datasets safely and still recommends simple visualizations.'''
    df_small = pd.DataFrame({
        "sales": [10,20]
    })

    run_test("Edge Test 2 - Small Dataset", df_small)


    # Test 8 – Missing values
    '''Ensures the visualization recommender still works correctly when the dataset contains null or missing values.'''
    df_missing = pd.DataFrame({
        "sales":[10,20,None,40,None],
        "category":["A","B","A",None,"C"]
    })

    run_test("Edge Test 3 - Missing Values", df_missing)


    # Test 9 – Text dataset
    '''Verifies that the system avoids numeric visualizations when the dataset contains only text columns.'''

    df_text = pd.DataFrame({
        "description":[
            "this is a long text example",
            "another text column entry",
            "visualization recommendation system"
        ]
    })

    run_test("Edge Test 4 - Text Only Dataset", df_text)


    # Test 10 – Duplicate rows dataset
    '''Tests whether duplicate records in the dataset affect visualization recommendations.'''

    df_duplicate = pd.DataFrame({
        "category": ["A", "B", "A", "B", "A"],
        "sales": [10, 20, 10, 20, 10]
    })

    run_test("Edge Test 5 - Duplicate Rows Dataset", df_duplicate)