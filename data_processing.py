import pandas as pd
import matplotlib.pyplot as plt

def filter_cities():

    df = pd.read_csv('./Indicators_for_SDG_1_CBG_level.csv')


    allowed_cities = ['New York city', 'Los Angeles city', 'Dallas city', 'Chicago city', 'Houston city', 'Phoenix city', 'Philadelphia city', 'San Antonio city', 'San Diego city', 'Jacksonville city']
    allowed_year = [2021]


    filtered_df = df[df['City Name'].isin(allowed_cities) & df['Year'].isin(allowed_year)]

    filtered_df.to_csv('filtered_cities.csv', index=False)

    print(f"Done! Reduced the file from {len(df)} to {len(filtered_df)} rows.")

def count_rows():
    df = pd.read_csv('./filtered_cities.csv')
    city_counts = df['City Name'].value_counts()

    print("Row counts per city:")
    print(city_counts)


def count_row_cbg():
    city_list = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Jacksonville']
    
    for city in city_list:
        df = pd.read_csv(f'./year2020_to_2023/images_within_CBGs_in_{city}_city.csv')
        cbg_counts = df['CBG Code'].value_counts()

        min_cgb = cbg_counts.idxmax()
        min_count = cbg_counts.max()
  
        print(f"Max photos for {city}: {min_count}:{min_cgb}")

def plot_cbg_count():
    city_list = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Jacksonville']
    
    for city in city_list:
        df = pd.read_csv(f'./year2020_to_2023/images_within_CBGs_in_{city}_city.csv')
        cbg_counts = df['CBG Code'].value_counts()

        plt.figure(figsize=(10, 6))
        cbg_counts.plot(kind='bar')
        plt.title(f'Number of Photos per CBG in {city} City')
        plt.xlabel('CBG Code')
        plt.ylabel('Number of Photos')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def clean_mapping():
    #city_list = ['New York', 'Los Angeles', 'Chicago', 'Dallas', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Jacksonville']
    city_list = ['Dallas']
    for city in city_list:
        df = pd.read_csv(f'./within/cleaned_{city} city.csv')

        # Convert Zoom 19 to Zoom 18
        df['x_18'] = df['x_tile19'] // 2
        df['y_18'] = df['y_tile19'] // 2

        df['filename'] = df['y_18'].astype(str) + '_' + df['x_18'].astype(str) + '.png'

        # Pick one CBG per filename
        df_final = df.drop_duplicates(subset=['filename'], keep='first')

        df_final[['filename', 'CBG Code']].to_csv(f'./image_to_cbg/{city}.csv', index=False)

def right_size():
    city_list = ['New York', 'Los Angeles', 'Chicago', 'Dallas','Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Jacksonville']
    for city in city_list:
        # threshold 
        X = 20 

        # Load data
        df = pd.read_csv(f'filtered_cities.csv')
        cbg_counts = df['CBG Code'].value_counts()

        # Filter for CBGs where the count is greater than X
        large_cbgs = cbg_counts[cbg_counts > X]

        distinct_count = len(large_cbgs)

        print(f"There are {distinct_count} distinct CBGs in {city} with more than {X} rows.")


def stratified_sample():
    #city_list = ['New York', 'Los Angeles', 'Dallas','Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Jacksonville']
    city_list = ['Dallas']      
    for city in city_list:
        filtered_df = pd.read_csv(f'./image_to_cbg/{city}.csv')
        # Configuration
        N = 20  # Number of rows per CBG
        M = 250 # Number of distinct CBGs to select

        # Find eligible CBGs 
        cbg_counts = filtered_df['CBG Code'].value_counts()
        eligible_cbgs = cbg_counts[cbg_counts >= N].index.tolist()

        selected_cbgs = eligible_cbgs[:M]

        # Filter and Sample
        final_df = (
            filtered_df[filtered_df['CBG Code'].isin(selected_cbgs)]
            .groupby('CBG Code')
            .head(N)
        )

        # Save the result
        final_df.to_csv(f'./labels/{city}.csv', index=False)

def left_semi_join():
    #city_list = ['New York', 'Los Angeles', 'Chicago', 'Dallas','Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Jacksonville']
    city_list = ['Dallas']  
    
    for city in city_list:
        mapping_df = pd.read_csv(f'./labels/{city}.csv')
        input_df = pd.read_csv(f'./satellite_imagery_collection/tilefile_unfiltered/{city} city.csv')

        input_df['filename_check'] = (
            input_df['y_tile'].astype(str) + 
            '_' + 
            input_df['x_tile'].astype(str) + 
            '.png'
        )

        filtered_input = input_df[input_df['filename_check'].isin(mapping_df['filename'])]

        final_output = filtered_input.drop(columns=['filename_check'])
        final_output.to_csv(f'./satellite_imagery_collection/tilefile_scd/{city}.csv', index=False)

def do_merge():
    #city_list = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Dallas', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Jacksonville']
    city_list = ['Dallas']
    for city in city_list:
        mapping_df = pd.read_csv(f'./labels/{city}.csv')
        input_df = pd.read_csv(f'./satellite_imagery_collection/tilefile_scd/{city}.csv')

        input_df['filename'] = (
            input_df['y_tile'].astype(str) + 
            '_' + 
            input_df['x_tile'].astype(str) + 
            '.png'
        )

        filtered_input_with_cbg = pd.merge(
            input_df, 
            mapping_df[['filename', 'CBG Code']], 
            on='filename', 
            how='inner'
        )

        # drop the temporary 'filename' column 
        final_output = filtered_input_with_cbg.drop(columns=['filename'])

        final_output.to_csv(f'./satellite_imagery_collection/tilefile_scd/{city}_filtered.csv', index=False)

def add_poverty_rate():

    city_list = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Dallas', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Jacksonville']
    
    for city in city_list:
        # Load the data 
        census_df = pd.read_csv('filtered_cities.csv')

        census_df['Total Pop for Poverty'] = (
            census_df['Population Above Poverty'] + census_df['Population Below Poverty']
        )

        # Avoid division by zero for empty tracts
        census_df['Poverty Rate'] = census_df['Population Below Poverty'] / census_df['Total Pop for Poverty']
        census_df['Poverty Rate'] = census_df['Poverty Rate'].fillna(0) # Handle 0/0 cases

        # Load city-specific image map 
        city_mapping_df = pd.read_csv(f'./labels/{city}.csv')

        # Merge the Poverty Rate into the mapping file w left join
        result_df = pd.merge(
            city_mapping_df,
            census_df[['CBG Code', 'Poverty Rate']],
            on='CBG Code',
            how='left'
        )

        result_df.to_csv(f'./labels/{city}.csv', index=False)

def filter_no_data():
    city_list = ['Dallas']
    
    for city in city_list:
        census_df = pd.read_csv('filtered_cities.csv', dtype={'CBG Code': str})
        valid_cbgs = census_df['CBG Code'].unique()
        mapping_df = pd.read_csv(f'./within/images_within_CBGs_in_{city} city.csv', dtype={'CBG Code': str})

        filtered_mapping_df = mapping_df[mapping_df['CBG Code'].isin(valid_cbgs)]

        filtered_mapping_df.to_csv(f'./within/cleaned_{city} city.csv', index=False)

#fixing issue with jacksonville
def remove_extra_col():

    #city_list  = ['New York', 'Los Angeles', 'Chicago','Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Jacksonville']

    city_list  = ['Jacksonville']
    for city in city_list:

       # df = pd.read_csv(f'./labels/{city}.csv')
        df = pd.read_csv(f'./satellite_imagery_collection/tilefile_scd/{city}.csv')
        df = df.drop(columns=['CBG Code_y'])
        df.to_csv(f'./satellite_imagery_collection/tilefile_scd/{city}.csv', index=False)
        #df.to_csv(f'./labels/{city}.csv', index=False)

def count_missing_data():
    city_list  = ['New York', 'Los Angeles', 'Chicago','Dallas','Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Jacksonville']
    for city in city_list:
        df_a = pd.read_csv(f'./labels/{city}.csv')
        df_b = pd.read_csv(f'./satellite_imagery_collection/tilefile_scd/{city}.csv')

        # Ensure the CBG column is treated as a string to avoid rounding
        cbg_a = df_a['CBG Code'].astype(str)
        cbg_b = df_b['CBG Code'].astype(str)

        # Find rows in A that are NOT in B
        missing_in_b = df_a[cbg_a.isin(cbg_b)]

        print(f"Number of tracts in {city} with valid CBG Code: {len(missing_in_b)/20}")

right_size()