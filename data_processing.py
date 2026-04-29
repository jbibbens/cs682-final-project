import pandas as pd
import matplotlib.pyplot as plt

def filter_cities():
    # 1. Load your data
    # Replace 'input_data.csv' with your actual filename
    df = pd.read_csv('./Indicators_for_SDG_1_CBG_level.csv')

    # 2. Define your list of allowed cities
    allowed_cities = ['New York city', 'Los Angeles city', 'Dallas city', 'Chicago city', 'Houston city', 'Phoenix city', 'Philadelphia city', 'San Antonio city', 'San Diego city', 'Jacksonville city']
    allowed_year = [2021]

    # 3. Filter the dataframe
    # .isin() checks if the value in "City Name" exists in your list
    filtered_df = df[df['City Name'].isin(allowed_cities) & df['Year'].isin(allowed_year)]

    # 4. Save the result
    # index=False prevents pandas from adding an extra ID column to your new file
    filtered_df.to_csv('filtered_cities.csv', index=False)

    print(f"Done! Reduced the file from {len(df)} to {len(filtered_df)} rows.")

def count_rows():
    df = pd.read_csv('./filtered_cities.csv')
    city_counts = df['City Name'].value_counts()

    # 2. Display the results in the console
    print("Row counts per city:")
    print(city_counts)


def count_row_cbg():
    city_list = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Jacksonville']
    
    for city in city_list:
        df = pd.read_csv(f'./year2020_to_2023/images_within_CBGs_in_{city}_city.csv')
        cbg_counts = df['CBG Code'].value_counts()

        min_cgb = cbg_counts.idxmax()
        min_count = cbg_counts.max()
        # 2. Display the results in the console
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

        # Create the filename string
        df['filename'] = df['y_18'].astype(str) + '_' + df['x_18'].astype(str) + '.png'

        # Pick one CBG per filename
        # 'first' is usually fine for a random-ish assignment
        df_final = df.drop_duplicates(subset=['filename'], keep='first')

        # Now you have a clean 1:1 mapping
        df_final[['filename', 'CBG Code']].to_csv(f'./image_to_cbg/{city}.csv', index=False)

def right_size():
    city_list = ['New York', 'Los Angeles', 'Chicago', 'Dallas','Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Jacksonville']
    for city in city_list:
        # 1. Define your threshold X
        X = 20 

        # 2. Load your data
        df = pd.read_csv(f'./image_to_cbg_unfiltered/final_image_to_cbg_{city}.csv')
        # 2. Get counts for each CBG
        cbg_counts = df['CBG Code'].value_counts()

        # 3. Filter for CBGs where the count is greater than X
        # This creates a boolean mask and returns only the rows that meet the criteria
        large_cbgs = cbg_counts[cbg_counts > X]

        # 4. Get the results
        distinct_count = len(large_cbgs)

        print(f"There are {distinct_count} distinct CBGs in {city} with more than {X} rows.")


def stratified_sample():
    #city_list = ['New York', 'Los Angeles', 'Dallas','Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Jacksonville']
    city_list = ['Dallas']      
    for city in city_list:
        filtered_df = pd.read_csv(f'./image_to_cbg/{city}.csv')
        # 1. Configuration
        N = 20  # Number of rows per CBG
        M = 250 # Number of distinct CBGs to select

        # 2. Find eligible CBGs (those with at least N rows)
        cbg_counts = filtered_df['CBG Code'].value_counts()
        eligible_cbgs = cbg_counts[cbg_counts >= N].index.tolist()

        # # 3. Check if we have enough distinct CBGs
        # if len(eligible_cbgs) < M:
        #     print(f"Warning: Only found {len(eligible_cbgs)} CBGs with {N}+ rows.")
        #     # You might want to adjust M here or stop the script
        #     selected_cbgs = eligible_cbgs 
        # else:
        #     # Pick M distinct CBGs (using [:M] for the first M, 
        #     # or use random.sample(eligible_cbgs, M) for a random selection)
        selected_cbgs = eligible_cbgs[:M]

        # 4. Filter and Sample
        # We filter the dataframe for only our selected CBGs, 
        # then group by CBG and take the first N rows of each group.
        final_df = (
            filtered_df[filtered_df['CBG Code'].isin(selected_cbgs)]
            .groupby('CBG Code')
            .head(N)
        )

        # 5. Save the result
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

        # 3. Filter the input_df
        # We only keep rows where the 'filename_check' exists in the mapping_df's 'filename' column
        filtered_input = input_df[input_df['filename_check'].isin(mapping_df['filename'])]

        # 4. Cleanup: Remove the helper column and save
        final_output = filtered_input.drop(columns=['filename_check'])
        final_output.to_csv(f'./satellite_imagery_collection/tilefile_scd/{city}.csv', index=False)

def do_merge():
    city_list = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Dallas', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Jacksonville']
    
    for city in city_list:
        mapping_df = pd.read_csv(f'./image_to_cbg/stratified_sample_{city}.csv')
        input_df = pd.read_csv(f'./satellite_imagery_collection/tilefile_scd/{city}_filtered.csv')

        # 2. Create the matching string in input_df
        # Ensure this matches your mapping format (y_x.png)
        input_df['filename'] = (
            input_df['y_tile'].astype(str) + 
            '_' + 
            input_df['x_tile'].astype(str) + 
            '.png'
        )

        # 3. Merge the DataFrames
        # This will keep rows from input_df and attach the 'CBG Code' where the filename matches.
        # 'how=inner' ensures we only keep tiles that have a matching CBG.
        filtered_input_with_cbg = pd.merge(
            input_df, 
            mapping_df[['filename', 'CBG Code']], 
            on='filename', 
            how='inner'
        )

        # 4. Cleanup
        # We can drop the temporary 'filename' column if you don't need it in the final download CSV
        final_output = filtered_input_with_cbg.drop(columns=['filename'])

        # 5. Save the result
        final_output.to_csv(f'./satellite_imagery_collection/tilefile_scd/{city}_filtered.csv', index=False)

def add_poverty_rate():

    city_list = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Dallas', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Jacksonville']
    
    for city in city_list:
        # 1. Load the census data (the file with all cities)
        census_df = pd.read_csv('filtered_cities.csv')

        # 2. Calculate the Poverty Rate
        # We sum Above and Below to get the total population evaluated for poverty
        census_df['Total Pop for Poverty'] = (
            census_df['Population Above Poverty'] + census_df['Population Below Poverty']
        )

        # Avoid division by zero for empty tracts
        census_df['Poverty Rate'] = census_df['Population Below Poverty'] / census_df['Total Pop for Poverty']
        census_df['Poverty Rate'] = census_df['Poverty Rate'].fillna(0) # Handle 0/0 cases

        # 3. Load your city-specific image mapping file
        city_mapping_df = pd.read_csv(f'./labels/{city}.csv')

        # 4. Merge the Poverty Rate into the mapping file
        # We use a 'left' join to keep all image filenames
        result_df = pd.merge(
            city_mapping_df,
            census_df[['CBG Code', 'Poverty Rate']],
            on='CBG Code',
            how='left'
        )

        # 5. Save the updated file
        result_df.to_csv(f'./labels/{city}.csv', index=False)

def filter_no_data():
    city_list = ['Dallas']
    
    for city in city_list:
        # 1. Load the census data (the file with all cities)
        census_df = pd.read_csv('filtered_cities.csv', dtype={'CBG Code': str})
        valid_cbgs = census_df['CBG Code'].unique()
        mapping_df = pd.read_csv(f'./within/images_within_CBGs_in_{city} city.csv', dtype={'CBG Code': str})

        filtered_mapping_df = mapping_df[mapping_df['CBG Code'].isin(valid_cbgs)]

        filtered_mapping_df.to_csv(f'./within/cleaned_{city} city.csv', index=False)

def remove_extra_col():

    city_list = city_list = ['New York', 'Los Angeles', 'Chicago','Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Jacksonville']

    for city in city_list:
        df = pd.read_csv(f'./labels/{city}.csv')
        df = df.drop(columns=['Poverty Rate_y'])
        df.to_csv(f'./labels/{city}.csv', index=False)

remove_extra_col()