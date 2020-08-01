import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def load_data(messages_filepath, categories_filepath):
    """
    Function to Load Messages Data with Categories 
    
    Inputs:
    
     Path to the CSV file of messages > messages_filepath
     Path to the CSV file of categories > categories_filepath
        
    Output:
        
     Data frame with combined messages and categories data > df
        
    """
    # Read message data
    messages = pd.read_csv(messages_filepath)
    
    # Read categories data
    categories = pd.read_csv(categories_filepath)

    #Merge messages and categories based on key value of id
    df = pd.merge(messages, categories, on='id')

    return df



def clean_data(df):
    """
    Function to Clean Categories Data 
    
    Arguments:
     
     Combined data of messages and categories > df
    
    Outputs:
    
     Data frame after cleaning up the categories column > df
     
    """
    # Spliting the categories column by making each value as a new column
    categories = df.categories.str.split(pat=';', expand=True)

    # Select the first row of the categories dataframe
    row = categories.iloc[0]

    # Use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])

    # Rename the categories columns
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int)

    #Drop the redundant categories column
    df.drop('categories', axis=1, inplace=True)

    # Concatenate the original dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(subset='id', inplace=True)

    return df



def save_data(df, database_filename):
    """
    Function to Save Data to SQLite DB
    
    Arguments:
       
    SQLite database destination - >  database_filename
    
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('df', engine, index=False, if_exists='replace')



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
