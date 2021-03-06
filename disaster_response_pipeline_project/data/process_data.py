import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Input: 
    messages_filepath - the message filepathe
    categories_filepath - the categories filepathe
    
    Output:
    Merged messages and categories files  
    
    load the two csv files and merge them based on the same id'''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)   
    return  messages.merge(categories,how = 'left',on='id')


def clean_data(df):
    '''
    Input:
    df - the merged dataframe
    
    Output:
    df - the dataframe after split categories into separate colums and each column only has numbers 0 or 1
    
    First, split categories into separate category columns.
    Second, convert category values to just numbers 0 or 1. 
    And replace categories column in df with new category columns. '''
    categories = df.categories.str.split(';',expand = True) 
    row = categories.iloc[1]
    category_colnames = row.apply(lambda x:x.split('-')[0])
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] =categories[column].apply(lambda x: x.split('-')[1])
    
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    categories = categories.replace(2,1)
    categories.head()
    # drop the original categories column from `df`
    df.drop(columns='categories',inplace= True)
    df = df.join(categories,how='left')
    df['id'].nunique()
    df.drop_duplicates(subset = 'id',inplace = True)
    return df
    
def save_data(df, database_filename):
    '''
    Input:
    df - the dataframe after cleaning
    database_filename - the file path of database
    
    Save the clean dataset into an sqlite database.'''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    
    df.to_sql('ETLTable', engine, if_exists='replace',index=False)


def main():
    ''''''
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