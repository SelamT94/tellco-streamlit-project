import unittest
import pandas as pd
from sqlalchemy import create_engine
import os

class TestDatabaseToCSV(unittest.TestCase):

    def setUp(self):
        # Define database connection parameters
        self.connection_params = {
            "host": "localhost",
            "user": "postgres",
            "password": "postgres",
            "port": "5432",
            "database": "telecom"
        }
        # Define SQL query to fetch data
        self.sql_query = 'SELECT * FROM xdr_data'
        # Define CSV file path
        self.csv_file = 'xdr_data.csv'

    def tearDown(self):
        # Remove the CSV file after testing
        if os.path.exists(self.csv_file):
            os.remove(self.csv_file)

    def test_export_to_csv(self):
        # Create database engine
        engine = create_engine(f"postgresql+psycopg2://{self.connection_params['user']}:{self.connection_params['password']}@{self.connection_params['host']}:{self.connection_params['port']}/{self.connection_params['database']}")
        
        # Read data from the database
        df = pd.read_sql(self.sql_query, con=engine)
        
        # Export data to CSV file
        df.to_csv(self.csv_file, index=False)
        
        # Check if CSV file is created
        self.assertTrue(os.path.exists(self.csv_file))
        
        # Check if CSV file is not empty
        self.assertTrue(os.path.getsize(self.csv_file) > 0)

if __name__ == '__main__':
    unittest.main()
