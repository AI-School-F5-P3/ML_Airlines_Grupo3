import sqlite3
import logging

class SQLiteHandler(logging.Handler):
    def __init__(self, db_name):
        logging.Handler.__init__(self)
        self.db_name = db_name
    
    def emit(self, record):
        log_entry = self.format(record)
        self.save_log_to_db(record.levelname, log_entry)
    
    def save_log_to_db(self, level, message):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO logs (level, message) VALUES (?, ?)
        ''', (level, message))
        conn.commit()
        conn.close()

# Usage
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create the SQLite handler
    sqlite_handler = SQLiteHandler('loggins.db')
    sqlite_handler.setLevel(logging.INFO)
    
    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sqlite_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(sqlite_handler)

setup_logging()
