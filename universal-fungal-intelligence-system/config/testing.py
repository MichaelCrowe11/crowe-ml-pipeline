# Configuration settings for the testing environment

import os

class Config:
    TESTING = True
    DEBUG = True
    DATABASE_URI = os.getenv('TEST_DATABASE_URI', 'sqlite:///:memory:')
    SECRET_KEY = os.getenv('SECRET_KEY', 'your_secret_key')
    LOGGING_LEVEL = os.getenv('LOGGING_LEVEL', 'DEBUG')

class TestingConfig(Config):
    pass