import os
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    """
    Common configurations
    """
    SECRET_KEY = os.getenv('SECRET_KEY', 'secret_key')
    DEFAULT_PAGE_SIZE = 15
    DEFAULT_PAGE = 1
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 465
    # MAIL_PORT = 465
    MAIL_USERNAME = 'thanhnguyen.it.work@gmail.com'
    MAIL_PASSWORD = 'Muoiad,23'
    MAIL_SUPPRESS_SEND = False
    MAIL_USE_TLS = False
    MAIL_USE_SSL = True
    TOKEN_EXPIRED_TIME = timedelta(days=50)
    PROPAGATE_EXCEPTIONS = True
    MIN_PASSWORD_CHARACTERS = 6
    FRONTEND_ADDRESS = 'http://localhost:3000'


class DevelopmentConfig(Config):
    """
    Development configurations
    """
    # FLASK_ENV = 'development'
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = "mysql://root:muoi@127.0.0.1:3306/fresh_fruit"
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    SERVER_ADDRESS = 'http://127.0.0.1:5000'
    MONGO_URI = "mongodb+srv://ThanhNguyen:thanhnguyen@springbootrestful.xlacd.mongodb.net/fresh?retryWrites=true&w=majority"


class MySqlConfig(Config):
    """
    Development configurations
    """
    # FLASK_ENV = 'development'
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = "mysql://root:muoi@127.0.0.1:3306/fresh_fruit"
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    SERVER_ADDRESS = 'http://127.0.0.1:5000'
    MONGO_URI = "mongodb+srv://ThanhNguyen:thanhnguyen@springbootrestful.xlacd.mongodb.net/fresh?retryWrites=true&w=majority"


class ProductionConfig(Config):
    """
    Production configurations
    """
    MONGO_URI = "mongodb+srv://ThanhNguyen:thanhnguyen@springbootrestful.xlacd.mongodb.net/fresh?retryWrites=true&w=majority"

    DEBUG = False


class TestingConfig(Config):
    """
    Testing configurations
    """

    TESTING = True


app_config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'mysql': MySqlConfig,
    'testing': TestingConfig
}
