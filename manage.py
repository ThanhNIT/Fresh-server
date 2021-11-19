import os

from flask_cors import CORS

from app import create_app
from flask_script import Manager
from flask_migrate import MigrateCommand

config_name = os.getenv('FLASK_CONFIG', 'development')

app = create_app(config_name)
manager = Manager(app)
manager.add_command('db', MigrateCommand)
CORS(app, resources={r"/*": {"origins": "*"}})
if __name__ == '__main__':
    manager.run()