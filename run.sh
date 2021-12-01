set FLASK_APP=manage.py
set FLASK_CONFIG=mysql
python3 manage.py db init
python3 manage.py db migrate
python3 manage.py db upgrade
python3 -m flask run -h 0.0.0.0
