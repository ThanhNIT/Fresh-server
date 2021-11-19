pip install -r requirements.txt
pip install --upgrade tensorflow
pip install  scikit-image



set FLASK_APP=manage.py
set FLASK_CONFIG=mysql
python manage.py db init
python manage.py db migrate
python manage.py db upgrade
python -m flask run -h 0.0.0.0

