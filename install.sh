sudo apt-get install python-pip python-dev python-virtualenv
echo "starting installation"
virtualenv tensorflow_tut
cd tensorflow_tut
source bin/activate
pip install --upgrade tensorflow
cd ..
python tensorflow_test.py
deactivate
