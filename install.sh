#echo "-------------------------------------------------------------------------------------------------"
#echo "downloading ROOT now"
#echo "-------------------------------------------------------------------------------------------------"
#wget https://root.cern.ch/download/root_v6.10.00.Linux-ubuntu16-x86_64-gcc5.4.tar.gz
#echo "-------------------------------------------------------------------------------------------------"
#echo "unzipping ROOT archive"
#echo "-------------------------------------------------------------------------------------------------"
#tar zxvf root_v6.10.00.Linux-ubuntu16-x86_64-gcc5.4.tar.gz
#echo "-------------------------------------------------------------------------------------------------"
#echo "setting up ROOT environment"
#echo "-------------------------------------------------------------------------------------------------"
#. root/bin/thisroot.sh
echo "-------------------------------------------------------------------------------------------------"
echo "installing python-pip and virtualenv"
echo "-------------------------------------------------------------------------------------------------"
sudo apt-get install python-pip python-dev python-virtualenv
echo "-------------------------------------------------------------------------------------------------"
echo "starting setup of virtual environment and tensorflow installation"
echo "-------------------------------------------------------------------------------------------------"
virtualenv tensorflow_tut
cd tensorflow_tut
source bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow
pip install --upgrade scipy
pip install --upgrade matplotlib
#pip install --upgrade rootpy
#NOTMVA=1 pip2 install --upgrade  root_numpy
#pip install pandas
#pip install tables
#pip install scikit-learn
cd ..
echo "-------------------------------------------------------------------------------------------------"
echo "executing smallest tensorflow test program"
echo "-------------------------------------------------------------------------------------------------"
python tensorflow_test.py
echo "-------------------------------------------------------------------------------------------------"
echo "deactivating virtual environment before exercive start"
echo "-------------------------------------------------------------------------------------------------"
deactivate
#echo "-------------------------------------------------------------------------------------------------"
#echo "deleting downloaded root archive"
#echo "-------------------------------------------------------------------------------------------------"
#rm root_v6.10.00.Linux-ubuntu16-x86_64-gcc5.4.tar.gz
echo "-------------------------------------------------------------------------------------------------"
echo "succesfully finished installation"
echo "-------------------------------------------------------------------------------------------------"
