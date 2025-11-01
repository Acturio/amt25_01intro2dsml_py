library(reticulate)

install_python(list = TRUE)
install_python(version = "3.11.8")
####################
#### VirtualEnv ####
####################

# create a new environment 
use_python("/usr/bin/python3.11.8")

# virtualenv_create(envname = "venv_dsml_py3_10", python_version = "python3.10")
virtualenv_create(envname = "dsml_py3_11", python_version = "python3.11.8")

virtualenv_list()

use_virtualenv("venv_dsml_py3_11")
py_install("numpy") # instala desde ambiente virtual

# import pandas 
pandas <- import("pandas")

# install pandas
virtualenv_install("venv_dsml_py3_11", requirements = "data/requirements.txt")
virtualenv_install("dsml_py3_11", requirements = "data/requirements.txt")
virtualenv_install("venv_dsml_py3_11", "numpy")
virtualenv_install("venv_dsml_py3_11", "pandas")
virtualenv_install("venv_dsml_py3_11", "openpyxl")
virtualenv_install("venv_dsml_py3_11", "siuba")
virtualenv_install("venv_dsml_py3_11", "plydata")
virtualenv_install("venv_dsml_py3_11", "scikit-learn")
virtualenv_install("venv_dsml_py3_11", "plotnine")
virtualenv_install("venv_dsml_py3_11", "mizani==0.9.2")
virtualenv_install("venv_dsml_py3_11", "mlxtend==0.23.0")
virtualenv_install("venv_dsml_py3_11", "seaborn==0.13.0")


# import pandas 
pandas <- import("pandas")
openpyxl <- import("openpyxl")

#virtualenv_remove("venv_dsml_py3_10")

