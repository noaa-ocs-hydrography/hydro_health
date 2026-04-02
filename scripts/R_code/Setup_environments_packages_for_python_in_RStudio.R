### R Script to begin using python in R Studio 

library(reticulate)
# Create new environment
version <- "3.12.4"
install_python(version = version)
virtualenv_create("my-python", python_version = version)

use_virtualenv("my-python", required = T)
virtualenv_install(envname = "my-python", "numpy", ignore_installed = F, pip_options = character() )
virtualenv_install(envname = "my-python", "pandas", ignore_installed = F, pip_options = character() )
virtualenv_install(envname = "my-python", "gdal", ignore_installed = F, pip_options = character() )