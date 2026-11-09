# Packages needed by titanic_model.R
# The base image (rocker/r-ver:4.4.1) pins CRAN to a fixed date, so versions are reproducible.
packages <- c("dplyr")

install.packages(packages)

# Fail the Docker build loudly if any package did not install
missing <- packages[!packages %in% rownames(installed.packages())]
if (length(missing) > 0) stop("Failed to install: ", paste(missing, collapse = ", "))
