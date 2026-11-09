FROM rocker/r-ver:4.4.1

WORKDIR /app

# Install R packages first so Docker can cache this layer
COPY install_packages.R .
RUN Rscript install_packages.R

COPY titanic_model.R .

# The data folder is mounted at runtime (see README), never baked into the image
ENV DATA_DIR=/data

CMD ["Rscript", "titanic_model.R"]
