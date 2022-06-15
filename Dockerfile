
FROM continuumio/miniconda3

LABEL author="Shreyan Shetty"

COPY deploy/conda/enivronment.yml environment.yml
COPY dist .

RUN conda env create -f environment.yml
RUN conda init bash

CMD ["/bin/bash"]
