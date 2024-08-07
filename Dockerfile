FROM python:3.11
MAINTAINER THE MLEXCHANGE TEAM

RUN ls
COPY pyproject.toml pyproject.toml
COPY README.md README.md

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libhdf5-dev

RUN pip install --upgrade pip &&\
    pip install .
WORKDIR /app/work
ENV HOME /app/work
COPY src src
COPY frontend.py frontend.py
COPY gunicorn_config.py gunicorn_config.py

CMD ["bash"]
CMD gunicorn -c gunicorn_config.py --reload frontend:server
