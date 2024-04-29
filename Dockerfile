FROM python:3.11
MAINTAINER THE MLEXCHANGE TEAM

RUN ls
COPY pyproject.toml pyproject.toml
COPY README.md README.md

RUN pip3 install --upgrade pip &&\
    pip3 install .
WORKDIR /app/work
ENV HOME /app/work
COPY src src
COPY frontend.py frontend.py

CMD ["bash"]
CMD python3 frontend.py
