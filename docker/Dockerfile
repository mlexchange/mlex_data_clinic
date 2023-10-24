FROM python:3.8
MAINTAINER THE MLEXCHANGE TEAM

RUN ls
COPY docker/requirements.txt requirements.txt

RUN pip3 install --upgrade pip &&\
    pip3 install -r requirements.txt\
    pip install git+https://github.com/taxe10/mlex_file_manager 

WORKDIR /app/work
ENV HOME /app/work
COPY src src
ENV PYTHONUNBUFFERED=1

CMD ["bash"]
CMD python3 src/frontend.py