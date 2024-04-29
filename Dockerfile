FROM python:3.9
MAINTAINER THE MLEXCHANGE TEAM

RUN ls
COPY pyproject.toml pyproject.toml
COPY README.md README.md

RUN pip3 install --upgrade pip &&\
    pip3 install . &&\
    pip install git+https://github.com/mlexchange/mlex_file_manager

RUN git clone https://github.com/mlexchange/mlex_dash_component_editor
WORKDIR /app/work
ENV HOME /app/work
COPY src src
COPY frontend.py frontend.py
RUN mv /mlex_dash_component_editor/src/dash_component_editor.py /app/work/src/dash_component_editor.py

CMD ["bash"]
CMD python3 frontend.py
