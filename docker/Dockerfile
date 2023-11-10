FROM python:3.9
MAINTAINER THE MLEXCHANGE TEAM

RUN ls
COPY docker/requirements.txt requirements.txt

RUN pip3 install --upgrade pip &&\
    pip3 install -r requirements.txt\
    pip install git+https://github.com/taxe10/mlex_file_manager
RUN pip3 uninstall dash-component-editor==0.0.7 -y

RUN git clone https://github.com/mlexchange/mlex_dash_component_editor
WORKDIR /app/work
ENV HOME /app/work
COPY src src
RUN mv /mlex_dash_component_editor/src/dash_component_editor.py /app/work/src/dash_component_editor.py

CMD ["bash"]
CMD python3 src/frontend.py