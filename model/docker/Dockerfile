FROM pytorch/pytorch:latest

COPY docker/requirements.txt requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential\
         libsm6 \
         libxext6 \
         libxrender-dev \
         ffmpeg  \
         tree \
         graphviz \
    python3-pip &&\
    pip install --upgrade pip

RUN pip install -r requirements.txt

WORKDIR /app/work/
COPY src/ src/

CMD ["bash"]
