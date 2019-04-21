FROM python:3.7

RUN apt-get update
RUN apt-get --yes install libomp-dev

COPY requirements.txt .
RUN pip3 install -r requirements.txt && rm requirements.txt

COPY . /mlrank/
ENV PYTHONPATH="$PYTHONPATH:/"

#ENV CACHE_FOLDER="./data/"

WORKDIR /mlrank

CMD ["python3", "./validation/synth.py"]
