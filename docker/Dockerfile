FROM pytorchlightning/pytorch_lightning:1.0.3-py3.8-torch1.6

RUN apt-get install byobu git -y

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
