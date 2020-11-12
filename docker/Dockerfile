FROM pytorchlightning/pytorch_lightning:1.0.3-py3.8-torch1.6

# RUN apt-get install byobu git -y

RUN pip install --upgrade pip \
    && pip install zarr pandas>=1.1.3 swifter>=1.0.7 albumentations>=0.5.0 scikit-learn>=0.23.2

# COPY ./requirements.txt requirements.txt

# RUN pip3 install -r requirements.txt
