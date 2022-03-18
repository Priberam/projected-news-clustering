FROM nvidia/cuda:11.1-base
ENV PYTHON_VERSION=3.6

RUN mkdir /app
WORKDIR /app


# Conda, fragments from: https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/Dockerfile
# Explicit install of Python 3.7 with:
# /opt/conda/bin/conda install -y python=$PYTHON_VERSION && \
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
  apt-get install -y wget bzip2 ca-certificates curl git && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh -O ~/miniconda.sh && \
  /bin/bash ~/miniconda.sh -b -p /opt/conda && \
  rm ~/miniconda.sh && \
  /opt/conda/bin/conda install -y python=$PYTHON_VERSION && \
  /opt/conda/bin/conda clean -tipsy && \
  ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
  echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
  echo "conda activate base" >> ~/.bashrc

# End Conda


RUN python3 --version
RUN python3 -m venv ./venv-local
ENV PATH="/app/venv-local/bin:$PATH"
ADD fastapi_web_handler.py /app/
ADD requirements.txt /app/
ADD imports.py /app/
RUN apt-get -y update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install cargo
RUN pip3 install --upgrade pip
RUN pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install -r requirements.txt
RUN python3 imports.py
CMD ["uvicorn", "--host", "0.0.0.0", "/app/fastapi_web_handler:app"]
