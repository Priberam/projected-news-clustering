FROM python:3.6
RUN mkdir /app
WORKDIR /app
RUN python -m venv ./venv-local
ENV PATH="/app/venv-local/bin:$PATH"
ADD fastapi_web_handler.py /app/
ADD requirements.txt /app/
ADD imports.py /app/
RUN apt-get -y update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install cargo
RUN pip install --upgrade pip
RUN pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt
RUN python imports.py
CMD ["uvicorn", "--host", "0.0.0.0", "/app/fastapi_web_handler:app"]