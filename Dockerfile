FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:23.02-py3
RUN apt-get --allow-insecure-repositories update && \
    apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/Europe/Zurich /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata
ENV TZ="Europe/Zurich"
RUN add-apt-repository -y ppa:ubuntugis/ppa
RUN apt-get --allow-insecure-repositories update && apt-get install -y sudo gdal-bin python3-gdal 
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
COPY . /SMARTIES
WORKDIR /SMARTIES
RUN pip install --no-cache-dir --upgrade pip 
RUN pip install --no-cache-dir -r requirements.txt