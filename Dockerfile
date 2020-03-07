FROM ubuntu:18.04

run apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    libcurl3-dev \
    libleptonica-dev \
    liblog4cplus-dev \
    libopencv-dev \
    libtesseract-dev \
    wget \
    python2.7 \
    python2.7-dev \
    python3 \
    python3-dev \
    libboost-all-dev -y

run curl -sL https://deb.nodesource.com/setup_10.x | bash -

run apt-get install nodejs ffmpeg -y

run wget https://bootstrap.pypa.io/get-pip.py

run python3 get-pip.py

run apt-get install build-essential \
    cmake unzip pkg-config libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev \
    software-properties-common \
    gfortran -y

RUN mkdir -p /opt/lib/boost
workdir /opt/lib/boost
RUN ln -s /usr/lib/x86_64-linux-gnu/libboost_* ./

# install MPI
run apt-get install libcr-dev mpich python3-mpi4py -y
run pip install --user gTTS opencv-python numpy mpi4py pandas matplotlib

RUN mkdir /home/openalpr
workdir /home/openalpr
RUN git clone https://github.com/openalpr/openalpr/

# Setup the build directory
run mkdir /home/openalpr/openalpr/src/build
workdir /home/openalpr/openalpr/src/build

# Setup the compile environment
run cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr -DCMAKE_INSTALL_SYSCONFDIR:PATH=/etc .. && \
    make -j2 && \
    make install

run python3 /home/openalpr/openalpr/src/bindings/python/setup.py install

workdir /home/dashcam/Text
RUN python2.7 setup.py build_ext -i

workdir /home/dashcam/ALPR
RUN python2.7 setup.py build_ext -i

RUN apt-get install -y openssh-server

COPY docker.pub /root/.ssh/authorized_keys

COPY user-config.jam /root
COPY project-config.jam /root

RUN mkdir /var/run/sshd
RUN sed -i 's/PermitRootLogin without-password/PermitRootLogin yes PasswordAuthentication yes PermitEmptyPasswords no/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 5678
CMD ["/usr/sbin/sshd", "-D"]

workdir /data

EXPOSE 5900
