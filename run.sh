dir=$(pwd)
if test -z "$dir/boost"
then
    git clone https://github.com/boostorg/boost/ boost
else
    echo "boost repository already existing"
fi
cp -Rf tools boost/
sudo docker run -d  -v `pwd`:/home/boost openalpr /home/boost/bootstrap.sh --prefix=/usr/local \
--libdir=/usr/local/lib \
--with-python-root=/usr/lib/python3.6 --without-icu \
--with-libraries=filesystem,system,thread,python,numpy

sudo docker run -d -v `pwd`:/home/boost openalpr /home/boost/boost/b2 --prefix=/usr/local \
--libdir=/usr/local/lib -d0 -j2 --layout=tagged \
--user-config=./project-config.jam -sNO_LZMA=1 install \
threading=multi link=shared cxxflags=-std=c++11 
