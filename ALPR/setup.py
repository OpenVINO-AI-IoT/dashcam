from distutils.core import setup, Extension
import os

module1 = Extension('libalpr',
       define_macros = [('MAJOR_VERSION', '1'),
              ('MINOR_VERSION', '0')],
       include_dirs = ['/usr/include', '/opt/dashcam/include', '/usr/include/python3.6', 
       '/home/openalpr/openalpr/src/openalpr/'],
       libraries = ['python2.7', 'boost_python3', 'boost_numpy3', 
       'boost_thread', 'boost_system', 
       'boost_filesystem', 'opencv_core', 'opencv_imgcodecs', 'openalpr'],
       library_dirs = ['/home/dashcam/build_opencv/lib', '/home/dashcam/libs/', 
       '/opt/dashcam/ALPR/', 
       '/opt/lib/boost', '/home/openalpr/openalpr/src/build/openalpr'],
       sources=['libalpr.cpp'],
       extra_compile_args=['-DUSE_BOOST_MODULE=ON', "-std=c++11"])

setup (name = 'PackageName',
       version = '1.0',
       description = 'This is a demo package',
       author = '',
       author_email = '',
       url = '',
       long_description = '''
This is really just a demo package.
''',
       ext_modules = [module1])