from distutils.core import setup, Extension
import sysconfig
import os

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-std=c++11"]

module1 = Extension('libalpr',
       define_macros = [('MAJOR_VERSION', '1'),
              ('MINOR_VERSION', '0')],
       include_dirs = ['/usr/include', '/home/dashcam/include', '/usr/include/python3.6', 
       '/home/openalpr/openalpr/src/openalpr/'],
       libraries = ['python2.7', 'boost_python3', 'boost_numpy3', 
       'boost_thread', 'boost_system', 
       'boost_filesystem', 'opencv_core', 'opencv_imgcodecs', 'openalpr'],
       library_dirs = ['/home/dashcam/build_opencv/lib', 
       '/opt/lib/boost', '/home/openalpr/openalpr/src/build/openalpr'],
       extra_compile_args=extra_compile_args,
       sources=['alpr.cpp'],
       language="c++11")

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