from distutils.core import setup, Extension
import os

module1 = Extension('libmain',
       define_macros = [('MAJOR_VERSION', '1'),
              ('MINOR_VERSION', '0')],
       include_dirs = ['/opt/dashcam/include', '/usr/include/python3.6', '/usr/include'],
       libraries = ['python2.7', 'boost_python3', 'boost_numpy3', 'boost_thread', 
       'boost_system', 'boost_filesystem', 
       'opencv_core', 'opencv_text', 'opencv_ml', 'opencv_objdetect', 'opencv_imgcodecs'],
       library_dirs = ['/usr/lib', '/home/dashcam/build_opencv/lib', '/opt/lib/boost', 
       '/opt/dashcam/Text/'],
       sources = ['libmain.cpp'],
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
