from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

from numpy import get_include

sourcefiles = ['plsa_em.pyx']
ext_modules = [Extension("plsa_em", sourcefiles,
               include_dirs=[get_include(),'.'],
               extra_compile_args=['-O3'])]

setup(name = 'PLSA EM Algorithm',
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules)
