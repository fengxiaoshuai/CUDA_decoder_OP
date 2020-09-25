import  os
import numpy
import subprocess
import tensorflow as tf 
from setuptools import setup
from os.path import join as pjoin
from distutils.extension import Extension
from distutils.command.build_ext import build_ext

def find_in_path(name, path):
    "Find a file in a search path"
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig




CUDA = locate_cuda()
mod = 'transformer'

is_debug_info = False

extra_compile_args={
    'gcc': [],
    'nvcc': ['-Xcompiler','-fPIC', '-O3',  '-shared', \
             '--expt-relaxed-constexpr', '-arch=sm_70',\
             '--std=c++11', '-D_GLIBCXX_USE_CXX11_ABI=0',\
             '-DGOOGLE_CUDA=1', '-DNDEBUG'],
    'g++': tf.sysconfig.get_link_flags() + ['-std=c++11'],
}

if is_debug_info:
    for _ in extra_compile_args:
        extra_compile_args[_].append('-DDEBUGINFO')

ext = Extension(os.path.join(mod, '_'+ mod),
                sources=['src/decoding_op.cc',
                        'src/decoding_op.cu.cc',
                        'src/decoding.cu',
                        'src/decoder.cu'],
               library_dirs=[CUDA['lib64']],
               runtime_library_dirs=[CUDA['lib64']],
               libraries=['cudart','cublas','tensorflow_framework'],
               extra_link_args=tf.sysconfig.get_link_flags(),
               extra_compile_args=extra_compile_args,
               include_dirs = [CUDA['include'], 'src', tf.sysconfig.get_include()])



def customize_compiler_for_nvcc(self):
    self.src_extensions.append('.cu')

    default_compiler_so = self.compiler_so
    super = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        self.set_executable('compiler_so', CUDA['nvcc'])
        postargs = extra_postargs['nvcc']
        super(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler_so = default_compiler_so

    self._compile = _compile


class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

setup(name=mod,
      author='s.feng@trip.com',
      version='0.0.1',
      packages=[mod],
      package_dir={mod: mod},
      ext_modules = [ext],
      install_requires=['tensorflow-gpu==1.15.4'],
      cmdclass={'build_ext': custom_build_ext},
)


