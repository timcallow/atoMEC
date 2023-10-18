import os
import re
import sys
import platform
import subprocess

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir]
        cmake_args += ['-DBUILD_SHARED_LIBS=ON']
        cmake_args += ['-DENABLE_PYTHON=ON']
        cmake_args += ['-DBUILD_TESTING=OFF']
        # Python tests need third derivatives
        cmake_args += ['-DDISABLE_KXC=OFF']
        cmake_args += ['-DENABLE_XHOST=OFF']  # EDIT for generality

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

extras = {
    'dev': ['bump2version'],
    'docs': open('docs/requirements.txt').read().splitlines(),
    'tests': open('tests/requirements.txt').read().splitlines(),
}

setup(
    name="atoMEC",
    version="1.3.0",
    description="KS-DFT average-atom code",
    long_description=readme,
    long_description_content_type='text/markdown',
    author="Tim Callow et al.",
    author_email="t.callow@hzdr.de",
    url="https://github.com/atomec-project/atoMEC",
    license=license,
    packages=[find_packages()[0], find_packages()[1], find_packages("libxc")[0]],
    package_dir={"pylibxc": "libxc"},
    install_requires=open('requirements.txt').read().splitlines(),
    ext_modules=[CMakeExtension('pylibxc.libxc', "libxc")],
    cmdclass=dict(build_ext=CMakeBuild),    
    extras_require=extras,
    python_requires=">=3.6",
)
