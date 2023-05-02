# from setuptools import setup
# setup(
# name='deepexploration',
# version='0.1',
# description='Testing installation of Package',
# url='#',
# author='auth',
# author_email='author@email.com',
# license='MIT',
# packages=['deepexploration', 'hello'],
# zip_safe=False
# )
import os

from setuptools import find_packages
from setuptools import setup

dir_path = os.path.dirname(os.path.realpath(__file__))

# with open(os.path.join(dir_path, 'src', 'airobot', 'version.py')) as fp:
#     exec(fp.read())


def read_requirements_file(filename):
    req_file_path = '%s/%s' % (dir_path, filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]


packages = find_packages('hello')
# Ensure that we don't pollute the global namespace.
# for p in packages:
#     assert p == 'airobot' or p.startswith('airobot.')


def pkg_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('../..', path, filename))
    return paths


# extra_pkg_files = pkg_files('src/airobot/urdfs')

setup(
    name='deepexploration',
    version='0.1.1',
    author='Us',
    url='https://github.com/ppfenninger/Sensorimotor_Learning_Final.git@package-testing',
    license='MIT',
    packages=packages,
    package_dir={'': 'hello'},
    python_requires='>=2.7, <3.11',
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Framework :: Robot Framework"
    ],
    # install_requires=read_requirements_file('requirements.txt'),
)
