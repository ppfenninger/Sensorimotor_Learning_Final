from setuptools import find_packages, setup
setup(
    name='deepexploration',
    packages=find_packages(
            where='deepexploration',
        ),
    package_dir={"": "deepexploration"},
    version='0.1.1',
    description='An deep exploration reinforcement learner',
    author='Me',
    license='MIT',
)