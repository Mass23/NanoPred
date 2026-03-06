from setuptools import setup, find_packages

setup(
    name='nanopred',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'nanopred=nanopred.cli:main',
        ],
    },
    install_requires=[
        'some_dependency',
        'another_dependency',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A brief description of your project.',
    license='MIT',
    url='https://github.com/Mass23/NanoPred',
)