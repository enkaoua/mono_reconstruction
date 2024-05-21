from setuptools import setup, find_packages

setup(
    name='LTRL',        # project's name
    version='0.1',                   #  project's version
    packages=find_packages(),        # Automatically find and include all Python packages
    install_requires=[               # List dependencies required for your project
        'numpy',
    ],
    author='Aure Enkaoua',              
    #author_email='@gmail.com',   
    description='recon',
    #long_description='A longer description of your project',
    url='https://github.com/enkaoua/mono_reconstruction',  # URL to your project's repository

)
