from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='alphadog',
   version='0.01',
   description='Alphadog systematic trading',
   long_description=long_description,
   author='Gurpreet Johl',
   author_email='gurpreetjohl@gmail.com',
   packages=[],
   install_requires=['numpy', 'pandas', 'pytest', 'scipy'],  # external packages as dependencies
   scripts=[]
)
