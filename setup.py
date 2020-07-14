from setuptools import setup

setup(name='fastrec',
	  version='0.0.1',
	  description='Rapidly deployed gnn based recommender',
	  url='https://github.com/devinjdangelo/FastRec',
	  author='Devin DAngelo',
	  packages=['fastrec'],
	  scripts=['fastrec/fastrec-deploy'],
	  zip_safe=False)