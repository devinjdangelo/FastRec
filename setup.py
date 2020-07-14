from setuptools import setup

def readme():
	with open('README.md') as f:
		return f.read()

setup(name='fastrec',
	  version='0.0.2',
	  description='Rapidly deployed gnn based recommender',
	  long_description=readme(),
	  url='https://github.com/devinjdangelo/FastRec',
	  author='Devin DAngelo',
	  packages=['fastrec'],
	  scripts=['fastrec/fastrec-deploy'],
	  keywords='recommender graph neural network gnn deployment deploy',
	  include_package_data=True,
	  long_description_content_type="text/markdown",
	  zip_safe=False)