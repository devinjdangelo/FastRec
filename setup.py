from setuptools import setup

def readme():
	with open('README.md') as f:
		return f.read()

setup(name='fastrec',
	  version='0.0.2post3',
	  description='Rapidly deployed gnn based recommender',
	  long_description=readme(),
	  url='https://github.com/devinjdangelo/FastRec',
	  author='Devin DAngelo',
	  packages=['fastrec'],
	  scripts=['fastrec/fastrec-deploy'],
	  install_requires=['torch==1.5.1','torchvision==0.6.1','dgl',
	  					'fastapi','uvicorn','tqdm','pandas'],
	  dependency_links=['https://download.pytorch.org/whl/torch_stable.html'],
	  keywords='recommender graph neural network gnn deployment deploy',
	  include_package_data=True,
	  long_description_content_type="text/markdown",
	  test_suite='nose.collector',
	  tests_require=['nose'],
	  zip_safe=False)