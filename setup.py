from setuptools import setup, find_packages

setup(name='embedding-cvx-projection',
      version='0.0.1',
      description='Embedding projection using CVXPY',
      author='Ivan Herreros',
      author_email='ivan.herreros@gmail.com',
      packages=find_packages(where='src'),
      package_dir={'': 'src'},
      install_requires=[
          'cvxpy'
        ])
