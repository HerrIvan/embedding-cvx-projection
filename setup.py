from setuptools import setup

setup(name='embedding-cvx-projection',
      version='0.0.1',
      description='Embedding projection using CVXPY',
      author='Ivan Herreros',
      author_email='ivan.herreros@gmail.com',
      package_dir={'embedding_cvx_projection': 'src'},
      install_requires=[
          'cvxpy'
        ])
