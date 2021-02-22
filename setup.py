from distutils.core import setup

setup(name='apricotexp',
      version='0.0.1',
      description="Example package",
      url='',
      author='Dávid Bartalis',
      author_email='bartalisdavid.98@gmail.com',
      packages=['apricot_exp'],
      install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'seaborn',
          'comet-ml',
          'apricot-select',
          'scikit-learn',
          'scipy',
          'pytest'
      ],
      zip_safe=False
)
