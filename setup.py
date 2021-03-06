from setuptools import setup

setup(name='tflab',
      version='0.1.3',
      description='A laboratory for experimenting with Tensorflow abstraction',
      url='https://github.com/mhamilton723/tflab',
      author='Mark Hamilton',
      author_email='mhamilton723@gmail.com',
      license=None,
      packages=['tflab'],
      zip_safe=False,
      install_requires=['tensorflow', 'numpy'],
      scripts=['bin/run_word2vec.py','bin/run_mmd.py','bin/run_network.py'],)
