from setuptools import setup

setup(name='ginkgo',
    version='0.1.0',
    desctiption='neural network shogi program implemented with TensorFlow',
    author='Yusuke Suzuki',
    license='Apache 2.0',
    packages=['ginkgo', 'ginkgo.tools', ],
    entry_points={
        'console_scripts':[
            'ginkgo_sfen_to_tfrecords = ginkgo.tools.convert_sfen_to_tfrecords:main'
        ]
    },
    install_requires=[
        'numpy',
        'tensorflow',
        'PyYAML',
        'python-shogi'
    ],
    zip_safe=False)


