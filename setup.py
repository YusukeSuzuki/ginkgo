from setuptools import setup, find_packages

setup(name='ginkgo',
    version='0.1.0',
    desctiption='neural network shogi program implemented with TensorFlow',
    author='Yusuke Suzuki',
    license='Apache 2.0',
    packages = find_packages(),
    entry_points={
        'console_scripts':[
            'ginkgo_sfen_to_tfrecords = ginkgo.tools.convert_sfen_to_tfrecords:main',
            'ginkgo_train_prophet_with_records = ginkgo.applications.train_prophet_with_records:main',
        ]
    },
    install_requires=[
        'numpy',
        'tensorflow',
        'PyYAML',
        'python-shogi'
    ],
    zip_safe=False)


