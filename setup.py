from distutils.core import setup

setup(
    name='slim',
    version='0.95a',

    packages=[
        'slim',
    ],

    package_data={
        'slim': ['defaults.cfg'],
    },

    install_requires=[
        'ctree',
    ]
)

