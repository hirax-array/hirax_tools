#!/usr/bin/env python

from setuptools import setup

setup(name='hirax_tools',
      version='0.1',
      description='HIRAX Quick Analysis Tools',
      author='Devin Crichton',
      author_email='devin.crichton@gmail.com',
      packages=['hirax_tools',],
      scripts=['hirax_tools/scripts/ht_transpose_vis',
               'hirax_tools/scripts/ht_waterfall_plot']
     )
