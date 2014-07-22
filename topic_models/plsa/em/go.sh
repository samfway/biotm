#!/bin/bash

rm -f plsa_em.c
rm -f plsa_em.so
rm -rf build/

python setup.py build_ext --inplace

./tests/test_em_c.py
