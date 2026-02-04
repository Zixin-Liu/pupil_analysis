# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 11:03:43 2026

@author: bbf2518
"""

from tobii_preprocess.step_1_cleanup import remove_empty_columns_tobii
from tobii_preprocess.step_1_cleanup import extract_relevant_rows_1
from tobii_preprocess.step_1_cleanup import extract_relevant_rows_2

remove_empty_columns_tobii("raw_tsv/confetti_child commonConfetti_62002_et_1.tsv", "preprocessed")
extract_relevant_rows_1("raw_tsv/62002_step_1.tsv")
extract_relevant_rows_2("raw_tsv/62002_step_2_1.tsv")
