# -*- coding: utf-8 -*-
import pysipfenn
from pysipfenn import Calculator     # The only thing needed for calculations
from pprint import pprint            # pretty printing
from collections import defaultdict  # convenience in the example
c = Calculator()
c.downloadModels(network='SIPFENN_Krajewski2020_NN24')
c.loadModels()
testFilesDir = "D:/SIPFENN_V0.9_Complete/SIPFENN_alpha/exampleInputFiles/"
c.runFromDirectory(directory=testFilesDir,descriptor='KS2022',mode='serial');