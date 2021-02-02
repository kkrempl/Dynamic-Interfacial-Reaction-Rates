#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 8

@author: Kevin Krempl
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import linregress
from scipy.integrate import cumtrapz

from ixdat.techniques.deconvolution import Kernel, DecoMeasurement
