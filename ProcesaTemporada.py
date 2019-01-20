#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
from statistics import mean, median, stdev
from time import gmtime, mktime, strftime, time

from configargparse import ArgumentParser
from pandas import DataFrame, ExcelWriter

from SMACB.ManageSMDataframes import (CATMERCADOFINAL, COLSPREC,
                                      calculaDFcategACB, calculaDFconVars,
                                      calculaDFprecedentes)
from SMACB.PartidoACB import PartidoACB
from SMACB.SMconstants import MINPRECIO, POSICIONES, PRECIOpunto
from SMACB.SuperManager import SuperManagerACB
from SMACB.TemporadaACB import TemporadaACB, calculaVars, calculaZ
from Utils.Misc import FORMATOtimestamp, SubSet

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add('-v', dest='verbose', action="count", env_var='SM_VERBOSE', required=False, default=0)
    parser.add('-d', dest='debug', action="store_true", env_var='SM_DEBUG', required=False, default=False)

    parser.add('-t', dest='temporada', type=str, env_var='SM_TEMPORADA', required=True)

    args = parser.parse_args()

    sm = SuperManagerACB()

    sm.loadData(args.infile)
    print("Cargados datos SuperManager de %s" % strftime(FORMATOtimestamp, sm.timestamp))
