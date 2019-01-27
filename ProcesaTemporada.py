#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from time import strftime

from configargparse import ArgumentParser

from SMACB.SuperManager import SuperManagerACB
from SMACB.TemporadaACB import TemporadaACB
from Utils.Misc import FORMATOtimestamp

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add('-v', dest='verbose', action="count", env_var='SM_VERBOSE', required=False, default=0)
    parser.add('-d', dest='debug', action="store_true", env_var='SM_DEBUG', required=False, default=False)

    parser.add('-i', dest='infile', type=str, env_var='SM_INFILE', required=True)
    parser.add('-t', dest='temporada', type=str, env_var='SM_TEMPORADA', required=True)

    args = parser.parse_args()

    sm = SuperManagerACB()
    sm.loadData(args.infile)
    print("Cargados datos SuperManager de %s" % strftime(FORMATOtimestamp, sm.timestamp))

    temporada = TemporadaACB()
    temporada.cargaTemporada(args.temporada)
    resultadoTemporada = temporada.extraeDatosJugadores()
    print("Cargada información de temporada de %s" % strftime(FORMATOtimestamp, temporada.timestamp))
