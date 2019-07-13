#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bz2
import csv
import logging
import re
from collections import defaultdict
from itertools import chain, product
from os import scandir
from pathlib import Path
from time import strftime

from configargparse import ArgumentParser

from SMACB.Guesser import (buildPosCupoIndex, getPlayersByPosAndCupoJornada,
                           indexGroup2Key, loadVar, varname2fichname)
from SMACB.SMconstants import CLAVESCSV
from SMACB.SuperManager import ResultadosJornadas, SuperManagerACB
from SMACB.TemporadaACB import TemporadaACB
from Utils.combinatorics import prod
from Utils.Misc import FORMATOtimestamp, creaPath

NJOBS = 4
LOCATIONCACHE = '/var/tmp/joblibCache'
LOCATIONCACHE = '/home/calba/devel/SuperManager/guesser'
indexGroups = [[0, 1, 2, 3], [4, 5], [6, 7, 8]]

indexes = buildPosCupoIndex()
# Planes con solucion usuario    Pabmm, J5

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(
    '%(asctime)s [%(process)d:%(threadName)10s@%(name)s %(levelname)s %(relativeCreated)14dms]: %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)


def procesaArgumentos():
    parser = ArgumentParser()

    parser.add('-i', dest='infile', type=str, env_var='SM_INFILE', required=True)
    parser.add('-t', dest='temporada', type=str, env_var='SM_TEMPORADA', required=True)
    parser.add('-j', dest='jornada', type=int, required=True)

    parser.add('-s', '--include-socio', dest='socioIn', type=str, action="append")
    parser.add('-e', '--exclude-socio', dest='socioOut', type=str, action="append")
    parser.add('-l', '--lista-socios', dest='listaSocios', action="store_true", default=False)

    parser.add("--data-dir", dest="datadir", type=str, default=LOCATIONCACHE)
    parser.add("--solution-file", dest="solfile", type=str)

    parser.add('-v', dest='verbose', action="count", env_var='SM_VERBOSE', required=False, default=0)
    parser.add('-d', dest='debug', action="store_true", env_var='SM_DEBUG', required=False, default=False)
    parser.add('--logdir', dest='logdir', type=str, env_var='SM_LOGDIR', required=False)

    parser.add('otherthings', nargs='*')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = procesaArgumentos()
    jornada = args.jornada
    destdir = args.datadir

    # Carga datos
    sm = SuperManagerACB()
    if 'infile' in args and args.infile:
        sm.loadData(args.infile)
        logger.info("Cargados datos SuperManager de %s" % strftime(FORMATOtimestamp, sm.timestamp))

    temporada = None
    resultadoTemporada = None
    if 'temporada' in args and args.temporada:
        temporada = TemporadaACB()
        temporada.cargaTemporada(args.temporada)
        resultadoTemporada = temporada.extraeDatosJugadores()
        logger.info("Cargada información de temporada de %s" % strftime(FORMATOtimestamp, temporada.timestamp))

    badTeams = args.socioOut if args.socioOut is not None else []
    resJornada = ResultadosJornadas(jornada, sm)
    goodTeams = args.socioIn if ('socioIn' in args and args.socioIn is not None) else resJornada.listaSocios()

    if args.listaSocios:
        for s in resJornada.socio2equipo:
            pref = "  "
            if s in goodTeams:
                pref = "SI"
            else:
                pref = "NO"

            if s in badTeams:
                pref = "NO"

            print("[%s] %-15s -> '%-28s': %s" % (pref, s, resJornada.socio2equipo[s], resJornada.resSocio2Str(s)))

        exit(0)

    sociosReales = [s for s in goodTeams if s in resJornada.socio2equipo and s not in badTeams]

    if not sociosReales:
        logger.error("No hay socios que procesar. Saliendo")
        exit(1)

    nombreFichGruposJugs = varname2fichname(jornada=jornada,
                                            varname="-".join([indexGroup2Key(indexGroups), "grupoJugs"]),
                                            basedir=destdir, ext="csv.bz2")

    posYcupos, jugadores, lenPosCupos = getPlayersByPosAndCupoJornada(jornada, sm, temporada)

    if 'solfile' in args and args.solfile:
        solLeidas = loadVar(args.solfile)
        if solLeidas is None:
            logger.info("Fichero suministrado no contiene soluciones: '%s'" % args.solfile)
        solList = [x for x in solLeidas if x[0] in sociosReales]
        if not solList:
            logger.info("Fichero suministrado '%s' no contiene soluciones para socios indicados: %s",
                        args.solfile, ", ".join(sociosReales))
            exit(1)
    else:
        solList = []
        solsPath = creaPath(destdir, "sols")
        for dirSol in scandir(solsPath):
            if not dirSol.is_dir():
                continue
            if dirSol.name in sociosReales:
                socio = dirSol.name
                socioCounter = 0
                socioList = []
                # "+".join([("J%03d" % plan['jornada']), plan['equipo'], planPart]) + ".pickle"
                patronFich = r"J%03d\+%s\+(\d_\d(-\d_\d)+).pickle" % (jornada, socio)

                for solFile in scandir(dirSol):
                    fileSol = solFile.name
                    if not solFile.is_file() or not re.match(patronFich, fileSol):
                        continue
                    solLeida = loadVar(Path(solFile.path))
                    if solLeida is None:
                        continue
                    socioCounter += 1
                    socioList.append(solLeida)

                logger.info("Encontrados %d ficheros de soluciones para socio %s (J:%d).", socioCounter, socio, jornada)
                socioFinal = list(chain.from_iterable(socioList))
                logger.info("Encontradas %d entradas de solución para socio %s (J:%d).", len(socioFinal), socio,
                            jornada)
                if socioFinal:
                    solList = solList + socioFinal

    if not solList:
        logger.info("No se han encontrado soluciones válidas")
        exit(1)

    gruposAencontrar = {'conts': defaultdict(int), 'entradas': defaultdict(list)}
    solucionesEncontradas = defaultdict(list)

    for sol in solList:
        socio, grupos, contador = sol
        for grupo in grupos:
            gruposAencontrar['conts'][grupo] += 1
        solucionesEncontradas[socio].append({'grupos': grupos, 'contador': contador})

    targetKeys = list(gruposAencontrar['conts'].keys())

    with bz2.open(filename=nombreFichGruposJugs, mode='rt') as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=CLAVESCSV, delimiter="|")
        headers = next(reader)
        for row in reader:
            if row['solkey'] not in targetKeys:
                continue
            gruposAencontrar['entradas'][row['solkey']].append(row['jugs'])

            # comb = row['grupo']

    for socio in solucionesEncontradas:
        solsSocio = []
        contSolsSocio = 1
        for i in range(len(solucionesEncontradas[socio])):
            combs = [gruposAencontrar['entradas'][x] for x in solucionesEncontradas[socio][i]['grupos']]
            prodLen = prod(map(len, combs))
            combProdSTR = " x ".join(map(str,sorted(combs)))
            print(socio,  contSolsSocio,i, solucionesEncontradas[socio][i]['contador'], prodLen,
                  sorted(solucionesEncontradas[socio][i]['grupos']),combProdSTR)
            for c in product(*combs):
                jugset = sorted(c[0].split("-") + c[1].split("-") + c[2].split("-"))
                solsSocio.append(jugset)
                solucionesEncontradas[socio][i]['jugset'] = jugset

                #print(contSolsSocio,i, socio, solucionesEncontradas[socio][i]['contador'], prodLen,                      sorted(solucionesEncontradas[socio][i]['grupos']), sorted(jugset))
                contSolsSocio += 1
        # print(socio, solsSocio)

    # solucionesEncontradas = {'Pabmm': [['105', '1L5', '1LU', '1N9', '52J', '57V', '586', 'A4B', 'BK4', 'FOX', 'Y3V'],
    #                                    ['105', '117', '1L5', '502', '519', '57F', 'BB2', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '502', '519', 'BB2', 'BBE', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '502', '519', 'BB2', 'BJP', 'FQ1', 'FQF', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '502', '519', 'B4P', 'BB2', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '502', '519', 'BB2', 'BGZ', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '502', '519', '54M', 'BB2', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '2B8', '502', '519', 'BB2', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '502', '519', '53A', 'BB2', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '502', '519', '577', 'BB2', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '502', '519', '51V', 'BB2', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '502', '519', '55I', 'BB2', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '502', '519', '8C4', 'BB2', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '502', '519', 'BB2', 'BJP', 'FQ1', 'UAD', 'XEU', 'Y7O'],
    #                                    ['105', '117', '1L5', '276', '502', '519', 'BB2', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '502', '519', '55Y', 'BB2', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1BG', '1L5', '502', '519', 'BB2', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '1PG', '502', '519', 'BB2', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '502', '519', 'BB2', 'BJP', 'FQ1', 'FRB', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '502', '519', 'BB2', 'BJP', 'FQ1', 'UAD', 'XDP', 'Y7O'],
    #                                    ['105', '117', '1GG', '1L5', '502', '519', 'BB2', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '502', '519', '57K', 'BB2', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '502', '519', '55B', 'BB2', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '502', '519', '526', 'BB2', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '502', '519', '52B', 'BB2', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['105', '117', '1L5', '502', '519', '581', 'BB2', 'BJP', 'FQ1', 'UAD', 'Y7O'],
    #                                    ['125', '1LU', '217', '519', '530', '87C', 'A8T', 'A9S', 'B8L', 'FOX', 'Y84'],
    #                                    ['117', '125', '164', '519', '580', '748', 'BB2', 'BIY', 'BK4', 'SCH', 'UAE'],
    #                                    ['117', '1L5', '502', '53J', '576', 'A8T', 'D04', 'D06', 'H05', 'SHG', 'Y3V'],
    #                                    ['117', '217', '218', '500', '52J', '53F', 'BIY', 'BLD', 'D06', 'T2Z', 'Y84'],
    #                                    ['117', '1LH', '55A', '56T', '57A', 'B86', 'BB2', 'BLP', 'FO1', 'FOX', 'FQH'],
    #                                    ['105', '1BA', '500', '81E', 'A1M', 'BB2', 'BIQ', 'D06', 'I3D', 'T78', 'Y84'],
    #                                    ['105', '117', '1N9', '500', '548', '748', 'A4B', 'B7G', 'BB2', 'BK4', 'UAE'],
    #                                    ['502', '542', '56L', 'A4B', 'BB2', 'BIY', 'BLP', 'FQ3', 'H05', 'H18', 'UAD']],
    #                          'mirza15': [['105', '117', '1L5', '217', '56T', '81E', 'A8T', 'BB2', 'BJP', 'D06','FQ1']]
    #                          }

    # print("-------------------++++++++++++++++++++++++-----------------------------")
    # print(solucionesEncontradas)
    # print("-------------------++++++++++++++++++++++++-----------------------------")

    for socio in solucionesEncontradas:
        jugEnSocio = defaultdict(int)
        print(socio, len(solucionesEncontradas[socio]))
        for dat in solucionesEncontradas[socio]:
            sol = dat['jugset']
            print(sol)
            precioAntes = sum([jugadores[j]['precioIni'] for j in sol])
            precioFin = sum([jugadores[j]['precioFin'] for j in sol])

            for j in sorted(sol, key=lambda x: (jugadores[x]['pos'], jugadores[x]['code'])):
                print(j, jugadores[j]['nombre'], jugadores[j].get('precioIni', "--------"),
                      jugadores[j].get('precioFin', "--------"))
                jugEnSocio[j] += 1

            print(sol, "%8i -> %8i" % (precioAntes, precioFin))
        for j in sorted(jugEnSocio, key=lambda x: (jugadores[x]['pos'], jugadores[x]['code'])):
            print(j, jugadores[j])

    # print(gruposAencontrar)
    # print(solucionesEncontradas)
