#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bz2
import csv
import errno
import gc
import logging
from collections import defaultdict
from itertools import product
from os import makedirs
from os.path import isfile, join
from time import strftime, time

import joblib
from babel.numbers import parse_decimal
from configargparse import ArgumentParser

from SMACB.Guesser import (buildPosCupoIndex, comb2Key, dumpVar,
                           getPlayersByPosAndCupoJornada, indexGroup2Key,
                           loadVar, plan2filename, varname2fichname)
from SMACB.SMconstants import CLAVESCSV, SEQCLAVES
from SMACB.SuperManager import SuperManagerACB
from SMACB.TemporadaACB import TemporadaACB
from Utils.CombinacionesConCupos import GeneraCombinaciones
from Utils.combinatorics import n_choose_m, prod
from Utils.Misc import FORMATOtimestamp, creaPath
from Utils.pysize import get_size

NJOBS = 2
MEMWORKER = "2GB"
BACKENDCHOICES = ['joblib', 'dasklocal', 'daskyarn', 'daskremote']
JOBLIBCHOICES = ['threads', 'processes']

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

indexGroups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
indexGroups = [[0, 1, 2], [3, 4], [5, 6], [7, 8]]
indexGroups = [[0, 1, 2, 3], [4, 5], [6, 7, 8]]
# indexGroups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

LOCATIONCACHE = '/home/calba/devel/SuperManager/guesser'

clavesParaNomFich = "+".join(SEQCLAVES)

indexes = buildPosCupoIndex()


def procesaArgumentos():
    parser = ArgumentParser()

    parser.add('-i', dest='infile', type=str, env_var='SM_INFILE', required=True)
    parser.add('-t', dest='temporada', type=str, env_var='SM_TEMPORADA', required=True)
    parser.add('-j', dest='jornada', type=int, required=True)

    parser.add("-o", "--output-dir", dest="outputdir", type=str, default=LOCATIONCACHE)

    parser.add('--nproc', dest='nproc', type=int, default=NJOBS)
    parser.add('--memworker', dest='memworker', default=MEMWORKER)
    parser.add('--joblibmode', dest='joblibmode', choices=JOBLIBCHOICES, default='threads')

    parser.add('-v', dest='verbose', action="count", env_var='SM_VERBOSE', required=False, default=0)
    parser.add('-d', dest='debug', action="store_true", env_var='SM_DEBUG', required=False, default=False)
    parser.add('--logdir', dest='logdir', type=str, env_var='SM_LOGDIR', required=False)

    return parser.parse_args()


class CuboCounter(object):
    def __init__(self, ocurrencias=0, contador=0):
        self.ocurrencias = ocurrencias
        self.contador = contador

    def pon(self, c):
        self.ocurrencias = self.ocurrencias + c.ocurrencias
        self.contador = self.contador + c.contador

    def __repr__(self):
        return "(%d,%d)" % (self.ocurrencias, self.contador)


def calculaCuentaResultados(comb, grupos2check, seqnum, jornada, **kwargs):
    def calculaValores(valores):
        result = defaultdict(CuboCounter)
        cubo = 0
        contcombs = 0
        for comb in product(*valores):
            valor = sum([c[0] for c in comb])
            contador = sum([c[1] for c in comb])
            cubo += 1
            contcombs += contador
            result[valor].pon(CuboCounter(1, contador))

        return result, cubo, contcombs

    planIn = time()

    claves = SEQCLAVES

    result = {'seqnum': i, 'comb': plan, 'equipo': 'cuenta', 'jornada': jornada, 'valores': {}, 'cubos': {},
              'conts': {}}

    for c in claves:

        valoresAcontar = [list(g['valores'][c].items()) for g in grupos2check]
        cuboIni = prod([len(g) for g in valoresAcontar])
        FORMATOIN = "%3d J:%2d %20s clave: '%-15s' equipos: %16d -> cubo inicial %10d reducción: %10f%%"
        ratio = 100.0 * cuboIni / (kwargs['numEquipos'])

        logger.info(FORMATOIN % (seqnum, jornada, comb, c, kwargs['numEquipos'], cuboIni, ratio))
        print(FORMATOIN % (seqnum, jornada, comb, c, kwargs['numEquipos'], cuboIni, ratio))

        timeIn = time()
        if c == 'broker':
            FORMATOOUT = "%3d J:%2d %20s clave: '%-15s' IGNORADO."
            logger.info(FORMATOOUT % (seqnum, jornada, comb, c))
            continue

        result['valores'][c], result['cubos'][c], result['conts'][c] = calculaValores(valoresAcontar)

        timeOut = time()
        durac = timeOut - timeIn
        vel = cuboIni / durac

        FORMATOOUT = "%3d J:%2d %20s clave: '%-15s' duración: %9.3fs (%15.3f c/s) valores: %6d"
        logger.info(FORMATOOUT % (seqnum, jornada, comb, c, durac, vel, len(result['valores'][c])))
        print(FORMATOOUT % (seqnum, jornada, comb, c, durac, vel, len(result['valores'][c])))

    msgIn = "%3d J:%2d Grabando plan. Memory %12d" % (seqnum, jornada, get_size(result))
    grabIn = time()
    logger.info(msgIn)
    print(msgIn)

    dumpVar(kwargs['filename'], result, False)
    gc.collect()

    grabOut = time()
    grabDur = grabOut - grabIn
    planDur = grabOut - planIn
    msgOut = "%3d J:%2d Grabado plan. Plan: %9.3fs Grab: %9.3fs" % (seqnum, jornada, planDur, grabDur)
    logger.info(msgOut)
    print(msgOut)

    return None


def cuentaCombinaciones(combList, jornada):
    result = []
    resultKeys = []
    for c in combList:
        newComb = []
        newCombKey = []

        for ig in indexGroups:
            grupoComb = {x: c[x] for x in ig}
            if sum(grupoComb.values()) == 0:
                continue
            newComb.append(grupoComb)
            newCombKey.append(comb2Key(grupoComb, jornada))

        result.append(newComb)
        resultKeys.append(newCombKey)

    return result, resultKeys


if __name__ == '__main__':
    logger.info("Comenzando ejecución")

    args = procesaArgumentos()
    jornada = args.jornada
    destdir = args.outputdir

    dh = logging.FileHandler(filename=join(destdir, "TeamGuesser.log"))
    dh.setFormatter(formatter)
    logger.addHandler(dh)

    if 'logdir' in args:
        fh = logging.FileHandler(filename=join(args.logdir, "J%03d.log" % jornada))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

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

    validCombs = GeneraCombinaciones()
    groupedCombs, groupedCombsKeys = cuentaCombinaciones(validCombs, jornada)

    logger.info("Cargando grupos de jornada %d" % (jornada))

    nombreFichCuentaValoresGrupos = varname2fichname(jornada=jornada, varname="-".join([
        indexGroup2Key(indexGroups), "cuentaValGrupos"]), basedir=destdir)

    nombreFichCuentaValoresEquipos = varname2fichname(jornada=jornada, varname="-".join([
        indexGroup2Key(indexGroups), "cuentaSoluciones"]), basedir=destdir)

    nombreFichGruposJugs = varname2fichname(jornada=jornada,
                                            varname="-".join([indexGroup2Key(indexGroups), "grupoJugs"]),
                                            basedir=destdir, ext="csv.bz2")

    logger.info("[fichero grupo Jugadores: %s]" % (nombreFichGruposJugs))

    cuentaGrupos = loadVar(nombreFichCuentaValoresGrupos)

    if cuentaGrupos is None:

        logger.info("Generando grupos para jornada %d" % (jornada))
        posYcupos, jugadores, lenPosCupos = getPlayersByPosAndCupoJornada(jornada, sm, temporada)

        newCuentaGrupos = defaultdict(dict)
        maxPosCupos = [0] * 9
        numCombsPosYCupos = [[]] * 9
        combsPosYCupos = [[]] * 9

        for i in posYcupos:
            maxPosCupos[i] = max([x[i] for x in validCombs])
            numCombsPosYCupos[i] = [0] * (maxPosCupos[i] + 1)
            combsPosYCupos[i] = [None] * (maxPosCupos[i] + 1)

            for n in range(maxPosCupos[i] + 1):
                numCombsPosYCupos[i][n] = n_choose_m(lenPosCupos[i], n)

        # indexGroups = {p: [indexes[p][c] for c in CUPOS] for p in POSICIONES}
        # ([{0: 0, 1: 0, 2: 3, 3: 0}, {4: 4, 5: 0}, {6: 1, 7: 1, 8: 2}], ['J014-0_0-1_0-2_3-3_0',
        # 'J014-4_4-5_0', 'J014-6_1-7_1-8_2'])
        # Solucion conocida para J14/mirza15
        # groupedCombs = [[{0: 0, 1: 0, 2: 3, 3: 0}, {4: 4, 5: 0}, {6: 1, 7: 1, 8: 2}]]
        # groupedCombsKeys = [['J014-0_0-1_0-2_3-3_0', 'J014-4_4-5_0', 'J014-6_1-7_1-8_2']]

        # Distribuciones de jugadores válidas por posición y cupo
        for c in groupedCombs:
            for grupoComb in c:
                claveComb = comb2Key(grupoComb, jornada)
                if claveComb not in newCuentaGrupos:
                    numCombs = prod([numCombsPosYCupos[x][grupoComb[x]] for x in grupoComb])
                    newCuentaGrupos[claveComb] = {'cont': 0, 'comb': grupoComb, 'numCombs': numCombs, 'key': claveComb}
                    newCuentaGrupos[claveComb]['valores'] = dict()
                    newCuentaGrupos[claveComb]['cont'] += 1
                    for k in SEQCLAVES:
                        newCuentaGrupos[claveComb]['valores'][k] = defaultdict(int)

        logger.info("Numero de combinaciones: %d en %d grupos",
                    sum([newCuentaGrupos[x]['numCombs'] for x in newCuentaGrupos]), len(newCuentaGrupos))

        if isfile(nombreFichGruposJugs):
            logger.info("Procesando archivo de grupos de jugadores indexGroup(%s): %s", indexGroup2Key(indexGroups),
                        nombreFichGruposJugs)

            parseFunc = {k: (int if k != "valJornada" else parse_decimal) for k in SEQCLAVES}

            with bz2.open(filename=nombreFichGruposJugs, mode='rt') as csv_file:
                reader = csv.DictReader(csv_file, fieldnames=CLAVESCSV, delimiter="|")
                headers = next(reader)
                for row in reader:
                    comb = row['grupo']
                    for k in SEQCLAVES:
                        newVal = (parseFunc[k])(row.get(k, 0))
                        newCuentaGrupos[comb]['valores'][k][newVal] = newCuentaGrupos[comb]['valores'][k][newVal] + 1

        logger.info("Grabando archivo de valores encontrados para grupos indexGroup(%s): %s",
                    indexGroup2Key(indexGroups),
                    nombreFichCuentaValoresGrupos)

        resDump = dumpVar(nombreFichCuentaValoresGrupos, newCuentaGrupos)
        cuentaGrupos = newCuentaGrupos
        gc.collect()
        logger.info("Grabado archivo de valores encontrados para grupos indexGroup(%s): %s",
                    indexGroup2Key(indexGroups),
                    nombreFichCuentaValoresGrupos)

    logger.info("%d grupos de valores para grupos indexGroup(%s): %d", len(indexGroups), indexGroup2Key(indexGroups),
                get_size(cuentaGrupos))

    dirCuentas = creaPath(destdir, "cuentas")
    try:
        makedirs(dirCuentas)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    planesAcorrer = []
    cuentasConocidas = []

    for i in range(len(groupedCombsKeys)):
        plan = groupedCombsKeys[i]

        planTotal = {'seqnum': i,
                     'comb': plan,
                     'equipo': 'cuenta',
                     'numEquipos': prod([cuentaGrupos[g]['numCombs'] for g in plan]),
                     'grupos2check': [cuentaGrupos[grupo] for grupo in plan],
                     'jornada': jornada}
        planTotal['filename'] = creaPath(dirCuentas, plan2filename(planTotal))

        sol = loadVar(planTotal['filename'])

        if sol is None:
            planesAcorrer.append(planTotal)
        else:
            cuentasConocidas.append(planTotal)
        # print(planTotal)

    if cuentasConocidas:
        logger.info("Encontradas cuentas para %d planes" % len(cuentasConocidas))

    logger.info("Cuentas para ejecutar: %d" % len(planesAcorrer))

    configParallel = {'verbose': 100}
    configParallel['n_jobs'] = args.nproc
    configParallel['prefer'] = args.joblibmode
    configParallel['max_nbytes'] = None

    result = joblib.Parallel(**configParallel)(
        joblib.delayed(calculaCuentaResultados)(**plan) for plan in planesAcorrer)

    logger.info("Terminando ejecución")
