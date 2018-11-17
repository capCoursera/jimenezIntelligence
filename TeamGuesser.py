#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bz2
import csv
import logging
from collections import defaultdict
from itertools import chain, product
from sys import getsizeof
from os.path import join
from time import strftime, time

import joblib
from configargparse import ArgumentParser
from dask.distributed import Client, LocalCluster

from SMACB.Guesser import (GeneraCombinacionJugs, agregaJugadores,
                           buildPosCupoIndex, comb2Key, dumpVar,
                           getPlayersByPosAndCupoJornada, loadVar,
                           varname2fichname)
from SMACB.SMconstants import CUPOS, POSICIONES, SEQCLAVES, solucion2clave
from SMACB.SuperManager import ResultadosJornadas, SuperManagerACB
from SMACB.TemporadaACB import TemporadaACB
from Utils.CombinacionesConCupos import GeneraCombinaciones
from Utils.Misc import FORMATOtimestamp, deepDict, deepDictSet
from Utils.combinatorics import n_choose_m, prod

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
formatter = logging.Formatter('%(asctime)s [%(process)d:%(threadName)s@%(name)s %(levelname)s %(relativeCreated)14dms]: %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

indexGroups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
indexGroups = [[0, 1, 2], [3, 4], [5, 6], [7, 8]]
indexGroups = [[0, 1, 2, 3], [4, 5], [6, 7, 8]]

LOCATIONCACHE = '/home/calba/devel/SuperManager/guesser'

CLAVESCSV = ['solkey', 'grupo', 'jugs', 'valJornada', 'broker', 'puntos', 'rebotes', 'triples', 'asistencias', 'Nones']

clavesParaNomFich = "+".join(SEQCLAVES)

indexes = buildPosCupoIndex()


def procesaArgumentos():
    parser = ArgumentParser()

    parser.add('-i', dest='infile', type=str, env_var='SM_INFILE', required=True)
    parser.add('-t', dest='temporada', type=str, env_var='SM_TEMPORADA', required=True)
    parser.add('-j', dest='jornada', type=int, required=True)

    parser.add('-s', '--include-socio', dest='socioIn', type=str, action="append")
    parser.add('-e', '--exclude-socio', dest='socioOut', type=str, action="append")
    parser.add('-l', '--lista-socios', dest='listaSocios', action="store_true", default=False)

    parser.add('-b', '--backend', dest='backend', choices=BACKENDCHOICES, default='joblib')
    parser.add('-x', '--scheduler', dest='scheduler', type=str, default='127.0.0.1')
    parser.add("-o", "--output-dir", dest="outputdir", type=str, default=LOCATIONCACHE)
    parser.add('-p', '--package', dest='package', type=str, action="append")

    parser.add('--nproc', dest='nproc', type=int, default=NJOBS)
    parser.add('--memworker', dest='memworker', default=MEMWORKER)
    parser.add('--joblibmode', dest='joblibmode', choices=JOBLIBCHOICES, default='threads')

    parser.add('-v', dest='verbose', action="count", env_var='SM_VERBOSE', required=False, default=0)
    parser.add('-d', dest='debug', action="store_true", env_var='SM_DEBUG', required=False, default=False)
    parser.add('--logdir', dest='logdir', type=str, env_var='SM_LOGDIR', required=False)

    args = parser.parse_args()

    return args


def validateCombs(comb, grupos2check, val2match, equipo, seqnum, jornada):
    result = []

    claves = SEQCLAVES.copy()

    contExcl = {'in': 0, 'out': 0, 'cubos': 0, 'depth': dict()}
    for i in range(len(claves) + 1):
        contExcl['depth'][i] = 0

    combVals = [g['valSets'] for g in grupos2check]
    combInt = [g['comb'] for g in grupos2check]

    def ValidaCombinacion(arbolSols, claves, val2match, curSol, equipo, combInt):
        if len(claves) == 0:
            return

        contExcl['depth'][len(claves)] += 1
        contExcl['in'] += 1
        contExcl['cubos'] += prod([len(g) for g in grupos2check])

        claveAct = claves[0]

        for prodKey in product(*arbolSols):
            sumKey = sum(prodKey)

            if sumKey != val2match[claveAct]:
                contExcl['out'] += 1
                continue

            nuevosCombVals = [c[v] for c, v in zip(arbolSols, prodKey)]

            if len(claves) == 1:
                nuevaSol = curSol + [prodKey]
                solAcum = {k: sum(s) for k, s in zip(SEQCLAVES, nuevaSol)}
                for k in SEQCLAVES:
                    assert (solAcum[k] == val2match[k])

                valsSolD = [dict(zip(SEQCLAVES, s)) for s in list(zip(*nuevaSol))]
                solClaves = [solucion2clave(c, s) for c, s in zip(comb, valsSolD)]

                regSol = (equipo, solClaves, prod([x for x in nuevosCombVals]))
                result.append(regSol)
                # TODO: logging
                logger.info("%-16s J:%2d P:%3d Sol: %s", equipo, jornada, seqnum, regSol)
                continue
            else:
                deeperSol = curSol + [prodKey]
                deeper = ValidaCombinacion(nuevosCombVals, claves[1:], val2match, deeperSol, equipo, combInt)
                if deeper is None:
                    continue
        return None

    solBusq = ", ".join(["%s: %s" % (k, str(val2match[k])) for k in SEQCLAVES])
    numCombs = prod([g['numCombs'] for g in grupos2check])
    tamCubo = prod([len(g['valSets']) for g in grupos2check])
    contExcl['cuboIni'] = tamCubo
    FORMATOIN = "%-16s P:%3d J:%2d %20s IN  numEqs %16d cubo inicial: %10d Valores a buscar: %s"
    logger.info(FORMATOIN % (equipo, seqnum, jornada, combInt, numCombs, tamCubo, solBusq))
    timeIn = time()
    ValidaCombinacion(combVals, claves, val2match, [], equipo, combInt)
    timeOut = time()
    durac = timeOut - timeIn

    numEqs = sum([eq[-1] for eq in result])
    ops = contExcl['cubos']
    FORMATOOUT = "%-16s P:%3d J:%2d %20s OUT %3d %3d %10.3fs %10.8f%% %16d -> %12d %s"
    logger.info(FORMATOOUT % (equipo, seqnum, jornada, combInt, len(result), numEqs, durac,
                              (100.0 * float(ops) / float(numCombs)), numCombs, ops, contExcl))

    return result


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

    dh = logging.FileHandler(filename=join(destdir,"TeamGuesser.log"))
    dh.setFormatter(formatter)
    logger.addHandler(dh)

    if 'logdir' in args:
        fh = logging.FileHandler(filename=join(args.logdir,"J%03d.log" % jornada))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    configParallel = {'verbose': 100}
    # TODO: Control de calidad con los parámetros
    if args.backend == 'joblib':
        configParallel['n_jobs'] = args.nproc
        configParallel['prefer'] = args.joblibmode
        # configParallel['require'] = 'sharedmem'

    elif args.backend == 'dasklocal':
        configParallel['backend'] = "dask"
        cluster = LocalCluster(n_workers=args.nproc, threads_per_worker=1, memory_limit=args.memworker)
        client = Client(cluster)
    elif args.backend == 'daskremote':
        configParallel['backend'] = "dask"
        error = 0
        if 'scheduler' not in args:
            logger.error("Backend: %s. Falta scheduler '-x' o '--scheduler'.")
            error += 1
        if 'package' not in args:
            logger.error("Backend: %s. Falta package '-p' o '--package'.")
            error += 1
        if error:
            logger.error("Backend: %s. Hubo %d errores. Saliendo." % (args.backend, error))
            exit(1)

        client = Client('tcp://%s:8786' % args.scheduler)
        for egg in args.package:
            client.upload_file(egg)
        configParallel['scheduler_host'] = (args.scheduler, 8786)
    elif args.backend == 'daskyarn':
        configParallel['backend'] = "dask"
        error = 0
        if 'package' not in args:
            logger.error("Backend: %s. Falta package '-p' o '--package'.")
            error += 1
        if error:
            logger.error("Backend: %s. Hubo %d errores. Saliendo." % (args.backend, error))
            exit(1)
    else:
        pass

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

    # Recupera resultados de la jornada
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

            print("[%s] %s -> '%s'" % (pref, s, resJornada.socio2equipo[s]))

        exit(0)

    sociosReales = [s for s in goodTeams if s in resJornada.socio2equipo and s not in badTeams]

    if not sociosReales:
        logger.error("No hay socios que procesar. Saliendo")
        exit(1)

    jugadores = None

    validCombs = GeneraCombinaciones()

    groupedCombs, groupedCombsKeys = cuentaCombinaciones(validCombs, jornada)

    logger.info("Cargando grupos de jornada %d (secuencia: %s)" % (jornada, ", ".join(SEQCLAVES)))

    nombrefichCuentaGrupos = varname2fichname(jornada=jornada, varname=(clavesParaNomFich + "-cuentaGrupos"),
                                              basedir=destdir)
    cuentaGrupos = loadVar(nombrefichCuentaGrupos)

    if cuentaGrupos is None:
        logger.info("Generando grupos para jornada %d Seq claves %s" % (jornada, ", ".join(SEQCLAVES)))
        posYcupos, jugadores, lenPosCupos = getPlayersByPosAndCupoJornada(jornada, sm, temporada)

        dumpVar(varname2fichname(jornada, "jugadores", basedir=destdir), jugadores)

        # groupedCombs = []
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

        indexGroups = {p: [indexes[p][c] for c in CUPOS] for p in POSICIONES}

        # Distribuciones de jugadores válidas por posición y cupo
        for c in groupedCombs:

            for grupoComb in c:
                claveComb = comb2Key(grupoComb, jornada)
                if claveComb not in newCuentaGrupos:
                    numCombs = prod([numCombsPosYCupos[x][grupoComb[x]] for x in grupoComb])
                    newCuentaGrupos[claveComb] = {'cont': 0, 'comb': grupoComb, 'numCombs': numCombs, 'key': claveComb}
                newCuentaGrupos[claveComb]['cont'] += 1

        logger.info("Numero de grupos: %d", sum([newCuentaGrupos[x]['numCombs'] for x in newCuentaGrupos]))

        with bz2.open(filename=varname2fichname(jornada, varname=(clavesParaNomFich + "-grupos"), basedir=destdir,
                                                ext="csv.bz2"),
                      mode='wt') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CLAVESCSV, delimiter="|")
            for comb in newCuentaGrupos:
                combList = []

                combGroup = newCuentaGrupos[comb]['comb']

                timeIn = time()
                for i in combGroup:
                    n = combGroup[i]
                    # Genera combinaciones y las cachea
                    if combsPosYCupos[i][n] is None:
                        combsPosYCupos[i][n] = GeneraCombinacionJugs(posYcupos[i], n)
                    if n != 0:
                        combList.append(combsPosYCupos[i][n])

                colSets = dict()

                for pr in product(*combList):
                    aux = []
                    for gr in pr:
                        for j in gr:
                            aux.append(j)

                    agr = agregaJugadores(aux, jugadores)
                    claveJugs = "-".join(aux)
                    indexComb = [agr[k] for k in SEQCLAVES]

                    agr['solkey'] = solucion2clave(comb, agr)
                    agr['grupo'] = comb
                    agr['jugs'] = claveJugs
                    writer.writerow(agr)

                    deepDictSet(colSets, indexComb, deepDict(colSets, indexComb, int) + 1)

                timeOut = time()
                duracion = timeOut - timeIn

                newCuentaGrupos[comb]['valSets'] = colSets
                formatoTraza = "Gen grupos %-20s %10.3fs cont: %3d numero combs %8d memoria %8d num claves L0: %d"
                logger.info(formatoTraza, comb, duracion, newCuentaGrupos[comb]['cont'],
                            newCuentaGrupos[comb]['numCombs'], getsizeof(colSets), len(colSets))

        resDump = dumpVar(nombrefichCuentaGrupos, newCuentaGrupos)
        del newCuentaGrupos
        import gc

        gc.collect()
        cuentaGrupos = loadVar(nombrefichCuentaGrupos)

    logger.info("Cargados %d grupos de combinaciones. Memory: %d" % (len(cuentaGrupos), getsizeof(cuentaGrupos)))

    resultado = dict()

    planesAcorrer = []
    sociosReales.sort()
    for i, socio in product(range(len(groupedCombsKeys)), sociosReales):
        plan = groupedCombsKeys[i]
        planTotal = {'seqnum': i,
                     'comb': plan,
                     'grupos2check': [cuentaGrupos[grupo] for grupo in plan],
                     'val2match': resJornada.resultados[socio],
                     'equipo': socio,
                     'jornada': jornada}
        planesAcorrer.append(planTotal)

    logger.info("Planes para ejecutar: %d" % len(planesAcorrer))

    if args.backend == 'joblib':

        result = joblib.Parallel(**configParallel)(joblib.delayed(validateCombs)(**plan) for plan in planesAcorrer)

        resultadoPlano = list(chain.from_iterable(result))

    elif 'dask' in args.backend:

        with joblib.parallel_backend('dask'):
            result = joblib.Parallel(**configParallel)(joblib.delayed(validateCombs)(**plan) for plan in planesAcorrer)

        # result = Parallel(**configParallel)(delayed(validateCombs)(**plan) for plan in planesAcorrer)
        resultadoPlano = list(chain.from_iterable(result))

    else:
        pass

    dumpVar(varname2fichname(jornada, "%s-resultado-socios-%s" % (clavesParaNomFich, "-".join(sociosReales)),
                             basedir=destdir), resultadoPlano)

    logger.info(resultadoPlano)
    logger.info("Terminando ejecución")
