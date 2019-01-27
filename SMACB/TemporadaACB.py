'''
Created on Jan 4, 2018

@author: calba
'''

from calendar import timegm
from collections import defaultdict
from copy import copy, deepcopy
from itertools import chain
from pickle import dump, load
from statistics import mean, median, stdev
from sys import setrecursionlimit
from time import gmtime, strftime

import numpy as np
import pandas as pd
from babel.numbers import decimal

from SMACB.CalendarioACB import CalendarioACB, calendario_URLBASE
from SMACB.PartidoACB import PartidoACB
from SMACB.SMconstants import (LISTACOMPOS, PARTIDOSTENDENCIA,
                               calculaValSuperManager)
from Utils.Misc import FORMATOfecha, FORMATOtimestamp, Seg2Tiempo
from Utils.Pandas import combinaPDindexes


class TemporadaACB(object):
    '''
    Aglutina calendario y lista de partidos
    '''

    def __init__(self, competicion="LACB", edicion=None, urlbase=calendario_URLBASE):
        self.timestamp = gmtime()
        self.Calendario = CalendarioACB(competicion=competicion, edicion=edicion, urlbase=urlbase)
        self.PartidosDescargados = set()
        self.Partidos = dict()
        self.changed = False
        self.translations = defaultdict(set)

    def actualizaTemporada(self, home=None, browser=None, config={}):
        self.Calendario.bajaCalendario(browser=browser, config=config)

        partidosBajados = set()

        for partido in self.Calendario.Partidos:
            if partido in self.PartidosDescargados:
                continue

            nuevoPartido = PartidoACB(**(self.Calendario.Partidos[partido]))
            nuevoPartido.descargaPartido(home=home, browser=browser, config=config)

            self.PartidosDescargados.add(partido)
            self.Partidos[partido] = nuevoPartido
            self.actualizaNombresEquipo(nuevoPartido)
            partidosBajados.add(partido)

            if config.justone:  # Just downloads a game (for testing/dev purposes)
                break

        if partidosBajados:
            self.changed = True
            self.timestamp = gmtime()

        return partidosBajados

    def actualizaNombresEquipo(self, partido):
        for loc in partido.Equipos:
            nombrePartido = partido.Equipos[loc]['Nombre']
            codigoParam = partido.CodigosCalendario[loc]
            if self.Calendario.nuevaTraduccionEquipo2Codigo(nombrePartido, codigoParam):
                self.changed = True

    def grabaTemporada(self, filename):
        aux = copy(self)

        # Clean stuff that shouldn't be saved
        for atributo in ('changed'):
            if hasattr(aux, atributo):
                aux.__delattr__(atributo)

        setrecursionlimit(50000)
        # TODO: Protect this
        dump(aux, open(filename, "wb"))

    def cargaTemporada(self, filename):
        # TODO: Protect this
        aux = load(open(filename, "rb"))

        for atributo in aux.__dict__.keys():
            if atributo in ('changed'):
                continue
            self.__setattr__(atributo, aux.__getattribute__(atributo))

    def nuevaTraduccionJugador(self, codigo, nombre):
        if (codigo not in self.translations) or (nombre not in self.translations[codigo]):
            self.changed = True

        (self.translations[codigo]).add(nombre)

    def listaJugadores(self, jornada=0, jornadaMax=0, fechaMax=None):

        def SacaJugadoresPartido(partido):
            for codigo in partido.Jugadores:
                (resultado['codigo2nombre'][codigo]).add(partido.Jugadores[codigo]['nombre'])
                resultado['nombre2codigo'][partido.Jugadores[codigo]['nombre']] = codigo

        resultado = {'codigo2nombre': defaultdict(set), 'nombre2codigo': dict()}

        for partido in self.Partidos:
            aceptaPartido = False
            if jornada and self.Partidos[partido].Jornada == jornada:
                aceptaPartido = True
            elif jornadaMax and self.Partidos[partido].Jornada >= jornadaMax:
                aceptaPartido = True
            elif fechaMax and self.Partidos[partido].FechaHora < fechaMax:
                aceptaPartido = True
            else:
                aceptaPartido = True

            if aceptaPartido:
                SacaJugadoresPartido(self.Partidos[partido])

        for codigo in self.translations:
            for trad in self.translations[codigo]:
                (resultado['codigo2nombre'][codigo]).add(trad)
                resultado['nombre2codigo'][trad] = codigo

        return resultado

    def resumen(self):
        print(self.__dict__.keys())
        print("Temporada. Timestamp %s" % strftime(FORMATOtimestamp, self.timestamp))
        print("Temporada. Cambios %s" % self.changed)
        print(self.Calendario.__dict__.keys())
        print("Temporada. Partidos cargados: %i,%i" % (len(self.Partidos), len(self.PartidosDescargados)))
        for partidoID in self.Partidos:
            partido = self.Partidos[partidoID]
            resumenPartido = " * %s: %s (%s) %i - %i %s (%s) " % (partidoID, partido.EquiposCalendario['Local'],
                                                                  partido.CodigosCalendario['Local'],
                                                                  partido.ResultadoCalendario['Local'],
                                                                  partido.ResultadoCalendario['Visitante'],
                                                                  partido.EquiposCalendario['Visitante'],
                                                                  partido.CodigosCalendario['Visitante'])

            print(resumenPartido)

    def maxJornada(self):
        acums = defaultdict(int)
        for claveP in self.Partidos:
            partido = self.Partidos[claveP]
            acums[partido.Jornada] += 1

        return max(acums.keys())

    def extraeDatosJugadores(self):
        resultado = dict()

        maxJ = self.maxJornada()

        def listaDatos():
            return [None] * maxJ

        clavePartido = ['FechaHora', 'URL', 'Partido', 'ResumenPartido', 'Jornada']
        claveJugador = ['esLocal', 'titular', 'nombre', 'haGanado', 'haJugado', 'equipo', 'CODequipo', 'rival',
                        'CODrival']
        claveEstad = ['Segs', 'P', 'T2-C', 'T2-I', 'T2%', 'T3-C', 'T3-I', 'T3%', 'T1-C', 'T1-I', 'T1%', 'REB-T',
                      'R-D', 'R-O', 'A', 'BR', 'BP', 'C', 'TAP-F', 'TAP-C', 'M', 'FP-F', 'FP-C', '+/-', 'V']
        claveDict = ['OrdenPartidos']
        claveDictInt = ['I-convocado', 'I-jugado']

        for clave in clavePartido + claveJugador + claveEstad:
            resultado[clave] = defaultdict(listaDatos)
        for clave in claveDict:
            resultado[clave] = dict()
        for clave in claveDictInt:
            resultado[clave] = defaultdict(int)

        for claveP in self.Partidos:
            partido = self.Partidos[claveP]
            jornada = partido.Jornada - 1  # Indice en el hash
            fechahora = partido.FechaHora
            segsPartido = partido.Equipos['Local']['estads']['Segs']

            resultadoPartido = "%i-%i" % (partido.DatosSuministrados['resultado'][0],
                                          partido.DatosSuministrados['resultado'][1])

            if partido.prorrogas:
                resultadoPartido += " %iPr" % partido.prorrogas

            for claveJ in partido.Jugadores:
                jugador = partido.Jugadores[claveJ]

                resultado['FechaHora'][claveJ][jornada] = fechahora
                resultado['Jornada'][claveJ][jornada] = partido.Jornada
                resultado['URL'][claveJ][jornada] = claveP
                nomPartido = ("" if jugador['esLocal'] else "@") + jugador['rival']
                resultado['Partido'][claveJ][jornada] = nomPartido

                for subClave in claveJugador:
                    resultado[subClave][claveJ][jornada] = jugador[subClave]

                for subClave in claveEstad:
                    if subClave in jugador['estads']:
                        resultado[subClave][claveJ][jornada] = jugador['estads'][subClave]

                textoResumen = "%s %s\n%s: %s\n%s\n\n" % (nomPartido,
                                                          ("(V)" if jugador['haGanado'] else "(D)"),
                                                          self.Calendario.nombresJornada()[jornada],
                                                          strftime(FORMATOfecha, fechahora),
                                                          resultadoPartido)

                if jugador['haJugado']:
                    estads = jugador['estads']

                    textoResumen += "Min: %s (%.2f%%)\n" % (Seg2Tiempo(estads['Segs']),
                                                            100.0 * estads['Segs'] / segsPartido)
                    textoResumen += "Val: %i\n" % estads['V']
                    textoResumen += "P: %i\n" % estads['P']
                    t2c = estads['T2-C']
                    t2i = estads['T2-I']
                    if t2i:
                        textoResumen += "T2: %i/%i (%.2f%%)\n" % (t2c, t2i, estads['T2%'])
                    else:
                        textoResumen += "T2: 0/0 (0.00%)\n"
                    t3c = estads['T3-C']
                    t3i = estads['T3-I']
                    if t3i:
                        textoResumen += "T3: %i/%i (%.2f%%)\n" % (t3c, t3i, estads['T3%'])
                    else:
                        textoResumen += "T3: 0/0 (0.00%)\n"

                    if t2i + t3i:
                        textoResumen += "TC: %i/%i (%.2f%%)\n" % (t2c + t3c, t2i + t3i,
                                                                  100 * (t2c + t3c) / (t2i + t3i))
                    else:
                        textoResumen += "TC: 0/0 (0.00%)\n"

                    textoResumen += "TL: %i/%i (%.2f%%)\n" % (estads['T1-C'], estads['T1-I'], estads['T1%'])
                    textoResumen += "R: %i+%i %i\n" % (estads['R-D'], estads['R-O'], estads['REB-T'])
                    textoResumen += "A: %i\n" % estads['A']
                    textoResumen += "BR: %i\n" % estads['BR']
                    textoResumen += "BP: %i\n" % estads['BP']
                    textoResumen += "Tap: %i\n" % estads['TAP-F']
                    textoResumen += "Tap Rec: %i\n" % estads['TAP-C']
                    textoResumen += "Fal: %i\n" % estads['FP-C']
                    textoResumen += "Fal Rec: %i\n" % estads['FP-F']
                else:
                    textoResumen += "No ha jugado"

                resultado['ResumenPartido'][claveJ][jornada] = textoResumen

        # Calcula el orden de las jornadas para mostrar los partidos jugados en orden cronológico
        for claveJ in resultado['FechaHora']:
            auxFH = [((timegm(resultado['FechaHora'][claveJ][x]) if resultado['FechaHora'][claveJ][x] else 0), x)
                     for x in range(len(resultado['FechaHora'][claveJ]))]
            auxFHsorted = [x[1] for x in sorted(auxFH, key=lambda x: x[0])]
            resultado['OrdenPartidos'][claveJ] = auxFHsorted

        for claveJ in resultado['haJugado']:
            convocados = [x for x in resultado['haJugado'][claveJ] if x is not None]
            jugados = sum([1 for x in convocados if x])
            resultado['I-convocado'][claveJ] = len(convocados)
            resultado['I-jugado'][claveJ] = jugados

        return resultado

    def extraeDataframeJugadores(self):

        def jorFech2periodo(dfTemp):
            periodoAct = 0
            jornada = dict()
            claveMin = dict()
            claveMax = dict()
            curVal = None
            jf2periodo = defaultdict(lambda: defaultdict(int))

            dfPairs = dfTemp.apply(lambda r: (r['Fecha'].date(), r['jornada']), axis=1).unique()
            for p in sorted(list(dfPairs)):
                if curVal is None or curVal[1] != p[1]:
                    if curVal:
                        periodoAct += 1

                    curVal = p
                    jornada[periodoAct] = p[1]
                    claveMin[periodoAct] = p[0]
                    claveMax[periodoAct] = p[0]

                else:
                    claveMax[periodoAct] = p[0]
                jf2periodo[p[1]][p[0]] = periodoAct

            p2k = {p: (("%s" % claveMin[p]) + (("\na %s" % claveMax[p]) if (claveMin[p] != claveMax[p]) else "") + (
                    "\n(J:%2i)" % jornada[p])) for p in jornada}

            result = dict()
            for j in jf2periodo:
                result[j] = dict()
                for d in jf2periodo[j]:
                    result[j][d] = p2k[jf2periodo[j][d]]

            return result

        dfPartidos = [partido.jugadoresAdataframe() for partido in self.Partidos.values()]
        dfResult = pd.concat(dfPartidos, axis=0, ignore_index=True, sort=True)

        periodos = jorFech2periodo(dfResult)

        dfResult['periodo'] = dfResult.apply(lambda r: periodos[r['jornada']][r['Fecha'].date()], axis=1)

        return (dfResult)

    def extraeDataframePartidos(self):
        # TODO: ¿Para qu� hice esto? Puede ser �til para el dossier
        """
        Devuelve un dataframe con una fila para cada jugador y columnas con las listas con las participaciones de los
        jugadores en todos los partidos.
        :return:
        """
        resultado = dict()

        maxJ = self.maxJornada()

        def listaDatos():
            return [None] * maxJ

        clavePartido = ['FechaHora', 'URL', 'Partido', 'ResumenPartido', 'Jornada']
        claveJugador = ['esLocal', 'titular', 'nombre', 'haGanado', 'haJugado', 'equipo', 'CODequipo', 'rival',
                        'CODrival']
        claveEstad = ['Segs', 'P', 'T2-C', 'T2-I', 'T2%', 'T3-C', 'T3-I', 'T3%', 'T1-C', 'T1-I', 'T1%', 'REB-T',
                      'R-D', 'R-O', 'A', 'BR', 'BP', 'C', 'TAP-F', 'TAP-C', 'M', 'FP-F', 'FP-C', '+/-', 'V']
        claveDict = ['OrdenPartidos']
        claveDictInt = ['I-convocado', 'I-jugado']

        for clave in clavePartido + claveJugador + claveEstad:
            resultado[clave] = defaultdict(listaDatos)
        for clave in claveDict:
            resultado[clave] = dict()
        for clave in claveDictInt:
            resultado[clave] = defaultdict(int)

        for claveP in self.Partidos:
            partido = self.Partidos[claveP]
            jornada = partido.Jornada - 1  # Indice en el hash
            fechahora = partido.FechaHora
            segsPartido = partido.Equipos['Local']['estads']['Segs']

            resultadoPartido = "%i-%i" % (partido.DatosSuministrados['resultado'][0],
                                          partido.DatosSuministrados['resultado'][1])

            if partido.prorrogas:
                resultadoPartido += " %iPr" % partido.prorrogas

            for claveJ in partido.Jugadores:
                jugador = partido.Jugadores[claveJ]

                resultado['FechaHora'][claveJ][jornada] = fechahora
                resultado['Jornada'][claveJ][jornada] = partido.Jornada
                resultado['URL'][claveJ][jornada] = claveP
                nomPartido = ("" if jugador['esLocal'] else "@") + jugador['rival']
                resultado['Partido'][claveJ][jornada] = nomPartido

                for subClave in claveJugador:
                    resultado[subClave][claveJ][jornada] = jugador[subClave]

                for subClave in claveEstad:
                    if subClave in jugador['estads']:
                        resultado[subClave][claveJ][jornada] = jugador['estads'][subClave]

                textoResumen = "%s %s\n%s: %s\n%s\n\n" % (nomPartido,
                                                          ("(V)" if jugador['haGanado'] else "(D)"),
                                                          self.Calendario.nombresJornada()[jornada],
                                                          strftime(FORMATOfecha, fechahora),
                                                          resultadoPartido)

                if jugador['haJugado']:
                    estads = jugador['estads']

                    textoResumen += "Min: %s (%.2f%%)\n" % (Seg2Tiempo(estads['Segs']),
                                                            100.0 * estads['Segs'] / segsPartido)
                    textoResumen += "Val: %i\n" % estads['V']
                    textoResumen += "P: %i\n" % estads['P']
                    t2c = estads['T2-C']
                    t2i = estads['T2-I']
                    if t2i:
                        textoResumen += "T2: %i/%i (%.2f%%)\n" % (t2c, t2i, estads['T2%'])
                    else:
                        textoResumen += "T2: 0/0 (0.00%)\n"
                    t3c = estads['T3-C']
                    t3i = estads['T3-I']
                    if t3i:
                        textoResumen += "T3: %i/%i (%.2f%%)\n" % (t3c, t3i, estads['T3%'])
                    else:
                        textoResumen += "T3: 0/0 (0.00%)\n"

                    if t2i + t3i:
                        textoResumen += "TC: %i/%i (%.2f%%)\n" % (t2c + t3c, t2i + t3i,
                                                                  100 * (t2c + t3c) / (t2i + t3i))
                    else:
                        textoResumen += "TC: 0/0 (0.00%)\n"

                    textoResumen += "TL: %i/%i (%.2f%%)\n" % (estads['T1-C'], estads['T1-I'], estads['T1%'])
                    textoResumen += "R: %i+%i %i\n" % (estads['R-D'], estads['R-O'], estads['REB-T'])
                    textoResumen += "A: %i\n" % estads['A']
                    textoResumen += "BR: %i\n" % estads['BR']
                    textoResumen += "BP: %i\n" % estads['BP']
                    textoResumen += "Tap: %i\n" % estads['TAP-F']
                    textoResumen += "Tap Rec: %i\n" % estads['TAP-C']
                    textoResumen += "Fal: %i\n" % estads['FP-C']
                    textoResumen += "Fal Rec: %i\n" % estads['FP-F']
                else:
                    textoResumen += "No ha jugado"

                resultado['ResumenPartido'][claveJ][jornada] = textoResumen

        # Calcula el orden de las jornadas para mostrar los partidos jugados en orden cronológico
        for claveJ in resultado['FechaHora']:
            auxFH = [((timegm(resultado['FechaHora'][claveJ][x]) if resultado['FechaHora'][claveJ][x] else 0), x)
                     for x in range(len(resultado['FechaHora'][claveJ]))]
            auxFHsorted = [x[1] for x in sorted(auxFH, key=lambda x: x[0])]
            resultado['OrdenPartidos'][claveJ] = auxFHsorted

        for claveJ in resultado['haJugado']:
            convocados = [x for x in resultado['haJugado'][claveJ] if x is not None]
            jugados = sum([1 for x in convocados if x])
            resultado['I-convocado'][claveJ] = len(convocados)
            resultado['I-jugado'][claveJ] = jugados

        return resultado

    def extraeDatosJornadaSM(self, jornada):
        result = dict()

        if jornada in self.Calendario.Jornadas:
            for linkPartido in self.Calendario.Jornadas[jornada]['partidos']:
                partido = self.Partidos[linkPartido]
                for jug in partido.Jugadores:
                    jugData = partido.Jugadores[jug]
                    if not jugData['esJugador']:
                        continue

                    aux = dict()

                    for c in LISTACOMPOS:
                        aux[c] = jugData['estads'].get(LISTACOMPOS[c], decimal.Decimal(0))

                    valP = jugData['estads'].get('V', decimal.Decimal(0))
                    # * (BONUSVICTORIA if (jugData['haGanado'] and (valP > 0)) else 1.0))
                    aux['valFromP'] = calculaValSuperManager(valP, jugData['haGanado'])

                    result[jug] = aux

        return result

    def extraePartidosPorEquipo(self):
        result = defaultdict(list)
        for e in chain.from_iterable([self.Partidos[p].estadsPartido() for p in self.Partidos]):
            result[e['codigo']].append(e)

        return result

    def obtenClasificacion(self):
        INTKEYS = ['Segs', 'P', 'T2-C', 'T2-I', 'T3-C', 'T3-I', 'T1-C', 'T1-I', 'REB-T', 'R-D', 'R-O',
                   'A', 'BR', 'BP', 'TAP-F', 'TAP-C', 'FP-F', 'FP-C', 'V', 'TC-I', 'TC-C', 'Prec']
        FLOKEYS = ['T2%', 'T3%', 'T1%', 'defAro', 'TC%', 'P2%', 'P3%', 'ataAro', 'Poss', 'OER', 'OEReff']
        entrada = {'partidos': {'J': 0, 'V': 0, 'D': 0, 'ratio': 0.0, 'F': 0, 'C': 0, 'dif': 0, 'prorrogas': 0},
                   'means': dict(),
                   'stds': dict(), 'medians': dict(), 'merged': dict()}
        el2cf = {True: 'casa', False: 'fuera'}
        hg2vd = {True: 'V', False: 'D'}

        result = dict()

        for e, ps in self.extraePartidosPorEquipo().items():
            result[e] = dict()
            result[e]['Nombre'] = ""
            result[e]['total'] = deepcopy(entrada)
            result[e]['casa'] = deepcopy(entrada)
            result[e]['fuera'] = deepcopy(entrada)
            result[e]['total']['partidos']['ultParts'] = tendenciaEquipo(ps, PARTIDOSTENDENCIA)

            for p in ps:
                locs = ('total', el2cf[p['esLocal']])

                for loc in locs:
                    result[e]['Nombre'] = p['nombre']
                    result[e][loc]['partidos']['J'] += 1
                    result[e][loc]['partidos'][hg2vd[p['haGanado']]] += 1
                    result[e][loc]['partidos']['ratio'] = 100.0 * result[e][loc]['partidos']['V'] / \
                                                          result[e][loc]['partidos']['J']
                    result[e][loc]['partidos']['F'] += p['yo-estads']['P']
                    result[e][loc]['partidos']['C'] += p['otro-estads']['P']
                    result[e][loc]['partidos']['dif'] = (
                            result[e][loc]['partidos']['F'] - result[e][loc]['partidos']['C'])
                    result[e][loc]['partidos']['prorrogas'] += p['prorrogas']

            for k in INTKEYS + FLOKEYS:
                obs = [p['yo-estads'][k] for p in ps]

                result[e]['total']['means'][k] = mean(obs)
                result[e]['total']['stds'][k] = stdev(obs)
                result[e]['total']['medians'][k] = median(obs)
                result[e]['total']['merged'][k] = (
                    result[e]['total']['means'][k], result[e]['total']['stds'][k], result[e]['total']['medians'][k])

            for cf in [True, False]:
                parts = [p for p in ps if p['esLocal'] == cf]
                loc = el2cf[cf]
                result[e][loc]['partidos']['ultParts'] = tendenciaEquipo(parts, PARTIDOSTENDENCIA)

                for k in INTKEYS + FLOKEYS:
                    obs = [p['yo-estads'][k] for p in parts]

                    result[e][loc]['means'][k] = mean(obs)
                    result[e][loc]['stds'][k] = stdev(obs)
                    result[e][loc]['medians'][k] = median(obs)
                    result[e][loc]['merged'][k] = (
                        result[e][loc]['means'][k], result[e][loc]['stds'][k], result[e][loc]['medians'][k])

        return result


def calculaTempStats(datos, clave, filtroFechas=None):
    if clave not in datos:
        raise (KeyError, "Clave '%s' no está en datos." % clave)

    if filtroFechas:
        datosWrk = datos
    else:
        datosWrk = datos

    agg = datosWrk.set_index('codigo')[clave].astype('float64').groupby('codigo').agg(['mean', 'std', 'count',
                                                                                       'median', 'min', 'max',
                                                                                       'skew'])
    agg1 = agg.rename(columns=dict([(x, clave + "-" + x) for x in agg.columns])).reset_index()
    return agg1


def calculaZ(datos, clave, useStd=True, filtroFechas=None):
    clZ = 'Z' if useStd else 'D'

    finalKeys = ['codigo', 'competicion', 'temporada', 'jornada', 'CODequipo', 'CODrival', 'esLocal',
                 'haJugado', 'Fecha', 'periodo', clave]
    finalTypes = {'CODrival': 'category', 'esLocal': 'bool', 'CODequipo': 'category',
                  ('half-' + clave): 'bool', ('aboveAvg-' + clave): 'bool', (clZ + '-' + clave): 'float64'}
    # We already merged SuperManager?
    if 'pos' in datos.columns:
        finalKeys.append('pos')
        finalTypes['pos'] = 'category'

    if filtroFechas:
        datosWrk = datos  # TODO: filtro de fechas
    else:
        datosWrk = datos

    agg1 = calculaTempStats(datos, clave, filtroFechas)

    dfResult = datosWrk[finalKeys].merge(agg1)
    stdMult = (1 / dfResult[clave + "-std"]) if useStd else 1
    dfResult[clZ + '-' + clave] = (dfResult[clave] - dfResult[clave + "-mean"]) * stdMult
    dfResult['half-' + clave] = (((dfResult[clave] - dfResult[clave + "-median"]) > 0.0)[~dfResult[clave].isna()]) * 100
    dfResult['aboveAvg-' + clave] = ((dfResult[clZ + '-' + clave] >= 0.0)[~dfResult[clave].isna()]) * 100

    return dfResult.astype(finalTypes)


def calculaVars(temporada, clave, useStd=True, filtroFechas=None):
    clZ = 'Z' if useStd else 'D'

    combs = {'R': ['CODrival'], 'RL': ['CODrival', 'esLocal'], 'L': ['esLocal']}
    if 'pos' in temporada.columns:
        combs['RP'] = ['CODrival', 'pos']
        combs['RPL'] = ['CODrival', 'esLocal', 'pos']

    colAdpt = {('half-' + clave + '-mean'): (clave + '-mejorMitad'),
               ('aboveAvg-' + clave + '-mean'): (clave + '-sobreMedia')}
    datos = calculaZ(temporada, clave, useStd=useStd, filtroFechas=filtroFechas)
    result = dict()

    for comb in combs:
        combfloat = combs[comb] + [(clZ + '-' + clave)]
        resfloat = datos[combfloat].groupby(combs[comb]).agg(['mean', 'std', 'count', 'min', 'median', 'max', 'skew'])
        combbool = combs[comb] + [('half-' + clave), ('aboveAvg-' + clave)]
        resbool = datos[combbool].groupby(combs[comb]).agg(['mean'])
        result[comb] = pd.concat([resbool, resfloat], axis=1, sort=True).reset_index()
        newColNames = [((comb + "-" + colAdpt.get(x, x)) if clave in x else x)
                       for x in combinaPDindexes(result[comb].columns)]
        result[comb].columns = newColNames
        result[comb]["-".join([comb, clave, (clZ.lower() + "Min")])] = (
                result[comb]["-".join([comb, clZ, clave, 'mean'])] - result[comb]["-".join([comb, clZ, clave, 'std'])])
        result[comb]["-".join([comb, clave, (clZ.lower() + "Max")])] = (
                result[comb]["-".join([comb, clZ, clave, 'mean'])] + result[comb]["-".join([comb, clZ, clave, 'std'])])

    return result


def datosClas2DF(datos, deslocs=['total']):
    """

    :param datos: resultado de           temporada.obtenClasificacion()
    :param deslocs: uno o más de ['total','casa','fuera']
    :return:
    """

    def getOrderPos(c):
        TARGETPOS = 2  # 0 media, 1 std, 2 mediana
        colName = c.name

        if colName[1] == 'partidos':
            return c

        print(c)
        npos = (np.argsort(-np.array(c.map(lambda x: x[TARGETPOS]))))
        print(npos)

        npos1 = map(lambda x: 1 + x[1], sorted(list(zip(npos, range(len(c))))))
        print(npos1)

        result = pd.Series(map(lambda x: tuple(list(x[0]) + [x[1]]), list(zip(c, npos1))), index=c.index)

        return result

    resultDFs = list()

    desdatos = ['partidos', 'merged']

    for e in datos:  # equipo
        nombre = datos[e]['Nombre']
        dfsEq = list()

        for l in deslocs:
            datosEq = datos[e][l]

            for d in desdatos:
                dictDatos = {nombre: datosEq[d]}
                dfDatos = pd.DataFrame.from_dict(dictDatos, orient='index')

                catsCol = [[l], [d], list(dfDatos.columns)]

                colMIDX = pd.MultiIndex.from_product(catsCol)
                dfDatosIDX = dfDatos.set_axis(colMIDX, axis=1, inplace=False)
                dfsEq.append(dfDatosIDX)

        finalDF = pd.concat(dfsEq, axis=1)
        resultDFs.append(finalDF)

    result = pd.concat(resultDFs).sort_values(
        by=[('total', 'partidos', 'V'), ('total', 'partidos', 'D'), ('total', 'partidos', 'dif')],
        ascending=[False, True, False]).apply(getOrderPos, axis=0)

    return result


def tendenciaEquipo(partidos, numPartidos=5):
    lv2k = {True: {True: 'v', False: 'd'}, False: {True: 'V', False: 'D'}}

    datosE = [(p['esLocal'], p['haGanado']) for p in sorted(partidos, key=lambda x: x['FechaHora'])]
    tendE = list(map(lambda x: lv2k[x[0]][x[1]], datosE))

    result = "".join(tendE[-numPartidos:])

    return result
