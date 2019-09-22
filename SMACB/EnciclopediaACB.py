'''
Created on Jan 4, 2018

@author: calba
'''

from argparse import Namespace
from calendar import timegm
from collections import defaultdict
from copy import copy
from pickle import dump, load
from sys import setrecursionlimit
from time import gmtime, strftime
from sys import exc_info
import pandas as pd
from babel.numbers import decimal

from .CalendarioACB import CalendarioACB, calendario_URLBASE, URL_BASE

from .PartidoACB import PartidoACB
from .SMconstants import LISTACOMPOS, calculaValSuperManager
from Utils.Misc import FORMATOfecha, FORMATOtimestamp, Seg2Tiempo
from Utils.Web import creaBrowser
from .CompoACB import CompoACB
from .TemporadaACB import TemporadaACB
from .FichaJugador import FichaJugador


class TemporadaACBXXX(object):
    '''
    Aglutina calendario y lista de partidos
    '''

    def __init__(self, **kwargs):
        competicion = kwargs.get('competicion', "LACB")
        edicion = kwargs.get('edicion', None)
        urlbase = kwargs.get('urlbase', calendario_URLBASE)
        descargaFichas = kwargs.get('descargaFichas', False)

        self.timestamp = gmtime()
        self.Calendario = CalendarioACB(competicion=competicion, edicion=edicion, urlbase=urlbase)
        self.PartidosDescargados = set()
        self.Partidos = dict()
        self.changed = False
        self.translations = defaultdict(set)
        self.descargaFichas = descargaFichas
        self.fichaJugadores = dict()
        self.fichaEntrenadores = dict()

    def actualizaTemporada(self, home=None, browser=None, config=Namespace()):

        if browser is None:
            browser = creaBrowser(config)
            browser.open(URL_BASE)

        self.Calendario.actualizaCalendario(browser=browser, config=config)

        if isinstance(config, dict):
            config = Namespace(**config)

        if config.procesaBio:
            self.descargaFichas = True

        partidosBajados = set()

        for partido in self.Calendario.Partidos:
            if partido in self.PartidosDescargados:
                continue

            try:
                nuevoPartido = PartidoACB(**(self.Calendario.Partidos[partido]))
                nuevoPartido.descargaPartido(home=home, browser=browser, config=config)
                self.PartidosDescargados.add(partido)
                self.Partidos[partido] = nuevoPartido
                self.actualizaNombresEquipo(nuevoPartido)
                partidosBajados.add(partido)

                if self.descargaFichas:
                    self.actualizaFichasPartido(nuevoPartido, browser=browser, config=config)
                if config.justone:  # Just downloads a game (for testing/dev purposes)
                    break

            except:
                print("actualizaTemporada: problemas descargando  partido'%s': %s" % (partido, exc_info()))

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

    def actualizaFichasPartido(self, nuevoPartido, browser=None, config=Namespace(), refrescaFichas=False):
        if browser is None:
            browser = creaBrowser(config)
            browser.open(URL_BASE)

        for codJ in nuevoPartido.Jugadores:
            if codJ not in self.fichaJugadores:
                self.fichaJugadores[codJ] = FichaJugador.fromURL(nuevoPartido.Jugadores[codJ]['linkPersona'],
                                                                 home=browser.get_url(),
                                                                 browser=browser, config=config)
            elif refrescaFichas:
                self.fichaJugadores[codJ] = FichaJugador.actualizaFicha(browser=browser, config=config)

            self.changed |= self.fichaJugadores[codJ].nuevoPartido(nuevoPartido)

        # TODO: Procesar ficha de entrenadores
        for codE in nuevoPartido.Entrenadores:
            pass


class EnciclopediaACB(object):

    def __init__(self):
        self.timestamp = gmtime()
        self.Competiciones = dict()
        self.partidos = dict()
        self.fichaJugadores = dict()
        self.fichaEntrenadores = dict()

        self.changed = False
        self.translationsEq = dict()
        self.translationsJug = dict()

    def cargaTemporada(self, tempData):
        # tempData = TemporadaACB()  # Para usar el completador. COMENTAR AL FINAL
        codCompo = tempData.Calendario.competicion
        idTemp = tempData.Calendario.edicion

        if self.Competiciones.get(codCompo, dict()).get(idTemp, None) is None:
            print("HERE1 ", codCompo, idTemp)
            if codCompo not in self.Competiciones:
                self.Competiciones[codCompo] = dict()
            self.Competiciones[codCompo][idTemp] = CompoACB(codCompo, idTemp)
            self.changed = True

        nuevosPartidos = self.Competiciones[codCompo][idTemp].actualizaDeTemp(tempData)
        if nuevosPartidos:
            self.changed = True

        for p in nuevosPartidos:
            self.partidos[p] = tempData.Partidos[p]
            self.changed = True


        for j in tempData.fichaJugadores:
            if j not in self.fichaJugadores:
                self.fichaJugadores[j] = copy(tempData.fichaJugadores[j])
                self.changed = True
            else:
                fichjug = tempData.fichaJugadores[j]
                # fichjug = FichaJugador()  # Para usar el completador. COMENTAR AL FINAL
                # self.fichaJugadores[j] = FichaJugador()  # Para usar el completador. COMENTAR AL FINAL
                for p in fichjug.partidos:

                    self.changed |= self.fichaJugadores[j].nuevoPartido(tempData.Partidos[p])
