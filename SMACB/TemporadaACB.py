'''
Created on Jan 4, 2018

@author: calba
'''

from calendar import timegm
from collections import defaultdict
from copy import copy
from pickle import dump, load
from sys import setrecursionlimit
from time import gmtime, strftime

from SMACB.CalendarioACB import CalendarioACB, calendario_URLBASE
from SMACB.PartidoACB import PartidoACB
from Utils.Misc import FORMATOtimestamp


class TemporadaACB(object):

    '''
    Aglutina calendario y lista de partidos
    '''

    def __init__(self, competition="LACB", edition=None, urlbase=calendario_URLBASE):
        self.timestamp = gmtime()
        self.Calendario = CalendarioACB(competition=competition, edition=edition, urlbase=urlbase)
        self.PartidosDescargados = set()
        self.Partidos = dict()
        self.changed = False

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

        clavePartido = ['FechaHora', 'URL', 'Partido', 'ResumenPartido']
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

            for claveJ in partido.Jugadores:
                jugador = partido.Jugadores[claveJ]

                resultado['FechaHora'][claveJ][jornada] = fechahora
                resultado['URL'][claveJ][jornada] = claveP
                resultado['Partido'][claveJ][jornada] = ("" if jugador['esLocal'] else "@") + jugador['rival']

                for subClave in claveJugador:
                    resultado[subClave][claveJ][jornada] = jugador[subClave]

                for subClave in claveEstad:
                    if subClave in jugador['estads']:
                        resultado[subClave][claveJ][jornada] = jugador['estads'][subClave]

        # Calcula el orden de las jornadas para mostrar los partidos jugados en orden cronológico
        for claveJ in resultado['FechaHora']:
            auxFH = [((timegm(resultado['FechaHora'][claveJ][x]) if resultado['FechaHora'][claveJ][x] else 0), x)
                     for x in range(len(resultado['FechaHora'][claveJ]))]
            auxFHsorted = [x[1] for x in sorted(auxFH, key=lambda x:x[0])]
            resultado['OrdenPartidos'][claveJ] = auxFHsorted

        for claveJ in resultado['haJugado']:
            convocados = [x for x in resultado['haJugado'][claveJ] if x is not None]
            jugados = sum([1 for x in convocados if x])
            resultado['I-convocado'][claveJ] = len(convocados)
            resultado['I-jugado'][claveJ] = jugados

        return resultado
