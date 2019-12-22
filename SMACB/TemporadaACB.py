'''
Created on Jan 4, 2018

@author: calba
'''

from argparse import Namespace
from collections import defaultdict
from copy import copy
from pickle import dump, load
from sys import exc_info, setrecursionlimit
from time import gmtime
from traceback import print_exception

import pandas as pd

from SMACB.CalendarioACB import URL_BASE, CalendarioACB, calendario_URLBASE
from SMACB.FichaJugador import FichaJugador
from SMACB.PartidoACB import PartidoACB
from Utils.Pandas import combinaPDindexes
from Utils.Web import creaBrowser


class TemporadaACB(object):
    '''
    Aglutina calendario y lista de partidos
    '''

    def __init__(self, **kwargs):
        self.competicion = kwargs.get('competicion', "LACB")
        self.edicion = kwargs.get('edicion', None)
        self.urlbase = kwargs.get('urlbase', calendario_URLBASE)
        descargaFichas = kwargs.get('descargaFichas', False)

        self.timestamp = gmtime()
        self.Calendario = CalendarioACB(competicion=self.competicion, edicion=self.edicion, urlbase=self.urlbase)
        self.PartidosDescargados = set()
        self.Partidos = dict()
        self.changed = False
        self.id2jugador = defaultdict(set)
        self.jugador2id = defaultdict(set)
        self.descargaFichas = descargaFichas
        self.fichaJugadores = dict()
        self.fichaEntrenadores = dict()

    def actualizaTemporada(self, home=None, browser=None, config=Namespace()):

        if isinstance(config, dict):
            config = Namespace(**config)

        if browser is None:
            browser = creaBrowser(config)
            browser.open(URL_BASE)

        self.Calendario.actualizaCalendario(browser=browser, config=config)

        if 'procesabio' in config and config.procesaBio:
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

            except BaseException:
                print("actualizaTemporada: problemas descargando  partido '%s': %s" % (partido, exc_info()))
                print_exception(*exc_info())

            if 'justone' in config and config.justone:  # Just downloads a game (for testing/dev purposes)
                break

        if partidosBajados:
            self.changed = True
            self.timestamp = gmtime()

        return partidosBajados

    def actualizaNombresEquipo(self, partido):
        for loc in partido.Equipos:
            nombrePartido = partido.Equipos[loc]['Nombre']
            codigoParam = partido.Equipos[loc]['abrev']
            idParam = partido.Equipos[loc]['id']
            if self.Calendario.nuevaTraduccionEquipo2Codigo(nombrePartido, codigoParam, idParam):
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

    def actualizaFichasPartido(self, nuevoPartido, browser=None, config=Namespace(), refrescaFichas=False):
        if browser is None:
            browser = creaBrowser(config)
            browser.open(URL_BASE)

        for codJ, datosJug in nuevoPartido.Jugadores.items():
            if codJ not in self.fichaJugadores:
                nuevaFicha = FichaJugador.fromURL(datosJug['linkPersona'], home=browser.get_url(), browser=browser,
                                                  config=config)

                self.jugador2id[nuevaFicha.nombre].add(nuevaFicha.id)
                self.jugador2id[nuevaFicha.alias].add(nuevaFicha.id)
                self.id2jugador[nuevaFicha.id].add(nuevaFicha.nombre)
                self.id2jugador[nuevaFicha.id].add(nuevaFicha.alias)
                self.fichaJugadores[codJ] = nuevaFicha

            elif refrescaFichas:
                self.fichaJugadores[codJ] = self.fichaJugadores[codJ].actualizaFicha(browser=browser, config=config)
                self.jugador2id[self.fichaJugadores[codJ].nombre].add(self.fichaJugadores[codJ].id)
                self.jugador2id[self.fichaJugadores[codJ].alias].add(self.fichaJugadores[codJ].id)
                self.id2jugador[self.fichaJugadores[codJ].id] = self.fichaJugadores[codJ].nombre
                self.id2jugador[self.fichaJugadores[codJ].id] = self.fichaJugadores[codJ].alias

            self.jugador2id[datosJug['nombre']].add(datosJug['codigo'])
            self.id2jugador[datosJug['codigo']].add(datosJug['nombre'])

            self.changed |= self.fichaJugadores[codJ].nuevoPartido(nuevoPartido)

        # TODO: Procesar ficha de entrenadores
        for codE in nuevoPartido.Entrenadores:
            pass


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
