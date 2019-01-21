'''
Created on Dec 31, 2017

@author: calba
'''

from time import gmtime, mktime, strptime
from traceback import print_exc

import pandas as pd
from bs4 import Tag

from Utils.Misc import BadParameters, BadString, ExtractREGroups
from Utils.Web import DescargaPagina, ExtraeGetParams

from .SMconstants import (BONUSVICTORIA, bool2esp, haGanado2esp, local2esp,
                          titular2esp)

templateURLficha = "http://www.acb.com/fichas/%s%i%03i.php"
reJornada = r".*J\s*(\d+)\s*"
rePublico = r".*ico:(\d+)"
reArbitro = r".*rb:\s*(.*)\s*$"
reResultadoEquipo = r"^\s*(.*)\s+(\d+)\s*$"

LocalVisitante = ('Local', 'Visitante')


class PartidoACB(object):

    def __init__(self, **kwargs):
        self.Jornada = None
        self.FechaHora = None
        self.Pabellon = None
        self.Asistencia = None
        self.Arbitros = []
        self.ResultadosParciales = []
        self.prorrogas = 0

        self.Equipos = {}
        for x in LocalVisitante:
            self.Equipos[x] = {}
            self.Equipos[x]['Jugadores'] = []
            self.Equipos[x]['haGanado'] = False

        self.Jugadores = {}

        self.EquiposCalendario = dict(zip(LocalVisitante, kwargs['equipos']))
        self.ResultadoCalendario = dict(zip(LocalVisitante, kwargs['resultado']))
        self.CodigosCalendario = dict(zip(LocalVisitante, kwargs['codigos']))

        self.VictoriaLocal = None

        self.DatosSuministrados = kwargs

        if 'url' in kwargs:
            self.url = kwargs['url']
        else:
            if 'URLparams' in kwargs:
                self.url = GeneraURLpartido(kwargs['URLparams'])

        if 'URLparams' in kwargs:
            self.competicion = kwargs['URLparams']['cod_competicion']
            self.temporada = kwargs['URLparams']['cod_edicion']
            self.idPartido = kwargs['URLparams']['partido']

    def descargaPartido(self, home=None, browser=None, config={}):

        if not hasattr(self, 'url'):
            raise BadParameters("PartidoACB: DescargaPartido: imposible encontrar la URL del partido")

        urlPartido = self.url

        partidoPage = DescargaPagina(urlPartido, home=home, browser=browser, config=config)

        self.procesaPartido(partidoPage)

    def procesaPartido(self, content):
        if 'timestamp' in content:
            self.timestamp = content['timestamp']
        else:
            self.timestamp = gmtime()
        if 'source' in content:
            self.url = content['source']

        tablasPartido = content['data'].find_all("table", {"class": "estadisticasnew"})

        # Encabezado de Tabla
        tabDatosGenerales = tablasPartido.pop(0)
        filas = tabDatosGenerales.find_all("tr")

        # Primera fila
        celdas = filas.pop(0).find_all("td")
        espTiempo = celdas.pop(0).get_text().split("|")

        # Jaux = ExtractREGroups(cadena=espTiempo.pop(0).strip(), regex=reJornada)
        self.Jornada = int(ExtractREGroups(cadena=espTiempo.pop(0).strip(), regex=reJornada)[0])
        self.FechaHora = strptime(espTiempo[0] + espTiempo[1], " %d/%m/%Y  %H:%M ")
        self.Pabellon = espTiempo[2].strip()
        self.Asistencia = int(ExtractREGroups(cadena=espTiempo[3], regex=rePublico)[0])
        celdas.pop(0)  # Spacer
        self.prorrogas = len(celdas) - 4

        # Segunda fila
        celdas = [x.get_text().strip() for x in filas.pop(0).find_all("td")]

        # reArbitro.match(celdas.pop(0))
        self.Arbitros = [x.strip() for x in ExtractREGroups(cadena=celdas.pop(0).strip(),
                                                            regex=reArbitro)[0].split(",")]
        celdas.pop(0)  # Spacer
        aux = map(lambda x: x.split("|"), celdas)
        self.ResultadosParciales = [(int(x[0]), int(x[1])) for x in aux]

        # Datos Partido
        tabDatosGenerales = tablasPartido.pop(0)
        filas = tabDatosGenerales.find_all("tr")
        fila0 = filas.pop(0)
        fila1 = filas.pop(0)

        colHeaders = self.extractPrefijosTabla(fila0, fila1)

        estado = "Local"

        self.getResultEquipo(fila0.find("td").get_text(), estado)

        while filas:
            fila = filas.pop(0)

            if "estverde" in fila.get('class', ""):
                estado = "Visitante"
                self.getResultEquipo(fila.find("td").get_text(), estado)
                self.VictoriaLocal = self.Equipos['Local']['Puntos'] > self.Equipos['Visitante']['Puntos']
                filas.pop(0)
                continue
            datos = self.procesaLineaTablaEstadistica(fila=fila, headers=colHeaders, estado=estado)
            if not datos:
                continue

            if datos['esJugador']:
                (self.Equipos[estado]['Jugadores']).append(datos['codigo'])
                self.Jugadores[datos['codigo']] = datos

            elif datos.get('noAsignado', False):
                self.Equipos[estado]['NoAsignado'] = datos['estads']

            elif datos.get('totalEquipo', False):
                self.prorrogas = datos['prorrogas']
                if '+/-' in datos['estads'] and datos['estads']['+/-'] is None:
                    datos['estads']['+/-'] = (self.ResultadoCalendario[estado] -
                                              self.ResultadoCalendario[OtherTeam(estado)])
                self.Equipos[estado]['estads'] = datos['estads']

            elif datos.get('entrenador', False):
                self.Equipos[estado]['Entrenador'] = datos

            else:
                BaseException("I am missing something: {}" % datos)

        # Asigna la victoria
        if self.VictoriaLocal:
            estadoGanador = 'Local'
        else:
            estadoGanador = 'Visitante'
        self.Equipos[estadoGanador]['haGanado'] = True
        self.Equipos[estadoGanador]['Entrenador']['haGanado'] = True
        for jug in self.Equipos[estadoGanador]['Jugadores']:
            self.Jugadores[jug]['haGanado'] = True

    def extractPrefijosTabla(self, filacolspans, filaheaders):
        """ Devuelve un array con las cabeceras de cada columna (con matices como los rebotes) y tiros
            Podría ser genérica pero Una de las celdas contiene información y no prefijo
        """

        colspans = [int(x.get('colspan', 1)) for x in filacolspans.find_all("td")]
        coltexts = [x.get_text().strip() for x in filacolspans.find_all("td")]
        coltexts[0] = ""  # La primera celda es el resultado del equipo. No pun prefijo
        prefixes = []

        for i in range(len(colspans)):
            prefixes += ([coltexts[i]] * colspans[i])

        estheaders = [x.get_text().strip() for x in filaheaders.find_all("td")]

        headers = [((x[0] + "-") if x[0] else "") + x[1] for x in zip(prefixes, estheaders)]
        assert (len(set(headers)) == len(headers))

        return (headers)

    def procesaLineaTablaEstadistica(self, fila, headers, estado):
        result = dict()
        result['competicion'] = self.competicion
        result['temporada'] = self.temporada
        result['jornada'] = self.Jornada
        result['equipo'] = self.EquiposCalendario[estado]
        result['CODequipo'] = self.CodigosCalendario[estado]
        result['rival'] = self.EquiposCalendario[OtherTeam(estado)]
        result['CODrival'] = self.CodigosCalendario[OtherTeam(estado)]
        result['estado'] = estado
        result['esLocal'] = (estado == "Local")

        result['esJugador'] = True
        result['entrenador'] = False
        result['haJugado'] = True
        textos = [x.get_text().strip() for x in fila.find_all("td")]
        celdas = fila.find_all("td")
        if (len(textos) == len(headers) - 1):  # El equipo no tiene dorsal
            result['isPlayer'] = False
            textos = [''] + textos
            celdas = [''] + celdas
        if (len(textos) == len(headers)):
            mergedTextos = dict(zip(headers[2:], textos[2:]))
            estads = self.procesaEstadisticas(mergedTextos)
            if None in estads.values():
                pass

            # mergedCeldas=dict(zip(headers[:2],celdas[:2])) ,mergedCeldas
            if textos[0]:
                result['titular'] = ("gristit" in celdas[0].get('class', ""))
                result['dorsal'] = textos[0]
                result['nombre'] = textos[1]
                result['haGanado'] = False
                linkdata = (celdas[1].find("a"))['href']
                linkdatapars = ExtraeGetParams(linkdata)
                try:
                    result['codigo'] = linkdatapars['id']
                except KeyError:
                    print("Exception: procesaLineaTablaEstadistica %s: unable to find id in %s '%s'" % (self.url,
                                                                                                        linkdata,
                                                                                                        textos[0]))
                    return None

                # (self.Equipos[estado]['Jugadores']).append(result['codigo'])
                if not estads:
                    result['haJugado'] = False

                result['estads'] = estads
                # self.Jugadores[result['codigo']]=result

            else:
                result['esJugador'] = False
                if textos[1].lower() == "equipo":
                    result['estads'] = estads
                    result['noAsignado'] = True
                    # self.Equipos[estado]['NoAsignado']=estads
                elif textos[1].lower() == "total":
                    result['totalEquipo'] = True
                    result['estads'] = estads
                    result['prorrogas'] = int(((estads['Segs'] / (5 * 60)) - 40) // 5)
                    # self.prorrogas = result['prorrogas']
                    # self.Equipos[estado]['estads']=estads
                    if estads['P'] != self.Equipos[estado]['Puntos']:
                        print(estads, self.Equipos[estado])
                        raise BaseException("ProcesaLineaTablaEstadistica: TOTAL '%s' puntos '%i' "
                                            "no casan con encabezado '%i' " % (estado, estads['P'],
                                                                               self.Equipos[estado]['Puntos']))
                else:
                    raise BaseException("ProcesaLineaTablaEstadistica: noplayer "
                                        "string '%s' unknown" % (textos[1].lower()))
        elif (len(textos) == 2):
            # Entrenador o faltas
            result['esJugador'] = False
            if textos[0].lower() == "e":
                result['entrenador'] = True
                result['nombre'] = textos[1]
                linkdata = (celdas[1].find("a"))['href']
                linkdatapars = ExtraeGetParams(linkdata)
                result['codigo'] = linkdatapars['id']
                result['haGanado'] = False
                # self.Equipos[estado]['Entrenador']=result
            elif textos[0].lower() == "5f":
                return dict()
                pass
            else:
                raise BaseException("ProcesaLineaTablaEstadistica: info string '%s' unknown" % (textos[0].lower()))

        # print(result)
        return (result)

        # print(len(textos),textos)

    def procesaEstadisticas(self, contadores):

        result = {}

        reTiempo = r"^\s*(\d+):(\d+)\s*$"
        reTiros = r"^\s*(\d+)/(\d+)\s*$"
        reRebotes = r"^\s*(\d+)\+(\d+)\s*$"
        rePorcentaje = r"^\s*(\d+)%\s*$"

        def ProcesaTiempo(cadena):
            auxTemp = ExtractREGroups(cadena=cadena, regex=reTiempo)
            if auxTemp:
                return (int(auxTemp[0]) * 60 + int(auxTemp[1]))
            else:
                raise BadString("ProcesaEstadisticas:ProcesaTiempo '%s' no casa RE '%s' " % (cadena, reTiempo))

        def ProcesaTiros(cadena):
            auxTemp = ExtractREGroups(cadena=cadena, regex=reTiros)
            if auxTemp:
                return (int(auxTemp[0]), int(auxTemp[1]))
            else:
                raise BadString("ProcesaEstadisticas:ProcesaTiros '%s' no casa RE '%s' " % (cadena, reTiros))

        def ProcesaRebotes(cadena):
            auxTemp = ExtractREGroups(cadena=cadena, regex=reRebotes)
            if auxTemp:
                return (int(auxTemp[0]), int(auxTemp[1]))
            else:
                raise BadString("ProcesaEstadisticas:ProcesaRebotes '%s' no casa RE '%s' " % (cadena, reRebotes))

        def ProcesaPorcentajes(cadena):
            auxTemp = ExtractREGroups(cadena=cadena, regex=rePorcentaje)
            if auxTemp:
                return (int(auxTemp[0]))
            else:
                raise BadString("ProcesaEstadisticas:ProcesaPorcentajes '%s' no casa RE '%s' " %
                                (cadena, rePorcentaje))

        for key in contadores.keys():
            val = contadores[key]
            if val == "":
                continue

            if key in ['Min']:
                result['Segs'] = ProcesaTiempo(val)
            elif key in ['T2', 'T3', 'T1']:
                aux = ProcesaTiros(val)
                result[key + "-C"] = aux[0]
                result[key + "-I"] = aux[1]
            elif key in ['T2 %', 'T3 %', 'T1 %']:
                result[key.replace(" ", "")] = ProcesaPorcentajes(val)
            elif key == 'REB-D+O':
                aux = ProcesaRebotes(val)
                result["R-D"] = aux[0]
                result["R-O"] = aux[1]
            else:
                try:
                    result[key] = int(val)
                except ValueError:
                    result[key] = None
                    print_exc()
                    print("ProcesaEstadisticas: Error: '%s'='%s' converting to INT. "
                          "URL Partido: %s -> %s" % (key, val, self.url, contadores))

        return (result)

    def getResultEquipo(self, cadena, estado):
        aux = ExtractREGroups(regex=reResultadoEquipo, cadena=cadena)

        if aux:
            self.Equipos[estado]['Nombre'] = aux[0]
            self.Equipos[estado]['Puntos'] = int(aux[1])
        else:
            raise BadString("GetResult: '%s' no casa RE '%s' " % (cadena, reResultadoEquipo))

    def resumenPartido(self):
        return " * J %i: %s (%s) %i - %i %s (%s) " % (self.Jornada, self.EquiposCalendario['Local'],
                                                      self.CodigosCalendario['Local'],
                                                      self.ResultadoCalendario['Local'],
                                                      self.ResultadoCalendario['Visitante'],
                                                      self.EquiposCalendario['Visitante'],
                                                      self.CodigosCalendario['Visitante'])

    def jugadoresAdataframe(self):
        typesDF = {'competicion': 'object', 'temporada': 'int64', 'jornada': 'int64', 'esLocal': 'bool',
                   'haJugado': 'bool', 'titular': 'category', 'haGanado': 'bool', 'enActa': 'bool', 'Vsm': 'float64'}

        # 'equipo': 'object', 'CODequipo': 'object', 'rival': 'object', 'CODrival': 'object', 'dorsal': 'object'
        # 'nombre': 'object', 'codigo': 'object'

        def jugador2dataframe(jugador):
            dictJugador = dict()
            dictJugador['enActa'] = True
            dictJugador['acta'] = 'S'

            for dato in jugador:
                if dato in ['esJugador', 'entrenador', 'estads', 'estado']:
                    continue
                dictJugador[dato] = jugador[dato]

            if jugador['haJugado']:
                for dato in jugador['estads']:
                    dictJugador[dato] = jugador['estads'][dato]
                    typesDF[dato] = 'float64'
                dictJugador['Vsm'] = (jugador['estads']['V'] * (BONUSVICTORIA if (
                        jugador['haGanado'] and (jugador['estads']['V'] > 0)) else 1.0)
                                      )
            else:
                dictJugador['V'] = 0.0
                dictJugador['Vsm'] = 0.0
                typesDF['V'] = 'float64'

            dfresult = pd.DataFrame.from_dict(dictJugador, orient='index').transpose()
            dfresult['Fecha'] = pd.to_datetime(mktime(self.FechaHora), unit='s')
            dfresult['local'] = dfresult['esLocal'].map(local2esp)
            dfresult['titular'] = dfresult['titular'].map(titular2esp)
            dfresult['resultado'] = dfresult['haGanado'].map(haGanado2esp)
            dfresult['jugado'] = dfresult['haJugado'].map(bool2esp)

            return (dfresult)

        dfJugs = [jugador2dataframe(self.Jugadores[x]) for x in self.Jugadores]
        dfResult = pd.concat(dfJugs, axis=0, ignore_index=True, sort=True).astype(typesDF)

        return (dfResult)

    def clasifPartido(self):
        """
        De cada partido extrae los datos de cada contendiente y los del rival. Creado para el dossier
        (sacar la clasificaci�n)
        :return: lista con diccionarios con los datos para cada partido y contendiente.
        """

        CLAVESGLOBALES = ['Jornada', 'FechaHora', 'prorrogas', 'url']
        result = []

        for l in self.Equipos:
            resEquipo = dict()
            resEquipo['nombre'] = self.Equipos[l]['Nombre']
            resEquipo['codigo'] = self.CodigosCalendario[l]
            resEquipo['rival'] = self.Equipos[OtherTeam(l)]['Nombre']
            resEquipo['cod-rival'] = self.CodigosCalendario[OtherTeam(l)]
            for k in CLAVESGLOBALES:
                resEquipo[k] = self.__getattribute__(k)
            resEquipo['haGanado'] = self.Equipos[l]['haGanado']
            resEquipo['esLocal'] = ('Local' == l)
            resEquipo['yo-estads'] = self.Equipos[l]['estads']
            resEquipo['otro-estads'] = self.Equipos[OtherTeam(l)]['estads']

            resEquipo['diferencia'] = self.Equipos[l]['Puntos'] - self.Equipos[OtherTeam(l)]['Puntos']
            resEquipo['yo-estads']['Prec'] = self.Equipos[OtherTeam(l)]['Puntos']
            resEquipo['yo-estads']['defAro'] = 100.0 / (
                        1 + (resEquipo['otro-estads']['R-O'] / resEquipo['yo-estads']['R-D']))
            resEquipo['yo-estads']['ataAro'] = 100.0 / (
                        1 + (resEquipo['otro-estads']['R-D'] / resEquipo['yo-estads']['R-O']))

            resEquipo['yo-estads']['TC-I'] = resEquipo['yo-estads']['T2-I'] + resEquipo['yo-estads']['T3-I']
            resEquipo['yo-estads']['TC-C'] = resEquipo['yo-estads']['T2-C'] + resEquipo['yo-estads']['T3-C']
            resEquipo['yo-estads']['TC%'] = 100.0 * resEquipo['yo-estads']['TC-C'] / resEquipo['yo-estads']['TC-I']
            resEquipo['yo-estads']['T1%'] = 100.0 * resEquipo['yo-estads']['T1-C'] / resEquipo['yo-estads']['T1-I']
            resEquipo['yo-estads']['T2%'] = 100.0 * resEquipo['yo-estads']['T2-C'] / resEquipo['yo-estads']['T2-I']
            resEquipo['yo-estads']['T3%'] = 100.0 * resEquipo['yo-estads']['T3-C'] / resEquipo['yo-estads']['T3-I']

            resEquipo['yo-estads']['P2%'] = 100.0 * 2 * resEquipo['yo-estads']['T2-C'] / resEquipo['yo-estads']['P']
            resEquipo['yo-estads']['P3%'] = 100.0 * 3 * resEquipo['yo-estads']['T3-C'] / resEquipo['yo-estads']['P']

            # TODO: Calculo posesiones, ...
            result.append(resEquipo)

        return result


def GeneraURLpartido(link):
    def CheckParameters(dictParams):
        requiredParamList = ('cod_competicion', 'cod_edicion', 'partido')
        errores = False
        missingParameters = set()

        for par in requiredParamList:
            if par not in dictParams:
                errores = True
                missingParameters.add(par)

        if errores:
            raise BadParameters("GeneraURLpartido: falta informacion en parámetro: %s." % missingParameters)

    if type(link) is Tag:  # Enlace a sacado con BeautifulSoup
        link2process = link['href']
    elif type(link) is str:  # Cadena URL
        link2process = link
    elif type(link) is dict:  # Diccionario con los parametros necesarios s(sacados de la URL, se supone)
        CheckParameters(link)
        return templateURLficha % (link['cod_competicion'], int(link['cod_edicion']), int(link['partido']))
    else:
        raise TypeError("GeneraURLpartido: incapaz de procesar %s (%s)" % (link, type(link)))

    liurlcomps = ExtraeGetParams(link2process)
    CheckParameters(liurlcomps)
    return templateURLficha % (liurlcomps['cod_competicion'], int(liurlcomps['cod_edicion']),
                               int(liurlcomps['partido']))


def OtherTeam(team):
    if team == 'Local':
        return 'Visitante'
    elif team == 'Visitante':
        return 'Local'
    else:
        raise BadParameters("OtherTeam: '%s' provided. It only accept 'Visitante' or 'Local'" % team)
