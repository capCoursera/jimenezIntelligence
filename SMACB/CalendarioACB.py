import re
from collections import defaultdict
from copy import copy
from time import gmtime, strptime

import bs4

from Utils.Misc import CompareBagsOfWords, PARSERfechaC
from Utils.Web import ComposeURL, DescargaPagina, ExtraeGetParams, MergeURL

from .PartidoACB import GeneraURLpartido
from .SMconstants import OtherTeam

URL_BASE = "http://www.acb.com"

calendario_URLBASE = "http://acb.com/calendario.php"
template_URLFICHA = "http://www.acb.com/fichas/%s%i%03i.php"
UMBRALbusquedaDistancia = 1  # La comparación debe ser >


class CalendarioACB(object):

    def __init__(self, urlbase=calendario_URLBASE, **kwargs):
        self.timestamp = gmtime()
        self.competicion = kwargs.get('competicion', "LACB")
        self.nombresCompeticion = defaultdict(int)
        self.edicion = kwargs.get('edicion')
        self.Partidos = {}
        self.Jornadas = {}
        self.equipo2codigo = {}
        self.codigo2equipo = defaultdict(set)
        self.url = urlbase

    def bajaCalendario(self, home=None, browser=None, config={}):
        urlCalendario = ComposeURL(self.url,
                                   {'cod_competicion': self.competicion, 'cod_edicion': self.edicion, 'vd': "1",
                                    'vh': "60"})
        calendarioPage = DescargaPagina(urlCalendario, home=home, browser=browser, config=config)
        self.procesaCalendario(calendarioPage)

    def procesaCalendario(self, content):
        if 'timestamp' in content:
            self.timestamp = content['timestamp']
        if 'source' in content:
            self.url = content['source']
        calendarioData = content['data']

        tablaCuerpo = calendarioData.table(recursive=False)[0]

        tablaCols = tablaCuerpo.find_all("td", recursive=False)
        colFechas = tablaCols[2]

        # Tomamos toda la informacion posible de la pagina del calendario.
        currJornada = None
        tablaPlayoffs = False
        for item in colFechas:
            if type(item) is bs4.element.NavigableString:  # Retornos de carro y cosas así
                continue
            elif item.name == 'div':
                divClasses = item.attrs.get('class', [])
                if (('menuseparacion' in divClasses) or ('piemenuclubs' in divClasses) or
                        ('cuerpobusca' in divClasses) or ('titulomenuclubsl' in divClasses)):
                    continue  # DIV estéticos o que no aportan información interesante
                elif 'titulomenuclubs' in divClasses:
                    tituloDiv = item.string
                    tituloFields = tituloDiv.split(" - ")
                    if len(tituloFields) == 1:  # Encabezado Selector de Jornadas a mostrar
                        continue
                    else:
                        self.nombresCompeticion[tituloFields[0]] += 1

                    jornadaMatch = re.match(r"JORNADA\s+(\d+)", tituloFields[1])
                    if jornadaMatch:  # Liga Endesa 2017-18 - JORNADA 34
                        currJornada = int(jornadaMatch.groups()[0])
                        self.Jornadas[currJornada] = dict()
                        self.Jornadas[currJornada]['nombre'] = tituloFields[1]
                        self.Jornadas[currJornada]['partidos'] = []
                        self.Jornadas[currJornada]['pendientes'] = []
                        self.Jornadas[currJornada]['equipos'] = set()
                        self.Jornadas[currJornada]['esPlayoff'] = False

                        continue
                    else:  # Liga Endesa 2017-18 - Calendario jornadas - Liga Regular
                        currJornada = None
                        if 'playoff' in tituloFields[2].lower():
                            tablaPlayoffs = True
                        else:
                            tablaPlayoffs = False

                elif ('cuerponaranja' in divClasses):  # Selector para calendario de clubes
                    self.procesaSelectorClubes(item)
                elif ('tablajornadas' in divClasses):  # Selector para calendario de clubes
                    if tablaPlayoffs:
                        self.procesaTablaCalendarioJornadas(item)
                    else:
                        continue
                else:
                    print("DIV Unprocessed: ", item.attrs)
            elif item.name == 'table':
                self.procesaTablaJornada(item, currJornada)
            elif item.name in ('br'):  # Otras cosas que no interesan
                continue
            else:
                print("Unexpected: ", item, item.__dict__.keys())

        # Detecta los cambios de nombre de equipo
        self.gestionaNombresDeEquipo()

        for jornada in self.Jornadas:
            if not self.Jornadas[jornada]['partidos']:
                continue

            for partido in self.Jornadas[jornada]['partidos']:
                self.Partidos[partido]['esPlayoff'] = self.Jornadas[jornada]['esPlayoff']
                codigos = [self.equipo2codigo.get(x, None) for x in self.Partidos[partido]['equipos']]
                if None in codigos:
                    raise BaseException("Equipo sin codigo en '%s' (%s) " %
                                        (" - ".join(self.Partidos[partido]['equipos']), partido))
                self.Partidos[partido]['codigos'] = codigos

    def procesaTablaJornada(self, tagTabla, currJornada):
        for row in tagTabla.find_all("tr"):
            cols = row.find_all("td", recursive=False)

            equipos = [x.strip() for x in cols[1].string.split(" - ")]
            for equipo in equipos:
                (self.Jornadas[currJornada]['equipos']).add(equipo)

            # si el partido ha sucedido, hay un enlace a las estadisticas en la col 0 (tambien en la del resultado)
            linksCol0 = cols[0].find_all("a")

            if linksCol0:
                linkGame = linksCol0[0]
                linkOk = GeneraURLpartido(linkGame)
                puntos = [int(x.strip()) for x in cols[2].string.split(" - ")]

                paramsURL = ExtraeGetParams(linkGame['href'])
                self.Partidos[linkOk] = {'url': linkOk, 'URLparams': paramsURL, 'jornada': currJornada,
                                         'equipos': equipos, 'resultado': puntos}
                (self.Jornadas[currJornada]['partidos']).append(linkOk)
            else:  # No ha habido partido
                partidoPendiente = dict()
                partidoPendiente['jornada'] = currJornada
                partidoPendiente['equipos'] = equipos
                textoFecha = cols[2].string
                if textoFecha:
                    try:
                        fechaPart = strptime(textoFecha, PARSERfechaC)
                        partidoPendiente['fecha'] = fechaPart
                    except ValueError:
                        partidoPendiente['fecha'] = None
                else:
                    partidoPendiente['fecha'] = None

                self.Jornadas[currJornada]['pendientes'].append(partidoPendiente)

    def procesaSelectorClubes(self, tagForm):
        optionList = tagForm.find_all("option")
        for optionTeam in optionList:
            equipoCodigo = optionTeam['value']
            equipoNombre = optionTeam.string
            if equipoCodigo == "0":
                continue
            self.nuevaTraduccionEquipo2Codigo(equipoNombre, equipoCodigo)

    def procesaTablaCalendarioJornadas(self, tagTabla):
        for table in tagTabla.find_all("table", attrs={'class': 'jornadas'}):
            for row in table.find_all("tr"):
                cols = row.find_all("td", recursive=False)
                if len(cols) == 2:  # Encabezamiento tabla
                    continue
                currJornada = int(cols[1].string)
                if currJornada not in self.Jornadas:
                    self.Jornadas[currJornada] = dict()
                    self.Jornadas[currJornada]['partidos'] = []
                    self.Jornadas[currJornada]['equipos'] = set()

                tituloFields = cols[2].string.split(":")
                self.Jornadas[currJornada]['nombre'] = tituloFields[0].strip()
                self.Jornadas[currJornada]['esPlayoff'] = True

    def gestionaNombresDeEquipo(self):
        """ Intenta tener en cuenta los nombres de equipos que cambian a lo largo de la temporada (patrocinios)
        """

        codigosTemporada = set(self.codigo2equipo.keys())
        combinacionesNoUsadas = dict()

        for jornada in sorted(self.Jornadas.keys(), reverse=True):
            if self.Jornadas[jornada]['esPlayoff']:
                continue

            if not self.Jornadas[jornada]['equipos']:
                continue

            codigosUsados = set()
            equiposNoAsignados = set()
            for equipo in self.Jornadas[jornada]['equipos']:
                if equipo in self.equipo2codigo:
                    codigosUsados.add(self.equipo2codigo[equipo])
                else:
                    equiposNoAsignados.add(equipo)
            codigosNoUsados = codigosTemporada - codigosUsados

            if not equiposNoAsignados:  # Se asignado todo!
                continue

            # TODO: Habra que hacer algo para cuando haya equipos impares

            # busca similitures entre el nombre que aparece en el formulario (recortado) y el de las jornadas
            changes = True
            while changes:
                changes = False
                if not equiposNoAsignados:
                    continue

                auxEquiposNoAsignados = copy(equiposNoAsignados)
                for equipo in auxEquiposNoAsignados:
                    auxCodigosNoUsados = copy(codigosNoUsados)
                    codigo = self.buscaEquipo2CodigoDistancia(equipo, codigosObjetivo=auxCodigosNoUsados)
                    if codigo:
                        self.nuevaTraduccionEquipo2Codigo(equipo, codigo)
                        equiposNoAsignados.remove(equipo)
                        codigosNoUsados.remove(codigo)
                        changes = True
                        break

                # Caso trivial, sólo queda uno por asignar
                if len(codigosNoUsados) == 1 and len(equiposNoAsignados) == 1:
                    codigo = codigosNoUsados.pop()
                    equipo = equiposNoAsignados.pop()
                    self.nuevaTraduccionEquipo2Codigo(equipo, codigo)
                    changes = True
                    continue

            if codigosNoUsados:
                combinacionesNoUsadas["|".join(sorted(list(codigosNoUsados)))] = equiposNoAsignados

        if not combinacionesNoUsadas:
            return
        else:
            print(combinacionesNoUsadas)

    def buscaEquipo2CodigoDistancia(self, equipoAbuscar, codigosObjetivo=None):
        if equipoAbuscar in self.equipo2codigo:
            return self.equipo2codigo[equipoAbuscar]

        if codigosObjetivo:
            listaCodigos = codigosObjetivo
        else:
            listaCodigos = self.codigo2equipo.keys()

        distancias = []
        for codigo in listaCodigos:
            for nombreObj in list(self.codigo2equipo[codigo]):
                tupla = (nombreObj, codigo, CompareBagsOfWords(equipoAbuscar, nombreObj))
                if tupla[2] > UMBRALbusquedaDistancia:
                    distancias.append(tupla)

        if distancias:
            resultados = sorted(distancias, key=lambda x: x[2], reverse=True)
            codigoResultado = resultados[0][1]

            self.nuevaTraduccionEquipo2Codigo(equipoAbuscar, codigoResultado)
            return codigoResultado
        else:
            print("No se han encontrado códigos posibles: %s" % equipoAbuscar)
            # TODO: Esto debería ser una excepción
            return None

    def nuevaTraduccionEquipo2Codigo(self, equipo, codigo):
        if equipo in self.equipo2codigo:
            return False

        self.equipo2codigo[equipo] = codigo
        (self.codigo2equipo[codigo]).add(equipo)
        return True

    def nombresJornada(self):
        result = [self.Jornadas[x]['nombre'].replace('JORNADA ', 'J ').replace(" P.", "").replace(",", "")
                  for x in self.Jornadas]

        return result

    def siguientesPartidos(self):
        """
        Devuelve un dict de listas con los partidos pendientes de cada equipo.
        :return:
        """
        LV = ['Local', 'Visitante']
        result = defaultdict(list)
        for j in self.Jornadas:
            if 'pendientes' not in self.Jornadas[j] or not self.Jornadas[j]['pendientes']:
                continue

            for p in self.Jornadas[j]['pendientes']:
                e2l = dict(zip(p['equipos'], LV))
                l2e = dict(zip(LV, p['equipos']))

                for e in p['equipos']:
                    codigo = self.equipo2codigo[e]
                    esLocal = (e2l[e] == 'Local')
                    rival = l2e[OtherTeam(e2l[e])]
                    codRival = self.equipo2codigo[rival]

                    partido = {'nombre': e, 'codigo': codigo, 'jornada': p['jornada'], 'fecha': p['fecha'],
                               'esLocal': esLocal, 'rival': rival, 'codRival': codRival}
                    result[codigo].append(partido)

        return result


def BuscaCalendario(url=URL_BASE, home=None, browser=None, config={}):
    link = None
    indexPage = DescargaPagina(url, home, browser, config)

    index = indexPage['data']

    # print (type(index),index)

    callinks = index.find_all("a", text="Calendario")

    if len(callinks) == 1:
        link = callinks[0]
    else:
        for auxlink in callinks:
            if 'calendario.php' in auxlink['href']:
                link = auxlink
                break
        else:
            raise SystemError("Too many links to Calendario. {}".format(callinks))

    result = MergeURL(url, link['href'])

    return result
