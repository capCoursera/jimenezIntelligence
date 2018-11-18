from collections import defaultdict
from itertools import combinations
from os.path import join
from pathlib import Path

import joblib
import logging
logger = logging.getLogger(__name__)


from SMACB.SMconstants import buildPosCupoIndex, calculaValSuperManager


def agregaJugadores(listaJugs, datosJugs):
    tradKEys = {'puntos': 'P', 'rebotes': 'REB-T', 'triples': 'T3-C', 'asistencias': 'A'}
    result = {'jugs': list(), 'valJornada': 0, 'broker': 0, 'puntos': 0, 'rebotes': 0, 'triples': 0, 'asistencias': 0,
              'Nones': 0}

    for j in listaJugs:
        if j is None:
            result['Nones'] += 1
            continue
        result['jugs'].append(j)
        for k in ['valJornada', 'broker', 'puntos', 'rebotes', 'triples', 'asistencias']:
            targKey = tradKEys.get(k, k)
            result[k] += datosJugs[j].get(targKey, 0)

    return result


def calcFileName(jornada, indice, cantidad, dirbase="/tmp"):
    nomfich = "J%02i-%02i-%02i.lst" % (jornada, indice, cantidad)
    return join(dirbase, nomfich)


def generaCombinacion(datos, n):
    if n == 0:
        return dict()

    numcombs = 0
    colvalores = defaultdict(list)

    for p in combinations(datos, n):
        total = [0] * (len(datos[0]) - 1)
        jugs = [x[0] for x in p]

        for x in p:
            for i in range(len(total)):
                total[i] += x[i + 1]

        numcombs += 1
        colvalores[total[0]].append(["-".join(jugs)] + total[1:])

    return colvalores


def GeneraCombinacionJugs(listaJugs, n):
    """
    Devuelve una lista con las combinaciones de codigos de jugadores tomadas de n en n

    :param listaJugs: Lista de CODIGOS
    :param n: numero de jugadores para usar
    :return: lista con las combinaciones de códigos
    """
    return [i for i in combinations(listaJugs, n)]


def getPartidosJornada(jornada, temporada):
    result = []

    if jornada not in temporada.Calendario.Jornadas:
        return result

    return [temporada.Partidos[x] for x in temporada.Calendario.Jornadas[jornada]['partidos']]


def getPlayersByPosAndCupoJornada(jornada, supermanager, temporada):
    result = []

    if jornada not in supermanager.mercadoJornada:
        return result

    mercadoFin = supermanager.mercado[supermanager.mercadoJornada[jornada]]

    partidos = getPartidosJornada(jornada, temporada)

    minTimestamp = min([x.timestamp for x in partidos])
    idMercadoIni = max([x for x in supermanager.mercado if supermanager.mercado[x].timestamp < minTimestamp])
    mercadoIni = supermanager.mercado[idMercadoIni]

    partidosOk = [x for x in partidos if x.timestamp < mercadoFin.timestamp]
    jugadoresEnPartidos = {y: x.Jugadores[y] for x in partidosOk for y in x.Jugadores}

    dictJugs = defaultdict(dict)

    for j in mercadoIni.PlayerData:
        aux = dict()
        aux['code'] = j
        aux['nombre'] = mercadoIni.PlayerData[j]['nombre']
        aux['cupo'] = mercadoIni.PlayerData[j]['cupo']
        aux['pos'] = mercadoIni.PlayerData[j]['pos']
        aux['precioIni'] = mercadoIni.PlayerData[j]['precio']
        if j not in mercadoFin.PlayerData:
            if j not in jugadoresEnPartidos:
                aux['precioFin'] = aux['precioIni']
                aux['valJornada'] = 0
            else:
                raise KeyError("Clave '%s' (%s) inexistente en mercadoFin y jugo partido" % (
                    j, mercadoIni.PlayerData[j]['nombre']))
        else:
            aux['precioFin'] = mercadoFin.PlayerData[j]['precio']
            aux['valJornada'] = mercadoFin.PlayerData[j]['valJornada']

        aux['broker'] = aux['precioFin'] - aux['precioIni']

        dictJugs[j] = aux

    for j in jugadoresEnPartidos:
        if j in dictJugs:
            for c in ['P', 'A', 'V', 'T3-C', 'REB-T']:
                dictJugs[j][c] = jugadoresEnPartidos[j]['estads'].get(c, 0)
            dictJugs[j]['valSM'] = calculaValSuperManager(jugadoresEnPartidos[j]['estads'].get('V', 0),
                                                          jugadoresEnPartidos[j]['haGanado'])

    indexPosCupo = buildPosCupoIndex()
    result = defaultdict(list)

    for j in dictJugs:
        pos = dictJugs[j]['pos']
        cupo = dictJugs[j]['cupo']
        i = indexPosCupo[pos][cupo]
        result[i].append(j)

    lengrupos = [len(result[x]) for x in range(len(result))]

    return result, dictJugs, lengrupos


def listaPosiciones():
    return [None] * 9


def dumpVar(pathFile, var2dump, compress=False):
    extraVars = dict()
    if compress:
        extraVars['compress'] = ('bz2', 3)

    res = joblib.dump(var2dump, pathFile, **extraVars)

    logger.debug("dumpVar %d", res)
    return res


def loadVar(pathFile, mmap_mode=None):
    if pathFile.exists():
        res = joblib.load(pathFile, mmap_mode=mmap_mode)

        return res

    return None


def varname2fichname(jornada, varname, basedir=".", ext="pickle"):
    return Path.joinpath(Path(basedir), Path("J%03d-%s.%s" % (jornada, varname, ext)))


def comb2Key(comb, jornada, joinerChar="-"):
    return ("J%03d" % jornada) + joinerChar + joinerChar.join("%1d_%1d" % (x, comb[x]) for x in comb)

def ig2Key(ig, jornada, joinerChar="-"):
    return ("J%03d" % jornada) + joinerChar + "+".join("%1d" % x for x in ig)

