import logging
from collections import defaultdict
from os.path import join

import joblib
from itertools import combinations, product

from Utils.Misc import creaPath
from .Constants import (buildPosCupoIndex, calculaValSuperManager, CUPOS, POSICIONES, SEQCLAVES)

id2key = {'t': 'triples', 'a': 'asistencias', 'r': 'rebotes', 'p': 'puntos', 'v': 'valJornada', 'b': 'broker'}
key2id = {v: k for k, v in id2key.items()}

ig = buildPosCupoIndex()
# {'posicion1': {'Extracomunitario': 0, 'Español': 1, 'normal': 2},
# 'posicion3': {'Extracomunitario': 3, 'Español': 4, 'normal': 5},
# 'posicion5': {'Extracomunitario': 6, 'Español': 7, 'normal': 8}}
ig2posYcupo = {ig[p][c]: (p, c) for p, c in product(POSICIONES, CUPOS)}

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


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

    # TODO: Control de calidad respecto a valSM vs valJornada
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


# TODO: Mensajes ilustrativos en las excepciones
def loadVar(pathFile, mmap_mode=None):
    try:
        res = joblib.load(pathFile, mmap_mode=mmap_mode)
        return res
    except EOFError as exc:
        print("loadVar: exc: {}".format(exc))
    except FileNotFoundError:
        pass

    return None


def varname2fichname(jornada, varname, basedir=".", ext="pickle"):
    return creaPath(basedir, "J%03d-%s.%s" % (jornada, varname, ext))


def comb2Key(comb, jornada, joinerChar="-"):
    return ("J%03d" % jornada) + joinerChar + joinerChar.join("%1d_%1d" % (x, comb[x]) for x in comb)


def keySearchOrderParameter(param):
    if param is None:
        return SEQCLAVES

    try:
        result = [id2key[k.lower()] for k in param]
    except KeyError as exc:
        raise ValueError(
                "keySearchOrderParameter: There was a problem with parameter '%s': bad key %s. Bye." % (param, exc))

    if len(result) != len(id2key):
        missing = ["%s(%s)" % (k, key2id[k]) for k in key2id if k not in result]

        raise ValueError("keySearchOrderParameter: There was a problem with parameter '%s': missing keys : %s. Bye." % (
                param, ", ".join(missing)))

    return result


def plan2filename(plan):
    combDicts = [g['comb'] for g in plan['grupos2check']]
    allCombs = dict()
    for g in combDicts:
        allCombs.update(g)

    planPart = "-".join(["%i_%i" % (k, allCombs[k]) for k in sorted(allCombs.keys())])

    filename = "+".join([("J%03d" % plan['jornada']), plan['equipo'], planPart]) + ".pickle"

    return filename


def indexGroup2Key(ig):
    return "_".join(["".join(map(str, g)) for g in ig])


def seq2name(seqk):
    return "".join([key2id[k] for k in seqk])


def indexPosCupo2str(i):
    pos2abr = {'posicion1': 'B', 'posicion3': 'A', 'posicion5': 'P'}
    cupo2abr = {'Extracomunitario': 'EXT', 'Español': 'ESP', 'normal': 'COM'}

    return "%s-%s" % (pos2abr[ig2posYcupo[i][0]], cupo2abr[ig2posYcupo[i][1]])
