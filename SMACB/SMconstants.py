from babel.numbers import decimal

from Utils.Misc import BadParameters

POSSTLRATIO = 0.44
PARTIDOSTENDENCIA = 5

MAXIMOextranjeros = 2
MINIMOnacionales = 4

PRESUPUESTOinicial = 6500000.0
PRECIOpunto = 70000.0
MINPRECIO = 45000.0

POSICIONES = {'posicion1': 'Base', 'posicion3': 'Alero', 'posicion5': 'Pivot'}
CUPOS = ['Extracomunitario', 'Español', 'normal']

LISTACOMPOS = {'puntos': 'P', 'rebotes': 'REB-T', 'triples': 'T3-C', 'asistencias': 'A'}

BONUSVICTORIA = 1.2

# Orden de busqueda de las claves y de almacenamiento procesado
# SEQCLAVES = ['asistencias', 'triples', 'rebotes', 'puntos', 'valJornada', 'broker']
# SEQCLAVES = ['asistencias', 'rebotes', 'puntos', 'triples', 'valJornada', 'broker']
SEQCLAVES = ['triples', 'asistencias', 'rebotes', 'puntos', 'valJornada', 'broker']

bool2esp = {True: "S", False: "N"}
haGanado2esp = {True: "V", False: "D"}
titular2esp = {True: "T", False: "B"}
local2esp = {True: "L", False: "V"}


def calculaValSuperManager(valoracion, haGanado=False):
    return round(
        decimal.Decimal.from_float(float(valoracion) * (BONUSVICTORIA if (haGanado and (valoracion > 0)) else 1.0)), 2)


def buildPosCupoIndex():
    """
    Genera un diccionario con los índices que corresponden a cada posición y cupo.
    :return:
    """
    indexResult = dict()

    aux = 0
    for pos in POSICIONES:
        indexResult[pos] = dict()
        for cupo in CUPOS:
            indexResult[pos][cupo] = aux
            aux += 1
    return indexResult


def claveGrupo(jornada, index, counters):
    FORMATO = "J%3d-%s"
    return FORMATO % (jornada, "+".join(["%1d_%1d" % (i, c) for i, c in zip(index, counters)]))


def solucion2clave(clave, sol, charsep="#"):
    CLAVESOL = ['valJornada', 'broker', 'puntos', 'asistencias', 'triples', 'rebotes']
    formatos = {'asistencias': "a_%03d", 'triples': "t_%03d", 'rebotes': "r_%03d", 'puntos': "p_%03d",
                'valJornada': "v_%05.2f",
                'broker': "b_%010d"}
    formatoTotal = charsep.join([formatos[k] for k in CLAVESOL])
    valores = [sol[k] for k in CLAVESOL]

    return clave + "#" + (formatoTotal % tuple(valores))


def OtherTeam(team):
    if team == 'Local':
        return 'Visitante'
    elif team == 'Visitante':
        return 'Local'
    else:
        raise BadParameters("OtherTeam: '%s' provided. It only accept 'Visitante' or 'Local'" % team)
