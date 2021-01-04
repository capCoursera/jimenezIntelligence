from collections import defaultdict
from copy import copy

import pandas as pd
import sys
from configargparse import ArgumentParser
from math import isnan
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Table, SimpleDocTemplate, Paragraph, TableStyle, Spacer, NextPageTemplate, PageTemplate, \
    Frame, PageBreak

from SMACB.CalendarioACB import NEVER
from SMACB.Constants import LocalVisitante, OtherLoc, haGanado2esp
from SMACB.FichaJugador import TRADPOSICION
from SMACB.TemporadaACB import TemporadaACB, extraeCampoYorden, precalculaOrdenEstadsLiga, COLSESTADSASCENDING, \
    auxEtiqPartido
from Utils.FechaHora import Time2Str

estadGlobales = None
estadGlobalesOrden = None

ESTAD_MEDIA = 0
ESTAD_MEDIANA = 1
ESTAD_DEVSTD = 2
ESTAD_MAX = 3
ESTAD_MIN = 4
ESTAD_COUNT = 5
ESTAD_SUMA = 6

ESTADISTICOEQ = 'mean'
ESTADISTICOJUG = 'mean'

FORMATOCAMPOS = {'entero': {'numero': '{:3.0f}'}, 'float': {'numero': '{:4.2f}'}, }

COLS_IDENTIFIC_JUG = ['competicion', 'temporada', 'CODequipo', 'IDequipo', 'codigo', 'dorsal', 'nombre']


def GENERADORETTIRO(*kargs, **kwargs):
    return lambda f: auxEtiqTiros(f, *kargs, **kwargs)


def GENERADORETREBOTE(*kargs, **kwargs):
    return lambda f: auxEtiqRebotes(f, *kargs, **kwargs)


def GENERADORFECHA(*kargs, **kwargs):
    return lambda f: auxEtFecha(f, *kargs, **kwargs)


def GENERADORTIEMPO(*kargs, **kwargs):
    return lambda f: auxEtiqTiempo(f, *kargs, **kwargs)


INFOESTADSEQ = {
    ('Eq', 'P'): {'etiq': 'PF', 'formato': 'float'},
    ('Rival', 'P'): {'etiq': 'PC', 'formato': 'float'},
    ('Eq', 'POS'): {'etiq': 'Pos', 'formato': 'float'},
    ('Eq', 'OER'): {'etiq': 'OER', 'formato': 'float'},
    ('Rival', 'OER'): {'etiq': 'DER', 'formato': 'float'},
    ('Eq', 'T2'): {'etiq': 'T2', 'generador': GENERADORETTIRO(tiro='2', entero=False, orden=True)},
    ('Eq', 'T3'): {'etiq': 'T3', 'generador': GENERADORETTIRO(tiro='3', entero=False, orden=True)},
    ('Eq', 'TC'): {'etiq': 'TC', 'generador': GENERADORETTIRO(tiro='C', entero=False, orden=True)},
    ('Eq', 'ppTC'): {'etiq': 'P / TC-I', 'formato': 'float'},
    ('Eq', 't3/tc-I'): {'etiq': 'T3-I / TC-I', 'formato': 'float'},
    ('Eq', 'FP-F'): {'etiq': 'F com', 'formato': 'float'},
    ('Eq', 'FP-C'): {'etiq': 'F rec', 'formato': 'float'},
    ('Eq', 'T1'): {'etiq': 'T3', 'generador': GENERADORETTIRO(tiro='3', entero=False, orden=True)},
    ('Eq', 'REB'): {'etiq': 'Rebs', 'ancho': 17, 'generador': GENERADORETREBOTE(entero=False, orden=True)},
    ('Eq', 'EffRebD'): {'etiq': 'F rec', 'formato': 'float'},
    ('Eq', 'EffRebO'): {'etiq': 'F rec', 'formato': 'float'},
    ('Eq', 'A'): {'formato': 'float'},
    ('Eq', 'BP'): {'formato': 'float'},
    ('Eq', 'BR'): {'formato': 'float'},
    ('Eq', 'A/BP'): {'formato': 'float'},
    ('Eq', 'A/TC-C'): {'etiq': 'A/Can', 'formato': 'float'},

    ('Rival', 'T2'): {'generador': GENERADORETTIRO(tiro='2', entero=False, orden=True)},
    ('Rival', 'T3'): {'generador': GENERADORETTIRO(tiro='3', entero=False, orden=True)},
    ('Rival', 'TC'): {'generador': GENERADORETTIRO(tiro='3', entero=False, orden=True)},
    ('Rival', 'ppTC'): {'etiq': 'P / TC-I', 'formato': 'float'},
    ('Rival', 't3/tc-I'): {'etiq': 'T3-I / TC-I', 'formato': 'float'},
    ('Rival', 'T1'): {'etiq': 'TL', 'generador': GENERADORETTIRO(tiro='3', entero=False, orden=True)},
    ('Rival', 'REB'): {'etiq': 'Rebs', 'ancho': 17, 'generador': GENERADORETREBOTE(entero=False, orden=True)},
    ('Rival', 'A'): {'formato': 'float'},
    ('Rival', 'BP'): {'formato': 'float'},
    ('Rival', 'BR'): {'formato': 'float'},
    ('Rival', 'A/BP'): {'formato': 'float'},
    ('Rival', 'A/TC-C'): {'etiq': 'A/Can', 'formato': 'float'},
}

INFOTABLAJUGS = {
    ('Jugador', 'dorsal'): {'etiq': 'D', 'ancho': 3},
    ('Jugador', 'nombre'): {'etiq': 'Nombre', 'ancho': 22, 'alignment': 'LEFT'},
    ('Jugador', 'pos'): {'etiq': 'Pos', 'ancho': 4, 'alignment': 'CENTER'},
    ('Jugador', 'altura'): {'etiq': 'Alt', 'ancho': 5},
    ('Jugador', 'licencia'): {'etiq': 'Lic', 'ancho': 5, 'alignment': 'CENTER'},
    ('Trayectoria', 'Acta'): {'etiq': 'Cv', 'ancho': 3, 'formato': 'entero'},
    ('Trayectoria', 'Jugados'): {'etiq': 'Ju', 'ancho': 3, 'formato': 'entero'},
    ('Trayectoria', 'Titular'): {'etiq': 'Tt', 'ancho': 3, 'formato': 'entero'},
    ('Trayectoria', 'Vict'): {'etiq': 'Vc', 'ancho': 3, 'formato': 'entero'},

    ('Promedios', 'etSegs'): {'etiq': 'Min', 'ancho': 7, 'generador': GENERADORTIEMPO(col='Segs')},
    ('Promedios', 'P'): {'etiq': 'P', 'ancho': 7, 'formato': 'float'},
    ('Promedios', 'etiqT2'): {'etiq': 'T2', 'ancho': 19, 'generador': GENERADORETTIRO('2', entero=False)},
    ('Promedios', 'etiqT3'): {'etiq': 'T3', 'ancho': 19, 'generador': GENERADORETTIRO(tiro='3', entero=False)},
    ('Promedios', 'etiqTC'): {'etiq': 'TC', 'ancho': 19, 'generador': GENERADORETTIRO('C', False)},
    ('Promedios', 'ppTC'): {'etiq': 'P/TC', 'ancho': 6, 'formato': 'float'},
    ('Promedios', 'FP-F'): {'etiq': 'F com', 'ancho': 6, 'formato': 'float'},
    ('Promedios', 'FP-C'): {'etiq': 'F rec', 'ancho': 6, 'formato': 'float'},
    ('Promedios', 'etiqT1'): {'etiq': 'TL', 'ancho': 19, 'generador': GENERADORETTIRO('1', False)},
    ('Promedios', 'etRebs'): {'etiq': 'Rebs', 'ancho': 17, 'generador': GENERADORETREBOTE(entero=False)},
    ('Promedios', 'A'): {'etiq': 'A', 'ancho': 6, 'formato': 'float'},
    ('Promedios', 'BP'): {'etiq': 'BP', 'ancho': 6, 'formato': 'float'},
    ('Promedios', 'BR'): {'etiq': 'BR', 'ancho': 6, 'formato': 'float'},
    ('Promedios', 'TAP-F'): {'etiq': 'Tap', 'ancho': 6, 'formato': 'float'},
    ('Promedios', 'TAP-C'): {'etiq': 'Tp R', 'ancho': 6, 'formato': 'float'},

    ('Totales', 'etSegs'): {'etiq': 'Min', 'ancho': 8, 'generador': GENERADORTIEMPO(col='Segs')},
    ('Totales', 'P'): {'etiq': 'P', 'ancho': 6, 'formato': 'entero'},
    ('Totales', 'etiqT2'): {'etiq': 'T2', 'ancho': 19, 'generador': GENERADORETTIRO('2', entero=True)},
    ('Totales', 'etiqT3'): {'etiq': 'T3', 'ancho': 19, 'generador': GENERADORETTIRO('3', entero=True)},
    ('Totales', 'etiqTC'): {'etiq': 'TC', 'ancho': 19, 'generador': GENERADORETTIRO('C', entero=True)},
    ('Totales', 'ppTC'): {'etiq': 'P/TC', 'ancho': 6, 'formato': 'float'},
    ('Totales', 'FP-F'): {'etiq': 'F com', 'ancho': 6, 'formato': 'entero'},
    ('Totales', 'FP-C'): {'etiq': 'F rec', 'ancho': 6, 'formato': 'entero'},
    ('Totales', 'etiqT1'): {'etiq': 'TL', 'ancho': 19, 'generador': GENERADORETTIRO('1', entero=True)},
    ('Totales', 'etRebs'): {'etiq': 'Rebs', 'ancho': 17, 'generador': GENERADORETREBOTE(entero=True)},
    ('Totales', 'A'): {'etiq': 'A', 'ancho': 6, 'formato': 'entero'},
    ('Totales', 'BP'): {'etiq': 'BP', 'ancho': 6, 'formato': 'entero'},
    ('Totales', 'BR'): {'etiq': 'BR', 'ancho': 6, 'formato': 'entero'},
    ('Totales', 'TAP-F'): {'etiq': 'Tap', 'ancho': 6, 'formato': 'entero'},
    ('Totales', 'TAP-C'): {'etiq': 'Tp R', 'ancho': 6, 'formato': 'entero'},

    ('UltimoPart', 'etFecha'): {'etiq': 'Fecha', 'ancho': 6, 'generador': GENERADORFECHA(col='Fecha'),
                                'alignment': 'CENTER'},
    ('UltimoPart', 'Partido'): {'etiq': 'Rival', 'ancho': 22, 'alignment': 'LEFT'},
    ('UltimoPart', 'resultado'): {'etiq': 'Vc', 'ancho': 3, 'alignment': 'CENTER'},
    ('UltimoPart', 'titular'): {'etiq': 'Tt', 'ancho': 3, 'alignment': 'CENTER'},
    ('UltimoPart', 'etSegs'): {'etiq': 'Min', 'ancho': 6, 'generador': GENERADORTIEMPO(col='Segs')},
    ('UltimoPart', 'P'): {'etiq': 'P', 'ancho': 4, 'formato': 'entero'},
    ('UltimoPart', 'etiqT2'): {'etiq': 'T2', 'ancho': 14, 'generador': GENERADORETTIRO('2', entero=True)},
    ('UltimoPart', 'etiqT3'): {'etiq': 'T3', 'ancho': 14, 'generador': GENERADORETTIRO('3', entero=True)},
    ('UltimoPart', 'etiqTC'): {'etiq': 'TC', 'ancho': 14, 'generador': GENERADORETTIRO('C', entero=True)},
    ('UltimoPart', 'ppTC'): {'etiq': 'P/TC', 'ancho': 6, 'formato': 'float'},
    ('UltimoPart', 'FP-F'): {'etiq': 'F com', 'ancho': 6, 'formato': 'entero'},
    ('UltimoPart', 'FP-C'): {'etiq': 'F rec', 'ancho': 6, 'formato': 'entero'},
    ('UltimoPart', 'etiqT1'): {'etiq': 'TL', 'ancho': 14, 'generador': GENERADORETTIRO('1', entero=True)},
    ('UltimoPart', 'etRebs'): {'etiq': 'Rebs', 'ancho': 10, 'generador': GENERADORETREBOTE(entero=True)},
    ('UltimoPart', 'A'): {'etiq': 'A', 'ancho': 4, 'formato': 'entero'},
    ('UltimoPart', 'BP'): {'etiq': 'BP', 'ancho': 4, 'formato': 'entero'},
    ('UltimoPart', 'BR'): {'etiq': 'BR', 'ancho': 4, 'formato': 'entero'},
    ('UltimoPart', 'TAP-C'): {'etiq': 'Tap', 'ancho': 4, 'formato': 'entero'},
    ('UltimoPart', 'TAP-F'): {'etiq': 'Tp R', 'ancho': 4, 'formato': 'entero'},
}

ESTILOS = getSampleStyleSheet()


def auxCalculaBalanceStr(record):
    victorias = record.get('V', 0)
    derrotas = record.get('D', 0)
    texto = f"{victorias}-{derrotas}"

    return texto


def auxEtiqRebotes(df, entero: bool = True) -> str:
    if isnan(df['R-D']):
        return "-"

    formato = "{:3}+{:3} {:3}" if entero else "{:6.2f}+{:6.2f} {:6.2f}"

    valores = [int(v) if entero else v for v in [df['R-D'], df['R-O'], df['REB-T']]]

    result = formato.format(*valores)

    return result


def auxEtiqTiempo(df, col='Segs'):
    t = df[col]
    if isnan(t):
        return "-"

    mins = t // 60
    segs = t % 60

    result = f"{mins:.0f}:{segs:02.0f}"

    return result


def auxEtiqTiros(df, tiro, entero=True):
    formato = "{:3}/{:3} {:6.2f}%" if entero else "{:6.2f}/{:6.2f} {:6.2f}%"

    etTC = f"T{tiro}-C"
    etTI = f"T{tiro}-I"
    etTpc = f"T{tiro}%"

    if df[etTI] == 0.0 or isnan(df[etTI]):
        return "-"

    valores = [int(v) if entero else v for v in [df[etTC], df[etTI]]] + [df[etTpc]]

    result = formato.format(*valores)

    return result


def auxEtFecha(f, col, formato="%d-%m"):
    if f is None:
        return "-"

    dato = f[col]
    result = dato.strftime(formato)

    return result


def auxGeneraTabla(dfDatos, collist, colSpecs, estiloTablaBaseOps, formatos=None, charWidth=10.0):
    dfColList = []
    filaCab = []
    anchoCols = []
    tStyle = TableStyle(estiloTablaBaseOps)

    if formatos is None:
        formatos = dict()

    for i, colkey in enumerate(collist):
        level, etiq = colkey
        colSpec = colSpecs.get(colkey, {})
        newCol = dfDatos[level].apply(colSpec['generador'], axis=1) if 'generador' in colSpec else dfDatos[[colkey]]

        if 'formato' in colSpec:
            etiqFormato = colSpec['formato']
            if colSpec['formato'] not in formatos:
                raise KeyError(
                    f"auxGeneraTabla: columna '{colkey}': formato '{etiqFormato}' desconocido. Formatos conocidos: {formatos}")
            formatSpec = formatos[etiqFormato]

            if 'numero' in formatSpec:
                newCol = newCol.apply(lambda c: c.map(formatSpec['numero'].format))

        newEtiq = colSpec.get('etiq', etiq)
        newAncho = colSpec.get('ancho', 10) * charWidth

        dfColList.append(newCol)
        filaCab.append(newEtiq)
        anchoCols.append(newAncho)
        if 'alignment' in colSpec:
            newCmdStyle = ["ALIGN", (i, 1), (i, -1), colSpec['alignment']]
            tStyle.add(*newCmdStyle)

    datosAux = pd.concat(dfColList, axis=1, join='outer', names=filaCab)

    datosTabla = [filaCab] + datosAux.to_records(index=False, column_dtypes='object').tolist()

    t = Table(datosTabla, style=tStyle, colWidths=anchoCols)

    return t


def cabeceraPortada(partido, tempData):
    datosLocal = partido['equipos']['Local']
    datosVisit = partido['equipos']['Visitante']
    compo = partido['cod_competicion']
    edicion = partido['cod_edicion']
    j = partido['jornada']
    fh = Time2Str(partido['fecha'])

    style = ParagraphStyle('cabStyle', align='center', fontName='Helvetica', fontSize=20, leading=22, )

    cadenaCentral = Paragraph(
        f"<para align='center' fontName='Helvetica' fontSize=20 leading=22><b>{compo}</b> {edicion} - J: <b>{j}</b><br/>{fh}</para>",
        style)

    cabLocal = datosCabEquipo(datosLocal, tempData, partido['fecha'])
    cabVisit = datosCabEquipo(datosVisit, tempData, partido['fecha'])

    tStyle = TableStyle([('BOX', (0, 0), (-1, -1), 2, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                         ('GRID', (0, 0), (-1, -1), 0.5, colors.black)])
    t = Table(data=[[cabLocal, cadenaCentral, cabVisit]], colWidths=[60 * mm, 80 * mm, 60 * mm], style=tStyle)  #

    return t


def cargaTemporada(fname):
    result = TemporadaACB()
    result.cargaTemporada(fname)

    return result


def datosCabEquipo(datosEq, tempData, fecha):
    # TODO: Imagen (descargar imagen de escudo y plantarla)
    nombre = datosEq['nombcorto']

    clasifAux = tempData.clasifEquipo(datosEq['abrev'], fecha)
    clasifStr = auxCalculaBalanceStr(clasifAux)

    result = [Paragraph(f"<para align='center' fontSize='16' leading='17'><b>{nombre}</b></para>"),
              Paragraph(f"<para align='center' fontSize='14'>{clasifStr}</para>")]

    return result


def recuperaEstadsGlobales(tempData):
    global estadGlobales, estadGlobalesOrden

    if estadGlobales is None:
        estadGlobales = tempData.dfEstadsLiga()
        estadGlobalesOrden = precalculaOrdenEstadsLiga(estadGlobales, COLSESTADSASCENDING)


def datosEstadsEquipoPortada(tempData: TemporadaACB, eq: str):
    recuperaEstadsGlobales(tempData)

    if eq not in estadGlobales.index:
        valCorrectos = ", ".join(sorted(estadGlobales.index))
        raise KeyError(f"extraeCampoYorden: equipo (abr) '{eq}' desconocido. Equipos validos: {valCorrectos}")

    estadsEq = estadGlobales.loc[eq]
    estadsEqOrden = estadGlobalesOrden.loc[eq]

    # targAbrev = list(tempData.Calendario.abrevsEquipo(eq).intersection(estadGlobales.keys()))[0]
    targAbrev = list(tempData.Calendario.abrevsEquipo(eq).intersection(estadGlobales.index))[0]

    pFav, pFavOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'P', ESTADISTICOEQ)
    pCon, pConOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'P', ESTADISTICOEQ)

    pos, posOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'POS', ESTADISTICOEQ)
    OER, OEROrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'OER', ESTADISTICOEQ)
    OERpot, OERpotOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'OERpot', ESTADISTICOEQ)
    DER, DEROrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'OER', ESTADISTICOEQ)

    T2C, T2COrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'T2-C', ESTADISTICOEQ)
    T2I, T2IOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'T2-I', ESTADISTICOEQ)
    T2pc, T2pcOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'T2%', ESTADISTICOEQ)
    T3C, T3COrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'T3-C', ESTADISTICOEQ)
    T3I, T3IOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'T3-I', ESTADISTICOEQ)
    T3pc, T3pcOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'T3%', ESTADISTICOEQ)
    TCC, TCCOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'TC-C', ESTADISTICOEQ)
    TCI, TCIOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'TC-I', ESTADISTICOEQ)
    TCpc, TCpcOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'TC%', ESTADISTICOEQ)
    ppTC, ppTCOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'ppTC', ESTADISTICOEQ)
    ratT3, ratT3Ord = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 't3/tc-I', ESTADISTICOEQ)
    Fcom, FcomOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'FP-F', ESTADISTICOEQ)
    Frec, FrecOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'FP-F', ESTADISTICOEQ)
    T1C, T1COrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'T1-C', ESTADISTICOEQ)
    T1I, T1IOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'T1-I', ESTADISTICOEQ)
    T1pc, T1pcOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'T1%', ESTADISTICOEQ)

    RebD, RebDOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'R-D', ESTADISTICOEQ)
    RebO, RebOOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'R-O', ESTADISTICOEQ)
    RebT, RebTOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'REB-T', ESTADISTICOEQ)
    EffRebD, EffRebDOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'EffRebD', ESTADISTICOEQ)
    EffRebO, EffRebOOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'EffRebO', ESTADISTICOEQ)

    A, AOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'A', ESTADISTICOEQ)
    BP, BPOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'BP', ESTADISTICOEQ)
    BR, BROrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'BR', ESTADISTICOEQ)
    ApBP, ApBPOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'A/BP', ESTADISTICOEQ)
    ApTCC, ApTCCOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Eq', 'A/TC-C', ESTADISTICOEQ)

    ### Valores del equipo rival

    rT2C, rT2COrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'T2-C', ESTADISTICOEQ)
    rT2I, rT2IOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'T2-I', ESTADISTICOEQ)
    rT2pc, rT2pcOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'T2%', ESTADISTICOEQ)
    rT3C, rT3COrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'T3-C', ESTADISTICOEQ)
    rT3I, rT3IOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'T3-I', ESTADISTICOEQ)
    rT3pc, rT3pcOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'T3%', ESTADISTICOEQ)
    rTCC, rTCCOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'TC-C', ESTADISTICOEQ)
    rTCI, rTCIOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'TC-I', ESTADISTICOEQ)
    rTCpc, rTCpcOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'TC%', ESTADISTICOEQ)
    rppTC, rppTCOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'ppTC', ESTADISTICOEQ)
    rratT3, rratT3Ord = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 't3/tc-I', ESTADISTICOEQ)
    rT1C, rT1COrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'T1-C', ESTADISTICOEQ)
    rT1I, rT1IOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'T1-I', ESTADISTICOEQ)
    rT1pc, rT1pcOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'T1%', ESTADISTICOEQ)

    rRebD, rRebDOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'R-D', ESTADISTICOEQ)
    rRebO, rRebOOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'R-O', ESTADISTICOEQ)
    rRebT, rRebTOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'REB-T', ESTADISTICOEQ)

    rA, rAOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'A', ESTADISTICOEQ)
    rBP, rBPOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'BP', ESTADISTICOEQ)
    rBR, rBROrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'BR', ESTADISTICOEQ)
    rApBP, rApBPOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'A/BP', ESTADISTICOEQ)
    rApTCC, rApTCCOrd = extraeCampoYorden(estadsEq, estadsEqOrden, 'Rival', 'A/TC-C', ESTADISTICOEQ)

    ###

    resultEq = f"""
<b>PF</b>:&nbsp;{pFav:.2f}({pFavOrd}) <b>/</b> <b>PC</b>:&nbsp;{pCon:.2f}({pConOrd}) <b>/</b>
<b>Pos</b>:&nbsp;{pos:.2f}({posOrd}) <b>/</b> <b>OER</b>:&nbsp;{OER:.2f}({OEROrd}) <b>/</b> <b>DER</b>:&nbsp;{DER:.2f}({DEROrd}) <b>/</b>
<b>T2</b>:&nbsp;{T2C:.2f}({T2IOrd})/{T2I:.2f}({T2IOrd})&nbsp;{T2pc:.2f}%({T2pcOrd}) <b>/</b> <b>T3</b>:&nbsp;{T3C:.2f}({T3IOrd})/{T3I:.2f}({T3IOrd})&nbsp;{T3pc:.2f}%({T3pcOrd}) <b>/</b>
<b>TC</b>:&nbsp;{TCC:.2f}({TCIOrd})/{TCI:.2f}({TCIOrd})&nbsp;{TCpc:.2f}%({TCpcOrd})&nbsp;<b>/</b> <b>P&nbsp;por&nbsp;TC-I</b>:&nbsp;{ppTC:.2f}({ppTCOrd}) <b>T3-I/TC-I</b>&nbsp;{ratT3:.2f}%({ratT3Ord}) <b>/</b>
<b>F&nbsp;com</b>:&nbsp;{Fcom:.2f}({FcomOrd})  <b>/</b> <b>F&nbsp;rec</b>:&nbsp;{Frec:.2f}({FrecOrd})  <b>/</b> <b>TL</b>:&nbsp;{T1C:.2f}({T1COrd})/{T1I:.2f}({T1IOrd})&nbsp;{T1pc:.2f}%({T1pcOrd}) <b>/</b>
<b>Reb</b>:&nbsp;{RebD:.2f}({RebDOrd})+{RebO:.2f}({RebOOrd}) {RebT:.2f}({RebTOrd}) <b>/</b> <b>EffRD</b>:&nbsp;{EffRebD:.2f}({EffRebDOrd}) <b>EffRO</b>:&nbsp;{EffRebO:.2f}({EffRebOOrd}) <b>/</b>
<b>A</b>:&nbsp;{A:.2f}({AOrd}) <b>/</b> <b>BP</b>:&nbsp;{BP:.2f}({BPOrd}) <b>/</b> <b>BR</b>:&nbsp;{BR:.2f}({BROrd}) <b>/</b> <b>A/BP</b>:&nbsp;{ApBP:.2f}({ApBPOrd}) <b>/</b> <b>A/Can</b>:&nbsp;{ApTCC:.2f}({ApTCCOrd})<br/>

<B>RIVAL</B><br/>
<b>T2</b>:&nbsp;{rT2C:.2f}({rT2IOrd})/{rT2I:.2f}({rT2IOrd})&nbsp;{rT2pc:.2f}%({rT2pcOrd}) <b>/</b> <b>T3</b>:&nbsp;{rT3C:.2f}({rT3IOrd})/{rT3I:.2f}({rT3IOrd})&nbsp;{rT3pc:.2f}%({rT3pcOrd}) <b>/</b>
<b>TC</b>:&nbsp;{rTCC:.2f}({rTCIOrd})/{rTCI:.2f}({rTCIOrd})&nbsp;{rTCpc:.2f}%({rTCpcOrd}) <b>/</b> <b>P&nbsp;por&nbsp;TC-I</b>:&nbsp;{rppTC:.2f}({rppTCOrd}) <b>T3-I/TC-I</b>&nbsp;{rratT3:.2f}%({rratT3Ord}) <b>/</b>
<b>TL</b>:&nbsp;{rT1C:.2f}({rT1COrd})/{rT1I:.2f}({rT1IOrd})&nbsp;{rT1pc:.2f}%({rT1pcOrd}) <b>/</b> <b>Reb</b>:&nbsp;{rRebD:.2f}({rRebDOrd})+{rRebO:.2f}({rRebOOrd})&nbsp;{rRebT:.2f}({rRebTOrd}) <b>/</b>
<b>A</b>:&nbsp;{rA:.2f}({rAOrd}) <b>/</b> <b>BP</b>:&nbsp;{rBP:.2f}({rBPOrd}) <b>/</b> <b>BR</b>:&nbsp;{rBR:.2f}({rBROrd}) <b>/</b> <b>A/BP</b>:&nbsp;{rApBP:.2f}({rApBPOrd}) <b>/</b> <b>A/Can</b>:&nbsp;{rApTCC:.2f}({rApTCCOrd})
"""

    return resultEq


def estadsEquipoPortada(tempData: TemporadaACB, abrevs: list):
    datLocal = datosEstadsEquipoPortada(tempData, abrevs[0])
    datVisitante = datosEstadsEquipoPortada(tempData, abrevs[1])

    style = ParagraphStyle('Normal', align='left', fontName='Helvetica', fontSize=10, leading=11, )

    parLocal = Paragraph(datLocal, style)
    parVisit = Paragraph(datVisitante, style)

    tStyle = TableStyle([('BOX', (0, 0), (-1, -1), 2, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                         ('GRID', (0, 0), (-1, -1), 0.5, colors.black)])
    t = Table(data=[[parLocal, parVisit]], colWidths=[100 * mm, 100 * mm], style=tStyle)

    return t


def estadsEquipoPortada_df(tempData: TemporadaACB, abrevs: list):
    datLocal = datosEstadsEquipoPortada(tempData, abrevs[0])
    datVisitante = datosEstadsEquipoPortada(tempData, abrevs[1])

    style = ParagraphStyle('Normal', align='left', fontName='Helvetica', fontSize=10, leading=11, )

    parLocal = Paragraph(datLocal, style)
    parVisit = Paragraph(datVisitante, style)

    tStyle = TableStyle([('BOX', (0, 0), (-1, -1), 2, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                         ('GRID', (0, 0), (-1, -1), 0.5, colors.black)])
    t = Table(data=[[parLocal, parVisit]], colWidths=[100 * mm, 100 * mm], style=tStyle)

    return t


def datosJugadores(tempData: TemporadaACB, abrEq, partJug):
    COLS_TRAYECT_TEMP_orig_names = ['enActa', 'haJugado', 'esTitular', 'haGanado', ]
    COLS_TRAYECT_TEMP_orig = [(col, 'sum') for col in COLS_TRAYECT_TEMP_orig_names]
    COLS_TRAYECT_TEMP = ['Acta', 'Jugados', 'Titular', 'Vict']
    COLS_FICHA = ['id', 'alias', 'pos', 'altura', 'licencia']
    VALS_ESTAD_JUGADOR = ['A', 'BP', 'BR', 'FP-C', 'FP-F', 'P', 'ppTC', 'R-D', 'R-O', 'REB-T', 'Segs', 'T1-C', 'T1-I',
                          'T1%', 'T2-C', 'T2-I', 'T2%', 'T3-C', 'T3-I', 'T3%', 'TC-I', 'TC-C', 'TC%', 'PTC', 'TAP-C',
                          'TAP-F']

    COLS_ESTAD_PROM = [(col, ESTADISTICOJUG) for col in VALS_ESTAD_JUGADOR]
    COLS_ESTAD_TOTAL = [(col, 'sum') for col in VALS_ESTAD_JUGADOR]

    abrevsEq = tempData.Calendario.abrevsEquipo(abrEq)
    keyDorsal = lambda d: -1 if d == '00' else int(d)

    urlPartsJug = [p.url for p in partJug]

    auxDF = tempData.extraeDataframeJugadores(listaURLPartidos=urlPartsJug)

    jugDF = auxDF.loc[auxDF['CODequipo'].isin(abrevsEq)]

    estadsJugDF = tempData.dfEstadsJugadores(jugDF, abrEq=abrEq)
    fichasJugadores = tempData.dataFrameFichasJugadores()
    fichasJugadores.posicion = fichasJugadores.posicion.map(TRADPOSICION)

    trayectTemp = estadsJugDF[COLS_TRAYECT_TEMP_orig]
    trayectTemp.columns = pd.Index(COLS_TRAYECT_TEMP)

    identifJug = pd.concat([estadsJugDF['Jugador'][COLS_IDENTIFIC_JUG], fichasJugadores[COLS_FICHA]], axis=1,
                           join="inner")

    estadsPromedios = estadsJugDF[COLS_ESTAD_PROM].droplevel(1, axis=1)
    estadsTotales = estadsJugDF[COLS_ESTAD_TOTAL].droplevel(1, axis=1)
    datosUltPart = jugDF.sort_values('Fecha').groupby('codigo').tail(n=1).set_index('codigo', drop=False)
    datosUltPart['Partido'] = datosUltPart.apply(
        lambda p: auxEtiqPartido(tempData, p['CODrival'], esLocal=p['esLocal']), axis=1)

    dataFramesAJuntar = {'Jugador': identifJug, 'Trayectoria': trayectTemp,
                         'Promedios': estadsPromedios,  # .drop(columns=COLS_IDENTIFIC_JUG + COLS_TRAYECT_TEMP)
                         'Totales': estadsTotales,  # .drop(columns=COLS_IDENTIFIC_JUG + COLS_TRAYECT_TEMP)
                         'UltimoPart': datosUltPart}  # .drop(columns=COLS_IDENTIFIC_JUG)
    result = pd.concat(dataFramesAJuntar.values(), axis=1, join='outer', keys=dataFramesAJuntar.keys()).sort_values(
        ('Jugador', 'dorsal'), key=lambda c: c.map(keyDorsal))

    return result


def datosTablaLiga(tempData: TemporadaACB):
    FONTSIZE = 10
    CELLPAD = 3 * mm

    estCelda = ParagraphStyle('celTabLiga', ESTILOS.get('Normal'), fontSize=FONTSIZE, leading=FONTSIZE,
                              alignment=TA_CENTER, borderPadding=CELLPAD, spaceAfter=CELLPAD, spaceBefore=CELLPAD)
    ESTILOS.add(estCelda)

    # Precalcula el contenido de la tabla
    auxTabla = defaultdict(dict)
    for jId, jDatos in tempData.Calendario.Jornadas.items():
        for part in jDatos['partidos']:
            idLocal = list(tempData.Calendario.tradEquipos['c2i'][part['equipos']['Local']['abrev']])[0]
            idVisitante = list(tempData.Calendario.tradEquipos['c2i'][part['equipos']['Visitante']['abrev']])[0]
            auxTabla[idLocal][idVisitante] = part
        for part in jDatos['pendientes']:
            idLocal = list(tempData.Calendario.tradEquipos['c2i'][part['equipos']['Local']['abrev']])[0]
            idVisitante = list(tempData.Calendario.tradEquipos['c2i'][part['equipos']['Visitante']['abrev']])[0]
            auxTabla[idLocal][idVisitante] = part

    # En la clasificación está el contenido de los márgenes, de las diagonales y el orden de presentación
    # de los equipos
    clasif = tempData.clasifLiga()
    seqIDs = [(pos, list(equipo['idEq'])[0]) for pos, equipo in enumerate(clasif)]

    datosTabla = []
    cabFila = [Paragraph('<b>Casa/Fuera</b>', style=estCelda)] + [
        Paragraph('<b>' + list(clasif[pos]['abrevsEq'])[0] + '</b>', style=estCelda) for pos, _ in seqIDs] + [
                  Paragraph('<b>Local</b>', style=estCelda)]
    datosTabla.append(cabFila)
    for pos, idLocal in seqIDs:
        fila = []
        nombreCorto = sorted(clasif[pos]['nombresEq'], key=lambda n: len(n))[0]
        abrev = list(clasif[pos]['abrevsEq'])[0]
        fila.append(Paragraph(f"{nombreCorto} (<b>{abrev}</b>)", style=estCelda))
        for _, idVisit in seqIDs:
            if idLocal != idVisit:
                part = auxTabla[idLocal][idVisit]
                fecha = part['fecha'].strftime("%d-%m") if (('fecha' in part) and (part['fecha'] != NEVER)) else 'TBD'
                jornada = part['jornada']

                texto = f"J:{jornada}<br/>@{fecha}"
                if not part['pendiente']:
                    pURL = part['url']
                    pTempFecha = tempData.Partidos[pURL].fechaPartido
                    fecha = pTempFecha.strftime("%d-%m")
                    pLocal = part['equipos']['Local']['puntos']
                    pVisit = part['equipos']['Visitante']['puntos']
                    texto = f"J:{jornada}<br/><b>{pLocal}-{pVisit}</b>"
            else:
                auxTexto = auxCalculaBalanceStr(clasif[pos])
                texto = f"<b>{auxTexto}</b>"
            fila.append(Paragraph(texto, style=estCelda))

        fila.append(Paragraph(auxCalculaBalanceStr(clasif[pos]['CasaFuera']['Local']), style=estCelda))
        datosTabla.append(fila)

    filaBalFuera = [Paragraph('<b>Visitante</b>', style=estCelda)]
    for pos, idLocal in seqIDs:
        filaBalFuera.append(Paragraph(auxCalculaBalanceStr(clasif[pos]['CasaFuera']['Visitante']), style=estCelda))
    filaBalFuera.append([])
    datosTabla.append(filaBalFuera)

    return datosTabla


def listaEquipos(tempData):
    print("Abreviatura -> nombre(s) equipo")
    for abr in sorted(tempData.Calendario.tradEquipos['c2n']):
        listaEquiposAux = sorted(tempData.Calendario.tradEquipos['c2n'][abr], key=lambda x: (len(x), x), reverse=True)
        listaEquiposStr = ",".join(listaEquiposAux)
        print(f'{abr}: {listaEquiposStr}')
    sys.exit(0)


def datosMezclaPartJugados(tempData, abrevs, partsIzda, partsDcha):
    partsIzdaAux = copy(partsIzda)
    partsDchaAux = copy(partsDcha)
    lineas = list()

    abrIzda, abrDcha = abrevs
    abrevsIzda = tempData.Calendario.abrevsEquipo(abrIzda)
    abrevsDcha = tempData.Calendario.abrevsEquipo(abrDcha)

    while (len(partsIzdaAux) > 0) or (len(partsDchaAux) > 0):
        bloque = dict()

        try:
            priPartIzda = partsIzdaAux[0]
        except IndexError:
            bloque['J'] = partsDchaAux[0].Jornada
            bloque['dcha'] = partidoTrayectoria(partsDchaAux.pop(0), abrevsDcha, tempData)
            lineas.append(bloque)
            continue
        try:
            priPartDcha = partsDchaAux[0]
        except IndexError:
            bloque['J'] = priPartIzda.Jornada
            bloque['izda'] = partidoTrayectoria(partsIzdaAux.pop(0), abrevsIzda, tempData)
            lineas.append(bloque)
            continue

        bloque = dict()
        if priPartIzda.Jornada == priPartDcha.Jornada:
            bloque['J'] = priPartIzda.Jornada
            bloque['izda'] = partidoTrayectoria(partsIzdaAux.pop(0), abrevsIzda, tempData)
            bloque['dcha'] = partidoTrayectoria(partsDchaAux.pop(0), abrevsDcha, tempData)
        else:
            if (priPartIzda.fechaPartido, priPartIzda.Jornada) < (priPartDcha.fechaPartido, priPartDcha.Jornada):
                bloque['J'] = priPartIzda.Jornada
                bloque['izda'] = partidoTrayectoria(partsIzdaAux.pop(0), abrevsIzda, tempData)
            else:
                bloque['J'] = priPartDcha.Jornada
                bloque['dcha'] = partidoTrayectoria(partsDchaAux.pop(0), abrevsDcha, tempData)

        lineas.append(bloque)

    return lineas


def paginasJugadores(tempData, abrEqs, juIzda, juDcha):
    result = []

    if len(juIzda):
        datosIzda = datosJugadores(tempData, abrEqs[0], juIzda)
        tablasJugadIzda = tablaJugadoresEquipo(datosIzda)

        result.append(NextPageTemplate('apaisada'))
        result.append(PageBreak())
        for t in tablasJugadIzda:
            result.append(Spacer(100 * mm, 2 * mm))
            result.append(t)

    if len(juDcha):
        datosIzda = datosJugadores(tempData, abrEqs[1], juDcha)
        tablasJugadIzda = tablaJugadoresEquipo(datosIzda)

        result.append(NextPageTemplate('apaisada'))
        result.append(PageBreak())
        for t in tablasJugadIzda:
            result.append(Spacer(100 * mm, 2 * mm))
            result.append(t)

    return result


def partidoTrayectoria(partido, abrevs, datosTemp):
    # Cadena de información del partido
    strFecha = partido.fechaPartido.strftime("%d-%m")
    abrEq = list(abrevs.intersection(partido.DatosSuministrados['participantes']))[0]
    abrRival = list(partido.DatosSuministrados['participantes'].difference(abrevs))[0]
    locEq = partido.DatosSuministrados['abrev2loc'][abrEq]
    locRival = OtherLoc(locEq)
    textRival = auxEtiqPartido(datosTemp, abrRival, locEq=locEq, usaLargo=False)
    clasifAux = datosTemp.clasifEquipo(abrRival, partido.fechaPartido)
    clasifStr = auxCalculaBalanceStr(clasifAux)
    strRival = f"{strFecha}: {textRival} ({clasifStr})"

    # Cadena del resultado del partido
    # TODO: Esto debería ir en HTML o Markup correspondiente
    marcador = {loc: str(partido.DatosSuministrados['resultado'][loc]) for loc in LocalVisitante}
    for loc in LocalVisitante:
        if partido.DatosSuministrados['equipos'][loc]['haGanado']:
            marcador[loc] = "<b>{}</b>".format(marcador[loc])
        if loc == locEq:
            marcador[loc] = "<u>{}</u>".format(marcador[loc])

    resAux = [marcador[loc] for loc in LocalVisitante]

    strResultado = "{} ({})".format("-".join(resAux),
                                    haGanado2esp[partido.DatosSuministrados['equipos'][locEq]['haGanado']])

    return strRival, strResultado


def reportTrayectoriaEquipos(tempData, abrEqs, juIzda, juDcha):
    listaTrayectoria = datosMezclaPartJugados(tempData, abrEqs, juIzda, juDcha)
    filas = []

    resultStyle = ParagraphStyle('trayStyle', fontName='Helvetica', fontSize=12, align='center')
    cellStyle = ParagraphStyle('trayStyle', fontName='Helvetica', fontSize=12)
    jornStyle = ParagraphStyle('trayStyle', fontName='Helvetica-Bold', fontSize=13, align='right')

    for f in listaTrayectoria:
        datosIzda = f.get('izda', ['', ''])
        datosDcha = f.get('dcha', ['', ''])
        jornada = f['J']

        aux = [Paragraph(f"<para align='center'>{datosIzda[1]}</para>"),
               Paragraph(f"<para>{datosIzda[0]}</para>"),
               Paragraph(f"<para align='center' fontName='Helvetica-Bold'>{str(jornada)}</para>"),
               Paragraph(f"<para>{datosDcha[0]}</para>"),
               Paragraph(f"<para align='center'>{datosDcha[1]}</para>")]
        filas.append(aux)

    tStyle = TableStyle([('BOX', (0, 0), (-1, -1), 1, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                         ('GRID', (0, 0), (-1, -1), 0.5, colors.black)])

    t = Table(data=filas, style=tStyle, colWidths=[23 * mm, 72 * mm, 10 * mm, 72 * mm, 23 * mm])

    return t


def tablaJugadoresEquipo(jugDF):
    result = []

    CELLPAD = 0.2 * mm
    FONTSIZE = 8
    ANCHOLETRA = FONTSIZE * 0.5

    COLSIDENT = [('Jugador', 'dorsal'),
                 ('Jugador', 'nombre'),
                 ('Jugador', 'pos'),
                 ('Jugador', 'altura'),
                 ('Jugador', 'licencia'),
                 ('Trayectoria', 'Acta'),
                 ('Trayectoria', 'Jugados'),
                 ('Trayectoria', 'Titular'),
                 ('Trayectoria', 'Vict')
                 ]
    COLSIDENT_UP = [('Jugador', 'dorsal'),
                    ('Jugador', 'nombre'),
                    ]

    COLS_PROMED = [('Promedios', 'etSegs'),
                   ('Promedios', 'P'),
                   ('Promedios', 'etiqT2'),
                   ('Promedios', 'etiqT3'),
                   ('Promedios', 'etiqTC'),
                   ('Promedios', 'ppTC'),
                   ('Promedios', 'FP-F'),
                   ('Promedios', 'FP-C'),
                   ('Promedios', 'etiqT1'),
                   ('Promedios', 'etRebs'),
                   ('Promedios', 'A'),
                   ('Promedios', 'BP'),
                   ('Promedios', 'BR'),
                   ('Promedios', 'TAP-F'),
                   ('Promedios', 'TAP-C'),
                   ]
    COLS_TOTALES = [
        ('Totales', 'etSegs'),
        ('Totales', 'P'),
        ('Totales', 'etiqT2'),
        ('Totales', 'etiqT3'),
        ('Totales', 'etiqTC'),
        ('Totales', 'ppTC'),
        ('Totales', 'FP-F'),
        ('Totales', 'FP-C'),
        ('Totales', 'etiqT1'),
        ('Totales', 'etRebs'),
        ('Totales', 'A'),
        ('Totales', 'BP'),
        ('Totales', 'BR'),
        ('Totales', 'TAP-F'),
        ('Totales', 'TAP-C'),
    ]
    COLS_ULTP = [('UltimoPart', 'etFecha'),
                 ('UltimoPart', 'Partido'),
                 ('UltimoPart', 'resultado'),
                 ('UltimoPart', 'titular'),
                 ('UltimoPart', 'etSegs'),
                 ('UltimoPart', 'P'),
                 ('UltimoPart', 'etiqT2'),
                 ('UltimoPart', 'etiqT3'),
                 ('UltimoPart', 'etiqTC'),
                 ('UltimoPart', 'ppTC'),
                 ('UltimoPart', 'FP-F'),
                 ('UltimoPart', 'FP-C'),
                 ('UltimoPart', 'etiqT1'),
                 ('UltimoPart', 'etRebs'),
                 ('UltimoPart', 'A'),
                 ('UltimoPart', 'BP'),
                 ('UltimoPart', 'BR'),
                 ('UltimoPart', 'TAP-F'),
                 ('UltimoPart', 'TAP-C'),
                 ]

    baseOPS = [('BOX', (0, 0), (-1, -1), 2, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
               ('ALIGN', (0, 0), (-1, 0), 'CENTER'), ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold'),
               ('ALIGN', (0, 1), (-1, -1), 'RIGHT'),
               ('GRID', (0, 0), (-1, -1), 0.5, colors.black), ('FONTSIZE', (0, 0), (-1, -1), FONTSIZE),
               ('LEADING', (0, 0), (-1, -1), FONTSIZE + 1), ('LEFTPADDING', (0, 0), (-1, -1), CELLPAD),
               ('RIGHTPADDING', (0, 0), (-1, -1), CELLPAD), ('TOPPADDING', (0, 0), (-1, -1), CELLPAD),
               ('BOTTOMPADDING', (0, 0), (-1, -1), CELLPAD), ]

    auxDF = jugDF.copy()

    for colList in [(COLSIDENT + COLS_PROMED), (COLSIDENT + COLS_TOTALES),
                    (COLSIDENT_UP + COLS_ULTP)]:  # , [COLSIDENT +COLS_TOTALES], [COLSIDENT +COLS_ULTP]
        t = auxGeneraTabla(auxDF, colList, INFOTABLAJUGS, baseOPS, FORMATOCAMPOS, ANCHOLETRA)

        result.append(t)

    return result


def tablaLiga(tempData: TemporadaACB):
    CELLPAD = 0.3 * mm
    FONTSIZE = 10

    tStyle = TableStyle([('BOX', (0, 0), (-1, -1), 2, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                         ('GRID', (0, 0), (-1, -1), 0.5, colors.black), ('FONTSIZE', (0, 0), (-1, -1), FONTSIZE),
                         ('LEADING', (0, 0), (-1, -1), FONTSIZE), ('LEFTPADDING', (0, 0), (-1, -1), CELLPAD),
                         ('RIGHTPADDING', (0, 0), (-1, -1), CELLPAD), ('TOPPADDING', (0, 0), (-1, -1), CELLPAD),
                         ('BOTTOMPADDING', (0, 0), (-1, -1), CELLPAD),
                         ("BACKGROUND", (-1, 1), (-1, -2), colors.lightgrey),
                         ("BACKGROUND", (1, -1), (-2, -1), colors.lightgrey)])
    datosAux = datosTablaLiga(tempData)
    alturas = [20] + [28] * (len(datosAux) - 2) + [20]
    anchos = [70] + [38] * (len(datosAux) - 2) + [30]

    for i in range(1, len(datosAux) - 1):
        tStyle.add("BACKGROUND", (i, i), (i, i), colors.lightgrey)

    t = Table(datosAux, style=tStyle, rowHeights=alturas, colWidths=anchos)

    return t


def preparaLibro(outfile, tempData, datosSig):
    MARGENFRAME = 2 * mm
    frameNormal = Frame(x1=MARGENFRAME, y1=MARGENFRAME, width=A4[0] - 2 * MARGENFRAME, height=A4[1] - 2 * MARGENFRAME,
                        leftPadding=MARGENFRAME,
                        bottomPadding=MARGENFRAME, rightPadding=MARGENFRAME, topPadding=MARGENFRAME)
    frameApaisado = Frame(x1=MARGENFRAME, y1=MARGENFRAME, width=A4[1] - 2 * MARGENFRAME, height=A4[0] - 2 * MARGENFRAME,
                          leftPadding=MARGENFRAME,
                          bottomPadding=MARGENFRAME, rightPadding=MARGENFRAME, topPadding=MARGENFRAME)
    pagNormal = PageTemplate('normal', pagesize=A4, frames=[frameNormal], autoNextPageTemplate='normal')
    pagApaisada = PageTemplate('apaisada', pagesize=landscape(A4), frames=[frameApaisado],
                               autoNextPageTemplate='apaisada')

    doc = SimpleDocTemplate(filename=outfile, pagesize=A4, bottomup=0, verbosity=4, initialFontName='Helvetica',
                            initialLeading=5 * mm,
                            leftMargin=5 * mm,
                            rightMargin=5 * mm,
                            topMargin=5 * mm,
                            bottomMargin=5 * mm, )
    doc.addPageTemplates([pagNormal, pagApaisada])

    story = []

    (sigPartido, abrEqs, juIzda, peEq, juDcha, peRiv, targLocal) = datosSig

    antecedentes = {p.url for p in juIzda}.intersection({p.url for p in juDcha})

    story.append(cabeceraPortada(sigPartido, tempData))

    story.append(Spacer(width=120 * mm, height=2 * mm))
    story.append(estadsEquipoPortada(tempData, abrEqs))

    if antecedentes:
        print("Antecedentes!")
    else:
        story.append(Spacer(width=120 * mm, height=3 * mm))
        story.append(Paragraph("Sin antecedentes esta temporada"))

    trayectoria = reportTrayectoriaEquipos(tempData, abrEqs, juIzda, juDcha)
    if trayectoria:
        story.append(Spacer(width=120 * mm, height=3 * mm))
        story.append(trayectoria)

    story.append(NextPageTemplate('apaisada'))
    story.append(PageBreak())
    story.append(tablaLiga(tempData))

    if (len(juIzda) or len(juDcha)):
        infoJugadores = paginasJugadores(tempData, abrEqs, juIzda, juDcha)
        story.extend(infoJugadores)

    doc.build(story)


def parse_arguments():
    descriptionTXT = "Prepares a booklet for the next game of a team"

    parser = ArgumentParser(description=descriptionTXT)
    parser.add_argument("-t", "--acbfile", dest="acbfile", action="store", required=True, env_var="ACB_FILE",
                        help="Nombre del ficheros de temporada", )
    parser.add_argument("-l", "--listaequipos", dest='listaEquipos', action="store_true", required=False,
                        help="Lista siglas para equipos", )

    parser.add_argument("-e", "--equipo", dest="equipo", action="store", required=False,
                        help="Abreviatura del equipo deseado (usar -l para obtener lista)", )
    parser.add_argument("-o", "--outfile", dest="outfile", action="store", help="Fichero PDF generado",
                        required=False, )

    parser.add_argument("-c", "--cachedir", dest="cachedir", action="store", required=False, env_var="ACB_CACHEDIR",
                        help="Ubicación de caché de ficheros", )

    result = parser.parse_args()

    return result


def main(args):
    tempData = cargaTemporada(args.acbfile)

    if args.listaEquipos:
        listaEquipos(tempData)

    REQARGS = ['equipo', 'outfile']
    missingReqs = {k for k in REQARGS if (k not in args) or (args.__getattribute__(k) is None)}
    if missingReqs:
        missingReqsStr = ",".join(sorted(missingReqs))
        print(f"Faltan argumentos (ver -h): {missingReqsStr}")
        sys.exit(1)
    try:
        datosSig = tempData.sigPartido(args.equipo)
    except KeyError as exc:
        print(f"Equipo desconocido '{args.equipo}': {exc}")
        sys.exit(1)

    preparaLibro(args.outfile, tempData, datosSig)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
