from time import gmtime
from SMACB.TemporadaACB import TemporadaACB
from SMACB.PartidoACB import PartidoACB


class CompoACB(object):
    def __init__(self, cod, temp):
        self.codCompo = cod
        self.idTemp = temp
        self.fechaPrimPart = None
        self.fechaUltPart = None
        self.timestamp = gmtime()
        self.tstampUltTempTratada = None
        self.Calendario = None
        self.partidos = set()

    def actualizaDeTemp(self, tempData):
        # tempData = TemporadaACB() # Para usar el completador. COMENTAR AL FINAL
        huboCambios = set()

        if self.codCompo != tempData.Calendario.competicion or self.idTemp != tempData.Calendario.edicion:
            raise ValueError("Competición o temporada no corresponden: esperados (%s,%s) suministrados (%s,%s)" % (
                self.codCompo, self.idTemp, tempData.Calendario.competicion, tempData.Calendario.edicion))

        if self.tstampUltTempTratada is not None and tempData.timestamp <= self.tstampUltTempTratada:
            return huboCambios

        for p in tempData.PartidosDescargados:
            if p in self.partidos:
                continue

            self.partidos.add(p)
            huboCambios.add(p)
            dataPart = tempData.Partidos[p]
            # dataPart = PartidoACB() # Para usar el completador. COMENTAR AL FINAL
            if self.fechaPrimPart is None:
                self.fechaPrimPart = dataPart.FechaHora
            elif dataPart.FechaHora < self.fechaPrimPart:
                dataPart.FechaHora = self.fechaPrimPart

            if self.fechaUltPart is None:
                self.fechaUltPart = dataPart.FechaHora
            elif dataPart.FechaHora > self.fechaUltPart:
                dataPart.FechaHora = self.fechaUltPart

        return huboCambios
