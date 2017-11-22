from bs4 import BeautifulSoup
from time import gmtime,strftime, strptime
from collections import defaultdict

import re

from babel.numbers import parse_decimal

class MercadoPageCompare():

    def __init__(self, old, new):

        if not ((type(old) is MercadoPageContent) and (type(new) is MercadoPageContent)):
            errorStr = ""
            if not (type(old) is MercadoPageContent):
                errorStr += "Type for original data '%s' is not supported. " % type(old)
            if not (type(new) is MercadoPageContent):
                errorStr += "Type for new data '%s' is not supported. " % type(new)

            raise TypeError(errorStr)


        # Am I a metadata freak?
        self.timestamps = {}
        self.timestamps['old'] = old.timestamp
        self.timestamps['new'] = new.timestamp

        self.sources = {}
        self.sources['old'] = old.source
        self.sources['new'] = new.source

        self.changes = False

        self.teamRenamed = False
        self.newTeams = []
        self.delTeams = []
        self.teamTranslationsOld2New = {}
        self.teamTranslationsNew2Old = {}

        self.playerChanges = defaultdict(lambda: defaultdict(str))

        self.bajas = []
        self.altas = []
        self.lesionado = []
        self.curado = []

        self.cambRival = 0
        self.contCambEquipo = 0
        cambEquipo = {}
        self.newRivals = defaultdict(int)
        origTeam = defaultdict(int)
        destTeam = defaultdict(int)

        oldPlayersID = set(old.PlayerData.keys())
        newPlayersID = set(new.PlayerData.keys())
        oldTeams = set(old.Team2Player.keys())
        newTeams = set(new.Team2Player.keys())
        self.teamsJornada = len(newTeams)

        if oldTeams != newTeams:
            self.teamRenamed = True
            newShowingTeams = newTeams - oldTeams
            nonShowingTeams = oldTeams - newTeams
            if len(newShowingTeams) == 1:
                self.teamTranslationsNew2Old[list(newShowingTeams)[0]] = list(nonShowingTeams)[0]
                self.teamTranslationsOld2New[list(nonShowingTeams)[0]] = list(newShowingTeams)[0]
            else:
                self.newTeams = list(newShowingTeams)
                self.delTeams = list(nonShowingTeams)

        bajasID = oldPlayersID - newPlayersID
        altasID = newPlayersID - oldPlayersID
        siguenID = oldPlayersID & newPlayersID


        if bajasID:
            self.changes = True
            self.bajas = [ old.PlayerData[x] for x in bajasID ]
            for key in bajasID:
                self.playerChanges[key]['baja'] += "{} ({}) es baja en '{}'. ".format(old.PlayerData[key]['nombre'], key, old.PlayerData[key]['equipo'])

        if altasID:
            self.changes = True
            self.altas = [ new.PlayerData[x] for x in altasID ]
            for key in altasID:
                self.playerChanges[key]['alta'] += "{} ({}) es alta en '{}'. ".format(new.PlayerData[key]['nombre'], key, new.PlayerData[key]['equipo'])

        for key in siguenID:
            oldPlInfo = old.PlayerData[key]
            newPlInfo = new.PlayerData[key]

            if oldPlInfo['equipo'] != newPlInfo['equipo']:
                oldTeam = oldPlInfo['equipo']
                newTeam = newPlInfo['equipo']

                self.contCambEquipo += 1
                origTeam[oldTeam] += 1
                destTeam[newTeam] += 1
                self.playerChanges[key]['cambio'] += "{} ({}) pasa de '{}' a '{}'. ".format(new.PlayerData[key]['nombre'],
                                                                                            key,
                                                                                            oldTeam,
                                                                                            newTeam)

                cambEquipo[key] = "{} pasa de {} a {}".format(key,
                                                             oldPlInfo['equipo'],
                                                             newPlInfo['equipo'])

            if oldPlInfo['rival'] != newPlInfo['rival']:
                self.changes = True
                oldRival = oldPlInfo['rival']
                newRival = newPlInfo['rival']

                self.newRivals[newRival] += 1
                self.cambRival += 1


            if oldPlInfo['lesion'] != newPlInfo['lesion']:
                self.changes = True
                if newPlInfo['lesion']:
                    self.playerChanges[key]['lesion'] += "{} ({},{}) se ha lesionado. ".format(newPlInfo['nombre'], key, newPlInfo['equipo'])
                    self.changes = True
                    self.lesionado.append(key)
                else:
                    self.playerChanges[key]['salud'] += "{} ({},{}) se ha recuperado. ".format(new.PlayerData[key]['nombre'], key, oldPlInfo['equipo'])
                    self.changes = True
                    self.curado.append(key)

            if 'info' in oldPlInfo or 'info' in newPlInfo:
                if 'info' in oldPlInfo:
                    if 'info' in newPlInfo:
                        if oldPlInfo['info'] != newPlInfo['info']:
                            self.changes = True
                            self.playerChanges[key]['info'] += "{} ({}) info pasa de '{}' a '{}'. ".format(new.PlayerData[key]['nombre'], key, oldPlInfo['info'], newPlInfo['info'])
                    else:
                        self.changes = True
                        self.playerChanges[key]['info'] += "{} ({}) info eliminada '{}'. ".format(new.PlayerData[key]['nombre'], key, oldPlInfo['info'])
                else:
                    self.changes = True
                    self.playerChanges[key]['info'] += "{} ({}) info nueva '{}'. ".format(new.PlayerData[key]['nombre'], key, newPlInfo['info'])

    def __repr__(self):
        changesMSG = "hubo cambios." if self.changes else "sin cambios."
        result ="Comparación entre {} ({}) y {} ({}): {}\n\n".format(self.sources['old'],
                                                               strftime("%Y-%m-%d %H:%M",self.timestamps['old']),
                                                               self.sources['new'],
                                                               strftime("%Y-%m-%d %H:%M",self.timestamps['new']),
                                                               changesMSG)

        if len(self.newRivals) == self.teamsJornada:
            result += "Cambio de jornada!\n\n"

        print(self.newRivals)

        if self.teamRenamed:
            result += "Equipos renombrados: {}\n".format(len(self.teamTranslationsNew2Old))
            for team in self.teamTranslationsOld2New.keys():
                result += "  '{}' pasa a ser '{}'\n".format(team,self.teamTranslationsOld2New[team])
        if self.newTeams:
            result += "Nuevos equipos ({}): {}\n".format(len(self.newTeams),self.newTeams.sort())
        if self.delTeams:
            result += "Equipos no juegan ({}): {}\n".format(len(self.delTeams),self.delTeams.sort())

        if self.teamRenamed or self.newTeams or self.delTeams:
            result += "\n"

        if self.contCambEquipo:
            result += "Cambios de equipo: {}\n".format(self.contCambEquipo)

        if self.altas:
            result += "Altas: {}\n".format(len(self.altas))

        if self.bajas:
            result += "Bajas: {}\n".format(len(self.bajas))

        if self.lesionado:
            result += "Lesionados: {}\n".format(len(self.lesionado))

        if self.curado:
            result += "Recuperados: {}\n".format(len(self.curado))

        if self.altas or self.bajas or self.lesionado or self.curado:
            result += "\n"

        for key in self.playerChanges:
            playerChangesInfo= self.playerChanges[key]

            for item in playerChangesInfo.keys():
                result += playerChangesInfo[item]
            result += "\n"

        if self.playerChanges:
            result += "\n"

        result += "\n"

        return result


        return str({'timestamp':self.timestamp,
                    'source':self.source,
                    'NoFoto2Nombre':self.NoFoto2Nombre,
                    'Nombre2NoFoto':self.Nombre2NoFoto,
                    'PositionsCounter':self.PositionsCounter,
                    'PlayerData':self.PlayerData,
                    'PlayerByPos':self.PlayerByPos,
                    'Team2Player':self.Team2Player
                    })


class MercadoPageContent():

    def __init__(self, textPage):
        self.timestamp = gmtime()
        self.source = textPage['source']
        self.contadorNoFoto = 0
        self.NoFoto2Nombre = {}
        self.Nombre2NoFoto = {}
        self.PositionsCounter = defaultdict(int)
        self.PlayerData = {}
        self.PlayerByPos = defaultdict(list)
        self.Team2Player = defaultdict(set)

        if (type(textPage['data']) is str):
            soup = BeautifulSoup(textPage['data'], "html.parser")
        elif (type(textPage['data']) is BeautifulSoup):
            soup = textPage['data']
        else:
            raise NotImplementedError("MercadoPageContent: type of content '%s' not supported" % type(textPage['data']))

        positions = soup.find_all("table", {"class":"listajugadores"})

        for pos in positions:
            position = pos['id']

            for player in pos.find_all("tr"):
                player_data = player.find_all("td")
                player_data or next

                fieldTrads = { 'foto' : ['foto'],
                               'jugador' : ['jugador'],
                               'equipo' : ['equipo'],
                               'promedio' : ['promVal', 'valJornada', 'seMantiene'],
                               'precio' : ['precio', 'enEquipos%'],
                               'val' : ['prom3Jornadas'],
                               'balance' : ['sube15%'],
                               'baja' : ['baja15%'],
                               'rival' : ['rival'],
                               'iconos' : ['iconos']
                               }

                result = { 'proxFuera': False , 'lesion': False,
                          'cupo': 'normal'  }
                result['pos'] = position
                self.PositionsCounter[position] += 1
                for data in player_data:
                    # print(data,data['class'])

                    dataid = (fieldTrads.get(data['class'][0])).pop(0)
                    if dataid == "foto":
                        img_link = data.img['src']
                        result['foto'] = img_link
                        result['nombre'] = data.img['title']
                        auxre = re.search(r'J(.{3})LACB([0-9]{2})\.jpg', img_link)
                        if auxre:
                            result['codJugador'] = auxre.group(1)
                            result['temp'] = auxre.group(2)
                        else:
                            jugCode = "NOFOTO%03i" % self.contadorNoFoto
                            self.contadorNoFoto += 1
                            self.NoFoto2Nombre[jugCode] = result['nombre']
                            self.Nombre2NoFoto[result['nombre']] = jugCode
                            result['codJugador'] = jugCode
                    elif dataid == 'jugador':
                        result['kiaLink'] = data.a['href']
                    elif dataid == 'iconos':
                        for icon in data.find_all("img"):
                            if icon['title'] == "Extracomunitario":
                                result['cupo'] = 'Extracomunitario'
                            elif icon['title'] == "Español":
                                result['cupo'] = "Español"
                            elif icon['title'] == "Lesionado":
                                result['lesion'] = True
                            elif icon['alt'] == "Icono de más información":
                                result['info'] = icon['title']
                            else:
                                print("No debería llegar aquí: ", icon)

                    elif dataid == 'equipo':
                        result['equipo'] = data.img['title']
                    elif dataid == 'rival':
                        for icon in data.find_all('img'):
                            if icon['title'] == "Partido fuera":
                                result['proxFuera'] = True
                            else:
                                result['rival'] = icon['title']
                    else:
                        auxval = data.get_text().strip()
                        if dataid == "enEquipos%":
                            auxval = auxval.replace("%", "")
                        result[dataid] = parse_decimal(auxval, locale="de")

                        # print("Not treated %s" % dataid, data,)
                if result.get('codJugador'):
                    self.PlayerData[result['codJugador']] = result
                    self.PlayerByPos[position].append(result['codJugador'])
                    self.Team2Player[result['equipo']].add(result['codJugador'])

    def __reprX__(self):
        return str({'timestamp':self.timestamp,
                    'source':self.source,
                    'NoFoto2Nombre':self.NoFoto2Nombre,
                    'Nombre2NoFoto':self.Nombre2NoFoto,
                    'PositionsCounter':self.PositionsCounter,
                    'PlayerData':self.PlayerData,
                    'PlayerByPos':self.PlayerByPos,
                    'Team2Player':self.Team2Player
                    })


    def SetTimestampFromStr(self,timeData):
        ERDATE=re.compile(".*-(\d{4}\d{2}\d{2}(\d{4})?)\..*")
        ermatch=ERDATE.match(timeData)
        if ermatch:
            if ermatch.group(2):
                self.timestamp=strptime(ermatch.group(1),"%Y%m%d%H%M")
            else:
                self.timestamp=strptime(ermatch.group(1),"%Y%m%d")

    def Diff(self, otherData):
        return MercadoPageCompare(self, otherData)

    def __ne__(self, other):
        diff = self.Diff(other)
        return diff.changes
