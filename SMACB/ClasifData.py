from time import gmtime
from babel.numbers import parse_decimal


class ClasifData(object):

    def __init__(self,content,label=None,source=None):
        self.timestamp = gmtime()
        self.label=label
        self.source = source
        self.data = self.processClasifPage(content)

    def processClasifPage(self, content):
        result = {}
        table = content.find("table", {"class":"general"})
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            entry = {}
            entry['team'] = cells[1].get_text()
            entry['socio'] = cells[2].get_text()
            entry['value'] = parse_decimal(cells[3].get_text(), locale="de")
            result[entry['team']] = entry
        return result

    def __repr__(self):
        return "{ "+", ".join([" '{}' ({}): {}".format(self.data[k]['team'],
                                              self.data[k]['socio'],
                                              self.data[k]['value']) for k in sorted(self.data, key=self.data.get('value'), reverse=True)])+"} "
