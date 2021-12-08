from io import StringIO
from xml.etree import cElementTree

import pandas as pd


class Box(object):
    def __init__(self):
        return

    def update(self, dic):
        self.__dict__.update(dic)


class Xml(object):
    def __init__(self, filename, needed=None):
        tree = cElementTree.ElementTree(file=filename)
        root = tree.getroot()
        self.box = Box()
        self.nodes = {}
        needed = [] if needed is None else needed
        for key in root[0].attrib:
            self.__dict__[key] = int(root[0].attrib[key])
        for element in root[0]:
            if element.tag == 'box':
                self.box.update(element.attrib)
                continue
            if (len(needed) > 0) and (element.tag not in needed):
                continue
            self.nodes[element.tag] = pd.read_csv(StringIO(element.text),
                                                     delim_whitespace=True,
                                                     squeeze=1,
                                                     header=None,
                                                     ).values
