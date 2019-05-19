import json
import csv
import logging

class DictStreamIterator(object):
    def __init__(self, infile):
        self.objid  = id(self)
        self.logger = logging.getLogger('DictStreamIterator' + str(self.objid))
        self.linenum = 1
        self.infile = infile

        self.logger.debug("initialized constructor")

    def __iter__(self):
        self.linenum = 1
        self.infile.seek(0, 0)
        return self

    def __next__(self):
        self.linenum += 1
        line = self.infile.readline()
        if len(line) == 0:
            raise StopIteration
        return line


class DictStreamIteratorJson(DictStreamIterator):
    def __next__(self):
        line  = super(DictStreamIteratorJson, self).__next__()
        d = None
        while d is None and len(line) > 0:
            try:
                d = json.loads(line)
            except Exception as ex:
                self.logger.error("could not convert to json line %d", self.linenum-1)
                line  = super(DictStreamIteratorJson, self).__next__()
                
        return d

class DictStreamIteratorCsv(DictStreamIterator):
    def __init__(self, infile):
        super(DictStreamIteratorCsv, self).__init__(infile)
        self.reader = csv.DictReader(self.infile)
    
    def __next__(self):
        self.linenum += 1
        return next(self.reader)

class DictStreamIteratorTsv(DictStreamIteratorCsv):
    def __init__(self, infile):
        super(DictStreamIteratorCsv, self).__init__(infile)
        self.reader = csv.DictReader(self.infile, delimiter="\t")
