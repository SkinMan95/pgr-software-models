import json
import csv

class DictStreamIterator(object):
    def __init__(self, infile):
        self.logger = logging.getLogger('DictStreamIterator' + str(self.objid))
        self.infile = infile

        self.logger.debug("initialized constructor")

    def __iter__(self):
        return self

    def __next__(self):
        line = self.infile.readline().strip()
        if len(line) == 0:
            raise StopIteration
        return line


class DictStreamIteratorJson(DictStreamIterator):
    def __next__(self):
        line  = super(DictStreamIteratorJson, self).__next__()
        d = json.loads(line)
        return d

class DictStreamIteratorCsv(DictStreamIterator):
    def __init__(self, infile):
        super(DictStreamIteratorCsv, self).__init__(infile)
        self.reader = csv.DictReader(self.infile)
    
    def __next__(self):
        return next(self.reader)

class DictStreamIteratorTsv(DictStreamIteratorCsv):
    def __init__(self, infile):
        super(DictStreamIteratorCsv, self).__init__(infile)
        self.reader = csv.DictReader(self.infile, delimiter="\t")
    
    def __next__(self):
        return next(self.reader)
