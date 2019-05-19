import os
import sys
from sys import stdin, stdout
import argparse
import time
import datetime
import logging
import traceback
import json
import csv
import emoji # pip install emoji
from unidecode import unidecode # pip install unidecode

def command_line_parser():
    """
    returns a dict with the options passed to the command line
    according with the options available
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--inputfile", type=str, required=True, default="-",
                        help="input file, stdin as default")
    
    parser.add_argument("--no_unidecode", action='store_false',
                        help="doesn't unidecode text, leaves it as is")
    
    parser.add_argument("--no_demojize", action='store_false',
                        help="doesn't demojize text, leaves it with original emoticons")

    parser.add_argument("--fields", nargs='*',
                        help="list of fields to get from tweet")

    parser.add_argument("-o", "--outputfile", type=str, default="-",
                        help="output file name, stdout as default")

    parser.add_argument("-v", "--verbose", action='store_true',
                        help="verbose output")

    args = parser.parse_args()

    return args

class TwitterParser(object):
    DEFAULT_FIELDS = ["text"]
    
    def __init__(self, fields=None, demojize=True, unidecode=True):
        self.objid         = id(self)
        self.logger        = logging.getLogger('TwitterParser' + str(self.objid))
        self.fields        = fields if fields is not None else TwitterParser.DEFAULT_FIELDS
        self.use_demojize  = demojize
        self.use_unidecode = unidecode
        
        self.logger.debug("initialized constructor")

    def get_unique_field(self, d, field):
        '''Returns only one instance of the field which is a leaf in the graph, None if not found'''
        if not isinstance(d, dict):
            return None
        if field in d.keys() and isinstance(d.get(field, None), str):
            return d[field]
        for value in d.values():
            v = get_unique_field(value, field)
            if v is not None:
                return v
        
        return None
        
    def get_field(self, d, field): # recursive dfs
        if not isinstance(d, dict):
            return []

        values_found = []
        if field in d.keys() and isinstance(d.get(field, None), str):
            values_found.append(d[field])
        for value in d.values():
            v = self.get_field(value, field)
            values_found.extend(v)

        # self.logger.debug("fields found for \'%s\': %s", field, values_found)
        return values_found

    def normalize_text(self, txt):
        # assert isinstance(txt, str), "type %s (%s)" % (type(txt), txt)
        if isinstance(txt, str):
            txt = emoji.demojize(txt) if self.use_demojize else txt
            txt = unidecode(txt) if self.use_unidecode else txt
        elif isinstance(txt, list):
            txt = [self.normalize_text(t) for t in txt]
        return txt

    def remove_junk(self, dat, prof=0):
        '''Removes anything that has length zero'''
        r = None
        if isinstance(dat, list):
            r = []
            for d in dat:
                v = self.remove_junk(d, prof+1)
                if v is not None:
                    r.append(v)
        else:
            r = None if len(dat) == 0 else dat

        if prof == 0:
            return r
        return None if r is None or len(r) == 0 else r
    
    def proc_field(self, fields):
        r = None
        self.logger.debug("proc_field input: %s", fields)
        r = [self.normalize_text(txt) for txt in fields]
        r = self.remove_junk(r)
        self.logger.debug("proc_field return: %s", r)
        return r

    def parse(self, inputfile, outputfile):
        line = inputfile.readline().strip()
        linenum = 1

        # writer = csv.DictWriter(outputfile, delimiter="\t", fieldnames=self.fields)
        # writer.writeheader()
        
        while len(line) != 0:
            # self.logger.debug("processing line #%d", linenum)
            try:
                data = json.loads(line)

                data_fields = {self.fields[0]: self.get_unique_field(data, self.fields[0])}
                for field in self.fields[1:]:
                    data_fields[field] = self.proc_field(self.get_field(data, field))

                # writer.writerow(data_fields)
                json.dump(data_fields, outputfile)
                outputfile.write("\n")
            except json.JSONDecodeError as jsonex:
                self.logger.error("Could not parse line #%d: %s", linenum, str(jsonex))
            except Exception as ex:
                traceback.print_exc()
                self.logger.error("Unknown error parsing line #%d: %s", linenum, str(ex))
                
            linenum += 1
            line = inputfile.readline().strip()

def main():
    options = command_line_parser()

    logging.basicConfig(level    = logging.DEBUG if options.verbose else logging.INFO,
                        format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt  = '%m-%d %H:%M',
                        filename = 'parser.log',
                        filemode = 'w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    ################ APPLICATION STARTS ################
    inputfile = open(options.inputfile, "r") if options.inputfile != '-' else stdin
    outputfile = open(options.outputfile + '.json', "w") if options.outputfile != '-' else stdout

    logging.debug("app options: %s", str(options))
    
    parser = TwitterParser(fields    = options.fields,
                           demojize  = options.no_demojize,
                           unidecode = options.no_unidecode)

    logging.debug("started to parse ...")
    parser.parse(inputfile, outputfile)
    logging.debug("finished parsing.")

    inputfile.close()
    outputfile.close()
    
if __name__ == "__main__":
    main()
