import json
import re


class RegexpManager:
    REGEXPS_PATH = "preprocessing/regexps.json"

    with open(REGEXPS_PATH) as regexps_file:
        regexps = json.load(regexps_file)

    def get_compiled(self):
        regexes = {k.lower(): re.compile(self.regexps[k]) for k, v in
                   self.regexps.items()}
        return regexes
