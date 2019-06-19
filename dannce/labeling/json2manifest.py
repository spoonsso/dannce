"""
Takes the json files with image ID onformation provided by SageMaker, and creates more readble manifest files.
This is a required first step before consolidating manual annotations

"""

import numpy
import json
import sys

def j2m(json_file):
	man_file = json_file.split('.json')[0] + '.manifest'

	with open(man_file,'w') as g:
		with open(json_file,'r') as f:
			for line in f:
				x = json.loads(line)
				ID = x['datasetObjectId']
				imname = json.loads(x['manifestLine'])['source-ref']
				g.write('{}\t{}\t0\n'.format(ID, imname))


