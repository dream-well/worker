import worker
import random
import asyncio
import time
synapse = worker.QueryZkProof(
	query_input = {'model_id': [0], 'public_inputs': [-0.2847691892031461, -0.08110789393296791, 0.6423077381281963, -0.19719430897525503, 0.1303140519756325]}
)

result = asyncio.run(worker.generateZkProof(synapse))

print(result)