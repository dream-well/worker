import worker
import random
import asyncio
import time
# synapse = new.QueryZkProof(
# 	query_input = {'model_id': [0], 'public_inputs': [-0.2847691892031461, -0.08110789393296791, 0.6423077381281963, -0.19719430897525503, 0.1303140519756325]}
# )

# result = new.generateZkProof(synapse)

# print(result)
retry = 0
start = time.time()
while retry < 10:
	synapse = worker.QueryZkProof(
		query_input = {'model_id': ['0a92bc32ea02abe54159da70aeb541d52c3cba27c8708669eda634e096a86f8b'], 'public_inputs': [[0.00390625], [3.655120544458697e-17], [False], [5280], [10.941974639892578], [60.0], [11.105813026428223], [55.0, 52.0, 100.0, 97.0, 56.0, 55.0, 55.0, 48.0, 57.0, 98.0, 55.0, 98.0, 51.0, 99.0, 101.0, 101.0, 102.0, 54.0, 97.0, 57.0, 97.0, 99.0, 100.0, 101.0, 49.0, 57.0, 49.0, 54.0, 50.0, 55.0, 48.0, 97.0, 100.0, 54.0, 98.0, 52.0, 56.0, 53.0, 57.0, 52.0, 102.0, 101.0, 99.0, 100.0, 100.0, 52.0, 54.0, 100.0, 98.0, 101.0, 54.0, 56.0, 57.0, 57.0, 99.0, 54.0, 56.0, 97.0, 100.0, 49.0, 100.0, 52.0, 53.0, 56.0], [3216754], [208]]}
	)

	asyncio.run(worker.generateZkProof(synapse))
	retry += 1
print("Time: ", (time.time() - start) / 10)
