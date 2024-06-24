import asyncio
import json
import multiprocessing
import os
import time
import traceback
from typing import Dict, Optional
import uuid
from itertools import chain

import bittensor as bt
import ezkl
from pydantic import BaseModel

from fastapi import FastAPI
app = FastAPI()

dir_path = os.path.dirname(os.path.realpath(__file__))

def gen_witness(input_path, circuit_path, witness_path, vk_path, srs_path):
    # bt.logging.debug("Generating witness")
    res = ezkl.gen_witness(
        input_path,
        circuit_path,
        witness_path,
        vk_path,
        srs_path,
    )
    # bt.logging.debug(f"Gen witness result: {res}")


async def gen_proof(
    input_path, vk_path, witness_path, circuit_path, pk_path, proof_path, srs_path
):
    start = time.time()
    gen_witness(input_path, circuit_path, witness_path, vk_path, srs_path)
    print(f"gen witness took {time.time() - start}")
    start = time.time()

    # bt.logging.debug("Generating proof")
    res = ezkl.prove(
        witness_path,
        circuit_path,
        pk_path,
        proof_path,
        "single",
        srs_path,
    )
    # bt.logging.debug(f"Proof generated: {proof_path}, result: {res} took {time.time() - start}")


async def proof_worker(
    input_path, vk_path, witness_path, circuit_path, pk_path, proof_path, srs_path
):
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    try:
        start = time.time()
        # result = loop.run_until_complete(
        #     gen_proof(
        #         input_path,
        #         vk_path,
        #         witness_path,
        #         circuit_path,
        #         pk_path,
        #         proof_path,
        #         srs_path,
        #     )
        # )
        result = await gen_proof(
                input_path,
                vk_path,
                witness_path,
                circuit_path,
                pk_path,
                proof_path,
                srs_path,
            )
        print("Proof generation took", time.time() - start)
        return result
    finally:
        # loop.close()
        pass


class VerifiedModelSession:

    def __init__(self, public_inputs=None, model_id=None):
        if public_inputs is None:
            public_inputs = []
        if model_id is None:
            model_id = [0]
        self.model_id = model_id[0]
        self.session_id = str(uuid.uuid4())
        self.model_name = f"model_{self.model_id}"
        model_path = os.path.join(
            dir_path, "model", self.model_name
        )

        self.pk_path = os.path.join(model_path, "pk.key")
        self.vk_path = os.path.join(model_path, "vk.key")
        self.srs_path = os.path.join(model_path, "kzg.srs")
        self.circuit_path = os.path.join(model_path, "model.compiled")
        self.settings_path = os.path.join(model_path, "settings.json")
        self.sample_input_path = os.path.join(model_path, "input.json")

        self.witness_path = os.path.join(
            dir_path, "temp", f"witness_{self.model_name}_{self.session_id}.json"
        )
        self.input_path = os.path.join(
            dir_path, "temp", f"input_{self.model_name}_{self.session_id}.json"
        )
        self.proof_path = os.path.join(
            dir_path, "temp", f"proof_{self.model_name}_{self.session_id}.json"
        )

        self.public_inputs = public_inputs

        self.py_run_args = ezkl.PyRunArgs()
        self.py_run_args.input_visibility = "public"
        self.py_run_args.output_visibility = "public"
        self.py_run_args.param_visibility = "fixed"

    # Generate the input.json file, which is used in witness generation
    def gen_input_file(self):
        input_data = self.public_inputs
        input_shapes = [[1]]
        data = {"input_data": input_data, "input_shapes": input_shapes}

        dir_name = os.path.dirname(self.input_path)
        os.makedirs(dir_name, exist_ok=True)

        with open(self.input_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        # bt.logging.info(f"Input data: {data}")

    def gen_proof_file(self, proof_string, inputs):
        dir_name = os.path.dirname(self.proof_path)
        os.makedirs(dir_name, exist_ok=True)
        proof_json = json.loads(proof_string)
        new_instances = list(chain.from_iterable(inputs))
        bt.logging.trace(f"New instances: {new_instances}")
        new_instances += proof_json["instances"][0][len(new_instances):]
        bt.logging.trace(
            f"New instances after appending with last instances from output: {new_instances}"
        )
        proof_json["instances"] = [new_instances]
        # Enforce EVM transcript type to be used for all proofs, ensuring all are valid for proving on EVM chains
        proof_json["transcript_type"] = "EVM"
        bt.logging.trace(f"Proof json: {proof_json}")

        with open(self.proof_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(proof_json))
            f.close()

    def aggregate_proofs(self, proofs):
        """
        Aggregates proofs for the previous weight adjustment period.
        """
        try:
            proof_paths = []
            for i, proof in enumerate(proofs):
                proof_path = os.path.join(
                    dir_path,
                    "temp",
                    f"proof_{self.model_name}_{self.session_id}_{i}.proof",
                )
                with open(proof_path, "w", encoding="utf-8") as f:
                    f.write(proof)
                proof_paths.append(proof_path)
            # bt.logging.debug(
                # f"Generating aggregated proof given proof paths: {proof_paths}"
            # )
            path = os.path.join(
                dir_path,
                "temp",
                f"aggregated_proof_{self.model_name}_{self.session_id}.proof",
            )
            time_start = time.time()
            ezkl.aggregate(aggregation_snarks=proof_paths, output_path=path)
            time_delta = time.time() - time_start
            # bt.logging.info(f"Proof aggregation took {time_delta} seconds")
            with open(path, "r", encoding="utf-8") as f:
                aggregated_proof = f.read()
            return aggregated_proof, time_delta
        except Exception as e:
            bt.logging.error(f"An error occurred during proof aggregation: {e}")
            traceback.print_exc()
            raise

    async def gen_proof(self):
        try:
            # bt.logging.debug("Generating input file")
            self.gen_input_file()
            # bt.logging.debug("Starting generating proof process...")
            start_time = time.time()
            # with multiprocessing.Pool(1) as p:
            #     # bt.logging.debug(
            #     #     f"Starting proof generation with paths: {self.input_path}, {self.vk_path}, "
            #     #     f"{self.witness_path}, {self.circuit_path}, {self.pk_path}, {self.proof_path}, "
            #     #     f"{self.srs_path}"
            #     # )
            #     p.apply(
            #         func=proof_worker,
            #         kwds={
            #             "input_path": self.input_path,
            #             "vk_path": self.vk_path,
            #             "witness_path": self.witness_path,
            #             "circuit_path": self.circuit_path,
            #             "pk_path": self.pk_path,
            #             "proof_path": self.proof_path,
            #             "srs_path": self.srs_path,
            #         },
            #     )
            await proof_worker(
                self.input_path, self.vk_path, self.witness_path, self.circuit_path, self.pk_path, self.proof_path, self.srs_path
            )
            end_time = time.time()
            proof_time = end_time - start_time
            # bt.logging.info(f"Proof generation took {proof_time} seconds")
            with open(self.proof_path, "r", encoding="utf-8") as f:
                proof_content = f.read()

            return proof_content, proof_time

        except Exception as e:
            print(f"An error occurred during proof generation: {e}")
            traceback.print_exc()
            raise

    def verify_proof(self):
        res = ezkl.verify(
            self.proof_path,
            self.settings_path,
            self.vk_path,
            self.srs_path,
        )

        return res

    def verify_proof_and_inputs(self, proof_string, inputs):
        if not proof_string:
            return False
        self.public_inputs = inputs
        self.gen_input_file()
        gen_witness(
            self.input_path,
            self.circuit_path,
            self.witness_path,
            self.vk_path,
            self.srs_path,
        )
        with open(self.witness_path, "r", encoding="utf-8") as f:
            witness_content = f.read()
        witness_json = json.loads(witness_content)
        self.gen_proof_file(proof_string, witness_json["inputs"])
        return self.verify_proof()

    def __enter__(self):
        return self

    def remove_temp_files(self):
        if os.path.exists(self.input_path):
            os.remove(self.input_path)

        if os.path.exists(self.witness_path):
            os.remove(self.witness_path)

        if os.path.exists(self.proof_path):
            os.remove(self.proof_path)

    def end(self):
        # self.remove_temp_files()
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):

        return None


class QueryZkProof(BaseModel):
    query_input: Optional[Dict] = None
    query_output: Optional[str] = None

total_time = 0
total_cnt = 0

@app.post("/generate_proof")
async def generateZkProof(synapse: QueryZkProof) -> QueryZkProof:
    time_in = time.time()
    # bt.logging.debug("Received request from validator")
    print(f"Input data: {synapse.query_input} \n")

    if not synapse.query_input or not synapse.query_input.get(
        "public_inputs", None
    ):
        bt.logging.error("Received empty query input")
        synapse.query_output = "Empty query input"
        return synapse

    model_id = synapse.query_input.get("model_id", [0])
    public_inputs = synapse.query_input["public_inputs"]
    if model_id == [0]:
        public_inputs = [public_inputs]

    # Run inputs through the model and generate a proof.
    try:
        with VerifiedModelSession(public_inputs, model_id) as model_session:
            model_session = VerifiedModelSession(public_inputs, model_id)
            # bt.logging.debug("Model session created successfully")
            synapse.query_output, proof_time = await model_session.gen_proof()
            # model_session.end()
    except Exception as e:
        synapse.query_output = "An error occurred"
        print(f"An error occurred while generating proven output\n{e}")
        proof_time = time.time() - time_in

    time_out = time.time()
    delta_t = time_out - time_in
    print(
        f"Total response time {delta_t}s. Proof time: {proof_time}s. "
        f"Overhead time: {delta_t - proof_time}s."
    )
    global total_time
    global total_cnt
    total_time += delta_t
    total_cnt += 1
    print(
        f"Average response time: {total_time / total_cnt}, cnt: {total_cnt}"
    )
    return synapse

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)