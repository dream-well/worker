from typing import Dict, Optional
import bittensor as bt
import time
import json
import os
import uuid
import ezkl
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryZkProof(BaseModel):
    query_input: Optional[Dict] = None
    query_output: Optional[str] = None

dir_path = os.path.dirname(os.path.realpath(__file__))

def gen_witness(input_path, circuit_path, witness_path, vk_path, srs_path):
    bt.logging.info("Generating witness")
    res = ezkl.gen_witness(input_path, circuit_path, witness_path, vk_path, srs_path)
    bt.logging.info(f"Gen witness result: {res}")

def generate_proof(input_path, vk_path, witness_path, circuit_path, pk_path, proof_path, srs_path):
    gen_witness(input_path, circuit_path, witness_path, vk_path, srs_path)
    bt.logging.info("Generating proof")
    ezkl.prove(witness_path, circuit_path, pk_path, proof_path, "single", srs_path)

class VerifiedModelSession:

    def __init__(self):
        self.model_id = 0
        self.session_id = str(uuid.uuid4())
        model_path = os.path.join(dir_path, f"model/model_{self.model_id}")

        self.pk_path = os.path.join(model_path, "pk.key")
        self.vk_path = os.path.join(model_path, "vk.key")
        self.srs_path = os.path.join(model_path, "kzg.srs")
        self.circuit_path = os.path.join(model_path, "model.compiled")
        self.settings_path = os.path.join(model_path, "settings.json")
        self.sample_input_path = os.path.join(model_path, "input.json")

        self.witness_path = os.path.join(dir_path, f"temp/witness_{self.model_id}_{self.session_id}.json")
        self.input_path = os.path.join(dir_path, f"temp/input_{self.model_id}_{self.session_id}.json")
        self.proof_path = os.path.join(dir_path, f"temp/model_{self.model_id}_{self.session_id}.proof")

        self.py_run_args = ezkl.PyRunArgs()
        self.py_run_args.input_visibility = "public"
        self.py_run_args.output_visibility = "public"
        self.py_run_args.param_visibility = "fixed"
        self.batch_size = 1

        self.worker_status = "idle"

    def gen_input_file(self):
        input_data = [self.public_inputs]
        input_shapes = [[self.batch_size]]
        data = {"input_data": input_data, "input_shapes": input_shapes}

        dir_name = os.path.dirname(self.input_path)
        os.makedirs(dir_name, exist_ok=True)

        with open(self.input_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def gen_proof_file(self, proof_string, instances):
        dir_name = os.path.dirname(self.proof_path)
        os.makedirs(dir_name, exist_ok=True)
        proof_json = json.loads(proof_string)
        new_instances = instances[0]
        bt.logging.trace(f"New instances: {new_instances}")
        new_instances.append(proof_json["instances"][0][-1])
        bt.logging.trace(f"New instances after appending with last instance from output: {new_instances}")
        proof_json["instances"] = [new_instances]
        bt.logging.trace(f"Proof json: {proof_json}")

        with open(self.proof_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(proof_json))

    def gen_proof(self, public_inputs):
        self.worker_status = "busy"
        self.public_inputs = public_inputs
        try:
            bt.logging.info("Generating input file")
            self.gen_input_file()
            bt.logging.info("Starting generating proof process...")
            start_time = time.time()
            generate_proof(
                input_path=self.input_path,
                vk_path=self.vk_path,
                witness_path=self.witness_path,
                circuit_path=self.circuit_path,
                pk_path=self.pk_path,
                proof_path=self.proof_path,
                srs_path=self.srs_path
            )
            end_time = time.time()
            proof_time = end_time - start_time
            bt.logging.info(f"Proof generation took {proof_time} seconds")
            with open(self.proof_path, "r", encoding="utf-8") as f:
                proof_content = f.read()

            self.worker_status = "idle"
            return proof_content, proof_time

        except Exception as e:
            bt.logging.error(f"An error occurred: {e}")
            self.worker_status = "idle"
            return f"An error occurred on miner proof: {e}", 0

    def verify_proof(self):
        res = ezkl.verify(self.proof_path, self.settings_path, self.vk_path, self.srs_path)
        return res

    def verify_proof_and_inputs(self, proof_string, inputs):
        if proof_string is None:
            return False
        self.public_inputs = inputs
        self.gen_input_file()
        gen_witness(self.input_path, self.circuit_path, self.witness_path, self.vk_path, self.srs_path)
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
        self.remove_temp_files()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

model_session = VerifiedModelSession()

def generateModel0Proof(synapse: QueryZkProof) -> QueryZkProof:
    query_input = synapse.query_input
    time_in = time.time()
    bt.logging.info("Received request from validator")
    bt.logging.info(f"Input data: {query_input}")
    if query_input is not None:
        public_inputs = query_input["public_inputs"]
    else:
        bt.logging.error("Received empty query input")
        return synapse

    try:
        bt.logging.info("Model session created successfully")
        query_output, proof_time = model_session.gen_proof(public_inputs)
        model_session.end()
    except Exception as e:
        synapse.query_output = "An error occurred"
        bt.logging.error("An error occurred while generating proven output", e)
        return synapse
    synapse.query_output = query_output
    bt.logging.info(f"✅ Proof completed {query_output}")
    time_out = time.time()
    delta_t = time_out - time_in
    bt.logging.info(
        f"Total response time {delta_t}s. Proof time: {proof_time}s. Overhead time: {delta_t - proof_time}s."
    )
    if delta_t > 300:
        bt.logging.error(
            "Response time is greater than validator timeout. This indicates your hardware is not processing validator's requests in time."
        )
    return synapse

