import argparse
import json
import logging
import openai # type: ignore
import os
import pandas as pd # type: ignore
import pprint as pp
import torch
import transformers

from retrying import retry # type: ignore
from typing import Optional



logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)




class ModelClient:
    """ base class for interacting with Model API """

class LlamaClient(ModelClient):
    def __init__(self, model_path="CodeLlama-7b"):
        super().__init__()
        self.pipeline = transformers.pipeline(
            "text-generation",
            model="meta-llama/Meta-Llama-3-8B",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="mps"
        )

    def send_and_get_response(self, query: str):
        return self.pipeline(query)[0]['generated_text']


class GPTClient(ModelClient):
    """ Class to interacti with OpenAI API """

    def __init__(self):
        super().__init__()
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def send_and_get_response(self, query: str):
        try:
            response = self.client.chat.completions.create(
                messages = [
                    {
                        "role": "user",
                        "content": query,
                    }
                ], model="gpt-4",
            )
        except Exception as e:
            logger.error(f"Error in getting response from OpenAI: {e}")
            raise

        return response.choices[0].message.content


class TrainingData:
    """ Class to load and preprocess the training data """

    def __init__(self, file_path: str = "ccm_json/ccm.json"):
        self.file_path = file_path
        self._data = None

    @property
    def data(self):
        return self._data 

    # Load the json file
    def load_data(self):
        file_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), self.file_path
        )
        try:
            file = open(file_path, 'r')
            self._data = json.load(file)
            file.close()
        except Exception as e:
            logger.error(f"Error in loading data: {e}")
            raise

    def get_dataframe(self, allowed_controls: list, cc: ModelClient) -> pd.DataFrame:
        df = pd.DataFrame(columns=['input', 'output',])

        controls = []

        for group in self._data['catalog']['groups']:
            print(group['id'])
            if (group['id'] not in allowed_controls): # for testing purposes, update later
                continue 
            for control in group['controls']:
                controls.append({
                    'title': str(control['title']),
                    'statement': str(control['parts'][0]['prose']),
                    'implementation': str(control['parts'][1]['prose'])
                })
        
        # print(controls[0]['title'])
        # print(controls[0]['statement'])
        # print(controls[0]['implementation'])

        for control in controls:
            query = "to do"
            json_response = json.load(send_and_get_response(cc, query))
            # df.append({'input':json_response['question'], 'output':json_response['answer']}, ignore_index=True)
        
        return df
            



def send_and_get_response(cc: ModelClient, query: str) -> Optional[str]:
    try:
        return cc.send_and_get_response(query)
    except Exception as e:
        logger.error(f"Error in getting response from Model: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['gpt', 'llama'], 
        help='Specify the model type (gpt or llama)',
        default='gpt'
    )

    args = parser.parse_args()
    td = TrainingData()
    logger.info("Loading data")
    td.load_data()
    if args.model == "llama":
        cc = LlamaClient()
    else: 
        cc = GPTClient()

    # td.get_dataframe(['IVS'], cc)

    query = """
Control: Infrastructure and Virtualization Security Policy and Procedures
Statement: Establish, document, approve, communicate, apply, evaluate and maintain
policies and procedures for infrastructure and virtualization security. Review
and update the policies and procedures at least annually.


The information above is for a "cloud control," which is a type of security measure taken to solve certain issues relating to cloud security. A user is alerted with a security risk relating to the control above, and wants to ask an LLM for help. Create a single question and answer relating to the control above and this statement: Governance and control VM lifecycle management. give response in json format, and both the question and answer should be simple such as a paragraph with no string formatting
"""
    logger.info(f"Sending query: {query}")
    response = send_and_get_response(cc, query)
    logger.info(f"Response: {json.load(response)}")


# prompt: 
"""
Control: Infrastructure and Virtualization Security Policy and Procedures
Statement: Establish, document, approve, communicate, apply, evaluate and maintain
policies and procedures for infrastructure and virtualization security. Review
and update the policies and procedures at least annually.


The information above is for a "cloud control," which is a type of security measure taken to solve certain issues relating to cloud security. A user is alerted with a security risk relating to the control above, and wants to ask an LLM for help. Create a single question and answer relating to the control above and this statement: Governance and control VM lifecycle management. give response in json format, and both the question and answer should be simple such as a paragraph with no string formatting
"""