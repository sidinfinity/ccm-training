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
        self.pipeline = transformers.pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B")  

    def send_and_get_response(self, query: str):
        return self.pipeline(query)


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

    def get_dataframe(controls: list, cc: ModelClient) -> pd.DataFrame:
        pd = pd.DataFrame()





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

    query = "how to make banana bread" 

    response = send_and_get_response(cc, query)
    logger.info(f"Response: {response}")