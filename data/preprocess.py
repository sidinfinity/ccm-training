import argparse
import json
import logging
import openai  # type: ignore
import os
import pandas as pd  # type: ignore
import torch
import transformers

from typing import Optional, List


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

    def send_and_get_response(self, query: str) -> str:
        try:
            response = self.client.chat.completions.create(
                messages=[
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


class Query:

    QUERY_FORMAT = {
        "1": "Control: ",
        "2": "Statement: ",
        "3": "Implementation Guideline: ",
        "4": (
                "The information above is for a 'cloud control,' which is a "
                "type of security measure taken to solve certain issues "
                "relating to cloud security. A user is alerted with a security "
                "risk relating to the control above, and wants to ask an LLM "
                "for help. Create a set of questions and answers that can help "
                "train the LLM. Give response in json format, with the keys "
                "starting off with: question1, answer1, question2, answer2, etc."
            )
    }

    def __init__(self, control: str, statement: str, implementation: str):
        self.control = control
        self.statement = statement
        self.implementation = implementation

    def create_query(self) -> str:
        query_format = self.QUERY_FORMAT
        query = (
            f"{query_format['1']}{self.control}\n"
            f"{query_format['2']}{self.statement}\n"
            f"{query_format['3']}{self.implementation}\n{query_format['4']}"
        )
        return query


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

    def get_queries(self, allowed_controls: Optional[List]) -> List[str]:
        queries = []

        for group in self._data['catalog']['groups']:
            print(group['id'])
            # for testing purposes, update later
            if allowed_controls and group['id'] not in allowed_controls:
                continue
            for control in group['controls']:
                query = Query(
                    control['title'],
                    control['parts'][0]['prose'],
                    control['parts'][1]['prose']
                )
                queries.append(query.create_query())

        return queries

    def create_dataset(
        self, allowed_controls: Optional[List], cc: ModelClient
    ) -> pd.DataFrame:
        df = pd.DataFrame(columns=['input', 'output'])

        queries = self.get_queries(allowed_controls)

        for query in queries:
            try:
                json_response = json.loads(send_and_get_response(cc, query))

                questions = [
                    key for key in json_response.keys()
                    if key.startswith('question')
                ]

                logger.info(f"Response: {json_response}")

                for question in questions:
                    result = {
                        'input': [json_response[question]],
                        'output': [json_response[question.replace('question', 'answer')]]
                    }
                    logger.info(f"Got result: {result}")
                    df = pd.concat(
                        [df, pd.DataFrame(result)], ignore_index=False
                    )
            except Exception as e:
                logger.error(
                    f"Error in sending query: {query}: {e}"
                )

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

    parser.add_argument(
        '--fname',
        type=str,
        help='CSV file name to save the data to',
        default='dataset.xml'
    )

    args = parser.parse_args()
    td = TrainingData()
    logger.info("Loading data")
    td.load_data()
    if args.model == "llama":
        cc = LlamaClient()
    else:
        cc = GPTClient()

    df = td.create_dataset(None, cc)
    df.to_xml(args.fname, index=True)
