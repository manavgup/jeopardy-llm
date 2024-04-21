import time
from pathlib import Path
from pprint import pprint
from typing import Optional

import pandas as pd
from datasets_test import load_dataset

from dotenv import load_dotenv
from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (
    DecodingMethod,
    FilePurpose,
    TextGenerationParameters,
    TuneAssetType,
    TuneParameters,
    TuneStatus,
)

class PromptTuner:
    def __init__(self, credentials: Credentials, 
                    num_training_samples: int = 100, 
                    num_validation_samples: int = 20, 
                    data_root: Optional[Path] = None):
        self.credentials: Credentials = credentials
        self.client = Client(credentials=credentials)
        self.num_training_samples = num_training_samples
        self.num_validation_samples =  num_validation_samples
        self.data_root = data_root or Path(__file__).parent.resolve() / ".data"
        self.training_file = self.data_root / "train.jsonl"
        self.validation_file = self.data_root / "validation.jsonl"

    def create_dataset(self, dataset_name: str, config: str):
        Path(self.data_root).mkdir(parents=True, exist_ok=True)
        if self.training_file.exists() and self.validation_file.exists():
            print("Dataset is already prepared")
            return

        data = load_dataset(dataset_name, config)
        df = pd.DataFrame(data["train"]).sample(n=self.num_training_samples + self.num_validation_samples)
        df.rename(columns={"sentence": "input", "label": "output"}, inplace=True)
        df["output"] = df["output"].astype(str)
        train_jsonl = df.iloc[:self.num_training_samples].to_json(orient="records", lines=True, force_ascii=True)
        validation_jsonl = df.iloc[self.num_training_samples:].to_json(orient="records", lines=True, force_ascii=True)
        with open(self.training_file, "w") as fout:
            fout.write(train_jsonl)
        with open(self.validation_file, "w") as fout:
            fout.write(validation_jsonl)

    def upload_files(self, update=True):
        files_info = self.client.file.list(search=self.training_file.name).results
        files_info += self.client.file.list(search=self.validation_file.name).results

        filenames_to_id = {f.file_name: f.id for f in files_info}
        for filepath in [self.training_file, self.validation_file]:
            filename = filepath.name
            if filename in filenames_to_id and update:
                print(f"File already present: Overwriting {filename}")
                self.client.file.delete(filenames_to_id[filename])
                response = self.client.file.create(file_path=filepath, purpose=FilePurpose.TUNE)
                filenames_to_id[filename] = response.result.id
            if filename not in filenames_to_id:
                print(f"File not present: Uploading {filename}")
                response = self.client.file.create(file_path=filepath, purpose=FilePurpose.TUNE)
                filenames_to_id[filename] = response.result.id
        return filenames_to_id[self.training_file.name], filenames_to_id[self.validation_file.name]

    def tune_model(
        self,
        model_id: str,
        name: str,
        hyperparams: TuneParameters,
        task_id: str = "classification",
    ):
        training_file_id, validation_file_id = self.upload_files(update=True)

        print(f"Tuning model: {name}")
        tune_result = self.client.tune.create(
            model_id=model_id,
            name=name,
            tuning_type="prompt_tuning",
            task_id=task_id,
            parameters=hyperparams,
            training_file_ids=[training_file_id],
            # validation_file_ids=[validation_file_id], # TODO: Broken at the moment - this causes tune to fail
        ).result

        while tune_result.status not in [TuneStatus.FAILED, TuneStatus.HALTED, TuneStatus.COMPLETED]:
            new_tune_result = self.client.tune.retrieve(tune_result.id).result
            print(f"Waiting for tune to finish, current status: {tune_result.status}")
            tune_result = new_tune_result
            time.sleep(10)

        if tune_result.status in [TuneStatus.FAILED, TuneStatus.HALTED]:
            print("Model tuning failed or halted")
            return None

        return tune_result

    def classify(self, tune_result, prompt):
        print(f"Prompt: {prompt}")
        gen_params = TextGenerationParameters(decoding_method=DecodingMethod.SAMPLE, max_new_tokens=1, min_new_tokens=1)
        gen_response = next(self.client.text.generation.create(model_id=tune_result.id, inputs=[prompt]))
        print("Answer: ", gen_response.results[0].generated_text)

    def get_tuned_models(self, limit=5, offset=0):
        interesting_metadata_fields = ["name", "id", "model_id", "created_at", "status"]
        tune_list = self.client.tune.list(limit=limit, offset=offset)
        for tune in tune_list.results:
            pprint(tune.model_dump(include=interesting_metadata_fields))

    def get_tune_details(self, tune_result):
        interesting_metadata_fields = ["name", "id", "model_id", "created_at", "status", "parameters"]
        tune_detail = self.client.tune.retrieve(id=tune_result.id).result
        pprint(tune_detail.model_dump(include=interesting_metadata_fields))

    def get_tune_logs(self, tune_result):
        logs = self.client.tune.read(id=tune_result.id, type=TuneAssetType.LOGS).decode("utf-8")
        print(logs)

    def delete_tuned_model(self, tune_result):
        self.client.tune.delete(tune_result.id)
        print("Tuned model deleted")

if __name__ == "__main__":
    from genai.credentials import Credentials

    load_dotenv()
    credentials = Credentials.from_env()

    tuner = PromptTuner(credentials, 100, 20, "./data")
    tuner.create_dataset()

    hyperparams = TuneParameters(num_epochs=2, verbalizer='classify { "0", "1", "2" } Input: {{input}} Output:')
    tune_result = tuner.tune_model(
        model_id="google/flan-t5-xl",
        name="classification-mpt-tune-api",
        hyperparams=hyperparams,
    )

    if tune_result:
        tuner.classify(tune_result, "Return on investment was 5.0 % , compared to a negative 4.1 % in 2009 .")
        tuner.get_tuned_models()
        tuner.get_tune_details(tune_result)
        tuner.get_tune_logs(tune_result)
        tuner.delete_tuned_model(tune_result)