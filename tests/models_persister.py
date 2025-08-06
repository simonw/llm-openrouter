import json
from vcr.persisters.filesystem import FilesystemPersister

# Only save response data for a few models to keep the recording size down.
TEST_MODEL_IDS = {
    "openai/gpt-3.5-turbo",
    "openai/gpt-4.1-mini",
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-sonnet-4",
}


class TruncatedModelsFilesystemPersister(FilesystemPersister):
    @staticmethod
    def save_cassette(cassette_path, cassette_dict, serializer):
        for request, response in zip(
            cassette_dict["requests"], cassette_dict["responses"]
        ):
            body = response.get("body", {})
            if request.url.endswith("/api/v1/models") and "string" in body:
                data = json.loads(body["string"])
                if "data" in data:
                    data["data"] = [
                        model
                        for model in data["data"]
                        if model.get("id") in TEST_MODEL_IDS
                    ]
                    body["string"] = json.dumps(data)

        FilesystemPersister.save_cassette(cassette_path, cassette_dict, serializer)
