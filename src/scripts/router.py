from pydantic import BaseModel
import json 

class Answer(BaseModel):
    choice: int
    reason: str

class RouterOutputParser:
    def parse(self, output: str) -> Answer:
        output = output.strip()
        json_output = output[output.find("["):output.find("]")+1]
        json_dict = json.loads(json_output)[0]
        return Answer(choice=json_dict["choice"], reason=json_dict["reason"])

    def format(self, prompt_template: str) -> str:
        format_str = (
            """The output should be formatted as a JSON instance that conforms to the JSON schema below. 
            {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "choice": {
                    "type": "integer"
                  },
                  "reason": {
                    "type": "string"
                  }
                },
                "required": [
                  "choice",
                  "reason"
                ],
                "additionalProperties": false
              }
            }
            """
        )
        return prompt_template + "\n\n" + format_str.replace("{", "{{").replace("}", "}}")
