import time
import re
import asyncio
import types
from typing import Union
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# Access variables
gemeni_api_key = os.getenv('gemeni_api_key')

genai.configure(api_key=gemeni_api_key)
model = genai.GenerativeModel('gemini-pro')
DEFAULT_PROMPT = ""

class Agent:
    def __init__(self, template, model=None, key_map: Union[dict, None] = None) -> None:
        self.message = []
        if isinstance(template, str):
            self.TEMPLATE = template
        elif isinstance(template, list) and len(template) > 1:
            self.TEMPLATE = template[0]
            self.template_list = template
        self.key_map = key_map
        
        if model is None:
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            self.model = model
        
        # Verify the model is correctly assigned
        if isinstance(self.model, str):
            raise ValueError("The model should be an instance of GenerativeModel, not a string.")
        
        self.func_dic = {
            'default': self.get_output,
            'padding_template': self.padding_template
        }


    def send_message(self):
        assert len(self.message) != 0 and self.message[-1]['role'] != 'assistant', 'ERROR in message format'
        try:
            ans = self.model.generate_content(
                self.message[-1]['content'],
                temperature= 0.2
            )
            self.parse_message(ans)
            return ans
        except Exception as e:
            print(e)
            time.sleep(20)
            ans = self.model.generate_content(
               self.message[-1]['content'],
               temperature = 0.2
            }
                
            )
            self.parse_message(ans)
            return ans

    
    async def send_message_async(self):
        try:
            ans = await self.model.generate_async(
                self.message[-1]['content'],
                temperature=0.2
              
            )
            self.parse_message(ans)
            return ans
        except Exception as e:
            print(e)
            await asyncio.sleep(20)
            ans = await self.model.generate_async(
                self.message[-1]['content'],
                temperature=0.2
                
            )
            self.parse_message(ans)
            return ans


    def padding_template(self, input):
        input = self.key_mapping(input)
        assert self._check_format(input.keys()), f"input lacks the necessary key"
        msg = self.TEMPLATE.format(**input)
        self.message.append({
            'role': 'user',
            'content': msg
        })

    def key_mapping(self, input):
        if self.key_map is not None:
            new_input = {}
            for key, val in input.items():
                if key in self.key_map.keys():
                    new_input[self.key_map[key]] = val
                else:
                    new_input[key] = val
            input = new_input
        return input

    def _check_format(self, key_list):
        placeholders = re.findall(r'\{([^}]+)\}', self.TEMPLATE)
        for key in placeholders:
            if key not in key_list:
                return False
        return True

    def get_output(self) -> str:
        assert len(self.message) != 0 and self.message[-1]['role'] == 'assistant'
        return self.message[-1]['content']

    def parse_message(self, completion):
        content = completion['candidates'][0]['output']  # Adapt to Gemini response structure
        role = 'assistant'  # Adjust role handling as needed
        record = {'role': role, 'content': content}
        self.message.append(record)
        return record

    def regist_fn(self, func, name):
        setattr(self, name, types.MethodType(func, self))
        self.func_dic[name] = getattr(self, name)
