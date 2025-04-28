import os
import json

class Target():
    '''
    Temporary implementation of Target class
    '''
    def __init__(self, model):
        self.model = model

    def respond(self, prompt):
        # 需要同时传入system和user参数
        system_prompt = "You are a helpful assistant."
        user_prompt = prompt  # 将用户输入作为user参数

        response = self.model.generate(
            system=system_prompt,
            user=user_prompt,  # 新增user参数
            max_tokens=10000,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
        if any(word in response for word in ["I cannot", "I am unable"]):
            return "[安全审查] 请求被拒绝"
        return response