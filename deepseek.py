'''
Author: mksyk cuirj04@gmail.com
Date: 2024-09-13 14:15:36
LastEditors: mksyk cuirj04@gmail.com
LastEditTime: 2024-09-14 08:19:21
FilePath: /LLM-medical-KG/deepseek.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V2-Lite-Chat", trust_remote_code=True,resume_download = True)