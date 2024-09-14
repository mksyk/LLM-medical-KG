'''
Author: mksyk cuirj04@gmail.com
Date: 2024-09-14 05:42:18
LastEditors: mksyk cuirj04@gmail.com
LastEditTime: 2024-09-14 05:42:19
FilePath: /LLM-medical-KG/try_ds.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import subprocess
import time

def run_command_with_retry(command, max_retries=5, retry_delay=5):
    retries = 0
    while retries < max_retries:
        try:
            print(f"Running command: {command}")
            subprocess.run(command, shell=True, check=True)
            print("Command executed successfully!")
            break
        except subprocess.CalledProcessError as e:
            print(f"Error encountered: {e}. Retrying {retries + 1}/{max_retries} in {retry_delay} seconds...")
            retries += 1
            time.sleep(retry_delay)
    if retries == max_retries:
        print("Max retries reached. Command failed.")

if __name__ == "__main__":
    run_command_with_retry("python deepseek.py")