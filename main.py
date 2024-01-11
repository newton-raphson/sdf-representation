from configgen.config_reader import Configuration
from executor.executor import Executor
from evaluations import post_process
import sys
if __name__ == '__main__':
    # pass the config file path to the function
    config_file_path = sys.argv[1]
    config = Configuration(config_file_path)
    executor = Executor(config) 
    if config.ppo:
        post_process.post_process(executor)
    else:
        executor.run()
    