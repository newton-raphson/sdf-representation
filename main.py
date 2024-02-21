from configgen.config_reader import Configuration
from executor.executor import Executor
from executor.executordistributed import DistributedExecutor
from evaluations import post_process
import sys
if __name__ == '__main__':
    # pass the config file path to the function
    config_file_path = sys.argv[1]
    print("Running with config file: ",config_file_path)
    config = Configuration(config_file_path)
    if config.distributed:
        executor = DistributedExecutor(config) 
        executor.run()
    else:
        executor = Executor(config) 
        executor.run()

    
    