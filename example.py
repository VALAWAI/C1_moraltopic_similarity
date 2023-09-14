#from mesa import Agent, Model
#from mesa.time import BaseScheduler
import multiprocessing
from random import seed, uniform
from moraltopic_similarity import create_app

multiprocessing.set_start_method('fork')


# WARNING: if running in OSX you need to execute the following instruction in each session
# export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

seed(100)

if __name__ == '__main__':
    app = create_app()
    p1 = multiprocessing.Process(target=app.run, kwargs={'debug': True})
    p1.start()

    import requests
    url = "http://127.0.0.1:5000"

    data = {
        'user_moral': [.8, .2], 
        'user_topic': [.5, .3, .2], 
        'item_moral': [.8, .2]
        'item_topic': [.5, .3, .2], 
    }

    def request_and_print():
        response = requests.get(f"{url}/moraltopic_similarity", json=data)
        print(response.status_code)
        print(response.json())

    p2 = multiprocessing.Process(target=request_and_print)
    p2.start()