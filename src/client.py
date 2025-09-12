import os

import evoml_client as ec
from dotenv import load_dotenv


def initialise_client(base_url: str):
    load_dotenv()
 
    username = os.getenv("USER_NAME")
    password = os.getenv("PASSWORD")

    print(username, password)

    ec.init(username=username, password=password, base_url=base_url)

initialise_client("https://evoml.ai")