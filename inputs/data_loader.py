import yaml
import pyupbit

def load_config(path="inputs/config.yaml"):
    with open(path, "r", encoding='utf-8') as file:
        return yaml.safe_load(file)

def get_upbit_keys():
    """Automatically gets account api keys."""
    ### UPBIT LOGIN ###
    f = open("../KEY.txt", "r")
    ACCESS_KEY = f.readline().strip()
    SECRET_KEY = f.readline().strip()

    try:
        upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
        print("Login Complete.")
        print("="*25)
        return upbit
    except Exception as e:
        print("login error occured:", e)
        return None