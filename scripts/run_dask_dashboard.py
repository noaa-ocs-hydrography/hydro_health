import sys
import os
import pathlib

HH_MODEL = pathlib.Path(__file__).parents[1] / 'src'
print(HH_MODEL)
sys.path.append(str(HH_MODEL))


from hydro_health.engines.Engine import Engine


if __name__ == "__main__":
    engine = Engine()
    engine.setup_dask('aws')
    print(engine.client.dashboard_link)
    while True:
        user_input = input("Press 'q' to quit: ")
        if user_input == 'q':
            engine.close_dask()
            break
        print("Closed dask")