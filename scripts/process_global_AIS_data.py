import requests

def get_density_map(output_path, file_name, start_time, end_time):
    url = "https://gmtds.maplarge.com/ogc/ais:density/wms"
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": "ais:density",
        "STYLES": "",
        "CRS": "EPSG:3857",
        "BBOX": "-180,-85.0511,180,85.0511",
        "WIDTH": 1024,
        "HEIGHT": 512,
        "FORMAT": "image/png",
        "TIME": f"{start_time}/{end_time}",
        "TRANSPARENT": "true",
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        full_path = f"{output_path}\\{file_name}.png"
        with open(full_path, "wb") as file:
            file.write(response.content)
        print(f"Density map saved as '{full_path}'")
    else:
        print(f"Error: {response.status_code} - {response.text}")

output_directory = r'C:\Users\aubrey.mccutchan\Repo\hydro_health\hydro_health\inputs\ais_data'
output_filename = "density_map_october_2020"
start_time = "2020-10-01T00:00:00Z"
end_time = "2020-10-31T23:59:59Z"

get_density_map(output_directory, output_filename, start_time, end_time)
