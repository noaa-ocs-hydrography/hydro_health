
import requests
import pathlib

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class LogfileDownloadEngine:
    """Read logfile for failed file downloads"""
    
    def __init__(self) -> None:
        self.log_file = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\log_prints_test_file.txt"
        self.digital_coast = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\raw\DigitalCoast"
        self.download_folder = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\raw\Digital_Cost_Manual_Downloads\laz_issues"

    def download_codes(self, code_errors) -> None:
        # https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/laz/geoid18/8407/20160510_424500e_2867500n.copc.laz
        pass

    def download_files(self, timeout_errors) -> None:
        # https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/laz/geoid18/6271/20160725_500500e_2730500n.copc.laz
        for url in timeout_errors:
            provider_path = url.split('.com/')[1] # laz/geoid18/6271/20160725_500500e_2730500n.copc.laz
            # find DigitalCoast provider folder with laz/geoid18/6271/

            output_file = pathlib.Path(self.download_folder) / pathlib.Path(provider_path)
            output_file.parents[0].mkdir(parents=True, exist_ok=True)
            
            retry_strategy = Retry(
                total=3,  # retries
                backoff_factor=1,  # delay in seconds
                status_forcelist=[404],  # Status codes to retry on
                allowed_methods=["GET"]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            request_session = requests.Session()
            request_session.mount("https://", adapter)
            request_session.mount("http://", adapter)

            intersected_response = request_session.get(url, timeout=5)
            if intersected_response.status_code == 200:
                with open(output_file, 'wb') as file:
                    file.write(intersected_response.content)
            else:
                print(f'Failed to download: {url}')


    def get_timeout_errors(self) -> list[str]:
        """Read all timeout errors from log file"""

        with open(self.log_file, 'r') as reader:
            time_out_errors = []
            for line in reader.readlines():
                if 'Timeout error' in line:
                    url = line.split(': ')[1].strip()
                    time_out_errors.append(url)
        return time_out_errors
    
    def get_code_errors(self) -> list[str]:
        """Read all code errors from log file"""

        with open(self.log_file, 'r') as reader:
            code_errors = []
            for line in reader.readlines():
                if 'LAZ Download failed' in line:
                    url = line.split(': ')[1].strip()
                    code_errors.append(url)
        return code_errors

    def run(self) -> None:
        # timeout_errors = self.get_timeout_errors()
        # self.download_files(timeout_errors)
        code_errors = self.get_code_errors()
        self.download_files(code_errors)

if __name__ == "__main__":
    engine = LogfileDownloadEngine()
    engine.run()

    # TODO delete these downloaded and converted files if the log file does not show any failed downloads