import csv
import json
import os
import re
import shutil
import threading
from abc import ABC, abstractmethod
from time import sleep

import diskcache as dc
import gsheets
import rl.utils.io
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from tqdm import tqdm

SOURCE_DATA_ROOT = str(rl.utils.io.get_data_path() / "source")
os.makedirs(os.path.dirname(SOURCE_DATA_ROOT), exist_ok=True)

STORAGE_CREDENTIALS_PATH = str(rl.utils.io.get_data_path() / "storage.json")
CLIENT_SECRETS_PATH = str(rl.utils.io.get_data_path() / "client_secrets.json")


cache = dc.Cache(os.path.join(SOURCE_DATA_ROOT, ".cache"))

_gdrive_auth = None


def gdrive_auth():
    global _gdrive_auth
    if _gdrive_auth is not None:
        return _gdrive_auth
    _gdrive_auth = GoogleAuth(settings={"client_config_file": CLIENT_SECRETS_PATH})
    if os.path.exists(STORAGE_CREDENTIALS_PATH):
        _gdrive_auth.LoadCredentialsFile(STORAGE_CREDENTIALS_PATH)
        return _gdrive_auth
    _gdrive_auth.LocalWebserverAuth()
    _gdrive_auth.SaveCredentialsFile(STORAGE_CREDENTIALS_PATH)
    return _gdrive_auth


class DataSource(ABC):
    name: str
    purpose_description: str
    format_description: str

    data_sources = {}

    def __init__(self, name, purpose_description, format_description):
        self.name = name
        self.purpose_description = purpose_description
        self.format_description = format_description
        self.data_sources[name] = self

    @abstractmethod
    def download(self):
        pass

    def complete(self):
        return os.path.exists(self.path())

    def path(self):
        return f"{SOURCE_DATA_ROOT}/{self.name}"

    def get(self):
        if not self.complete():
            self.download()
        return self.path()

    def refresh(self):
        if not self.complete():
            shutil.rmtree(self.path())
        self.download()


class GoogleDriveDataSource(DataSource):
    file_id: str
    share_url: str

    def __init__(self, name, purpose_description, format_description, share_url):
        self.share_url = share_url
        self.file_id = re.search(r"/file/d/([^/]+)/", share_url).group(1)
        super().__init__(name, purpose_description, format_description)

    def download(self):
        auth = gdrive_auth()
        drive = GoogleDrive(auth)
        file = drive.CreateFile({"id": self.file_id})
        os.makedirs(os.path.dirname(self.path()), exist_ok=True)
        file.FetchContent()
        with open(self.path(), "wb") as f:
            f.write(file.content.getvalue())


class GoogleDriveFolderDataSource(DataSource):
    folder_id: str
    share_url: str
    files: list

    def __init__(self, name, purpose_description, format_description, share_url):
        self.share_url = share_url
        self.folder_id = re.search(r"/folders/([^/]+)[/?]", share_url).group(1)
        super().__init__(name, purpose_description, format_description)

    def complete(self):
        if not os.path.exists(self.path()) or not os.path.exists(
            f"{self.path()}/.folder_manifest.csv"
        ):
            return False
        with open(f"{self.path()}/.folder_manifest.csv") as f:
            reader = csv.DictReader(f)
            manifest = {row["title"] for row in reader}
        downloaded = {os.path.basename(f) for f in os.listdir(self.path())}
        return len(manifest - downloaded) == 0

    def download(self):
        auth = gdrive_auth()
        drive = GoogleDrive(auth)
        file_list = drive.ListFile(
            {"q": f"'{self.folder_id}' in parents and trashed=false"}
        ).GetList()
        os.makedirs(self.path(), exist_ok=True)
        with open(f"{self.path()}/.folder_manifest.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=["title", "id"])
            writer.writeheader()
            for file in file_list:
                writer.writerow({"title": file["title"], "id": file["id"]})
        already_downloaded = {os.path.basename(f) for f in os.listdir(self.path())}
        to_download = [f for f in file_list if f["title"] not in already_downloaded]
        pbar = tqdm(
            to_download, total=len(to_download), desc=f"Downloading {self.name}"
        )

        def download_files(files):
            for file in files:
                try:
                    file.FetchContent()
                except Exception:
                    sleep(1)
                    file.FetchContent()
                with open(f"{self.path()}/{file['title']}", "wb") as f:
                    f.write(file.content.getvalue())
                pbar.update(1)

        threads = []
        n_threads = 3
        for i in range(n_threads):
            thread = threading.Thread(
                target=download_files, args=(to_download[i::n_threads],)
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()


class GoogleSheetsDataSource(DataSource):
    share_url: str

    def __init__(
        self,
        name,
        purpose_description,
        format_description,
        share_url,
        worksheet_title=None,
    ):
        self.share_url = share_url
        self.worksheet_title = worksheet_title
        super().__init__(name, purpose_description, format_description)

    def download(self):
        sheets = gsheets.Sheets.from_files(
            CLIENT_SECRETS_PATH, STORAGE_CREDENTIALS_PATH
        )
        sheet = sheets.get(self.share_url)
        worksheet = sheet.sheets[0]
        if self.worksheet_title:
            worksheet = None
            for ws in sheet.sheets:
                if ws.title == self.worksheet_title:
                    worksheet = ws
                    break
            if worksheet is None:
                raise ValueError(f"Worksheet {self.worksheet_title} not found")
        os.makedirs(os.path.dirname(self.path()), exist_ok=True)
        worksheet.to_csv(self.path())


class DataLoopDataSource(DataSource):
    dataset_name: str

    def __init__(self, name, purpose_description, format_description, dataset_name):
        super().__init__(name, purpose_description, format_description)
        self.dataset_name = dataset_name

    def download(self):
        import cacafo.dataloop

        os.makedirs(os.path.dirname(self.path()), exist_ok=True)
        data = cacafo.dataloop.get_dataset(self.dataset_name)
        with open(self.path(), "w") as f:
            f.write("\n".join([json.dumps(row) for row in data]))


def set_root(path):
    global SOURCE_DATA_ROOT
    SOURCE_DATA_ROOT = path


def get(name):
    return DataSource.data_sources[name].get()


def list():
    return DataSource.data_sources.keys()


def get_all():
    return {name: source.get() for name, source in DataSource.data_sources.items()}


def refresh_all():
    for name, source in DataSource.data_sources.items():
        source.refresh()


GoogleDriveDataSource(
    "labels/annotated.csv",
    "List of images that were annotated by CF in the original batch, c. 2022",
    "CSV with columns: jpeg_path, tif_path",
    "https://drive.google.com/file/d/1gwxYcRfguDpExGNbGDTrAXIBH91qEZOr/view?usp=drive_link",
)

GoogleDriveDataSource(
    "labels/empty_files.csv",
    "List of empty images that were annotated by CF in the original batch, c. 2022",
    "CSV with columns: jpeg_path, tif_path, lon_min, lat_max, lon_max, lat_min",
    "https://drive.google.com/file/d/1O35Xm8MqZbw2drxW3miIpl2jiu5NQVIn/view?usp=drive_link",
)

GoogleDriveDataSource(
    "labels/labeled_boxes.csv",
    "List of positive, labeled images that were annotated by CF in the original batch, c. 2022",
    "CSV with columns: jpeg_path, tif_path, x_min, y_min, x_max, y_max, lon_min, lat_max, lon_max, lat_min, class_id, class_name",
    "https://drive.google.com/file/d/1a3CyJwVnUAoKe_XoWzVMe8ili3g8f_NY/view?usp=drive_link",
)


GoogleDriveDataSource(
    "labels/non_med_recall.jsonl",
    "Annotations for non-med and other county groups to improve recall",
    "DataLoop-formatted JSON, in lines format",
    "https://drive.google.com/file/d/1hb2lpdJ5_JQjo2LECQBA5kl6Mgyh8PWB/view?usp=drive_link",
)

GoogleDriveDataSource(
    "labels/labeled_boxes.csv",
    "List of positive, labeled images that were annotated by CF in the original batch, c. 2022",
    "CSV with columns: jpeg_path, tif_path, x_min, y_min, x_max, y_max, lon_min, lat_max, lon_max, lat_min, class_id, class_name",
    "https://drive.google.com/file/d/1a3CyJwVnUAoKe_XoWzVMe8ili3g8f_NY/view?usp=drive_link",
)

GoogleDriveFolderDataSource(
    "labels/missed_active_permits",
    "Annotations for permits missed by active learner",
    "Directory of dataloop-formatted annotation JSON files",
    "https://drive.google.com/drive/folders/1n0r1ATGsKkr688ChDcQyT5-jUdzOZVZZ?usp=drive_link",
)


GoogleDriveFolderDataSource(
    "labels/missed_historical_permits",
    "Annotations for historical permits missed by active learner",
    "Directory of dataloop-formatted annotation JSON files",
    "https://drive.google.com/drive/folders/1m_MFZjboApL_5VVi1rAqT5-dKQ9H4oLn?usp=drive_link",
)

GoogleDriveFolderDataSource(
    "labels/adjacent",
    "Annotations for adjacent tiles of all previous positive detections",
    "Directory of dataloop-formatted annotation JSON files",
    "https://drive.google.com/drive/folders/1mWliqP6bEFAyJtnjCmDODGEVDlgmzcke?usp=sharing",
)

GoogleSheetsDataSource(
    "adjacent_image_flag_descriptions.csv",
    "List of flagged adjacent tile annotations, with a code describing the action to take",
    "CSV with columns: file, CF Flag description, coords, action, , labeler -- the blank column is the action code",
    "https://docs.google.com/spreadsheets/d/1w9yCKe9I6Kaifhl63Nsn0awKbh9R4vUXk-fZUqrcX1A/edit?usp=sharing",
)

GoogleSheetsDataSource(
    "construction_dating_adjacent_images_v1.csv",
    "Construction dating annotations for adjacent tiles",
    "CSV with columns: processing date, annotator, cafo id, cafo uuid, latitude, longitude, lat/lon,...",
    "https://docs.google.com/spreadsheets/d/1ivMOmfPb3Rqtj9eiiA0oldlSA_p69A-pe70ni2A1nKA/edit?usp=sharing",
)

GoogleSheetsDataSource(
    "construction_dating_ex_ante_permits.csv",
    "Construction dating annotations for ex ante permits + cleanup",
    "CSV with columns: processing date, annotator, cafo id, cafo uuid, latitude, longitude, lat/lon,...",
    "https://docs.google.com/spreadsheets/d/1mrSutjzzfqmYfodSIFbvRgo76Xq87usjrvfCDuQHJHM/edit?usp=sharing",
)

GoogleSheetsDataSource(
    "construction_dating_v1.csv",
    "Construction dating annotations for base tiles.",
    "CSV with columns: processing date, annotator, cafo id, cafo uuid, latitude, longitude, lat/lon,...",
    "https://docs.google.com/spreadsheets/d/1RoNWqaSSYwGB8sGbl2kRAiPw1Ail5SKmxkfrz6-qOb8/edit?usp=sharing",
)

GoogleSheetsDataSource(
    "construction_dating_ex_ante_adjacents.csv",
    "Construction dating annotations for ex ante permit adjacents.",
    "CSV with columns: processing date, annotator, cafo id, cafo uuid, latitude, longitude, lat/lon,...",
    "https://docs.google.com/spreadsheets/d/1P40ImWtcb_EyRdXUILrE1iAzXnjwnNsZ5lQT-A2puJ0/edit?usp=sharing",
)

GoogleSheetsDataSource(
    "construction_dating_missed_adjacents.csv",
    "Missed adjacent construction dating annotations",
    "CSV with columns: processing date, annotator, cafo id, cafo uuid, latitude, longitude, lat/lon,...",
    "https://docs.google.com/spreadsheets/d/18ZgGqDZs0LoCW13LsN860AN5VGuVezCXaYsMSYnzeZ4/edit?usp=sharing",
)


GoogleSheetsDataSource(
    "permits_parcels.csv",
    "permit parcel join",
    "CSV with columns representing permits and also parcels",
    "https://docs.google.com/spreadsheets/d/1B14BrSq5p9za45o-zrTMBAP-l3DW_0-87igdr6LfX3A/edit?usp=sharing",
)

GoogleDriveDataSource(
    "labels/ex_ante_permits.jsonl",
    "Annotations for ex ante permits",
    "DataLoop-formatted JSON, in lines format",
    "https://drive.google.com/file/d/1PBZ8rOOgIY5rJoNcT7BehtKpEqpKNaBG/view?usp=drive_link",
)

GoogleDriveDataSource(
    "labels/ex_ante_permit_adjacents.jsonl",
    "Annotations for ex ante permit adjacents",
    "DataLoop-formatted JSON, in lines format",
    "https://drive.google.com/file/d/1O7JNidkhLrP-djVu4n_avHQSSLLoa0kY/view?usp=drive_link",
)

GoogleDriveDataSource(
    "labels/missed_adjacents.jsonl",
    "Annotations for missed adjacents due to bug",
    "DataLoop-formatted JSON, in lines format",
    "https://drive.google.com/file/d/1bNppgyWXqaBfdFIXyPMTlaKCDQdNyiJz/view?usp=sharing",
)

for worksheet in (
    "facilities_w_no_animal_type_or_permit",
    "facilities_w_no_animal_type_or_permit_v2",
    "facilities_still_without_animal_types",
    "facilities_w_two_or_more_types",
    "animal_typing_2024-06-10",
):
    GoogleSheetsDataSource(
        f"new_animal_typing/{worksheet}.csv",
        "Facilities with no animal type or permit",
        "CSV with columns: id,uuid,latitude,longitude,coords,labler,isAFO?,isCAFO?,animal type,subtype,notes",
        "https://docs.google.com/spreadsheets/d/1Ynj9h2iBnvvxQx_IGskpcVl5bU7ezD-DQQKJLfgS-tE/edit?usp=sharing",
        worksheet_title=worksheet,
    )


GoogleDriveFolderDataSource(
    "parcels/polygon-centroids-merge-v2",
    "Polygon centroids matched to parcels",
    "folder of csvs with parcel heading",
    "https://drive.google.com/drive/folders/1nDihOmvW2NY92z5aUWscD6Te4efqM-9A?usp=drive_link",
)

GoogleDriveFolderDataSource(
    "afo-permits-geocoded",
    "Parcels matched to permits",
    "folder of csvs with permit heading",
    "https://drive.google.com/drive/folders/1LAirMDQEmFgRcznrZTTtyTJ5-D1ifBa5?usp=drive_link",
)

################

GoogleDriveDataSource(
    "parcels.csv",
    "Parcels",
    "csv with number, county, owner, address, data",
    "https://drive.google.com/file/d/18SnzOmuQfCKg_Ze-SMaQXdH8y5Rs1ETd/view?usp=sharing",
)

GoogleDriveDataSource(
    "parcel_locations.csv",
    "Parcel locations",
    "csv with county_name, number, lat, lon",
    "https://drive.google.com/file/d/1ott9PR6B17jakxZLiu44ygl7yXLVb3ji/view?usp=drive_link",
)

GoogleDriveDataSource(
    "geocoded_addresses.csv",
    "Geocoded addresses",
    "csv with Latitude, Longitude, Address, City, State, Zip",
    "https://drive.google.com/file/d/1YiNvq94ctJ6WE8a_uwn6dU__Gy7nm2gp/view?usp=drive_link",
)

GoogleDriveDataSource(
    "permits.csv",
    "All permit locations from CWIQS",
    "CSV with columns including Latitude, Longitude, Facility Address, etc.",
    "https://drive.google.com/file/d/1ut68DkRFRMhs14-RNgHya6Yo1ZZok8-r/view?usp=drive_link",
)


GoogleSheetsDataSource(
    "county_groups.csv",
    "County super groups",
    "CSV with columns: County, Group Name",
    "https://docs.google.com/spreadsheets/d/1TKMLlPGi7Dh5KYr5ISfF5o10pP5UHeVp_or5rEn7KsI/edit?usp=sharing",
)

GoogleDriveDataSource(
    "counties.csv",
    "List of counties in California",
    "CSV with columns: county, the_geom, lat, lon",
    "https://drive.google.com/file/d/1eF8OuHcobFcxeULsqYdNeuqnP2U7TnZu/view?usp=sharing",
)
