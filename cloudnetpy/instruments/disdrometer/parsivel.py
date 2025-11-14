import csv
import datetime
import logging
import re
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import islice
from os import PathLike
from typing import Any
from uuid import UUID
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numpy import ma

from cloudnetpy import output
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.constants import MM_TO_M, SEC_IN_HOUR
from cloudnetpy.exceptions import DisdrometerDataError
from cloudnetpy.instruments import instruments
from cloudnetpy.utils import get_uuid

from .common import ATTRIBUTES, Disdrometer


# def parsivel2nc(
#     disdrometer_file: str | PathLike | Iterable[str | PathLike],
#     output_file: str | PathLike,
#     site_meta: dict,
#     uuid: str | UUID | None = None,
#     date: str | datetime.date | None = None,
#     telegram: Sequence[int | None] | None = None,
#     timestamps: Sequence[datetime.datetime] | None = None,
# ) -> UUID:
#     """Converts OTT Parsivel-2 disdrometer data into Cloudnet Level 1b netCDF
#     file.

#     Args:
#         disdrometer_file: Filename of disdrometer file or list of filenames.
#         output_file: Output filename.
#         site_meta: Dictionary containing information about the site. Required key
#             is `name`.
#         uuid: Set specific UUID for the file.
#         date: Expected date of the measurements as YYYY-MM-DD.
#         telegram: List of measured value numbers as specified in section 11.2 of
#             the instrument's operating instructions. Unknown values are indicated
#             with None. Telegram is required if the input file doesn't contain a
#             header.
#         timestamps: Specify list of timestamps if they are missing in the input file.

#     Returns:
#         UUID of the generated file.

#     Raises:
#         DisdrometerDataError: Timestamps do not match the expected date, or unable
#             to read the disdrometer file.

#     Examples:
#         >>> from cloudnetpy.instruments import parsivel2nc
#         >>> site_meta = {'name': 'Lindenberg', 'altitude': 104, 'latitude': 52.2,
#         'longitude': 14.1}
#         >>> uuid = parsivel2nc('parsivel.log', 'parsivel.nc', site_meta)

#     """
#     if isinstance(date, str):
#         date = datetime.date.fromisoformat(date)
#     uuid = get_uuid(uuid)
#     if isinstance(disdrometer_file, str | PathLike):
#         disdrometer_file = [disdrometer_file]
#     disdrometer = Parsivel(disdrometer_file, site_meta, telegram, date, timestamps)
#     disdrometer.sort_timestamps()
#     disdrometer.remove_duplicate_timestamps()
#     disdrometer.mask_invalid_values()
#     if len(disdrometer.data["time"].data) < 2:
#         msg = "Too few data points"
#         raise DisdrometerDataError(msg)
#     disdrometer.convert_units()
#     disdrometer.add_meta()
#     attributes = output.add_time_attribute(ATTRIBUTES, disdrometer.date)
#     output.update_attributes(disdrometer.data, attributes)
#     output.save_level1b(disdrometer, output_file, uuid)
#     return uuid




def parsivel2nc(
    disdrometer_file: str | PathLike | Iterable[str | PathLike],
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
    telegram: Sequence[int | None] | None = None,
    timestamps: Sequence[datetime.datetime] | None = None,
    file_contains_date_range: bool | None = False
) -> UUID:
    """Converts OTT Parsivel-2 disdrometer data into Cloudnet Level 1b netCDF
    file.
    Args:
        disdrometer_file: Filename of disdrometer file or list of filenames.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            is `name`.
        uuid: Set specific UUID for the file.
        date: Expected date of the measurements as YYYY-MM-DD.
        telegram: List of measured value numbers as specified in section 11.2 of
            the instrument's operating instructions. Unknown values are indicated
            with None. Telegram is required if the input file doesn't contain a
            header.
        timestamps: Specify list of timestamps if they are missing in the input file.
        file_contains_date_range: If True, extract data for the specified date from
            a multi-date file before processing.
    Returns:
        UUID of the generated file.
    Raises:
        DisdrometerDataError: Timestamps do not match the expected date, or unable
            to read the disdrometer file.
    Examples:
        >>> from cloudnetpy.instruments import parsivel2nc
        >>> site_meta = {'name': 'Lindenberg', 'altitude': 104, 'latitude': 52.2,
                'longitude': 14.1}
        >>> uuid = parsivel2nc('parsivel.log', 'parsivel.nc', site_meta)
    """
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    
    uuid = get_uuid(uuid)
    
    # Handle multi-date file extraction
    if file_contains_date_range:
        if date is None:
            msg = "Date must be specified when file_contains_date_range is True"
            raise DisdrometerDataError(msg)
        
        logging.info("File contains date range, extracting data for specified date.")
        
        # Convert to single file if it's a list
        temp_file_path = handle_date_range(date, disdrometer_file, output_file)
        disdrometer_file = [temp_file_path]

        
    
    elif isinstance(disdrometer_file, str | PathLike):
        disdrometer_file = [disdrometer_file]
    
    disdrometer = Parsivel(disdrometer_file, site_meta, telegram, date, timestamps)
    disdrometer.sort_timestamps()
    disdrometer.remove_duplicate_timestamps()
    disdrometer.mask_invalid_values()
    
    if len(disdrometer.data["time"].data) < 2:
        msg = "Too few data points"
        raise DisdrometerDataError(msg)
    
    disdrometer.convert_units()
    disdrometer.add_meta()
    attributes = output.add_time_attribute(ATTRIBUTES, disdrometer.date)
    output.update_attributes(disdrometer.data, attributes)
    output.save_level1b(disdrometer, output_file, uuid)
    
    # Clean up temporary file if it was created
    if temp_file_path is not None:
        temp_file_path.unlink()
    
    return uuid

class Parsivel(Disdrometer):
    def __init__(
        self,
        filenames: Iterable[str | PathLike],
        site_meta: dict,
        telegram: Sequence[int | None] | None = None,
        expected_date: datetime.date | None = None,
        timestamps: Sequence[datetime.datetime] | None = None,
    ) -> None:
        super().__init__()
        self.site_meta = site_meta
        self.raw_data = _read_parsivel(filenames, telegram, timestamps)
        self._screen_time(expected_date)
        self.n_velocity = 32
        self.n_diameter = 32
        self.serial_number = None
        self.instrument = instruments.PARSIVEL2

        self._append_data()
        self._create_velocity_vectors()
        self._create_diameter_vectors()

    def _screen_time(self, expected_date: datetime.date | None = None) -> None:
        if expected_date is None:
            self.date = self.raw_data["time"][0].astype(object).date()
            return
        self.date = expected_date
        valid_mask = self.raw_data["time"].astype("datetime64[D]") == self.date
        if np.count_nonzero(valid_mask) == 0:
            msg = f"No data found on {expected_date}"
            raise DisdrometerDataError(msg)
        for key in self.raw_data:
            self.raw_data[key] = self.raw_data[key][valid_mask]

    def _append_data(self) -> None:
        for key, values in self.raw_data.items():
            if key.startswith("_"):
                continue
            name = key
            values_out = values
            match key:
                case "spectrum":
                    name = "data_raw"
                    dimensions = ["time", "diameter", "velocity"]
                case "number_concentration" | "fall_velocity":
                    dimensions = ["time", "diameter"]
                case "time":
                    dimensions = []
                    base = values[0].astype("datetime64[D]")
                    values_out = (values - base) / np.timedelta64(1, "h")
                case _:
                    dimensions = ["time"]
            self.data[name] = CloudnetArray(values_out, name, dimensions=dimensions)
        if "_sensor_id" in self.raw_data:
            first_id = self.raw_data["_sensor_id"][0]
            for sensor_id in self.raw_data["_sensor_id"]:
                if sensor_id != first_id:
                    msg = "Multiple sensor IDs are not supported"
                    raise DisdrometerDataError(msg)
            self.serial_number = first_id

    def _create_velocity_vectors(self) -> None:
        n_values = [10, 5, 5, 5, 5, 2]
        spreads = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
        self.store_vectors(n_values, spreads, "velocity")

    def _create_diameter_vectors(self) -> None:
        n_values = [10, 5, 5, 5, 5, 2]
        spreads = [0.125, 0.25, 0.5, 1, 2, 3]
        self.store_vectors(n_values, spreads, "diameter")

    def mask_invalid_values(self) -> None:
        if variable := self.data.get("number_concentration"):
            variable.data = ma.masked_where(variable.data == -9.999, variable.data)
        if variable := self.data.get("fall_velocity"):
            variable.data = ma.masked_where(variable.data == 0, variable.data)

    def convert_units(self) -> None:
        mmh_to_ms = SEC_IN_HOUR / MM_TO_M
        c_to_k = 273.15
        self._convert_data(("rainfall_rate",), mmh_to_ms)
        self._convert_data(("snowfall_rate",), mmh_to_ms)
        self._convert_data(("diameter", "diameter_spread", "diameter_bnds"), 1e3)
        self._convert_data(("V_sensor_supply",), 10)
        self._convert_data(("T_sensor",), c_to_k, method="add")
        if variable := self.data.get("number_concentration"):
            variable.data = np.power(10, variable.data).round().astype(np.uint32)


CSV_HEADERS = {
    "Date": "_date",
    "Time": "_time",
    "Intensity of precipitation (mm/h)": "rainfall_rate",
    "Precipitation since start (mm)": "_rain_accum",
    "Radar reflectivity (dBz)": "radar_reflectivity",
    "MOR Visibility (m)": "visibility",
    "Signal amplitude of Laserband": "sig_laser",
    "Number of detected particles": "n_particles",
    "Temperature in sensor (°C)": "T_sensor",
    "Heating current (A)": "I_heating",
    "Sensor voltage (V)": "V_power_supply",
    "Kinetic Energy": "kinetic_energy",
    "Snow intensity (mm/h)": "snowfall_rate",
    "Weather code SYNOP WaWa": "synop_WaWa",
    "Weather code METAR/SPECI": "_metar",
    "Weather code NWS": "_nws",
    "Optics status": "state_sensor",
    "Spectrum": "spectrum",
}

TOA5_HEADERS = {
    "RECORD": "_record",
    "TIMESTAMP": "_datetime",
    "datetime_utc": "_datetime",
    "rainIntensity": "rainfall_rate",
    "rain_intensity": "rainfall_rate",
    "rain rate [mm/h]": "rainfall_rate",
    "snowIntensity": "snowfall_rate",
    "snow_intensity": "snowfall_rate",
    "accPrec": "_rain_accum",
    "precipitation": "_rain_accum",
    "rain accum [mm]": "_rain_accum",
    "weatherCodeWaWa": "synop_WaWa",
    "wawa": "synop_WaWa",
    "weather_code_wawa": "synop_WaWa",
    "radarReflectivity": "radar_reflectivity",
    "radar_reflectivity": "radar_reflectivity",
    "Z [dBz]": "radar_reflectivity",
    "morVisibility": "visibility",
    "mor_visibility": "visibility",
    "MOR visibility [m]": "visibility",
    "kineticEnergy": "kinetic_energy",
    "kinetic_energy": "kinetic_energy",
    "signalAmplitude": "sig_laser",
    "signal_amplitude": "sig_laser",
    "Signal amplitude": "sig_laser",
    "sensorTemperature": "T_sensor",
    "sensor_temperature": "T_sensor",
    "Temperature sensor [°C]": "T_sensor",
    "pbcTemperature": "_T_pcb",
    "pbc_temperature": "_T_pcb",
    "rightTemperature": "_T_right",
    "right_temperature": "_T_right",
    "leftTemperature": "_T_left",
    "left_temperature": "_T_left",
    "heatingCurrent": "I_heating",
    "heating_current": "I_heating",
    "sensorVoltage": "V_power_supply",
    "sensor_voltage": "V_power_supply",
    "Power supply voltage in the sensor [V]": "V_power_supply",
    "sensorStatus": "state_sensor",
    "sensor_status": "state_sensor",
    "Sensor status": "state_sensor",
    "errorCode": "error_code",
    "error_code": "error_code",
    "Error code": "error_code",
    "numberParticles": "n_particles",
    "number_particles": "n_particles",
    "Number of detected particles": "n_particles",
    "N": "number_concentration",
    "V": "fall_velocity",
    "spectrum": "spectrum",
    "Current heating system [A]": "I_heating",
    "sample interval [s]": "interval",
    "Serial number": "_sensor_id",
    "IOP firmware version": "_iop_firmware_version",
    "Station name": "_station_name",
    "Rain amount absolute [mm]": "_rain_amount_absolute",
}

TELEGRAM = {
    1: "rainfall_rate",
    2: "_rain_accum",
    3: "synop_WaWa",
    4: "synop_WW",
    5: "_metar_speci",
    6: "_nws",
    7: "radar_reflectivity",
    8: "visibility",
    9: "interval",
    10: "sig_laser",
    11: "n_particles",
    12: "T_sensor",
    13: "_sensor_id",
    14: "_iop_firmware_version",
    15: "_dsp_firmware_version",
    16: "I_heating",
    17: "V_power_supply",
    18: "state_sensor",
    19: "_datetime",
    20: "_time",
    21: "_date",
    22: "_station_name",
    23: "_station_number",
    24: "_rain_amount_absolute",
    25: "error_code",
    26: "_T_pcb",
    27: "_T_right",
    28: "_T_left",
    30: "rainfall_rate",
    31: "rainfall_rate",
    32: "_rain_accum",
    33: "radar_reflectivity",
    34: "kinetic_energy",
    35: "snowfall_rate",
    # 60, 61 = all particles detected
    90: "number_concentration",
    91: "fall_velocity",
    93: "spectrum",
}


def _parse_int(tokens: Iterator[str]) -> int:
    return int(next(tokens))


def _parse_float(tokens: Iterator[str]) -> float:
    token = next(tokens)
    token = token.replace(",", ".")
    return float(token)


def _parse_date(tokens: Iterator[str]) -> datetime.date:
    token = next(tokens)
    if "/" in token:
        year, month, day = token.split("/")
    elif "." in token:
        day, month, year = token.split(".")
    else:
        msg = f"Unsupported date: '{input}'"
        raise ValueError(msg)
    if len(year) != 4:
        msg = f"Unsupported date: '{input}'"
        raise ValueError(msg)
    return datetime.date(int(year), int(month), int(day))


def _parse_time(tokens: Iterator[str]) -> datetime.time:
    token = next(tokens)
    hour, minute, second = token.split(":")
    return datetime.time(int(hour), int(minute), int(second))


def _parse_datetime(tokens: Iterator[str]) -> datetime.datetime:
    token = next(tokens)
    year = int(token[:4])
    month = int(token[4:6])
    day = int(token[6:8])
    hour = int(token[8:10])
    minute = int(token[10:12])
    second = int(token[12:14])
    return datetime.datetime(
        year,
        month,
        day,
        hour,
        minute,
        second,
    )


def _parse_vector(tokens: Iterator[str]) -> npt.NDArray:
    return np.array([_parse_float(tokens) for _i in range(32)])


def _parse_spectrum(tokens: Iterator[str]) -> npt.NDArray:
    first = next(tokens)
    if first == "<SPECTRUM>ZERO</SPECTRUM>":
        return np.zeros((32, 32), dtype="i2")
    if first.startswith("<SPECTRUM>"):
        raw = [first.removeprefix("<SPECTRUM>")]
        raw.extend(islice(tokens, 1023))
        if next(tokens) != "</SPECTRUM>":
            msg = "Invalid spectrum format"
            raise ValueError(msg)
        values = [int(x) if x != "" else 0 for x in raw]
    elif "/" in first:
        values = [int(x) for x in first.removesuffix("/R").split("/")]
    else:
        values = [int(first)]
        values.extend(int(x) for x in islice(tokens, 1023))
    if len(values) != 1024:
        msg = f"Invalid spectrum length: {len(values)}"
        raise ValueError(msg)
    return np.array(values, dtype="i2").reshape((32, 32))


ParserType = Callable[[Iterator[str]], Any]


PARSERS: dict[str, ParserType] = {
    "I_heating": _parse_float,
    "T_sensor": _parse_int,
    "_T_pcb": _parse_int,
    "_T_right": _parse_int,
    "_T_left": _parse_int,
    "V_power_supply": _parse_float,
    "_date": _parse_date,
    "_rain_accum": _parse_float,
    "_rain_amount_absolute": _parse_float,
    "_time": _parse_time,
    "error_code": _parse_int,
    "fall_velocity": _parse_vector,
    "interval": _parse_int,
    "kinetic_energy": _parse_float,
    "n_particles": _parse_int,
    "number_concentration": _parse_vector,
    "_datetime": _parse_datetime,
    "radar_reflectivity": _parse_float,
    "rainfall_rate": _parse_float,
    "sig_laser": _parse_int,
    "snowfall_rate": _parse_float,
    "spectrum": _parse_spectrum,
    "state_sensor": _parse_int,
    "synop_WaWa": _parse_int,
    "synop_WW": _parse_int,
    "visibility": _parse_int,
}

EMPTY_VALUES: dict[ParserType, Any] = {
    _parse_int: 0,
    _parse_float: 0.0,
    _parse_date: datetime.date(2000, 1, 1),
    _parse_time: datetime.time(12, 0, 0),
    _parse_datetime: datetime.datetime(2000, 1, 1),
    _parse_vector: np.zeros(32, dtype=float),
    _parse_spectrum: np.zeros((32, 32), dtype="i2"),
}


def _parse_headers(line: str) -> list[str]:
    return [CSV_HEADERS[header.strip()] for header in line.split(";")]


def _parse_telegram(telegram: Sequence[int | None]) -> list[str]:
    return [
        TELEGRAM[num] if num is not None else f"_unknown_{i}"
        for i, num in enumerate(telegram)
    ]


def _read_rows(headers: list[str], rows: list[str]) -> dict[str, list]:
    result: dict[str, list] = {header: [] for header in headers}
    invalid_rows = 0
    for row in rows:
        if row == "":
            continue
        try:
            parsed = _parse_row(row, headers)
            for header, value in zip(headers, parsed, strict=True):
                result[header].append(value)
        except (ValueError, StopIteration):
            invalid_rows += 1
            continue
    if invalid_rows == len(rows):
        msg = "No valid data in file"
        raise DisdrometerDataError(msg)
    if invalid_rows > 0:
        logging.info("Skipped %s invalid rows", invalid_rows)
    return result


def _parse_row(row_in: str, headers: list[str]) -> list:
    tokens = iter(row_in.removesuffix(";").split(";"))
    parsed = [PARSERS.get(header, next)(tokens) for header in headers]
    if unread_tokens := list(tokens):
        msg = f"Unused tokens: {unread_tokens}"
        raise ValueError(msg)
    return parsed


def _read_toa5(filename: str | PathLike) -> dict[str, list]:
    """Read ASCII data from Campbell Scientific datalogger such as CR1000.

    References:
        CR1000 Measurement and Control System.
        https://s.campbellsci.com/documents/us/manuals/cr1000.pdf
    """
    with open(filename, errors="ignore") as file:
        reader = csv.reader(file)
        _origin_line = next(reader)
        header_line = next(reader)
        headers = [
            TOA5_HEADERS.get(re.sub(r"\(.*", "", field)) for field in header_line
        ]
        if unknown_headers := [
            header_line[i] for i in range(len(header_line)) if headers[i] is None
        ]:
            msg = "Unknown headers: " + ", ".join(unknown_headers)
            logging.warning(msg)
        _units_line = next(reader)
        _process_line = next(reader)
        data: dict[str, list] = {header: [] for header in headers if header is not None}
        n_rows = 0
        n_invalid_rows = 0
        for data_line in reader:
            n_rows += 1
            scalars: dict[str, datetime.datetime | int | float | str] = {}
            arrays: dict[str, list] = {
                "number_concentration": [],
                "fall_velocity": [],
                "spectrum": [],
            }
            try:
                for header, value in zip(headers, data_line, strict=True):
                    if header is None:
                        continue
                    if header == "_datetime":
                        scalars[header] = datetime.datetime.strptime(
                            value,
                            "%Y-%m-%d %H:%M:%S",
                        )
                    elif header in ("number_concentration", "fall_velocity"):
                        arrays[header].append(float(value))
                    elif header == "spectrum":
                        arrays[header].append(int(value))
                    elif PARSERS.get(header) == _parse_int:
                        scalars[header] = int(value)
                    elif PARSERS.get(header) == _parse_float:
                        scalars[header] = float(value)
                    else:
                        scalars[header] = value
            except ValueError:
                n_invalid_rows += 1
                continue
            for header, scalar in scalars.items():
                data[header].append(scalar)
            if "spectrum" in headers:
                data["spectrum"].append(
                    np.array(arrays["spectrum"], dtype="i2").reshape((32, 32)),
                )
            if "number_concentration" in headers:
                data["number_concentration"].append(arrays["number_concentration"])
            if "fall_velocity" in headers:
                data["fall_velocity"].append(arrays["fall_velocity"])
        if n_invalid_rows == n_rows:
            msg = "No valid data in file"
            raise DisdrometerDataError(msg)
        if n_invalid_rows > 0:
            logging.info("Skipped %s invalid rows", n_invalid_rows)
        return data


def _read_bucharest_file(filename: str | PathLike) -> dict[str, list]:
    with open(filename, errors="ignore") as file:
        reader = csv.reader(file)
        header_line = next(reader)[0].split(";")
        headers = [
            TOA5_HEADERS.get(
                re.sub(
                    r"N[0-9][0-9]",
                    "N",
                    re.sub(r"v[0-9][0-9]", "V", re.sub(r"M\_.*", "spectrum", field)),
                ),
            )
            for field in header_line
        ]
        if unknown_headers := [
            header_line[i] for i in range(len(header_line)) if headers[i] is None
        ]:
            msg = "Unknown headers: " + ", ".join(unknown_headers)
            logging.warning(msg)

        data: dict[str, list] = {header: [] for header in headers if header is not None}
        n_rows = 0
        n_invalid_rows = 0
        for data_line in reader:
            data_line_splat = data_line[0].split(";")
            data_line_splat = [d for d in data_line_splat if d != ""]
            n_rows += 1
            scalars: dict[str, datetime.datetime | int | float | str] = {}
            arrays: dict[str, list] = {
                "number_concentration": [],
                "fall_velocity": [],
                "spectrum": [],
            }
            try:
                for header, value in zip(headers, data_line_splat, strict=True):
                    if header is None:
                        continue
                    if header == "_datetime":
                        scalars[header] = datetime.datetime.strptime(
                            value,
                            "%Y-%m-%d %H:%M:%S",
                        )
                    elif header in ("number_concentration", "fall_velocity"):
                        arrays[header].append(float(value))
                    elif header == "spectrum":
                        arrays[header].append(int(value))
                    elif PARSERS.get(header) == _parse_int:
                        scalars[header] = int(value)
                    elif PARSERS.get(header) == _parse_float:
                        scalars[header] = float(value)
                    else:
                        scalars[header] = value
            except ValueError:
                n_invalid_rows += 1
                continue
            for header, scalar in scalars.items():
                data[header].append(scalar)
            if "spectrum" in headers:
                data["spectrum"].append(
                    np.array(arrays["spectrum"], dtype="i2").reshape((32, 32)),
                )
            if "number_concentration" in headers:
                data["number_concentration"].append(arrays["number_concentration"])
            if "fall_velocity" in headers:
                data["fall_velocity"].append(arrays["fall_velocity"])
        if n_invalid_rows == n_rows:
            msg = "No valid data in file"
            raise DisdrometerDataError(msg)
        if n_invalid_rows > 0:
            logging.info("Skipped %s invalid rows", n_invalid_rows)
        return data


def _read_typ_op4a(lines: list[str]) -> dict[str, Any]:
    """Read output of "CS/PA" command. The output starts with line "TYP OP4A"
    followed by one line per measured variable in format: <number>:<value>.
    Output ends with characters: <ETX><CR><LF><NUL>. Lines are separated by
    <CR><LF>.
    """
    data = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.strip().split(":", maxsplit=1)
        # Skip datetime and 16-bit values.
        if key in ("19", "30", "31", "32", "33"):
            continue
        varname = TELEGRAM.get(int(key))
        if varname is None:
            continue
        parser = PARSERS.get(varname, next)
        tokens = value.split(";")
        data[varname] = parser(iter(tokens))
    return data


def _read_fmi(content: str) -> dict[str, list]:
    r"""Read format used by Finnish Meteorological Institute and University of
    Helsinki.

    Format consists of sequence of the following:
    - "[YYYY-MM-DD HH:MM:SS\n"
    - output of "CS/PA" command without non-printable characters at the end
    - "]\n"
    """
    output: dict[str, list] = {"_datetime": []}
    for m in re.finditer(
        r"\[(?P<year>\d+)-(?P<month>\d+)-(?P<day>\d+) "
        r"(?P<hour>\d+):(?P<minute>\d+):(?P<second>\d+)"
        r"(?P<output>[^\]]*)\]",
        content,
    ):
        try:
            record = _read_typ_op4a(m["output"].splitlines())
        except ValueError:
            continue

        for key, value in record.items():
            if key not in output:
                output[key] = [None] * len(output["_datetime"])
            output[key].append(value)
        for key in output:
            if key not in record and key != "_datetime":
                output[key].append(None)

        output["_datetime"].append(
            datetime.datetime(
                int(m["year"]),
                int(m["month"]),
                int(m["day"]),
                int(m["hour"]),
                int(m["minute"]),
                int(m["second"]),
            )
        )
    return output


def _read_parsivel(
    filenames: Iterable[str | PathLike],
    telegram: Sequence[int | None] | None = None,
    timestamps: Sequence[datetime.datetime] | None = None,
) -> dict[str, npt.NDArray]:
    combined_data = defaultdict(list)
    for filename in filenames:
        with open(filename, encoding="latin1", errors="ignore") as file:
            content = file.read()
            lines = content.splitlines()
        if not lines:
            msg = f"File '{filename}' is empty"
            raise DisdrometerDataError(msg)
        if "TOA5" in lines[0]:
            data = _read_toa5(filename)
        elif "N00" in lines[0]:
            data = _read_bucharest_file(filename)
        elif "TYP OP4A" in lines[0]:
            data = _read_typ_op4a(lines)
            data = {key: [value] for key, value in data.items()}
        elif "Date" in lines[0]:
            headers = _parse_headers(lines[0])
            data = _read_rows(headers, lines[1:])
        elif "[" in lines[0]:
            data = _read_fmi(content)
        elif telegram is not None:
            headers = _parse_telegram(telegram)
            data = _read_rows(headers, lines)
        else:
            msg = "telegram must be specified for files without header"
            raise ValueError(msg)
        if "_datetime" not in data and timestamps is None:
            data["_datetime"] = [
                datetime.datetime.combine(date, time)
                for date, time in zip(data["_date"], data["_time"], strict=True)
            ]
        for key, values in data.items():
            combined_data[key].extend(values)
    if timestamps is not None:
        combined_data["_datetime"] = list(timestamps)
    result = {}
    for key, value in combined_data.items():
        array = np.array(
            [
                x
                if x is not None
                else (EMPTY_VALUES[PARSERS[key]] if key in PARSERS else "")
                for x in value
            ]
        )
        mask = [np.full(array.shape[1:], x is None) for x in value]
        result[key] = ma.array(array, mask=mask)
    result["time"] = result["_datetime"].astype("datetime64[s]")
    return result

#####################################################
# Implement the cleaning and splitting routine here? #
######################################################

def find_files_by_date(target_date, input_dir):
    """
    Find files whose date range contains the given date YYYYMMDD.
    Works for .txt files.
    """
    target_date = datetime.strptime(target_date, "%Y%m%d").date()

    # match paths
    file_paths = glob.glob(os.path.join(input_dir, "*.txt"))

    date_pattern = re.compile(r"(\d{8})_(\d{8})")
    matching_files = []

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        m = date_pattern.search(file_name)
        if not m:
            continue

        start_date = datetime.strptime(m.group(1), "%Y%m%d").date()
        end_date   = datetime.strptime(m.group(2), "%Y%m%d").date()

        # check if target_date lies in the interval [start_date, end_date]
        if start_date <= target_date <= end_date:
            matching_files.append(file_name)

    return matching_files



def clean_array_line(line: str) -> str:
    """Clean array lines to remove trailing semicolons."""
    # Match and clean only array lines starting with "90:", "91:", ..., "98:"
    if re.match(r"^(9[0-8]):", line):
        # Split the identifier (e.g., "90:") from the values
        identifier, values = line.split(":", 1)
        # Remove trailing semicolons from the values
        values = re.sub(r";+$", "", values.strip())
        return f"{identifier}:{values}"
    return line



def extract_date_segments(input_file, target_date):
    """
    Extract segments for a specific date from the input file.
    Args:
        input_file (str): Path to the Parsivel file.
        target_date (str): Target date in YYYYMMDD format (e.g., "20231025").
    Returns:
        list: List of cleaned segment strings for the target date.
    """
    segment_lines = []
    current_date = ""
    result_segments = []
    telegram = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 16, 17, 18,
                20, 21, 24, 25, 26, 27, 28, 34, 35, 90, 91, 93]
    
    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            # Clean array lines 
            line = clean_array_line(line)
            
            # Check for line format like 21: for date
            if re.match(r"^21:", line):
                raw_date = line.split(':')[1]
                current_date = raw_date.replace('.', '') if raw_date else ""
                if len(current_date) == 8:  # Ensure format like DDMMYYYY
                    current_date = f"{current_date[4:8]}{current_date[2:4]}{current_date[0:2]}"  # Convert to YYYYMMDD
            
            # Only process lines for the target date
            if current_date != target_date:
                continue
            
            # Add only the lines with valid telegram numbers
            match = re.match(r"^(\d+):", line)
            if match:
                number = int(match.group(1))
                if number in telegram:
                    # Remove numbers before ":" and clean up the rest of the line
                    cleaned_line = line.split(":", 1)[1]
                    segment_lines.append(cleaned_line)
            
            # If line is 98:, assume the end of the current segment
            if re.match(r"^98:", line):
                cleaned_segment = ';'.join(segment_lines)
                result_segments.append(cleaned_segment)
                segment_lines = []  # Reset for the next segment
    
    # Remove the first line if it exists (it's broken)
    if result_segments:
        result_segments = result_segments[1:]
    
    # with open(f"parsivel_extracted_{target_date}.txt", "w") as f:
    #     f.write("\n".join(result_segments) )
    
    return result_segments


def handle_date_range(date, disdrometer_file, output_file):

    # Convert to single file if it's a list
    if not isinstance(disdrometer_file, str | PathLike):
        if len(disdrometer_file) > 1:
            msg = "Only single file supported with file_contains_date_range=True"
            raise DisdrometerDataError(msg)
        disdrometer_file = disdrometer_file[0]

    # Convert date to YYYYMMDD format
    target_date = date.strftime("%Y%m%d")

    # Extract segments for the target date
    segments = extract_date_segments(str(disdrometer_file), target_date)

    if not segments:
        msg = f"No data found for date {target_date} in file {disdrometer_file}"
        raise DisdrometerDataError(msg)

    # Create temporary file with extracted data
    output_path = Path(output_file)
    temp_file = output_path.parent / f"{target_date}_parsivel_temp.txt"

    with open(temp_file, 'w') as f:
        f.write('\n'.join(segments))

    # Use the temporary file for processing
    # disdrometer_file = [temp_file]
    logging.info(f"Extracted {len(segments)} segments for date {target_date}")

    return temp_file