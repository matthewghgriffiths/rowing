"""
peach.py — Peach binary telemetry parser
============================================

Processing pipeline (all contained in PeachData.from_path):

  Stage 1 — load_header(bin)
      Reads the first ~50 kB: session timestamp, boat parameters, sensor list,
      event log, and GPS init coordinates.  No data records are touched.

  Stage 2 — _locate_records(bin, n_sensors)
      Finds GPS and stroke record starts using their fixed 2-word magic headers
      (0x8013 / 0x800A).  Locates periodic records by constructing the per-file
      flag from the sensor count and trying both possible header sizes (14 and
      18 uint16 words).  Returns start positions and widths for all three
      stream types.

  Stage 3 — _parse_raw(bin, record_locs)
      Extracts the three raw DataFrames (GPS, stroke, periodic) and stores them
      in self.raw_data.

  Stage 4 — _map_channels()
      Applies channel-naming and scaling tables (derived purely from sensor
      metadata) to produce the named (channel, position) MultiIndex DataFrames
      exposed as self.gps, self.stroke, and self.periodic.

Column conventions
------------------
All three public DataFrames use a two-level MultiIndex column:
    (channel_name, position)
where position is 'Boat' for vessel-level channels or a seat-number string
('1'–'8') for per-seat oarlock channels.

Notes on UTC time
-----------------
GPS records contain a sequential fix counter in raw column 10 that increments
by 1 each second once the GPS receiver has a lock.  The reference export
converts this to wall-clock milliseconds by adding a session-specific offset
that is not stored anywhere we have found in the binary.  Raw column 10 is
therefore exposed as 'GPS Fix Counter' rather than absolute UTC time.
"""

import re
import io
from pathlib import Path
from typing import Mapping
import logging

import numpy as np
import pandas as pd
from scipy import stats


_UTF_RE = re.compile(b'(?:[\x20-\x7E]\x00){4,}')
_ASCII_RE = re.compile(b'(?:[\x20-\x7E]){4,}')

# ─────────────────────────────────────────────────────────────────────────────
# Channel / sensor metadata tables  (pure functions of the sensor list)
# ─────────────────────────────────────────────────────────────────────────────

# NK Peach channel-type codes (byte 2 of each sensor record in the header).
# Gate / oarlock sensors use codes 2 and 4 for sweep, or 9–14 for sculling.
# Boat-logger channels use codes 1, 3, 6, 7, 8, 27, 28 — but their stream order
# is the reverse of the metadata order, so names are assigned by fixed position
# in BOAT_PERIODIC_NAMES rather than by code lookup.
CHANNEL_TYPE_NAMES = {
    # Gate / oarlock sensor codes — sweep (one sensor per seat)
    2: 'GateForceX',
    4: 'GateAngle',
    5: 'GateForceY',
    # Gate / oarlock sensor codes — sculling (two sensors per seat: P and S)
    # Codes appear in pairs; higher code = Port, lower = Starboard within each type.
    # GateAngle pair:
    14: ('P', 'GateAngle'),
    13: ('S', 'GateAngle'),
    # GateForceX pair:
    12: ('P', 'GateForceX'),
    11: ('S', 'GateForceX'),
    # GateForceY pair (sculling only):
    10: ('P', 'GateForceY'),
    9:  ('S', 'GateForceY'),
    1: 'Speed',
    3: 'Accel',
    # 6: '',
    # 7: '',
    8: 'Roll Angle',
    27: 'Pitch Angle',
    28: 'Yaw Angle'
}
_BOAT_CHANNEL_CODES = {1, 3, 6, 7, 8, 27, 28}

# The boat-logger periodic channels always appear in this fixed stream order,
# regardless of the order they are listed in the sensor metadata.
BOAT_PERIODIC_ORDER = {
    6: 'Speed',       # code 3
    7: 'Accel',       # code 1
    10: 'Roll Angle',  # code 8
    11: 'Pitch Angle',  # code 27
    12: 'Yaw Angle',   # code 28
}
# Physical-unit scaling applied as: value = (raw + 1) * scale - shift
# Channels absent from these dicts are passed through unscaled.
PERIODIC_SCALES = {
    'GateAngle': 1/16, 'GateForceX': 1/16, 'GateForceY': 1/16,
    'GateAngleVel': 1/16,
    'Speed': 1/256, 'Accel': 1/256,
    'Roll Angle': 1/64, 'Pitch Angle': 1/64, 'Yaw Angle': 1/64,
}
PERIODIC_SHIFT = {
    'Distance': 0,
    'Accel': 32, 'Speed': 32,
    'Pitch Angle': 128, 'Roll Angle': 128, 'Yaw Angle': 128,
    'GateAngle': 512, 'GateForceX': 512, 'GateForceY': 512,
}
STROKE_SCALES = {
    'SwivelPower': 1, 'Rower Swivel Power': 1,
    'MinAngle': 1/16, 'MaxAngle': 1/16,
    'CatchSlip': 1/16, 'FinishSlip': 1/16,
    'Drive Start T': 1,
    'Rating': 1/2, 'AvgBoatSpeed': 1/256,
    'StrokeNumber': 1, 'Dist/Stroke': 1/256, 'Average Power': 1,
}
STROKE_SHIFT = {
    'Average Power': 1002, 'SwivelPower': 1002,
    'AvgBoatSpeed': 1/128, 'Dist/Stroke': 1/128,
    'CatchSlip': 512, 'FinishSlip': 512,
    'MaxAngle': 512, 'MinAngle': 512,
    'Drive Start T': 2, 'Rating': 1, 'StrokeNumber': 1,
}

_AVG_EARTH_RADIUS_KM = 6371.0088
_GPS_SCALE = _AVG_EARTH_RADIUS_KM * 1000 * np.pi / 180

# GPS column mapping: raw DataFrame key (int or computed name) → output name
# Raw integer keys are uint16 column indices; string keys are pre-computed columns.
# Column 10 is a sequential GPS fix counter (increments by 1/s after lock).
# It cannot be converted to wall-clock UTC without a session-specific offset
# that is not present in the binary, so it is exposed as-is.
_GPS_NAMED_COLS = {
    'lat': 'lat', 'long': 'long',
    6:  'GPS X Lo',  7:  'GPS X Hi',
    8:  'GPS Y Lo',  9:  'GPS Y Hi',
    10: 'GPS Fix Counter',   # sequential; not absolute UTC time
    11: 'GPS Status',
    12: 'Chanb880', 13: 'Chanb881', 14: 'Chanb882',
}
GPS_SCALES = {
    'GPS X Hi': 64, 'GPS Y Hi': 64,
    'GPS X Lo': 1/256, 'GPS Y Lo': 1/256,
    'lat': 1/256, 'long': 1/256,
    'Chanb880': 1,
    'Chanb881': 1/16,
    'Chanb882': 1,
}
GPS_SHIFT = {
    'GPS X Lo': 1/128, 'GPS Y Lo': 1/128,
    'GPS X Hi': 524352, 'GPS Y Hi': 524352,
    'lat': 524288 + 1/128, 'long': 524288 + 1/128,
    'Chanb880': 8193,
    'Chanb881': 512 + 1/16,
    'Chanb882': 8192,
}

# Fixed stroke header column positions (0-indexed uint16 columns in each record)
_STROKE_HEADER_COLS = {
    5: 'StrokeNumber', 6: 'Rating', 7: 'AvgBoatSpeed',
    9: 'Dist/Stroke',  10: 'Average Power',
}
# Per-seat column offsets within each stroke seat block
_STROKE_SEAT_SWEEP = {      # 7-column sweep block (offset 6 is a SwivelPower repeat)
    0: 'SwivelPower', 1: 'MinAngle', 2: 'MaxAngle',
    3: 'CatchSlip', 4: 'FinishSlip', 5: 'Drive Start T',
}
_STROKE_SEAT_SCULL = {      # 12-column sculling block (P then S for each channel)
    0:  ('P', 'SwivelPower'), 1:  ('S', 'SwivelPower'),
    2:  ('P', 'MinAngle'),    3:  ('S', 'MinAngle'),
    4:  ('P', 'MaxAngle'),    5:  ('S', 'MaxAngle'),
    6:  ('P', 'CatchSlip'),   7:  ('S', 'CatchSlip'),
    8:  ('P', 'FinishSlip'),  9:  ('S', 'FinishSlip'),
    10: ('P', 'Drive Start T'), 11: ('S', 'Drive Start T'),
}

# uint32 basis vectors for combining two consecutive uint16 words
C16 = np.r_[1, 2 ** 16].astype(np.uint32)   # full 32-bit recombination
C14 = np.r_[1, 2 ** 14].astype(np.uint32)   # 14-bit GPS coordinate words

# Fixed binary sequence used to locate the GPS init-coordinate block
_GPS_LOC_FLAG = np.r_[
    20, 0, 0, 0, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 17
].astype(np.uint16)

PARAM_NAMES = [
    'Impeller Scaling', 'Average Rating over', 'Sweep Drive Start',
    'Sweep Recovery', 'Sweep Oar Inboard', 'Sweep Oar Length',
    'Scull Drive Start', 'Scull Recovery', 'Scull Oar Inboard',
    'Scull Oar Length',
]
PARAM_UNITS = ['%', 'strokes', 'kg', 'kg', 'cm', 'cm', 'kg', 'kg', 'cm', 'cm']
PARAM_SCALES = [100, 1, 100, 100, 10, 10, 100, 100, 10, 10]

# for processing reference/export files
ALLOWED_POSITIONS = ['Boat'] + [str(i) for i in range(1, 9)]
EXPORT_EXTRA_COLS = [
    'Chanb885', 'GPS RTCM', 'UTC Time',
    'GateAngleVel', 'Angle Max F', 'Work PC Q4', 'Drive Time',
    'Angle 0.7 F', 'Work PC Q3', 'Work PC Q1', 'Rower Swivel Power',
    'Recovery Time', 'Max Force PC', 'Work PC Q2',
    'Normalized Time',
    'P GateAngleVel', 'S GateAngleVel', 'P Angle 0.7 F', 'P Angle Max F',
    'P Drive Time', 'P Max Force PC', 'P Min Angle', 'P Recovery Time',
    'P Swivel Power', 'P Work PC Q1', 'P Work PC Q2', 'P Work PC Q3',
    'P Work PC Q4', 'S Angle 0.7 F', 'S Angle Max F', 'S Drive Time',
    'S Max Force PC', 'S Min Angle', 'S Recovery Time', 'S Swivel Power',
    'S Work PC Q1', 'S Work PC Q2', 'S Work PC Q3', 'S Work PC Q4'
]
# ─────────────────────────────────────────────────────────────────────────────
# Low-level binary search helper
# ─────────────────────────────────────────────────────────────────────────────


def search(arr, seq):
    """Return a boolean mask, True wherever *seq* starts in *arr*."""
    match = np.ones_like(arr, dtype=bool)
    n = len(arr)
    pairs = seq.items() if isinstance(seq, Mapping) else enumerate(seq)
    for i, v in pairs:
        match[:n - i] &= (arr[i:] == v)
        match[n - i:] = False
    return match


# ─────────────────────────────────────────────────────────────────────────────
# Raw block extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_aperiodic(bin_u16, starts, n_cols, n_rows=10):
    """Extract n_rows × n_cols uint16 blocks at each offset in *starts*."""
    return pd.concat(
        [pd.DataFrame(bin_u16[s:s + n_cols * n_rows].reshape(n_rows, n_cols))
         for s in starts],
        ignore_index=True,
    )


def extract_aperiodic32(bin_u16, starts, n_cols, n_rows=1):
    """Like extract_aperiodic but interprets uint16 pairs as uint32."""
    return pd.concat(
        [pd.DataFrame(
            bin_u16[s:s + 2 * n_cols * n_rows]
            .view(np.uint32).reshape(n_rows, n_cols))
         for s in starts],
        ignore_index=True,
    )


def extract_periodic(bin_u16, starts, n_cols, n_rows=50):
    """Extract periodic blocks; data is stored column-major (transposed)."""
    groups = {}
    for g, s in enumerate(starts):
        sn = s + n_cols * n_rows
        if sn < bin_u16.size:
            groups[g] = pd.DataFrame(
                bin_u16[s:sn].reshape(n_cols, n_rows).T)

    return pd.concat(
        groups, names=['group']
    ).droplevel(1).reset_index().set_index('group', append=True)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — header / metadata
# ─────────────────────────────────────────────────────────────────────────────


def _parse_block(data_u8, header_len, row_width, dtype=np.uint8):
    """Split *data_u8* into (records_df, header_bytes, tail_bytes)."""
    n_rows = (data_u8.size - header_len) // row_width
    end = header_len + n_rows * row_width
    header, body, tail = np.split(data_u8, [header_len, end])
    return pd.DataFrame(body.reshape(n_rows, row_width).view(dtype)), header, tail


def load_header(bin_u16: np.ndarray) -> dict:
    """Stage 1: parse the file header and return a metadata dict.

    Only the first ~50 kB (parameter and sensor sections) are read.
    Data records are not touched at this stage.

    Returns
    -------
    dict containing:
        date, timestamp, params, sensors, sensors info,
        stroke info, periodic info, gps info, events, gps_coords
    """
    sbin = bin_u16[:50_000]
    sbin_u8 = sbin.view(np.uint8)

    # Split the header into named sections using known flag sequences.
    # The GPS (0x8013) and stroke (0x800A) magic words delimit where data starts.
    flags = [
        # 'Bo' UTF-16 → 'Boat'
        np.array([66, 0, 111, 0], dtype=np.uint8),
        np.frombuffer(b'Logger', dtype=np.uint8),
        np.frombuffer(b'Boat',   dtype=np.uint8),
        np.array([0x8013, 10], dtype=np.uint16).view(np.uint8),  # GPS magic
        np.array([0x800A, 10], dtype=np.uint16).view(np.uint8),  # stroke magic
    ]
    sep_mask = np.zeros(sbin_u8.size, dtype=bool)
    for flag in flags:
        if flag.dtype == np.uint8:
            sep_mask |= search(sbin_u8, flag)
        else:
            sep_mask |= np.repeat(search(sbin, flag.view(np.uint16)), 2)

    section_names = [
        'Header', 'ParamHeader', 'Parameters',
        'VersionsHeader', 'Sensors', 'EventsHeader', 'Events',
    ]
    sections = dict(zip(section_names, np.split(
        sbin_u8, sep_mask.nonzero()[0])))
    if len(sections) != len(section_names):
        return

    meta = {}

    # ── Session timestamp ─────────────────────────────────────────────────────
    hdr = sections['Header']
    meta['Header'] = hdr
    meta['serial'], meta['session'] = map(int, hdr[40:48].view(np.uint32))
    meta['timestamp'] = int(hdr[24:28].view(np.uint32)[0])
    meta['date'] = pd.Timestamp(meta['timestamp'], unit='s')

    # ── Boat parameters ───────────────────────────────────────────────────────
    params, *_ = _parse_block(sections['Parameters'], 27, 8, dtype=np.uint16)
    params.index = PARAM_NAMES
    params['value'] = params[2] / PARAM_SCALES
    params['unit'] = PARAM_UNITS
    meta['params'] = params

    # ── Sensor list ───────────────────────────────────────────────────────────
    sensors, *_ = _parse_block(sections['Sensors'], 24, 20)
    sensors['seat'] = sensors[0]
    sensors['channel'] = sensors[2]
    sensors['version'] = (
        sensors[[4, 5, 6, 7]] % 128).astype(str).apply('.'.join, axis=1)
    sensors['S/N'] = sensors[[16, 17]].values.copy().view(np.uint16)
    meta['sensors'] = sensors

    # Derived tables used by stages 2–4
    meta['sensors info'] = infer_sensor_table(sensors)
    meta['stroke info'] = infer_stroke_table(meta['sensors info'])
    meta['periodic info'] = infer_periodic_table(meta['sensors info'])
    meta['gps info'] = infer_gps_table()

    # ── Event log ─────────────────────────────────────────────────────────────
    ev_start = search(sections['Events'], [
                      1, 64, 0, 0, 0, 0, 128, 59]).argmax()
    events, *_ = _parse_block(sections['Events'],
                              ev_start, 16, dtype=np.uint16)
    meta['events'] = events

    # ── GPS initialisation coordinates ────────────────────────────────────────
    # A fixed 16-word pattern in the binary precedes the init GPS fix.
    # Bytes 25–32 of a small surrounding block hold (lat, lon) as int32 / 2^23.
    init_pos, = search(bin_u16, _GPS_LOC_FLAG).nonzero()
    coords = np.r_[np.nan, np.nan]
    if init_pos.size:
        i = init_pos[0]
        gps_init_u8 = bin_u16[i + 7: i + 31].view(np.uint8)
        coords = gps_init_u8[25:33].view(np.int32) / 2 ** 23  # (lat, lon) °
    meta['latitude'], meta['longitude'] = meta['gps_coords'] = coords
    return meta


def infer_sensor_table(sensors_raw: pd.DataFrame) -> pd.DataFrame:
    """Convert the raw sensor bytes into a human-readable table.

    Gate sensors are sorted seat-ascending to match their order in the
    periodic stream; boat channels follow.

    Returns
    -------
    pd.DataFrame
        Columns: sensor_index, seat, channel (code), name, side_prefix,
                 sn, is_boat, is_scull, n_stroke_cols.
    """
    rows = []
    for i, row in sensors_raw.reset_index(drop=True).iterrows():
        code = int(row['channel'])
        sn = int(row['S/N'])
        seat = int(row['seat'])
        is_boat = code in _BOAT_CHANNEL_CODES
        raw_name = CHANNEL_TYPE_NAMES.get(code, f'Chan {code:04X}')

        if isinstance(raw_name, tuple):
            side_prefix, channel_name = raw_name
        else:
            side_prefix, channel_name = None, raw_name

        rows.append(dict(
            sensor_index=i, seat=seat, channel=code,
            name=channel_name, side_prefix=side_prefix, sn=sn,
            is_boat=is_boat, is_scull=bool(side_prefix),
            n_stroke_cols=0 if is_boat else (12 if side_prefix else 7),
        ))

    df = pd.DataFrame(rows)
    gate = df[~df['is_boat']].sort_values('seat').reset_index(drop=True)
    boat = df[df['is_boat']].reset_index(drop=True)
    return pd.concat([gate, boat], ignore_index=True)


def infer_periodic_table(st: pd.DataFrame) -> pd.DataFrame:
    """Build a column-index → (channel, position, scale, shift) table.

    Column indices start at 6; the first six columns come from the group
    header and are handled separately in _parse_raw.
    """
    gate = st[~st['is_boat']].set_index('seat').rename(index=int)
    gate['order'] = gate['channel'].replace({
        5: -1  # GateForceY comes last
    })
    boat = st[st['is_boat']].set_index('name')

    rows = {}
    # Boat channels in fixed stream order (independent of metadata order)
    for col, name in BOAT_PERIODIC_ORDER.items():
        rows[col] = dict(
            channel=name, position='Boat',
            name=name, side=None,
            sn=boat.sn[name],
            code=boat.channel[name],
        )

    col = 13
    # Gate channels interleaved per seat ascending
    for seat in sorted(gate.index.dropna().unique()):
        for _, r in gate.loc[seat].sort_values('order', ascending=False).iterrows():
            side = r['side_prefix']
            ch = r['name']
            rows[col] = dict(
                channel=f'{side} {ch}' if side else ch,
                position=str(seat), name=ch, side=side,
                sn=r.sn,
                code=r.channel,
            )
            col += 1

    df = pd.DataFrame.from_dict(rows, orient='index')
    df['scale'] = df['name'].map(PERIODIC_SCALES)
    df['shift'] = df['name'].map(PERIODIC_SHIFT)
    return df


def infer_stroke_table(st: pd.DataFrame) -> pd.DataFrame:
    """Build a column-index → (channel, position, scale, shift) table for stroke."""
    rows = {i: dict(channel=n, position='Boat', name=n, side=None)
            for i, n in _STROKE_HEADER_COLS.items()}

    block_start = 12
    for seat_idx, seat_row in st[~st['is_boat']].groupby('seat').first().sort_index().iterrows():
        seat_map = _STROKE_SEAT_SCULL if seat_row['is_scull'] else _STROKE_SEAT_SWEEP
        for offset, label in seat_map.items():
            if seat_row['is_scull']:
                side, ch_name = label
                col_name = f'{side} {ch_name}'
            else:
                side, ch_name = None, label
                col_name = ch_name
            rows[block_start + offset] = dict(
                channel=col_name, position=str(seat_idx), name=ch_name, side=side)
        block_start += seat_row['n_stroke_cols']

    df = pd.DataFrame.from_dict(rows, orient='index')
    df['scale'] = df['name'].map(STROKE_SCALES)
    df['shift'] = df['name'].map(STROKE_SHIFT)
    return df


def infer_gps_table() -> pd.DataFrame:
    """Build the GPS column-index → (channel, position, scale, shift) table."""
    rows = {k: dict(channel=v, position='Boat', name=v, side=None)
            for k, v in _GPS_NAMED_COLS.items()}
    df = pd.DataFrame.from_dict(rows, orient='index')
    df['scale'] = df['name'].map(GPS_SCALES)
    df['shift'] = df['name'].map(GPS_SHIFT)
    return df


# def infer_periodic_column_names(st: pd.DataFrame) -> pd.MultiIndex:
#     """Return a (channel, position) MultiIndex matching periodic stream cols 6, 7, …"""
#     meta = infer_periodic_table(st)
#     return pd.MultiIndex.from_tuples(
#         [(r['channel'], r['position']) for _, r in meta.iterrows()],
#         names=['channel', 'position'],
#     )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — locate data records
# ─────────────────────────────────────────────────────────────────────────────

def _locate_records(bin_u16: np.ndarray, n_sensors: int) -> dict:
    """Stage 2: find start positions for GPS, stroke, and periodic records.

    GPS and stroke records are identified by their fixed 2-word magic headers.
    Periodic records are located using a flag value computed from the sensor
    count:

        flag = n_sensors * 100 + header_bytes

    where header_bytes is either 28 (14 uint16 words) or 36 (18 uint16 words)
    depending on the logger firmware version.  We try both and use the one
    that finds records.

    Parameters
    ----------
    bin_u16 : np.ndarray (uint16 view of the complete file)
    n_sensors : int
        Total number of sensors (boat + gate) from the sensor metadata.
        This equals the number of uint16 values per periodic sample.

    Returns
    -------
    dict with keys:
        gps_starts, stroke_starts, periodic_starts  — uint16 index arrays
        stroke_width      — uint16 columns per stroke record row
        periodic_width    — uint16 columns per periodic sample (= n_sensors)
        periodic_hdr_words — uint16 words in the periodic record header (14 or 18)
    """
    gps_starts,    = search(bin_u16, [0x8013, 10]).nonzero()
    stroke_starts, = search(bin_u16, [0x800A, 10]).nonzero()
    if not (gps_starts.size and stroke_starts.size):
        logging.info("Could not locate GPS/stroke records")
        return

    # Stroke record width: byte +2 of each record stores 2× the uint16 column count
    stroke_widths = np.unique(bin_u16[stroke_starts + 2]) // 2
    if len(stroke_widths) != 1:
        logging.info("Could not identify stroke_width")
        return

    stroke_width = int(stroke_widths.item())

    # Periodic flag encodes width and header size.
    # We only need to search from where the data records begin.
    datastart = int(min(gps_starts[0], stroke_starts[0]))

    periodic_starts = None
    periodic_hdr_words = None
    for h_words in (18, 14):               # 36-byte header is more common
        flag = n_sensors * 100 + h_words * 2
        cands, = search(bin_u16[datastart:], [flag, 0]).nonzero()
        if len(cands) > 10:
            periodic_starts = cands + datastart
            periodic_hdr_words = h_words
            break

    if periodic_starts is None:
        logging.info("Could not locate GPS/stroke records")
        return

    return dict(
        gps_starts=gps_starts,
        stroke_starts=stroke_starts,
        periodic_starts=periodic_starts,
        stroke_width=stroke_width,
        periodic_width=n_sensors,           # one uint16 sample per sensor
        periodic_hdr_words=periodic_hdr_words,
    )


def _parse_raw(
        bin_u16: np.ndarray, rl: dict[str, np.ndarray], gps_coords: tuple[float, float]
) -> dict[str, pd.DataFrame]:
    """Stage 3: extract raw DataFrames.  Populates ``self.raw_data``."""
    # rl = self.record_locs
    raw = {}

    # ── GPS (0x8013) ──────────────────────────────────────────────────────
    # Record layout: 3-word header | 16 uint16 columns × 10 rows
    gps = extract_aperiodic(bin_u16, rl['gps_starts'] + 3, 16, 10)
    gps['Time'] = gps[[1, 2]] @ C16
    gps['Distance'] = gps[[3, 4]] @ C16 / 256 - 16385 / 256
    gps['lat'] = gps[[6, 7]] @ C14
    gps['long'] = gps[[8, 9]] @ C14

    raw['GPS'] = gps.set_index('Time')

    # ── Stroke (0x800A) ───────────────────────────────────────────────────
    sw = rl['stroke_width']
    stroke = extract_aperiodic(bin_u16, rl['stroke_starts'] + 3, sw, 10)
    stroke['Time'] = stroke[[1, 2]] @ C16
    stroke['Distance'] = stroke[[3, 4]] @ C16 / 256 - 16385 / 256
    raw['stroke'] = stroke = stroke.set_index('Time')

    # ── Periodic (50 Hz) ──────────────────────────────────────────────────
    pw = rl['periodic_width']
    # h_off = rl['periodic_hdr_words']  # uint16 words to skip per record
    h_off = 14
    # Group header: 6 uint32 words at offset +2 from the record start.
    # Column 4 of the group header is the validity flag: 1000 = valid data,
    # other values indicate that this slot was occupied by a GPS or stroke
    # record rather than a full periodic group.
    raw['periodic_group'] = groups = extract_aperiodic32(
        bin_u16, rl['periodic_starts'] + 2, 6, 1)

    # Sensor data: pw columns × 50 rows, stored column-major.
    periodic = extract_periodic(
        bin_u16, rl['periodic_starts'] + h_off, pw, 50)

    # Gate sensor values use the top bit as a sign/overflow flag; strip it.
    for c in periodic.columns[7:]:
        periodic[c] = periodic[c] % 2 ** 15

    # Distance is stored as two uint16 words in a non-standard interleaved
    # byte order; permute to restore the correct uint32 value.
    periodic.loc[:, [2, 3]] = (
        np.permute_dims(
            periodic[[2, 3]].values.reshape(-1, 2, 25, 2), (0, 3, 1, 2)
        ).reshape(-1, 2)
    )

    # Merge group header (cols 0–5) with sensor data (cols shifted to 6–6+pw-1).
    raw['raw_periodic'] = periodic = (
        periodic.rename(columns=lambda x: x + 6)
        .join(groups, on='group')
        .reset_index('group', drop=True)
    )

    # Column 3 is the group base time (ms); add per-sample offset (20 ms each).
    periodic['Time'] = periodic[3] = (
        periodic[3] + (np.arange(len(periodic)) % 50) * 20)
    periodic['Distance'] = periodic[[8, 9]] @ C14 / 256 - 16385 / 256
    periodic = periodic[periodic[4] == 1000].set_index('Time')

    periodic['StrokeNumber'] = 0
    periodic.loc[
        stroke.index.intersection(periodic.index), 'StrokeNumber'
    ] = 1
    strokenum = periodic.StrokeNumber = periodic.StrokeNumber.cumsum().squeeze()
    timestamp = periodic.index.to_series().astype(int)

    stroke_end = timestamp.groupby(strokenum).max()
    stroke_start = stroke_end.shift().fillna(0)
    stroke_length = stroke_end.diff(1)
    stroke_length.iloc[0] = stroke_end.iloc[0]
    norm = (
        (timestamp * 1. - strokenum.map(stroke_start))
        / strokenum.map(stroke_length)
    )
    periodic['Normalized Time'] = (norm - 0.5) * 100

    # Keep only groups where the validity flag is 1000.
    raw['periodic'] = periodic  # [periodic[4] == 1000].set_index('Time')
    return raw

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 helper — apply channel metadata to a raw DataFrame
# ─────────────────────────────────────────────────────────────────────────────


def _map_channels(raw_data, metadata) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stage 4: apply channel metadata.  Populates gps, stroke, periodic."""
    m = metadata
    r = raw_data

    gps = map_data(
        r['GPS'], m['gps info'], keep_cols=['Distance', 'latitude', 'longitude'],)
    stroke = map_data(
        r['stroke'], m['stroke info'], keep_cols=['Distance'])

    periodic = map_data(
        r['periodic'], m['periodic info'],
        keep_cols=['Distance', 'Normalized Time', 'StrokeNumber']
    )

    lat0, lon0 = metadata['gps_coords']
    long_scale = _GPS_SCALE * np.cos(np.deg2rad(lat0))
    gps[('latitude', 'Boat')] = (
        gps['lat'] / _GPS_SCALE + lat0).where(gps['lat'] != 0)
    gps[('longitude', 'Boat')] = (
        gps['long'] / long_scale + lon0).where(gps['long'] != 0)

    for side in ['', 'P ', 'S ']:
        if f'{side}MaxAngle' in stroke:
            length = \
                stroke[f'{side}MaxAngle'] - stroke[f'{side}MinAngle']
            effect = length - \
                stroke[side + 'CatchSlip'] - stroke[side + 'FinishSlip']
            new_stroke_data = pd.concat({
                side + "Length": length, side + "Effective": effect}, axis=1
            ).rename_axis(columns=stroke.columns.names)

            stroke = pd.concat([
                stroke, new_stroke_data
            ], axis=1)

        if (ga := side + 'GateAngle') in periodic:
            gateangelvel = np.gradient(
                periodic[ga], periodic.index / 1000, axis=0
            ) + periodic[ga] * 0
            periodic = pd.concat([
                periodic,
                pd.concat({side + 'GateAngleVel': gateangelvel}, axis=1)
                .rename_axis(columns=periodic.columns.names)
            ], axis=1)

    # if 'MaxAngle' in stroke:
    #     stroke_length = stroke.MaxAngle - stroke.MinAngle
    #     effect = stroke_length - stroke.CatchSlip - stroke.FinishSlip
    #     stroke = pd.concat([
    #         stroke,
    #         pd.concat({"Length": stroke_length, "Effective": effect}, axis=1)
    #     ], axis=1)
    #     gateangelvel = np.gradient(
    #         periodic.GateAngle, periodic.index / 1000, axis=0
    #     ) + periodic.GateAngle * 0
    #     periodic = pd.concat([
    #         periodic,
    #         pd.concat({'GateAngleVel': gateangelvel}, axis=1)
    #         .rename_axis(columns=periodic.columns.names)
    #     ], axis=1)
    # elif 'P MaxAngle' in stroke:
    #     for side in ['P', 'S']:
    #         pass

    return gps, stroke, periodic


def map_data(raw: pd.DataFrame, meta: pd.DataFrame, keep_cols=()) -> pd.DataFrame:
    """Apply channel naming and scaling to a raw integer-indexed DataFrame.

    Parameters
    ----------
    raw : DataFrame with integer column indices
    meta : channel-metadata DataFrame (from infer_*_metadata) with columns
           channel, position, scale, shift
    keep_cols : names of already-computed columns in *raw* to pass through
                as-is under position='Boat'

    Returns
    -------
    pd.DataFrame with a (channel, position) MultiIndex column.
    """
    results = {(c, 'Boat'): raw[c] for c in keep_cols if c in raw}

    if meta is not None:
        for i, ch in meta.iterrows():
            if i not in raw.columns:
                continue
            k = (ch['channel'], ch['position'])
            scale = ch['scale']
            shift = ch['shift']
            if pd.isna(scale):
                results[k] = raw[i].where(raw[i] > 0)
            else:
                results[k] = ((raw[i] + 1) * scale - shift).where(raw[i] > 0)

    return pd.concat(results, axis=1, names=['channel', 'position'])


# ─────────────────────────────────────────────────────────────────────────────
# Crew parsing
# ─────────────────────────────────────────────────────────────────────────────

_CREW_RECORD_RE = re.compile(b'\x05\x00\x00\x00\xe0([\x00-\xff])\x00\x00')
_NAME_RE = re.compile(b'\x80([\x01-\x20])\x80([\x20-\x7E]+)')


def parse_crew_from_index(index_data: bytes) -> pd.DataFrame:
    """Parse the crew list from a peach-data-index binary.

    The index is a proprietary binary database (similar to JET/MDB).
    BoatConfig records follow a detectable pattern encoding position, side,
    and name.  Positions 0 and >9 are template slots and are excluded.
    Named records are preferred over unnamed ones at the same position.

    Returns
    -------
    pd.DataFrame with columns: position (int), side, name.
    """
    records: dict[int, dict] = {}

    for m in _CREW_RECORD_RE.finditer(index_data):
        pos = m.group(1)[0]
        if pos == 0 or pos > 9:
            continue
        after = index_data[m.end(): m.end() + 40]
        side_byte = after[15] if len(after) > 15 else 0
        nm = _NAME_RE.search(after)
        name = nm.group(2).decode(
            'ascii', errors='replace').strip() if nm else None

        side = 'Cox' if pos == 9 else ('Port' if side_byte == 1 else 'Stbd')

        if pos not in records or name is not None:
            records[pos] = dict(
                position=int(pos),
                side=side,
                name=name if name is not None else f'Seat {pos}',
            )

    if not records:
        return pd.DataFrame(columns=['position', 'side', 'name'])
    return (pd.DataFrame(records.values())
            .sort_values('position')
            .reset_index(drop=True))


def _load_crew(peach_path: str) -> pd.DataFrame:
    """Try to load the crew index file alongside *peach_path*."""
    idx = re.sub(r'\.peach-data$', '.peach-data-index', peach_path)
    try:
        with open(idx, 'rb') as f:
            return parse_crew_from_index(f.read())
    except FileNotFoundError:
        return None
        # return pd.DataFrame(columns=['position', 'side', 'name'])


# ─────────────────────────────────────────────────────────────────────────────
# Alignment / validation
# ─────────────────────────────────────────────────────────────────────────────

def check_alignment(parsed: pd.DataFrame, ref: pd.DataFrame):
    """Compute regression statistics between *parsed* and *ref* DataFrames.

    Returns (calibration_df, missing_columns_index).
    Perfect alignment: slope=1, intercept=0, rvalue=1, rmse=0.
    """
    ref = ref.drop(EXPORT_EXTRA_COLS, axis=1, level=0, errors='ignore')
    missing = ref.columns.difference(parsed.columns)
    p, r = parsed.align(ref, join='inner')
    regs = {}
    for c, v in r.items():
        if v.std() > 0:
            x, y = v.dropna().align(p[c].dropna(), join='inner')
            regs[c] = pd.Series(stats.linregress(x, y)._asdict())
    cal = pd.concat(regs, names=['reference', 'parsed']).unstack()
    cal['rmse'] = np.square(p - r).mean() ** 0.5
    return cal, missing


# ─────────────────────────────────────────────────────────────────────────────
# PeachData — main public interface
# ─────────────────────────────────────────────────────────────────────────────


class PeachData:
    """Container for all data parsed from a ``.peach-data`` file.

    Processing pipeline
    -------------------
    1. :meth:`_load_header`   — parse session metadata from the file header.
    2. :meth:`_locate_records` — find GPS, stroke and periodic record positions.
    3. :meth:`_parse_raw`      — extract raw integer DataFrames.
    4. :meth:`_map_channels`   — apply naming and scaling to produce the public
                                  DataFrames (gps, stroke, periodic).

    Attributes
    ----------
    path : str
        Path to the source ``.peach-data`` file.
    metadata : dict
        Parsed header fields (date, params, sensors, sensors info,
        stroke info, periodic info, gps info, events, gps_coords).
    record_locs : dict
        Record start positions and widths from Stage 2.
    raw_data : dict
        Raw uint16 DataFrames: 'GPS', 'stroke', 'periodic', 'periodic_group'.
    gps : pd.DataFrame
        GPS aperiodic data with (channel, position) MultiIndex columns.
        Includes computed latitude/longitude (WGS84 °) and metre-offset
        lat/long, plus GPS Fix Counter (sequential, not absolute UTC).
    stroke : pd.DataFrame
        Per-stroke summary with (channel, position) MultiIndex columns.
    periodic : pd.DataFrame
        50 Hz sensor data with (channel, position) MultiIndex columns.
    crew : pd.DataFrame
        Columns: position (int), side, name.  Empty if no index file found.
    """

    def __init__(self, path=None):
        self.path:        str | None = None if path is None else str(path)
        self.metadata:    dict | None = None
        self.record_locs: dict | None = None
        self.raw_data:    dict | None = None
        self.gps:         pd.DataFrame | None = None
        self.stroke:      pd.DataFrame | None = None
        self.periodic:    pd.DataFrame | None = None
        self.crew:        pd.DataFrame | None = None

    # ── Public factory method ─────────────────────────────────────────────────

    @classmethod
    def from_path(cls, peach_path: str) -> 'PeachData':
        """Load a ``.peach-data`` file.  The matching ``.peach-data-index``
        (crew list) is discovered automatically if present.
        """

        with open(peach_path, 'rb') as f:
            raw_bytes = f.read()

        bin_u16 = np.frombuffer(raw_bytes, dtype=np.uint16)

        self = cls(peach_path)._load_from_bin(bin_u16)
        self._crew = _load_crew(self.path)

        return self

    @classmethod
    def from_bytes(cls, raw_bytes, index_bytes=None, path=None):
        bin_u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
        self = cls(path)._load_from_bin(bin_u16)
        if index_bytes:
            self._crew = parse_crew_from_index(index_bytes)

        return self

    def _load_from_bin(self, bin_u16):
        self.metadata = load_header(bin_u16)
        if self.metadata:
            n_sensors = len(self.metadata['sensors info'])
            self.record_locs = _locate_records(bin_u16, n_sensors)

        if self.record_locs:
            self.raw_data = _parse_raw(
                bin_u16, self.record_locs, self.gps_coords)
            self.gps, self.stroke, self.periodic = _map_channels(
                self.raw_data, self.metadata)

        return self

    def add_details(self, names=True, side=True):
        if (self.crew is not None) and ('position' in self.crew):
            crew_list = self.crew.astype(str).set_index('position')
            crew_list.loc['Boat'] = pd.Series({
                'side': 'Boat', 'name': 'Boat'})

            cols = ['channel', 'name' if names else 'position']
            if side:
                cols.append('side')

            if self.is_sculling:
                def _add_details(df):
                    merged = (
                        df.columns.to_frame(index=False).astype(str)
                        .join(crew_list, on='position', how='left')
                    )
                    merged['side'] = (
                        merged.channel.str.extract("^(P|S)\s")[0]
                        .replace({'P': 'Port', 'S': 'Stbd'})
                        .fillna('Boat')
                    )
                    merged['channel'] = merged.channel.str.replace(
                        "^(P |S )", "", regex=True)

                    new_df = df.copy(False)
                    new_df.columns = pd.MultiIndex.from_frame(merged[cols])
                    return new_df

            else:
                def _add_details(df):
                    merged = (
                        df.columns.to_frame(index=False).astype(str)
                        .join(crew_list, on='position', how='left')
                    )

                    new_df = df.copy(False)
                    new_df.columns = pd.MultiIndex.from_frame(merged[cols])
                    return new_df
        else:
            def _add_details(x):
                return x

        updated = type(self)()
        updated.path = self.path
        updated.metadata = self.metadata
        updated.record_locs = self.record_locs
        updated.raw_data = self.raw_data
        updated.gps = self.gps
        if self.stroke is not None:
            updated.stroke = _add_details(self.stroke)
        if self.periodic is not None:
            updated.periodic = _add_details(self.periodic)
        updated.crew = self.crew

        return updated

        # ── Repr ─────────────────────────────────────────────────────────────────

    def __repr__(self):
        if self.metadata is None:
            return 'PeachData(unloaded)'

        try:
            from rowing import utils
            duration = utils.format_timedelta_hours(
                self.duration, hundreths=False)
        except (ImportError, AttributeError):
            duration = self.duration

        return (
            "PeachData("
            f"id={self.id}, boat={self.boat_type}, distance={self.distance}, "
            f"duration={duration}, strokes={self.n_strokes}, date={self.date}"
            ")"
        )

    # ── Convenience properties ────────────────────────────────────────────────
    @property
    def crew(self):
        if self._crew is None:
            return pd.DataFrame(dict(
                position=self.seats,
                name=self.seats,
                side=['Port'] * len(self.seats),  # TODO Fix
            ))
        return self._crew

    @crew.setter
    def crew(self, crew):
        self._crew = crew

    @property
    def oar_type(self):
        if self.metadata:
            return 'sculling' if self.is_sculling else 'sweep'
        return ''

    @property
    def boat_type(self) -> str:
        if self.metadata:
            nrower = self.seats[-1]
            mod = '-'
            if self.is_sculling:
                mod = 'x'
            elif nrower == 8:
                mod = '+'
            elif len(self.crew) and (self.crew.name.iloc[-1] != 'Seat 9'):
                mod = '+'

            return f"{nrower}{mod}"

        return ''

    @property
    def id(self):
        if self.serial:
            return f"{self.serial:06d}-{self.session:06d}"

    @property
    def distance(self) -> int:
        if self.gps is not None:
            return int(self.gps.Distance.iloc[-1, 0])
        return 0

    @property
    def n_strokes(self) -> int:
        if self.stroke is not None:
            return int(self.stroke.StrokeNumber.iloc[-1, 0])
        return 0

    @property
    def n_periodic(self) -> int:
        if self.periodic is not None:
            return len(self.periodic)
        return 0

    @property
    def duration(self) -> pd.Timedelta:
        if self.gps is not None:
            return pd.Timedelta(self.gps.index[-1], unit='ms')
        return pd.Timedelta(0)

    @property
    def date(self) -> pd.Timestamp | None:
        """Session timestamp."""
        if self.metadata:
            return self.metadata['date']

        return pd.Timestamp(0)

    @property
    def gps_coords(self) -> tuple[float, float] | None:
        """GPS initialisation point as ``(latitude_deg, longitude_deg)``."""
        if self.metadata:
            return tuple(self.metadata['gps_coords'])

    @property
    def serial(self) -> int | None:
        if self.metadata:
            return self.metadata['serial']

    @property
    def session(self) -> int | None:
        if self.metadata:
            return self.metadata['session']

    @property
    def details(self) -> pd.DataFrame | None:
        if self.metadata:
            ks = [
                'serial', 'session', 'date', 'latitude', 'longitude'
            ]
            details = {
                k: self.metadata[k]
                for k in ks
            }
            details['type'] = self.boat_type
            details['distance'] = self.distance
            details['strokes'] = self.n_strokes
            details['duration'] = self.duration
            crew = (
                self.crew
                .set_index('position')
                .stack().swaplevel().sort_index()
                .to_frame().T
            )
            crew.index = [self.id]
            return pd.concat([
                pd.DataFrame.from_dict(
                    {('Boat', self.id): details}, orient='index'
                ).unstack(0),
                crew
            ], axis=1)

    @property
    def data(self) -> dict[str, pd.DataFrame] | None:
        if self.metadata:
            return dict(
                details=self.details,
                params=self.params,
                gps=self.gps,
                stroke=self.stroke,
                periodic=self.periodic
            )

    def app_data(self, names=True, with_timings=True):
        from rowing.analysis import files, telemetry
        detailed = self.add_details(names=names, side=True)
        app_data = {
            # 'power':
            'details': detailed.details,
            'Crew Info': detailed.crew,
            'Parameter Info': detailed.params
            .rename_axis('Parameter')
            .rename(columns={'value': 'Value'})
            .reset_index(),
            'Sensor Info': detailed.sensor_table,
        }
        positions = (
            detailed.gps.reset_index()
            .droplevel(1, axis=1)
            .dropna(subset=['latitude', 'longitude'])
        )
        positions['time'] = detailed.date + \
            pd.to_timedelta(positions.Time, unit='ms')
        app_data['positions'] = files.process_latlontime(positions)

        app_data['Periodic'] = periodic = (
            detailed.periodic.reset_index()
            .rename_axis(columns=['channel', 'rower', 'side'])
        )
        periodic['timestamp', 'Boat', 'Boat'] = periodic.Time
        periodic['Time'] = detailed.date + \
            pd.to_timedelta(periodic.Time, unit='ms')

        power = detailed.stroke.reset_index().rename_axis(
            columns=['channel', 'rower', 'side'])
        power['timestamp', 'Boat', 'Boat'] = power.Time
        power['Time'] = detailed.date + \
            pd.to_timedelta(power.Time, unit='ms')

        if self.is_sculling:
            rower_power = (
                power.SwivelPower
                .xs('Stbd', level=1, axis=1, drop_level=False)
                + power.SwivelPower
                .xs('Port', level=1, axis=1, drop_level=True)
            )
        else:
            rower_power = power.SwivelPower

        app_data['power'] = pd.concat([
            power, pd.concat({'Rower Swivel Power': rower_power}, axis=1)
        ], axis=1)
        if with_timings:
            app_data['power'] = telemetry.add_timings(app_data)

        return app_data

    @property
    def crew_list(self):
        if self.crew is not None:
            return self.crew.set_index('position').Name.rename(index=str)

    def save(self, base_path):
        base = Path(base_path)
        export_data = self.data
        if export_data:
            for folder, df in export_data.items():
                folder = base / folder
                folder.mkdir(exist_ok=True, parents=True)
                if df is not None:
                    (
                        df.rename(columns=str)
                        .to_parquet(folder / f"{self.id}.parquet")
                    )

        return self

    @property
    def sensor_table(self) -> pd.DataFrame | None:
        """Human-readable sensor table (gate sensors first, seat-ascending)."""
        if self.metadata:
            return self.metadata['sensors info']

    @property
    def seats(self) -> list[int] | None:
        """Sorted list of gate-sensor seat numbers present in this file."""
        if self.metadata:
            st = self.sensor_table
            return sorted(int(s) for s in st.loc[~st['is_boat'], 'seat'].dropna().unique())

    @property
    def is_sculling(self) -> bool | None:
        """True if this file contains sculling (P/S) gate sensors."""
        if self.metadata:
            st = self.sensor_table
            return bool(self.sensor_table['is_scull'].any())

    @property
    def params(self) -> pd.DataFrame | None:
        """Boat configuration parameters (oar lengths, drive thresholds, …)."""
        if self.metadata:
            return self.metadata['params'][['value', 'unit']]

    # ── Validation ────────────────────────────────────────────────────────────

    def check_alignment(self, ref_data: dict) -> tuple[dict, pd.DataFrame]:
        """Validate parsed data against a reference export.

        Parameters
        ----------
        ref_data : dict
            Output of :func:`parse_reference_file`.

        Returns
        -------
        cals : dict[str, pd.DataFrame]
            Regression statistics keyed by 'periodic', 'gps', 'stroke'.
            Perfect alignment: slope=1, intercept=0, rvalue=1, rmse=0.
        missing : pd.DataFrame
            Reference columns absent from the parsed data, with a 'data' column
            indicating which stream they came from.
        """
        cals, missing_parts = {}, {}
        for key, ref_key in [
            ('periodic', 'Periodic'),
            ('gps',      'Aperiodic 0x8013'),
            ('stroke',   'Aperiodic 0x800A'),
        ]:
            parsed_df = getattr(self, key)
            cals[key], missing_parts[key] = check_alignment(
                parsed_df, ref_data[ref_key])

        missing = pd.concat(
            {k: v.to_frame(index=False) for k, v in missing_parts.items()},
            names=['data'],
        ).reset_index(0)
        return cals, missing


# ─────────────────────────────────────────────────────────────────────────────
# Reference-file parsing  (validation / alignment only)
# ─────────────────────────────────────────────────────────────────────────────


def parse_reference_lines(lines, **kws):
    k = None
    last = 0
    groups = {}
    for i, line in enumerate(lines):
        if line.startswith("====="):
            if k:
                groups[k] = lines[last + 1:i]
            k = " ".join(line.strip().split("\t")[1:])
            last = i
    groups[k] = lines[last + 1:]

    data = {}
    for k, ls in groups.items():
        kwargs = {**kws}
        if k == 'Rig Info':
            ls = ['Position\tSide\n'] + ls[2:]
        elif 'eriodic' in k:
            kwargs['header'] = list(range(2))
        if ls:
            data[k] = df = pd.read_table(io.StringIO("".join(ls)), **kwargs)
        if 'eriodic' in k:
            unnamed_cols = df.droplevel(
                axis=1, level=1).filter(regex='Unnamed').columns
            data[k] = (
                df
                .drop(unnamed_cols, axis=1, level=0)
                .rename(
                    lambda x: x if x in ALLOWED_POSITIONS else 'Boat', axis=1, level=1
                )
                .rename_axis(columns=['channel', 'position'])
                .set_index(('Time', 'Boat'))
                .rename_axis(index='Time')
            )
    return data


def parse_reference_file(filepath, **kws):
    if filepath.endswith("peach-data"):
        filepath = filepath.replace("peach-data", 'txt')
    with open(filepath, 'r') as f:
        return parse_reference_lines(f.readlines(), **kws)


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'row005235-000119D.peach-data'
    data = PeachData.from_path(path)
    print(data)
    print(data.details)


if __name__ == '__main__':
    main()
