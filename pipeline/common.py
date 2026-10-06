"""Shared helpers for the pipeline scripts that turn data/source into data/base."""
from __future__ import annotations

import io
import struct
import sys
import zipfile
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "framework"))

from factorlab.data import BASE, SOURCE, save_base  # noqa: E402,F401

CRSP_DAILY_ZIP = next((SOURCE / "crsp").glob("CRSP-Annual Update-Stock daily*.zip"), SOURCE / "crsp" / "missing.zip")
COMPUSTAT_QUARTERLY_ZIP = SOURCE / "compustat" / "Comp_Quarterly6126.csv.zip"
CCM_LINK_ZIP = SOURCE / "ccm" / "link.zip"
FRED = SOURCE / "fred"


class NestedZipFirstCsvReader(io.RawIOBase):
    """Stream the first CSV inside the nested CRSP _csv.zip without extracting it."""

    def __init__(self, outer_zip: Path):
        self.outer = zipfile.ZipFile(outer_zip)
        inner_name = [n for n in self.outer.namelist() if n.endswith("_csv.zip")][0]
        self.stream = self.outer.open(inner_name)
        sig = self.stream.read(4)
        if sig != b"PK\x03\x04":
            raise ValueError("Inner file is not a local ZIP entry stream.")
        header = self.stream.read(26)
        version, flag, method, *_rest, name_len, extra_len = struct.unpack("<HHHHHIIIHH", header)
        self.inner_csv_name = self.stream.read(name_len).decode("utf-8", errors="replace")
        self.stream.read(extra_len)
        if method != 8:
            raise ValueError(f"Unsupported inner ZIP compression method: {method}")
        self.decompressor = zlib.decompressobj(-15)
        self.buffer = bytearray()
        self.closed_flag = False

    def readable(self) -> bool:
        return True

    def readinto(self, b: bytearray) -> int:
        data = self.read(len(b))
        if not data:
            return 0
        b[: len(data)] = data
        return len(data)

    def read(self, size: int = -1) -> bytes:
        if self.closed_flag:
            return b""
        if size is None or size < 0:
            chunks = [bytes(self.buffer)]
            self.buffer.clear()
            while not self.decompressor.eof:
                chunk = self.stream.read(1024 * 1024)
                if not chunk:
                    break
                chunks.append(self.decompressor.decompress(chunk))
            return b"".join(chunks)
        while len(self.buffer) < size and not self.decompressor.eof:
            chunk = self.stream.read(1024 * 1024)
            if not chunk:
                break
            self.buffer.extend(self.decompressor.decompress(chunk))
        out = bytes(self.buffer[:size])
        del self.buffer[:size]
        return out

    def close(self) -> None:
        if not self.closed_flag:
            self.closed_flag = True
            self.stream.close()
            self.outer.close()
        super().close()
