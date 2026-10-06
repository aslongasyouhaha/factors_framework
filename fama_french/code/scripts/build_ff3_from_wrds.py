from __future__ import annotations

import argparse
import io
import struct
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))
sys.path.insert(0, str(CODE_ROOT))

from factors import FactorBuildTools
from fama_french import FamaFrench3Builder


RAW = PROJECT_ROOT / "data" / "raw"
CRSP_ZIP = next(RAW.glob("CRSP-Annual Update-Stock daily*.zip"))
COMPUSTAT_ZIP = RAW / "20250102updated north fundamental_dta.zip"
COMPUSTAT_QUARTERLY_ZIP = RAW / "Comp_Quarterly6126.csv.zip"
LINK_ZIP = RAW / "link.zip"
YIELD_1M = RAW / "yield_data" / "DGS1MO.csv"
OUT = PROJECT_ROOT / "results"


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


def exchange_code(primary_exchange: pd.Series) -> pd.Series:
    return primary_exchange.map({"N": 1, "A": 2, "Q": 3}).astype("float")


def build_crsp_monthly(start_year: int, end_year: int, chunksize: int) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"crsp_monthly_permco_{start_year}_{end_year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    usecols = [
        "PERMNO",
        "PERMCO",
        "DlyCalDt",
        "DlyRet",
        "DlyPrc",
        "DlyCap",
        "DlyVol",
        "PrimaryExch",
        "SecurityType",
        "SecuritySubType",
        "ShareType",
        "USIncFlg",
        "TradingStatusFlg",
        "ConditionalType",
        "SecurityBegDt",
    ]
    parts = []
    reader = NestedZipFirstCsvReader(CRSP_ZIP)
    text = io.TextIOWrapper(reader, encoding="utf-8", newline="")
    for i, chunk in enumerate(pd.read_csv(text, usecols=usecols, chunksize=chunksize, low_memory=False)):
        chunk = chunk.rename(
            columns={
                "PERMNO": "asset_id",
                "PERMCO": "company_id",
                "DlyCalDt": "date",
                "DlyRet": "ret",
                "DlyPrc": "price",
                "DlyCap": "market_equity",
                "DlyVol": "volume",
                "PrimaryExch": "primary_exchange",
                "SecurityBegDt": "first_trade_date",
            }
        )
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk[chunk["date"].dt.year.between(start_year - 1, end_year + 1)]
        if chunk.empty:
            continue
        chunk = chunk[
            chunk["SecurityType"].eq("EQTY")
            & chunk["SecuritySubType"].eq("COM")
            & chunk["USIncFlg"].eq("Y")
            & chunk["TradingStatusFlg"].eq("A")
            & chunk["ConditionalType"].eq("RW")
            & chunk["primary_exchange"].isin(["N", "A", "Q"])
        ].copy()
        if chunk.empty:
            continue
        chunk["ret"] = pd.to_numeric(chunk["ret"], errors="coerce")
        chunk["company_id"] = pd.to_numeric(chunk["company_id"], errors="coerce")
        chunk["market_equity"] = pd.to_numeric(chunk["market_equity"], errors="coerce")
        chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce").abs()
        chunk["volume"] = pd.to_numeric(chunk["volume"], errors="coerce")
        chunk = chunk.dropna(subset=["company_id"])
        chunk["company_id"] = chunk["company_id"].astype("int64")
        chunk["exchange_code"] = exchange_code(chunk["primary_exchange"])
        chunk["month"] = chunk["date"] + pd.offsets.MonthEnd(0)
        chunk["ret_gross"] = 1.0 + chunk["ret"]
        chunk["dollar_volume"] = chunk["price"] * chunk["volume"]
        monthly = (
            chunk.sort_values(["asset_id", "date"])
            .groupby(["asset_id", "month"], as_index=False)
            .agg(
                company_id=("company_id", "last"),
                ret_gross=("ret_gross", "prod"),
                n_ret=("ret", "count"),
                last_trade_date=("date", "last"),
                market_equity=("market_equity", "last"),
                exchange_code=("exchange_code", "last"),
                dollar_volume=("dollar_volume", "mean"),
                first_trade_date=("first_trade_date", "first"),
            )
        )
        monthly["ret"] = monthly["ret_gross"] - 1.0
        parts.append(monthly.drop(columns="ret_gross"))
        if i % 25 == 0:
            print(f"CRSP chunks read: {i:,}; monthly parts: {len(parts):,}")
    if not parts:
        raise RuntimeError("No CRSP rows found for requested date range.")
    monthly = pd.concat(parts, ignore_index=True)
    monthly = (
        monthly.sort_values(["asset_id", "month", "last_trade_date"])
        .groupby(["asset_id", "month"], as_index=False)
        .agg(
            company_id=("company_id", "last"),
            ret=("ret", lambda x: (1.0 + x).prod() - 1.0),
            n_ret=("n_ret", "sum"),
            last_trade_date=("last_trade_date", "last"),
            market_equity=("market_equity", "last"),
            exchange_code=("exchange_code", "last"),
            dollar_volume=("dollar_volume", "mean"),
            first_trade_date=("first_trade_date", "first"),
        )
    )
    monthly = monthly.rename(columns={"month": "date"})
    monthly["first_trade_date"] = pd.to_datetime(monthly["first_trade_date"], errors="coerce")
    monthly = assign_permco_market_equity(monthly)
    monthly.to_parquet(cache, index=False, compression="zstd")
    return monthly


def assign_permco_market_equity(monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multiple PERMNO share classes to PERMCO-level market equity.

    Fama-French style construction sums market equity across securities of the
    same company and assigns the company market equity to the largest security.
    """
    out = monthly.copy()
    out["company_market_equity"] = out.groupby(["date", "company_id"])["market_equity"].transform("sum")
    rank = out.groupby(["date", "company_id"])["market_equity"].rank(method="first", ascending=False)
    out["ff_primary_share"] = rank.eq(1)
    out["market_equity_security"] = out["market_equity"]
    out["market_equity"] = out["company_market_equity"]
    return out


def build_compustat_be(start_year: int, end_year: int) -> pd.DataFrame:
    cache = OUT / f"compustat_be_{start_year}_{end_year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    cols = ["gvkey", "datadate", "fyear", "indfmt", "consol", "popsrc", "datafmt", "seq", "ceq", "txditc", "pstkrv", "pstkl", "pstk", "at", "lt"]
    parts = []
    with zipfile.ZipFile(COMPUSTAT_ZIP) as z:
        dta_name = [n for n in z.namelist() if n.endswith(".dta")][0]
        with z.open(dta_name) as f:
            for chunk in pd.read_stata(f, columns=cols, chunksize=100_000, convert_categoricals=False):
                chunk = chunk[
                    chunk["fyear"].between(start_year - 2, end_year)
                    & chunk["indfmt"].eq("INDL")
                    & chunk["consol"].eq("C")
                    & chunk["popsrc"].eq("D")
                    & chunk["datafmt"].eq("STD")
                ].copy()
                if chunk.empty:
                    continue
                chunk["book_equity"] = FactorBuildTools.book_equity(
                    seq=chunk["seq"],
                    ceq=chunk["ceq"],
                    txditc=chunk["txditc"],
                    pstkrv=chunk["pstkrv"],
                    pstkl=chunk["pstkl"],
                    pstk=chunk["pstk"],
                    at=chunk["at"],
                    lt=chunk["lt"],
                )
                parts.append(chunk[["gvkey", "datadate", "fyear", "book_equity"]].dropna(subset=["book_equity"]))
    if not parts:
        raise RuntimeError("No Compustat book equity rows found.")
    be = pd.concat(parts, ignore_index=True)
    be["gvkey"] = be["gvkey"].astype("string").str.zfill(6)
    be["datadate"] = pd.to_datetime(be["datadate"])
    be["fyear"] = be["fyear"].astype(int)
    be.to_parquet(cache, index=False, compression="zstd")
    return be


def build_compustat_be_quarterly(start_year: int, end_year: int, chunksize: int) -> pd.DataFrame:
    """Approximate annual book equity from Compustat quarterly fiscal Q4 rows."""
    cache = OUT / f"compustat_be_quarterly_q4_{start_year}_{end_year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    cols = [
        "gvkey",
        "datadate",
        "fyearq",
        "fqtr",
        "indfmt",
        "consol",
        "datafmt",
        "seqq",
        "ceqq",
        "txditcq",
        "pstkrq",
        "pstkq",
        "atq",
        "ltq",
    ]
    parts = []
    with zipfile.ZipFile(COMPUSTAT_QUARTERLY_ZIP) as z:
        csv_name = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")][0]
        with z.open(csv_name) as f:
            for chunk in pd.read_csv(f, usecols=cols, chunksize=chunksize, low_memory=False):
                chunk = chunk[
                    chunk["fyearq"].between(start_year - 2, end_year)
                    & chunk["fqtr"].eq(4)
                    & chunk["indfmt"].eq("INDL")
                    & chunk["consol"].eq("C")
                    & chunk["datafmt"].eq("STD")
                ].copy()
                if chunk.empty:
                    continue
                chunk["book_equity"] = FactorBuildTools.book_equity(
                    seq=chunk["seqq"],
                    ceq=chunk["ceqq"],
                    txditc=chunk["txditcq"],
                    pstkrv=chunk["pstkrq"],
                    pstkl=None,
                    pstk=chunk["pstkq"],
                    at=chunk["atq"],
                    lt=chunk["ltq"],
                )
                out = chunk.rename(columns={"fyearq": "fyear"})[["gvkey", "datadate", "fyear", "book_equity"]]
                parts.append(out.dropna(subset=["book_equity"]))
    if not parts:
        raise RuntimeError("No quarterly Compustat fiscal Q4 book equity rows found.")
    be = pd.concat(parts, ignore_index=True)
    be["gvkey"] = be["gvkey"].astype("string").str.zfill(6)
    be["datadate"] = pd.to_datetime(be["datadate"])
    be["fyear"] = pd.to_numeric(be["fyear"], errors="coerce").astype("int64")
    be = be[be["book_equity"].gt(0)].copy()
    be = be.sort_values(["gvkey", "fyear", "datadate"]).drop_duplicates(["gvkey", "fyear"], keep="last")
    be.to_parquet(cache, index=False, compression="zstd")
    return be


def choose_compustat_source(source: str) -> str:
    if source != "auto":
        return source
    if COMPUSTAT_QUARTERLY_ZIP.exists():
        return "quarterly"
    return "annual"


def load_link_table() -> pd.DataFrame:
    cache = OUT / "ccm_link_filtered.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    usecols = ["gvkey", "LINKPRIM", "LINKTYPE", "LPERMNO", "LINKDT", "LINKENDDT"]
    with zipfile.ZipFile(LINK_ZIP) as z:
        with z.open(z.namelist()[0]) as f:
            link = pd.read_csv(f, usecols=usecols, dtype={"gvkey": "string"})
    link = link[link["LPERMNO"].notna() & link["LINKTYPE"].isin(["LU", "LC"]) & link["LINKPRIM"].isin(["P", "C"])].copy()
    link["gvkey"] = link["gvkey"].astype("string").str.zfill(6)
    link["asset_id"] = link["LPERMNO"].astype("int64")
    link["linkdt"] = pd.to_datetime(link["LINKDT"])
    link["linkenddt"] = pd.to_datetime(link["LINKENDDT"].replace("E", "2262-04-11"))
    link = link[["gvkey", "asset_id", "linkdt", "linkenddt", "LINKTYPE", "LINKPRIM"]]
    link.to_parquet(cache, index=False, compression="zstd")
    return link


def link_be_to_permno(be: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    merged = be.merge(link, on="gvkey", how="inner")
    merged = merged[merged["datadate"].between(merged["linkdt"], merged["linkenddt"])]
    merged = merged.sort_values(["asset_id", "fyear", "datadate"]).drop_duplicates(["asset_id", "fyear"], keep="last")
    return merged[["asset_id", "fyear", "datadate", "book_equity"]]


def build_characteristics(monthly: pd.DataFrame, be_permno: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    m = monthly[monthly["ff_primary_share"]].copy()
    m["year"] = m["date"].dt.year
    m["month_num"] = m["date"].dt.month
    june = m[m["month_num"].eq(6)][["asset_id", "company_id", "year", "date", "market_equity", "exchange_code", "first_trade_date", "dollar_volume"]]
    june = june.rename(columns={"date": "rebalance_date"})
    dec = m[m["month_num"].eq(12)][["asset_id", "year", "market_equity"]].rename(columns={"year": "be_year", "market_equity": "dec_market_equity"})
    chars = june[june["year"].between(start_year, end_year)].copy()
    chars["be_year"] = chars["year"] - 1
    chars = chars.merge(dec, on=["asset_id", "be_year"], how="inner")
    chars = chars.merge(be_permno.rename(columns={"fyear": "be_year"}), on=["asset_id", "be_year"], how="inner")
    # Compustat fundamentals are in USD millions; CRSP DlyCap is in USD thousands.
    chars["bm"] = chars["book_equity"] * 1000.0 / chars["dec_market_equity"]
    chars = chars.rename(columns={"market_equity": "market_equity"})
    chars = chars[chars["bm"].gt(0) & chars["market_equity"].gt(0) & chars["dec_market_equity"].gt(0)]
    chars = FactorBuildTools.filter_listing_age(
        FactorBuildTools.to_panel_index(chars, date_col="rebalance_date", asset_col="asset_id"),
        min_days=180,
    ).reset_index().rename(columns={"date": "rebalance_date"})
    return chars[["rebalance_date", "asset_id", "company_id", "market_equity", "bm", "exchange_code", "book_equity", "dec_market_equity", "dollar_volume"]]


def load_risk_free(start_year: int, end_year: int) -> pd.DataFrame | None:
    if not YIELD_1M.exists():
        return None
    rf = pd.read_csv(YIELD_1M, parse_dates=["observation_date"]).rename(columns={"observation_date": "date", "DGS1MO": "rate"})
    rf["rate"] = pd.to_numeric(rf["rate"], errors="coerce")
    rf = rf[rf["date"].dt.year.between(start_year, end_year + 1)]
    rf["month"] = rf["date"] + pd.offsets.MonthEnd(0)
    out = rf.sort_values("date").groupby("month", as_index=False).tail(1)
    out = out.rename(columns={"date": "rf_observation_date", "month": "date"})
    out["RF"] = out["rate"].fillna(0.0) / 100.0 / 12.0
    return out[["date", "RF"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Fama-French 3 factors from local CRSP/Compustat/link data.")
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--compustat-source", choices=["auto", "annual", "quarterly"], default="auto")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    monthly = build_crsp_monthly(args.start_year, args.end_year, args.chunksize)
    compustat_source = choose_compustat_source(args.compustat_source)
    if compustat_source == "quarterly":
        be = build_compustat_be_quarterly(args.start_year, args.end_year, args.chunksize)
    else:
        be = build_compustat_be(args.start_year, args.end_year)
    link = load_link_table()
    be_permno = link_be_to_permno(be, link)
    chars = build_characteristics(monthly, be_permno, args.start_year, args.end_year)

    returns = monthly[monthly["ff_primary_share"]][["date", "asset_id", "ret", "market_equity"]]
    rf = load_risk_free(args.start_year, args.end_year)
    builder = FamaFrench3Builder()
    factor = builder.build({"characteristics": chars, "returns": returns, "risk_free": rf})

    chars.to_parquet(OUT / f"ff3_characteristics_{args.start_year}_{args.end_year}.parquet", index=False, compression="zstd")
    factor.holdings.to_parquet(OUT / f"ff3_holdings_{args.start_year}_{args.end_year}.parquet", index=False, compression="zstd")
    factor.returns.to_csv(OUT / f"ff3_factors_{args.start_year}_{args.end_year}.csv", index=False)
    factor.diagnostics["compustat_source"] = compustat_source
    factor.diagnostics.to_csv(OUT / f"ff3_diagnostics_{args.start_year}_{args.end_year}.csv", index=False)

    print("Compustat source:", compustat_source)
    print("Characteristics:", chars.shape)
    print("Holdings:", factor.holdings.shape)
    print("Factors:", factor.returns.shape)
    print(factor.returns.head().to_string(index=False))
    print(f"Wrote outputs under {OUT.resolve()}")


if __name__ == "__main__":
    main()




