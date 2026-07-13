"""data.go.kr file-dataset collector (no login needed for file downloads).

Flow per dataset page: fileData.do HTML -> main uddi (fn_fileDataDown args)
-> selectFileDataDownload.do metadata JSON -> atchFileId/fileDetailSn ->
cmm/fileDownload.do binary. Files land in curated/raw/kpx/.

Usage: uv run python -m src.data.collectors.datagokr
"""

from __future__ import annotations

import io
import json
import os
import re
import zipfile

import requests

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
RAW_DIR = os.path.join(ROOT, "curated", "raw", "kpx")

DATASETS = {
    "jeju_gen": "15127502",       # 제주 태양광/풍력 시간대별 발전량 (2019-12~2023-12)
    "regional_gen": "15065269",   # 지역별 시간별 태양광 및 풍력 발전량 (최신 연도 연장용)
    "jeju_demand": "15065239",    # 시간별 제주전력수요
    "national_demand": "15065266",  # 시간별 전국 전력수요량
}

BASE = "https://www.data.go.kr"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; norm-boundary)"}


def main_uddi(pk: str) -> str:
    html = requests.get(f"{BASE}/data/{pk}/fileData.do", headers=HEADERS,
                        timeout=60).text
    m = re.search(r"fn_fileDataDown\('%s',\s*'(uddi:[a-z0-9-]+)'" % pk, html)
    if not m:
        raise RuntimeError(f"{pk}: no fn_fileDataDown uddi found")
    return m.group(1)


def all_uddis(pk: str) -> list[str]:
    """Every uddi on the page — includes previous (annual) vintages."""
    html = requests.get(f"{BASE}/data/{pk}/fileData.do", headers=HEADERS,
                        timeout=60).text
    return sorted(set(re.findall(r"uddi:[a-z0-9-]{36}", html)))


def file_ids(pk: str, uddi: str) -> tuple[str, str, str]:
    j = requests.get(
        f"{BASE}/tcs/dss/selectFileDataDownload.do",
        params={"publicDataPk": pk, "publicDataDetailPk": uddi},
        headers=HEADERS, timeout=60,
    ).json()
    name = j["dataSetFileDetailInfo"]["dataNm"]
    return j["atchFileId"], str(j.get("fileDetailSn", 1)), name


def download(pk: str, tag: str, uddi: str | None = None) -> list[str]:
    uddi = uddi or main_uddi(pk)
    atch, sn, name = file_ids(pk, uddi)
    resp = requests.get(
        f"{BASE}/cmm/cmm/fileDownload.do",
        params={"atchFileId": atch, "fileDetailSn": sn},
        headers=HEADERS, timeout=300,
    )
    resp.raise_for_status()
    os.makedirs(RAW_DIR, exist_ok=True)
    out_dir = os.path.join(RAW_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    content = resp.content
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for zi in zf.infolist():
                # zip member names are cp437-mangled cp949
                try:
                    fname = zi.filename.encode("cp437").decode("cp949")
                except (UnicodeDecodeError, UnicodeEncodeError):
                    fname = zi.filename
                target = os.path.join(out_dir, os.path.basename(fname))
                with open(target, "wb") as f:
                    f.write(zf.read(zi))
                saved.append(target)
    else:
        safe = re.sub(r"[^\w가-힣.-]+", "_", name)[:120]
        target = os.path.join(out_dir, f"{safe}.csv")
        with open(target, "wb") as f:
            f.write(content)
        saved.append(target)
    meta = {"publicDataPk": pk, "uddi": uddi, "atchFileId": atch,
            "dataNm": name, "files": [os.path.basename(s) for s in saved]}
    with open(os.path.join(out_dir, f"meta_{uddi.split(':')[1][:8]}.json"),
              "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return saved


def download_all_vintages(pk: str, tag: str) -> None:
    for uddi in all_uddis(pk):
        try:
            saved = download(pk, tag, uddi=uddi)
            print(f"{tag} {uddi[:13]}...: {len(saved)} file(s): "
                  f"{[os.path.basename(s) for s in saved]}")
        except Exception as exc:
            print(f"{tag} {uddi[:13]}...: skip ({str(exc)[:60]})")


def main():
    for tag, pk in DATASETS.items():
        try:
            download_all_vintages(pk, tag)
        except Exception as exc:
            print(f"{tag} ({pk}): FAILED — {exc}")


if __name__ == "__main__":
    main()
