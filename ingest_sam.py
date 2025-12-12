# ingest_sam.py
import os
import time
import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SAM_API_KEY")
BASE_URL = "https://api.sam.gov/opportunities/v2/search"

OUT_FILE = Path("rfp.csv")
MAX_RECORDS_HARD_CAP = 200  # don't pull more than this per run


def _is_url(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    try:
        u = urlparse(text.strip())
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def mmddyyyy(iso_date: str) -> str:
    """
    Convert 'YYYY-MM-DD' -> 'MM/DD/YYYY' for SAM.
    """
    y, m, d = iso_date.split("-")
    return f"{m}/{d}/{y}"


def build_params(posted_from_iso: str,
                 posted_to_iso: str,
                 limit: int,
                 offset: int,
                 naics: str | None,
                 state: str | None) -> dict:
    params = {
        "api_key": API_KEY,
        "postedFrom": mmddyyyy(posted_from_iso),
        "postedTo": mmddyyyy(posted_to_iso),
        "limit": min(max(1, limit), 1000),
        "offset": max(0, offset),
        "sort": "-postedDate",
    }
    if naics:
        params["ncode"] = naics
    if state:
        params["state"] = state
    return params


def fetch_description_text(desc_url: str) -> str:
    """
    Fetch full description text from a SAM notice URL.

    Handles cases like:
      - JSON: {"description": "<p>Some HTML...</p>"}
      - Raw HTML

    Returns plain text with collapsed whitespace.
    """
    if not desc_url or not _is_url(desc_url):
        return ""
    params = {}
    if API_KEY:
        params["api_key"] = API_KEY
    try:
        resp = requests.get(desc_url, params=params, timeout=60)
        if resp.status_code != 200:
            return ""

        raw = resp.text or ""

        # Try JSON with 'description'
        desc = raw
        try:
            data = resp.json()
            if isinstance(data, dict) and "description" in data:
                desc = data.get("description", "") or ""
        except Exception:
            # Not JSON; keep raw
            pass

        # Strip simple HTML tags
        desc = re.sub(r"<[^>]+>", " ", desc)
        # Collapse whitespace
        desc = " ".join(desc.split())
        return desc.strip()
    except Exception:
        return ""


def flatten_notice(n: dict) -> dict:
    """
    Map a single SAM notice JSON into a flat row for CSV.
    """
    notice_id = n.get("noticeId") or n.get("id") or ""

    title = (n.get("title") or "").strip()
    summary_raw = (n.get("summary") or n.get("shortDescription") or "").strip()

    # description + additionalInfoLink are both possible URLs to long description
    raw_desc_field = (n.get("description") or "").strip()
    addl = (n.get("additionalInfoLink") or "").strip()

    desc_fetch_url = ""
    if _is_url(raw_desc_field):
        desc_fetch_url = raw_desc_field
    elif _is_url(addl):
        desc_fetch_url = addl

    # organization
    org_obj = n.get("organization") or {}
    if isinstance(org_obj, dict):
        org_name = (org_obj.get("name") or "").strip()
        org_code = (org_obj.get("code") or "").strip()
    else:
        org_name = (n.get("organizationName") or "").strip()
        org_code = (n.get("organizationCode") or "").strip()

    parent_path = (n.get("fullParentPathName") or "").strip()

    # type info
    t = n.get("type") or {}
    if isinstance(t, dict):
        base_type = (t.get("baseType") or "").strip()
        type_name = (t.get("name") or "").strip()
    else:
        base_type = (n.get("baseType") or "").strip()
        type_name = (n.get("type") or "").strip()

    posted = (n.get("postedDate") or n.get("publishDate") or "").strip()
    response = (
        n.get("responseDeadLine")
        or n.get("reponseDeadLine")
        or n.get("responseDate")
        or n.get("closeDate")
        or ""
    )
    response = str(response).strip()

    jurisdiction = (n.get("jurisdiction") or "").strip()

    # place of performance / state
    state_val = ""
    pop = n.get("placeOfPerformance") or {}
    if isinstance(pop, dict):
        st = pop.get("state") or pop.get("stateCode") or ""
        if isinstance(st, dict):
            state_val = (st.get("code") or st.get("name") or "").strip()
        else:
            state_val = str(st).strip()

    # NAICS
    naics_list: list[str] = []
    nc = n.get("naicsCode")
    if nc:
        naics_list.append(str(nc).strip())
    for x in (n.get("naics") or []):
        if isinstance(x, str):
            naics_list.append(x.strip())
        elif isinstance(x, dict):
            naics_list.append((x.get("code") or x.get("value") or "").strip())

    # PSC
    psc_list: list[str] = []
    cc = n.get("classificationCode")
    if isinstance(cc, str):
        psc_list.append(cc.strip())
    elif isinstance(cc, list):
        psc_list.extend(str(x).strip() for x in cc if x)

    set_aside = (n.get("setAside") or "").strip()

    # links
    source_url = (
        n.get("publicLink")
        or n.get("permalink")
        or n.get("uiLink")
        or n.get("additionalInfoLink")
        or ""
    )
    source_url = source_url.strip()

    ui_link = (n.get("uiLink") or "").strip()

    # Start description_text with summary if it's real text, not URL
    description_text = summary_raw if not _is_url(summary_raw) else ""

    # Pull full long-form description if we have a URL
    if desc_fetch_url:
        full_desc = fetch_description_text(desc_fetch_url)
        if full_desc:
            description_text = (description_text + " " + full_desc).strip()

    return {
        "id": notice_id,
        "source": "sam",
        "source_url": source_url,
        "ui_link": ui_link,
        "title": title,
        "summary": summary_raw if not _is_url(summary_raw) else "",
        "solicitation_number": (n.get("solicitationNumber") or "").strip(),
        "organization_name": org_name,
        "organization_code": org_code,
        "full_parent_path_name": parent_path,
        "base_type": base_type,
        "type": type_name,
        "posted_date": posted,
        "response_date": response,
        "jurisdiction": jurisdiction,
        "state": state_val,
        "naics": ";".join([x for x in naics_list if x]),
        "psc": ";".join([x for x in psc_list if x]),
        "set_aside": set_aside,
        "place_of_performance": state_val,
        "budget_low": "",
        "budget_high": "",
        "additional_info_link": addl,
        "resource_links": "",
        "url_pdf": "",
        "description_text": description_text,
    }


def fetch_to_csv(posted_from: str,
                 posted_to: str,
                 limit: int = 100,
                 max_records: int = MAX_RECORDS_HARD_CAP,
                 naics: str | None = None,
                 state: str | None = None,
                 sleep: float = 0.3) -> None:
    if not API_KEY:
        raise SystemExit(
            "ERROR: SAM_API_KEY not set. Put it in .env or environment."
        )

    if max_records > MAX_RECORDS_HARD_CAP:
        print(f"Clamping max_records from {max_records} to {MAX_RECORDS_HARD_CAP}.")
        max_records = MAX_RECORDS_HARD_CAP

    rows = []
    offset = 0
    total_records = None

    while True:
        params = build_params(posted_from, posted_to, limit, offset, naics, state)
        resp = requests.get(BASE_URL, params=params, timeout=60)

        if resp.status_code != 200:
            print("WARN: status", resp.status_code, resp.text[:400])
            break

        data = resp.json()
        if total_records is None:
            total_records = data.get("totalRecords") or 0
            print(f"totalRecords reported: {total_records}")

        items = data.get("opportunitiesData") or data.get("data") or []
        if not items:
            print("No items on this page.")
            break

        for n in items:
            rows.append(flatten_notice(n))
            if len(rows) >= max_records:
                break

        print(f"Fetched {len(items)} items; accumulated {len(rows)}; offset {offset}")

        if len(rows) >= max_records:
            break

        offset += limit
        if total_records is not None and offset >= total_records:
            break

        time.sleep(sleep)

    if not rows:
        print("No rows fetched.")
        return

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["id"])
    df.to_csv(OUT_FILE, index=False)
    print(f"Wrote {len(df)} rows to {OUT_FILE.resolve()}")


def parse_args():
    p = argparse.ArgumentParser(description="Fetch RFPs from SAM.gov into rfp.csv")
    p.add_argument("--posted-from", required=True, help="YYYY-MM-DD")
    p.add_argument("--posted-to", required=True, help="YYYY-MM-DD")
    p.add_argument("--limit", type=int, default=100, help="max records per page (<=1000)")
    p.add_argument("--max-records", type=int, default=MAX_RECORDS_HARD_CAP)
    p.add_argument("--naics", type=str, default=None, help="e.g. 541620")
    p.add_argument("--state", type=str, default=None, help="two-letter state code")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fetch_to_csv(
        posted_from=args.posted_from,
        posted_to=args.posted_to,
        limit=args.limit,
        max_records=args.max_records,
        naics=args.naics,
        state=args.state,
    )
