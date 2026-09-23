from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd

OFFICIAL_NATIONAL_CANDIDATE_COUNT: int = 1070000

RAW_COLUMN_MAP: Dict[str, str] = {
    "dm1": "toan",
    "dm2": "ngu_van",
    "dm3": "ngoai_ngu",
    "dm4": "lich_su",
    "dm5": "dia_li",
    "dm6": "gdkt_pl",
    "dm7": "vat_li",
    "dm8": "hoa_hoc",
    "dm9": "sinh_hoc",
    "dm10": "tin_hoc",
    "dm11": "gdcd",
    "dm12": "cn_cong_nghiep",
    "dm13": "cn_nong_nghiep",
}

SUBJECT_DISPLAY_NAMES: Dict[str, str] = {
    "toan": "Toán",
    "ngu_van": "Ngữ văn",
    "ngoai_ngu": "Ngoại ngữ",
    "lich_su": "Lịch sử",
    "dia_li": "Địa lí",
    "gdkt_pl": "GDKT & PL",
    "vat_li": "Vật lí",
    "hoa_hoc": "Hóa học",
    "sinh_hoc": "Sinh học",
    "tin_hoc": "Tin học",
    "gdcd": "Giáo dục công dân",
    "cn_cong_nghiep": "CN Công nghiệp",
    "cn_nong_nghiep": "CN Nông nghiệp",
}

PROVINCE_MAP: Dict[str, str] = {
    "01": "Thành phố Hà Nội",
    "02": "Thành phố Hồ Chí Minh",
    "03": "Thành phố Hải Phòng",
    "04": "Tỉnh Cao Bằng",
    "05": "Tỉnh Lạng Sơn",
    "06": "Tỉnh Quảng Ninh",
    "07": "Tỉnh Bắc Giang",
    "08": "Tỉnh Phú Thọ",
    "09": "Tỉnh Vĩnh Phúc",
    "10": "Tỉnh Bắc Ninh",
    "11": "Tỉnh Điện Biên",
    "12": "Tỉnh Lai Châu",
    "13": "Tỉnh Sơn La",
    "14": "Tỉnh Yên Bái",
    "15": "Tỉnh Hòa Bình",
    "16": "Tỉnh Thái Nguyên",
    "17": "Tỉnh Tuyên Quang",
    "18": "Tỉnh Hà Giang",
    "19": "Tỉnh Lào Cai",
    "20": "Tỉnh Bắc Kạn",
    "21": "Tỉnh Hải Dương",
    "22": "Tỉnh Hưng Yên",
    "23": "Tỉnh Nam Định",
    "24": "Tỉnh Thái Bình",
    "25": "Tỉnh Ninh Bình",
    "26": "Tỉnh Hà Nam",
    "27": "Tỉnh Thanh Hóa",
    "28": "Tỉnh Nghệ An",
    "29": "Tỉnh Hà Tĩnh",
    "30": "Tỉnh Quảng Bình",
    "31": "Tỉnh Quảng Trị",
    "32": "Thừa Thiên Huế",
    "33": "Thành phố Đà Nẵng",
    "34": "Tỉnh Quảng Nam",
    "35": "Tỉnh Quảng Ngãi",
    "36": "Tỉnh Bình Định",
    "37": "Tỉnh Phú Yên",
    "38": "Tỉnh Khánh Hòa",
    "39": "Tỉnh Ninh Thuận",
    "40": "Tỉnh Bình Thuận",
    "41": "Tỉnh Kon Tum",
    "42": "Tỉnh Gia Lai",
    "43": "Tỉnh Đắk Lắk",
    "44": "Tỉnh Đắk Nông",
    "45": "Tỉnh Lâm Đồng",
    "46": "Tỉnh Bình Phước",
    "47": "Tỉnh Tây Ninh",
    "48": "Tỉnh Bình Dương",
    "49": "Tỉnh Đồng Nai",
    "50": "Tỉnh Bà Rịa - Vũng Tàu",
    "51": "Tỉnh Long An",
    "52": "Tỉnh Tiền Giang",
    "53": "Tỉnh Bến Tre",
    "54": "Tỉnh Trà Vinh",
    "55": "Tỉnh Vĩnh Long",
    "56": "Tỉnh Đồng Tháp",
    "57": "Tỉnh An Giang",
    "58": "Tỉnh Kiên Giang",
    "59": "Thành phố Cần Thơ",
    "60": "Tỉnh Hậu Giang",
    "61": "Tỉnh Sóc Trăng",
    "62": "Tỉnh Bạc Liêu",
    "63": "Tỉnh Cà Mau",
    "66": "Tỉnh Đắk Lắk",
    "68": "Tỉnh Lâm Đồng",
    "75": "Tỉnh Đồng Nai",
    "79": "Thành phố Hồ Chí Minh",
    "80": "Tỉnh Tây Ninh",
    "82": "Tỉnh Đồng Tháp",
    "86": "Tỉnh Vĩnh Long",
    "91": "Tỉnh An Giang",
    "92": "Thành phố Cần Thơ",
    "96": "Tỉnh Cà Mau",
}

NORTHERN_CODES = {
    "01", "03", "04", "05", "06", "07", "08", "09", "10", "11",
    "12", "13", "14", "15", "16", "17", "18", "19", "20", "21",
    "22", "23", "24", "25", "26"
}

CENTRAL_CODES = {
    "27", "28", "29", "30", "31", "32", "33", "34", "35", "36",
    "37", "38", "39", "40", "41", "42", "43", "44", "45", "66", "68"
}


def get_province_mapping() -> Dict[str, str]:
    return PROVINCE_MAP.copy()


def get_region_name(code: str) -> str:
    if code in NORTHERN_CODES:
        return "Miền Bắc"
    elif code in CENTRAL_CODES:
        return "Miền Trung"
    else:
        return "Miền Nam"


def clean_thpt_2026_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df.empty:
        quality_report = {
            "total_records": 0,
            "duplicate_records": 0,
            "cleaned_records": 0,
            "official_candidates": OFFICIAL_NATIONAL_CANDIDATE_COUNT,
            "coverage_pct": 0.0,
            "coverage_label": "Partial public-data coverage",
            "missing_rates": {},
        }
        return df, quality_report

    clean_df = df.copy()
    clean_df.rename(columns=RAW_COLUMN_MAP, inplace=True)

    if "sbd" in clean_df.columns:
        clean_df["sbd"] = clean_df["sbd"].astype(str).str.zfill(8)
        clean_df["province_code"] = clean_df["sbd"].str[:2]
        clean_df["province_name"] = (
            clean_df["province_code"].map(PROVINCE_MAP).fillna("Unknown")
        )
        clean_df["region"] = clean_df["province_code"].apply(get_region_name)
    elif "province_code" in clean_df.columns:
        clean_df["province_code"] = (
            clean_df["province_code"].astype(str).str.zfill(2)
        )
        clean_df["province_name"] = (
            clean_df["province_code"].map(PROVINCE_MAP).fillna("Unknown")
        )
        clean_df["region"] = clean_df["province_code"].apply(get_region_name)

    initial_count = len(clean_df)
    if "sbd" in clean_df.columns:
        clean_df.drop_duplicates(subset=["sbd"], keep="first", inplace=True)
    duplicate_count = initial_count - len(clean_df)

    subject_cols = [c for c in RAW_COLUMN_MAP.values() if c in clean_df.columns]
    missing_rates = {}

    for col in subject_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
        clean_df.loc[(clean_df[col] < 0.0) | (clean_df[col] > 10.0), col] = (
            np.nan
        )
        missing_count = int(clean_df[col].isna().sum())
        missing_rates[col] = {
            "subject_name": SUBJECT_DISPLAY_NAMES.get(col, col),
            "valid_candidates": int(clean_df[col].notna().sum()),
            "missing_count": missing_count,
            "missing_pct": round(missing_count / len(clean_df) * 100, 2),
        }

    cleaned_count = len(clean_df)
    coverage_pct = round(
        (cleaned_count / OFFICIAL_NATIONAL_CANDIDATE_COUNT) * 100, 2
    )
    coverage_label = (
        "Nationwide Full Coverage"
        if coverage_pct >= 95.0
        else "Partial public-data coverage"
    )

    quality_report = {
        "total_records": initial_count,
        "duplicate_records": duplicate_count,
        "cleaned_records": cleaned_count,
        "official_candidates": OFFICIAL_NATIONAL_CANDIDATE_COUNT,
        "coverage_pct": coverage_pct,
        "coverage_label": coverage_label,
        "missing_rates": missing_rates,
    }

    return clean_df, quality_report
