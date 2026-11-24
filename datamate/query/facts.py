"""The canonical tables for the query side: the same table names and column names at every shop, with the rulebook's general rules attached.
Temporary tables carry the prefix _f_: a bare name could coincide with a real shop table (a shop may have its own phieu_nhap table) and hide it.

Built from the SQL verified against training answers (tests/verify_answers.py), creating temporary tables when needed, not writing to the shop's DB.
Plans (query/plan.py) work only with these tables, so the LLM does not have to remember the cancel / return rules, row types,
unit conversion... for each question. A shop without enough data for a table: NoSource (the caller leaves blank, no guessing).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from datamate.tests.verify_answers import (OK_ORDER, _cols, _tables, base_unit_of, debt_sql, deliveries_sql, flow_sql,
                                           group_maps, lots_sql, orders_sql, product_names, purchase_lines_sql,
                                           purchases_sql, rx_ids, sales_sql, stock_at, text_key)


class NoSource(Exception):
    """The shop has no data to build the needed table / column."""


# table name -> {column: meaning}; the meaning is written for the LLM to read. For money columns see MONEY.
FIELDS = {
    "dong_ban": {
        "don": "mã đơn", "ngay": "ngày đặt", "gio": "giờ tạo đơn (0-23)", "ca": "'sang' (6-13h) / 'chieu' (14-21h)",
        "thu": "thứ trong tuần (1 = thứ Hai ... 7 = Chủ nhật)", "thang": "tháng 'YYYY-MM'",
        "trang_thai": "'hoan_thanh' / 'huy' / 'tra_hang' / 'dang_giao'",
        "tien": "thành tiền của dòng (doanh thu)", "la_mat_hang": "TRUE nếu dòng là mặt hàng (không phải phí phục vụ...)",
        "hang": "mặt hàng", "ten_hang": "tên hàng", "sl_co_ban": "số lượng quy về đơn vị cơ bản",
        "sl": "số lượng theo đơn vị ghi trên dòng", "gia_von": "giá vốn một đơn vị bán", "don_gia": "đơn giá ghi trên dòng",
        "lai": "lãi gộp của dòng", "gia_1_dv": "đơn giá cho một đơn vị cơ bản, trước chiết khấu dòng",
        "khach": "khách (không gồm khách lẻ)", "khach_le": "TRUE nếu khách lẻ",
        "kenh": "kênh bán: tai_quay, dien_thoai, zalo, facebook, shopee, lazada, tiktok, grab, website",
        "thanh_toan": "hình thức: tien_mat, chuyen_khoan, the, vi, cod, ghi_no",
        "nhan_vien": "nhân viên bán", "nhom_hang": "nhóm hàng", "nhom_khach": "nhóm khách",
        "don_vi_co_ban": "đơn vị cơ bản của hàng", "ke_don": "TRUE nếu thuốc kê đơn", "hoat_chat": "hoạt chất của thuốc",
        "giam_dong": "tiền giảm giá ghi trên dòng (đã trừ trong tien)",
    },
    "don": {
        "don": "mã đơn", "ngay": "ngày đặt", "thang": "tháng 'YYYY-MM'", "thu": "thứ (1 = thứ Hai ... 7 = Chủ nhật)",
        "trang_thai": "'hoan_thanh' / 'huy' / 'tra_hang' / 'dang_giao'", "khach": "khách (không gồm khách lẻ)",
        "doanh_thu": "doanh thu của đơn", "nhan_vien": "nhân viên", "kenh": "kênh bán", "thanh_toan": "hình thức thanh toán",
        "giam_don": "tiền giảm giá cả đơn (voucher, chiết khấu đơn; không trừ vào doanh_thu)",
    },
    "phieu_nhap": {"phieu": "mã phiếu", "ngay": "ngày phiếu", "tong_tien": "tổng tiền phiếu nhập",
                   "nha_cung_cap": "nhà cung cấp"},
    "dong_nhap": {"hang": "mặt hàng", "ngay": "ngày phiếu", "sl": "số lượng theo đơn vị nhập", "tien": "tiền của dòng nhập"},
    "dot_giao": {"ngay": "ngày giao", "don": "mã đơn", "dot": "đợt thứ mấy của đơn", "tai_xe": "tài xế", "bien_so": "biển số xe"},
    "khoan_chi": {"ngay": "ngày chi", "tien": "số tiền chi", "khoan_muc": "khoản mục chi", "ghi_chu": "ghi chú"},
    "cong_no": {"don": "mã đơn", "ngay": "ngày đặt", "con_no": "số còn nợ của đơn tại ngày tính (> 0)",
                "tuoi_no": "số ngày từ ngày đặt đến ngày tính", "khach": "khách (không gồm khách lẻ)"},
    "dong_tien_no": {"khach": "khách", "no_them": "phần chưa trả ngay của đơn đặt trong kỳ",
                     "tra_no": "khoản trả sau ngày mua, có ngày trả trong kỳ"},
    "lo_hang": {"hang": "mặt hàng", "so_lo": "số lô", "han_dung": "hạn dùng", "con_ton": "còn tồn tại ngày tính (đv cơ bản)",
                "gia_tri_con": "giá trị phần còn tồn theo giá nhập của lô"},
    "ton": {"hang": "mặt hàng", "ton": "tồn hết ngày tính (đv cơ bản)", "gia_tri": "tồn x giá vốn trong danh mục"},
}
MONEY = {"tien", "lai", "don_gia", "gia_1_dv", "gia_von", "doanh_thu", "tong_tien", "con_no", "no_them", "tra_no",
         "gia_tri_con", "gia_tri", "giam_dong", "giam_don"}
# tables not by period but by "state at the end of day X" (§2) or that take a whole range as parameters
AT_DATE = {"cong_no", "lo_hang", "ton"}
BY_RANGE = {"dong_tien_no"}
# tables with the notion of a valid order (§1: not cancelled, not returned)
HAS_VALID = {"dong_ban", "don"}


def _frame(con, name: str, df: pd.DataFrame, types: dict | None = None) -> str:
    """DataFrame -> temporary table. types: cast columns (a fully empty column left to DuckDB's guess becomes an integer)."""
    con.register("_f", df)
    cols = ", ".join(f'CAST("{c}" AS {(types or {}).get(c, "VARCHAR")}) AS "{c}"' for c in df.columns)
    con.execute(f'CREATE OR REPLACE TEMP TABLE "{name}" AS SELECT {cols} FROM _f')
    con.unregister("_f")
    return name


# name key in SQL, the same operation as loader.build.text_key: lowercase, đ -> d, strip accents, characters that are not letters / digits become whitespace
# , collapse whitespace
_NAME_KEY = ("NULLIF(trim(regexp_replace(strip_accents(replace(lower(CAST({x} AS VARCHAR)), 'đ', 'd')), "
             "'[^a-z0-9_]+', ' ', 'g')), '')")


class Facts:
    """The canonical tables of ONE shop (one Facts per connection)."""

    def __init__(self, con):
        self.con = con
        self._made: dict[str, str] = {}

    # lookup tables
    def _dim_hang(self) -> str:
        if "dim_hang" in self._made:
            return "_f_dim_hang"
        rows: dict[str, dict] = {}

        def put(k, col, v):
            if k is not None and v is not None:
                rows.setdefault(str(k), {"k": str(k)}).setdefault(col, v)
        for g, keys in group_maps(self.con, False).items():
            for k in keys:
                put(k, "nhom_hang", g)
        for k, u in base_unit_of(self.con).items():
            put(k, "don_vi_co_ban", u)
        ids = rx_ids(self.con)
        names = product_names(self.con)
        if ids is not None:
            for p, nm in names.items():
                put(p, "ke_don", p in ids)
                put(text_key(nm), "ke_don", p in ids)
        for p, nm in names.items():
            put(p, "ten", nm)
        for t in _tables(self.con, "drug_info") + _tables(self.con, "products"):
            c = _cols(self.con, t)
            if "active_ingredient" in c:
                for idc in ("product_id", "product_key"):
                    if idc in c:
                        for k, v in self.con.execute(f'SELECT DISTINCT CAST("{idc}" AS VARCHAR), active_ingredient '
                                                     f'FROM "{t}"').fetchall():
                            put(k, "hoat_chat", v)
                            if k in names:      # sales rows that record only the product name: look up by name key
                                put(text_key(names[k]), "hoat_chat", v)
        cols = ["k", "nhom_hang", "don_vi_co_ban", "ke_don", "hoat_chat", "ten"]
        df = pd.DataFrame([{c: r.get(c) for c in cols} for r in rows.values()], columns=cols).astype(object)
        self._made["dim_hang"] = _frame(self.con, "_f_dim_hang", df, {"ke_don": "BOOLEAN"})
        return "_f_dim_hang"

    def _dim_khach(self) -> str:
        if "dim_khach" in self._made:
            return "_f_dim_khach"
        rows = [{"k": k, "nhom_khach": g} for g, keys in group_maps(self.con, True).items() for k in keys]
        df = pd.DataFrame(rows, columns=["k", "nhom_khach"]).drop_duplicates("k").astype(object)
        self._made["dim_khach"] = _frame(self.con, "_f_dim_khach", df)
        return "_f_dim_khach"

    def code_labels(self) -> dict:
        """The canonical code of a channel / payment method ('tai_quay', 'chuyen_khoan') -> the most common spelling in the ledger
        ('Tại cửa hàng'): a submitted answer gives the name as written in the file (§9), the code is only for matching words in the question."""
        if "labels" not in self._made:
            out = {}
            s = sales_sql(self.con)
            if s is not None:
                for code, raw in (("ch", "chraw"), ("pm", "pmraw")):
                    for c, r in self.con.execute(
                            f"SELECT {code}, {raw} FROM (SELECT {code}, {raw}, row_number() OVER (PARTITION BY {code} "
                            f"ORDER BY COUNT(*) DESC, {raw}) rn FROM ({s}) WHERE {code} IS NOT NULL AND {raw} IS NOT NULL "
                            f"GROUP BY 1, 2) WHERE rn = 1").fetchall():
                        out.setdefault(str(c), str(r))
            self._made["labels"] = out
        return self._made["labels"]

    # tables by period
    def dong_ban(self) -> str:
        s = sales_sql(self.con)
        if s is None:
            raise NoSource("không có dòng bán")
        have = {r[0] for r in self.con.execute(f"DESCRIBE {s}").fetchall()}
        lt = "s.lt" if "lt" in have else "'hang'"
        disc = "CAST(s.disc AS DOUBLE)" if "disc" in have else "NULL::DOUBLE"
        # all times are 00:00 means the shop records only dates: no time, no shift (not every order is created at midnight)
        timed = self.con.execute(f"SELECT COUNT(*) FROM ({s}) WHERE hour(CAST(t AS TIMESTAMP)) <> 0 "
                                 f"OR minute(CAST(t AS TIMESTAMP)) <> 0").fetchone()[0]
        hour = "hour(CAST(s.t AS TIMESTAMP))" if timed else "NULL::INTEGER"
        dh, dk = self._dim_hang(), self._dim_khach()
        # staff recorded by name only ("Lan", "LAN", "lan "): for each name key (accents stripped, lowercase, punctuation as in text_key) take the most
        # common spelling as the representative name. Grouping by the raw name would turn one person into several; keep the spelling in the ledger (not
        # the accent-stripped key) so the answer keeps its accents and a filter "nhan_vien = 'Lan'" still matches
        key = _NAME_KEY.format(x="staff")
        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _f_dim_nv AS
            SELECT k, staff AS rep FROM (
                SELECT {key} AS k, staff, row_number() OVER (PARTITION BY {key} ORDER BY COUNT(*) DESC, staff) AS rn
                FROM ({s}) WHERE staff IS NOT NULL GROUP BY 1, 2) WHERE rn = 1 AND k IS NOT NULL""")
        self.con.execute(f"""
            CREATE OR REPLACE TEMP TABLE _f_dong_ban AS
            SELECT COALESCE(CAST(s.order_id AS VARCHAR), 'dong' || row_number() OVER ()) AS don,
                   s.d AS ngay, {hour} AS gio,
                   CASE WHEN {hour} BETWEEN 6 AND 13 THEN 'sang' WHEN {hour} BETWEEN 14 AND 21 THEN 'chieu' END AS ca,
                   isodow(s.d) AS thu, strftime(s.d, '%Y-%m') AS thang,
                   CAST(s.st AS VARCHAR) AS trang_thai, {OK_ORDER.replace('st ', 's.st ')} AS hop_le,
                   s.amt AS tien, ({lt}) = 'hang' AS la_mat_hang,
                   CAST(s.pid AS VARCHAR) AS ma_hang, CAST(s.pkey AS VARCHAR) AS khoa_hang,
                   COALESCE(CAST(s.pid AS VARCHAR), CAST(s.pkey AS VARCHAR)) AS hang,
                   COALESCE(s.pname, a.ten, b.ten) AS ten_hang,
                   CAST(s.qb AS DOUBLE) AS sl_co_ban, CAST(s.qty AS DOUBLE) AS sl, CAST(s.uc AS DOUBLE) AS gia_von,
                   CAST(s.up AS DOUBLE) AS don_gia, s.amt - CAST(s.qty AS DOUBLE) * CAST(s.uc AS DOUBLE) AS lai,
                   CAST(s.up AS DOUBLE) * COALESCE(CAST(s.qty AS DOUBLE) / NULLIF(CAST(s.qb AS DOUBLE), 0), 1) AS gia_1_dv,
                   CASE WHEN COALESCE(s.walk, FALSE) THEN NULL
                        ELSE COALESCE(CAST(s.cid AS VARCHAR), CAST(s.ckey AS VARCHAR)) END AS khach,
                   s.cname AS ten_khach, COALESCE(s.walk, FALSE) AS khach_le,
                   s.ch AS kenh, s.pm AS thanh_toan, COALESCE(nv.rep, s.staff) AS nhan_vien,
                   COALESCE(a.nhom_hang, b.nhom_hang) AS nhom_hang,
                   COALESCE(a.don_vi_co_ban, b.don_vi_co_ban) AS don_vi_co_ban,
                   COALESCE(a.ke_don, b.ke_don) AS ke_don, COALESCE(a.hoat_chat, b.hoat_chat) AS hoat_chat,
                   CASE WHEN COALESCE(s.walk, FALSE) THEN NULL ELSE COALESCE(c.nhom_khach, e.nhom_khach) END AS nhom_khach,
                   {disc} AS giam_dong
            FROM ({s}) s
            LEFT JOIN "{dh}" a ON a.k = CAST(s.pid AS VARCHAR) LEFT JOIN "{dh}" b ON b.k = CAST(s.pkey AS VARCHAR)
            LEFT JOIN "{dk}" c ON c.k = CAST(s.cid AS VARCHAR) LEFT JOIN "{dk}" e ON e.k = CAST(s.ckey AS VARCHAR)
            LEFT JOIN _f_dim_nv nv ON nv.k = {_NAME_KEY.format(x="s.staff")}""")
        return "_f_dong_ban"

    def don(self) -> str:
        s = orders_sql(self.con)
        lines = self.table("dong_ban") if sales_sql(self.con) else None
        if s is None and lines is None:
            raise NoSource("không có đơn")
        agg = ("SELECT don, SUM(tien) doanh_thu, ANY_VALUE(khach) khach, ANY_VALUE(ten_khach) ten_khach, "
               "ANY_VALUE(nhan_vien) nhan_vien, "
               "ANY_VALUE(kenh) kenh, ANY_VALUE(thanh_toan) thanh_toan, MIN(ngay) ngay, MIN(trang_thai) trang_thai "
               "FROM _f_dong_ban GROUP BY don") if lines else None
        if s is None:
            base = f"SELECT don, ngay, trang_thai, doanh_thu, khach, ten_khach, nhan_vien, kenh, thanh_toan FROM ({agg})"
        elif agg is None:
            base = ("SELECT CAST(order_id AS VARCHAR) don, d ngay, st trang_thai, NULL doanh_thu, NULL khach, NULL ten_khach, "
                    f"NULL nhan_vien, NULL kenh, NULL thanh_toan FROM ({s})")
        else:
            base = (f"SELECT CAST(o.order_id AS VARCHAR) don, o.d ngay, o.st trang_thai, a.doanh_thu, a.khach, a.ten_khach, "
                    f"a.nhan_vien, a.kenh, a.thanh_toan FROM ({s}) o LEFT JOIN ({agg}) a "
                    f"ON CAST(o.order_id AS VARCHAR) = a.don")
        # order discount in the order table; if absent, in the sales-row table (order info repeated on every row) or in a separate order-info table
        disc = next((f'SELECT CAST(order_id AS VARCHAR) k, ANY_VALUE(order_discount) g FROM "{t}" GROUP BY 1'
                     for role in ("orders", "sales_lines", "order_info", "other") for t in _tables(self.con, role)
                     if {"order_id", "order_discount"} <= _cols(self.con, t)), None)
        self.con.execute(f"""CREATE OR REPLACE TEMP TABLE _f_don AS SELECT b.*, isodow(b.ngay) thu,
                             strftime(b.ngay, '%Y-%m') thang, {"CAST(g.g AS DOUBLE)" if disc else "NULL::DOUBLE"} giam_don,
                             (b.trang_thai IS NULL OR b.trang_thai NOT IN ('huy', 'tra_hang')) hop_le
                             FROM ({base}) b{f" LEFT JOIN ({disc}) g ON g.k = b.don" if disc else ""}""")
        return "_f_don"

    def phieu_nhap(self) -> str:
        s = purchases_sql(self.con)
        if s is None:
            raise NoSource("không có phiếu nhập")
        sup = next((f'SELECT CAST(purchase_id AS VARCHAR) k, ANY_VALUE(supplier_name) n FROM "{t}" GROUP BY 1'
                    for role in ("purchases", "purchase_lines") for t in _tables(self.con, role)
                    if {"purchase_id", "supplier_name"} <= _cols(self.con, t)), None)
        # purchases_sql returns d, total, cancelled; the supplier is joined separately when present
        self.con.execute(f"""CREATE OR REPLACE TEMP TABLE _f_phieu_nhap AS
            SELECT d ngay, total tong_tien, NULL::VARCHAR nha_cung_cap, NULL::VARCHAR phieu
            FROM ({s}) WHERE NOT COALESCE(cancelled, FALSE)""")
        if sup:
            for P in _tables(self.con, "purchases"):
                c = _cols(self.con, P)
                if {"purchase_id", "purchase_total", "purchase_date"} <= c:
                    cx = "COALESCE(purchase_cancelled, FALSE)" if "purchase_cancelled" in c else "FALSE"
                    self.con.execute(f"""CREATE OR REPLACE TEMP TABLE _f_phieu_nhap AS
                        SELECT CAST(p.d AS DATE) ngay, p.t tong_tien, s.n nha_cung_cap, p.k phieu FROM
                        (SELECT CAST(purchase_id AS VARCHAR) k, MIN(purchase_date) d, ANY_VALUE(purchase_total) t,
                                BOOL_OR({cx}) x FROM "{P}" GROUP BY 1) p LEFT JOIN ({sup}) s USING (k) WHERE NOT p.x""")
                    break
            else:       # a purchase ledger with product rows only: the slip and supplier are written on the row itself
                for L in _tables(self.con, "purchase_lines"):
                    c = _cols(self.con, L)
                    amt = "line_amount" if "line_amount" in c else "qty * unit_price" if {"qty", "unit_price"} <= c else None
                    if amt and {"purchase_id", "purchase_date", "supplier_name"} <= c:
                        cx = "BOOL_OR(purchase_cancelled)" if "purchase_cancelled" in c else "FALSE"
                        self.con.execute(f"""CREATE OR REPLACE TEMP TABLE _f_phieu_nhap AS
                            SELECT CAST(MIN(purchase_date) AS DATE) ngay, SUM({amt}) tong_tien, ANY_VALUE(supplier_name) nha_cung_cap,
                                   CAST(purchase_id AS VARCHAR) phieu
                            FROM "{L}" GROUP BY purchase_id HAVING NOT {cx}""")
                        break
        return "_f_phieu_nhap"

    def dong_nhap(self) -> str:
        s = purchase_lines_sql(self.con)
        if s is None:
            raise NoSource("không có dòng nhập")
        self.con.execute(f"""CREATE OR REPLACE TEMP TABLE _f_dong_nhap AS
            SELECT CAST(pid AS VARCHAR) ma_hang, CAST(pkey AS VARCHAR) khoa_hang,
                   COALESCE(CAST(pid AS VARCHAR), CAST(pkey AS VARCHAR)) hang, qty sl, amt tien, d ngay
            FROM ({s}) WHERE NOT cancelled""")
        return "_f_dong_nhap"

    def dot_giao(self) -> str:
        """One row per delivery batch of each order as in deliveries_sql (§6), with the driver and licence plate when the ledger records them."""
        if deliveries_sql(self.con) is None:
            raise NoSource("không có sổ giao hàng")
        t = next(t for t in _tables(self.con, "deliveries") if "delivery_date" in _cols(self.con, t))
        c = _cols(self.con, t)
        key = [k for k in ("order_id", "delivery_round") if k in c] or ["delivery_date"]
        cols = ["CAST(MIN(delivery_date) AS DATE) ngay",
                "CAST(ANY_VALUE(order_id) AS VARCHAR) don" if "order_id" in c else "NULL::VARCHAR don",
                "TRY_CAST(ANY_VALUE(delivery_round) AS INTEGER) dot" if "delivery_round" in c else "NULL::INTEGER dot",
                "ANY_VALUE(driver) tai_xe" if "driver" in c else "NULL::VARCHAR tai_xe",
                "ANY_VALUE(plate) bien_so" if "plate" in c else "NULL::VARCHAR bien_so"]
        self.con.execute(f'CREATE OR REPLACE TEMP TABLE _f_dot_giao AS SELECT {", ".join(cols)} FROM "{t}" GROUP BY {", ".join(key)}')
        return "_f_dot_giao"

    def khoan_chi(self) -> str:
        E = next((e for e in _tables(self.con, "expenses") if {"expense_amount", "expense_date"} <= _cols(self.con, e)),
                 None)
        if E is None:
            raise NoSource("không có sổ chi phí")
        c = _cols(self.con, E)
        item = "expense_item" if "expense_item" in c else "NULL::VARCHAR"
        note = "note" if "note" in c else "NULL::VARCHAR"
        self.con.execute(f'CREATE OR REPLACE TEMP TABLE _f_khoan_chi AS SELECT CAST(expense_date AS DATE) ngay, '
                         f'expense_amount tien, {item} khoan_muc, {note} ghi_chu FROM "{E}"')
        return "_f_khoan_chi"

    # tables by as-of date / range
    def cong_no(self, x: dt.date) -> str:
        s = debt_sql(self.con)
        if s is None:
            raise NoSource("không có sổ thu / bảng đơn để tính nợ")
        name = f"_f_cong_no_{x:%Y%m%d}"
        self.con.execute(f"""CREATE OR REPLACE TEMP TABLE "{name}" AS
            SELECT CAST(order_id AS VARCHAR) don, d ngay, owe con_no, DATE '{x}' - d tuoi_no,
                   CASE WHEN walk THEN NULL ELSE cid END khach, cname ten_khach
            FROM ({s.replace('?', f"DATE '{x}'")}) WHERE owe > 0""")
        return name

    def dong_tien_no(self, a: dt.date, b: dt.date) -> str:
        s = flow_sql(self.con)
        if s is None:
            raise NoSource("không có sổ thu / bảng đơn để tính nợ")
        name = f"_f_dong_tien_no_{a:%Y%m%d}_{b:%Y%m%d}"
        filled = s
        for v in (a, b, a, b):
            filled = filled.replace("?", f"DATE '{v}'", 1)
        self.con.execute(f"""CREATE OR REPLACE TEMP TABLE "{name}" AS
            SELECT CASE WHEN walk THEN NULL ELSE cid END khach, cname ten_khach, add no_them, paid_later tra_no
            FROM ({filled})""")
        return name

    def lo_hang(self, x: dt.date) -> str:
        s = lots_sql(self.con)
        if s is None:
            raise NoSource("không có dữ liệu lô / hạn dùng")
        name = f"_f_lo_hang_{x:%Y%m%d}"
        self.con.execute(f"""CREATE OR REPLACE TEMP TABLE "{name}" AS
            SELECT pid ma_hang, NULL::VARCHAR khoa_hang, pid hang, batch_id so_lo, exp han_dung, remain con_ton,
                   remain * cost_base gia_tri_con FROM ({s.replace('?', f"DATE '{x}'")})""")
        return name

    def ton(self, x: dt.date) -> str:
        name = f"_f_ton_{x:%Y%m%d}"
        cost: dict[str, float] = {}
        items: dict[str, tuple] = {}          # code (key by name if no code) -> (code, key): one row per product
        by_name: dict[str, str] = {}          # one code may have several spellings of the name; a row that records only the name maps to that code
        rows_in = []
        for role in ("products", "stock"):
            for t in _tables(self.con, role):
                c = _cols(self.con, t)
                if "product_name" not in c:
                    continue
                idc = "CAST(product_id AS VARCHAR)" if "product_id" in c else "NULL"
                uc = "unit_cost" if "unit_cost" in c else "NULL"
                rows_in += self.con.execute(f'SELECT DISTINCT {idc}, product_name, {uc} FROM "{t}" '
                                            f'WHERE product_name IS NOT NULL').fetchall()
        for pid, nm, u in sorted(rows_in, key=lambda r: r[0] is None):
            k = text_key(nm)
            ident = pid or by_name.get(k) or k
            if pid:
                by_name.setdefault(k, pid)
            items.setdefault(ident, (pid, k))
            if u is not None:
                cost.setdefault(ident, u)
        rows = []
        for pid, k in items.values():
            q, _src = stock_at(self.con, pid, k, x)
            if q is not None:
                c = cost.get(pid or k)
                rows.append({"ma_hang": pid, "khoa_hang": k, "hang": pid or k, "ton": q,
                             "gia_tri": q * c if (c and q > 0) else None})
        if not rows:
            raise NoSource("không có nguồn tồn")
        return _frame(self.con, name, pd.DataFrame(rows), {"ton": "DOUBLE", "gia_tri": "DOUBLE"})

    # entry points
    def table(self, fact: str, at: dt.date | None = None, rng: tuple | None = None) -> str:
        """Temporary table name of `fact` (built when needed). at: the as-of date; rng: the range (a, b)."""
        key = f"{fact}|{at}|{rng}"
        if key in self._made:
            return self._made[key]
        if fact in AT_DATE:
            if at is None:
                raise NoSource(f"{fact} cần ngày tính")
            name = getattr(self, fact)(at)
        elif fact in BY_RANGE:
            if rng is None:
                raise NoSource(f"{fact} cần khoảng ngày")
            name = getattr(self, fact)(*rng)
        elif fact in FIELDS:
            name = getattr(self, fact)()
        else:
            raise NoSource(f"không có bảng {fact}")
        self._made[key] = name
        return name
