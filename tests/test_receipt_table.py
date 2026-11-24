from datamate.loader.mapping import ColumnMap, _resolve_context
from datamate.loader.build import table_role


def test_receipt_table_no_receipt_id(monkeypatch):
    monkeypatch.setenv("DATAMATE_NV3", "1")   # NV3 is off by default
    # A table named phieuthu with 3 columns (Date order_date, Order code order_id, Amount order_total)
    # -> payment_date, order_id, payment_amount, role payments.
    maps = [
        ColumnMap("phieuthu", "Ngày", "order_date", "name_dict", "Ngày"),
        ColumnMap("phieuthu", "Mã đơn", "order_id", "name_dict", "Mã đơn"),
        ColumnMap("phieuthu", "Số tiền", "order_total", "name_dict", "Số tiền"),
    ]
    _resolve_context(maps)
    res = {m.column: m.canonical for m in maps}
    assert res["Ngày"] == "payment_date"
    assert res["Mã đơn"] == "order_id"
    assert res["Số tiền"] == "payment_amount"

    canons = {m.canonical for m in maps}
    role = table_role(canons)
    assert role == "payments"


def test_receipt_table_don_hang_untouched():
    # A table named don_hang with the same 3 columns -> unchanged
    maps = [
        ColumnMap("don_hang", "Ngày", "order_date", "name_dict", "Ngày"),
        ColumnMap("don_hang", "Mã đơn", "order_id", "name_dict", "Mã đơn"),
        ColumnMap("don_hang", "Số tiền", "order_total", "name_dict", "Số tiền"),
    ]
    _resolve_context(maps)
    res = {m.column: m.canonical for m in maps}
    assert res["Ngày"] == "order_date"
    assert res["Mã đơn"] == "order_id"
    assert res["Số tiền"] == "order_total"
    role = table_role({m.canonical for m in maps})
    assert role == "orders"


def test_receipt_table_with_product_untouched():
    # A table named phieuthu with an extra product_name -> unchanged
    maps = [
        ColumnMap("phieuthu", "Ngày", "order_date", "name_dict", "Ngày"),
        ColumnMap("phieuthu", "Mã đơn", "order_id", "name_dict", "Mã đơn"),
        ColumnMap("phieuthu", "Số tiền", "order_total", "name_dict", "Số tiền"),
        ColumnMap("phieuthu", "Sản phẩm", "product_name", "name_dict", "Sản phẩm"),
    ]
    _resolve_context(maps)
    res = {m.column: m.canonical for m in maps}
    assert res["Ngày"] == "order_date"
    assert res["Mã đơn"] == "order_id"
    assert res["Số tiền"] == "order_total"
