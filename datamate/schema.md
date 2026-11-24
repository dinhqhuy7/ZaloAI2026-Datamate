# Canonical column names and the DuckDB file (DataMate — the data-processing layer)

A version for the whole team to agree on. Source code: `loader/mapping.py` (`DESC_RULES`), `loader/build.py`, `loader/shopmeta.py`.

## How to use it (the query side reads this part first)

- **Each shop is a separate file.** The path is `solve data/<split>/<shop>/<shop>.duckdb`. A question with a `business_id` opens exactly that file; no table is shared across shops.
- **Canonical names are a shared vocabulary, not a shared table.**
  - Each shop has only the columns its data really has.
  - Industry-specific columns (`imei`, `active_ingredient`, `table_no`, `delivery_round`, `plate`…) exist only in shops of that industry.
  - A column that could not be mapped keeps its **original name**.
- **When a question arrives, read in this order:**
  1. `_meta_shop`: the last date that has data, the industry, the rulebook §6 section.
  2. `_meta_tables`: the tables and their roles.
  3. `_meta_columns`: the columns, original names, README descriptions.
  4. `_rulebook`: fetch the rule section needed (§1–§10 and the industry section).
- **With canonical names, a rulebook rule can be written as one SQL template for every shop.** For example revenue is the sum of `line_amount`, excluding `status_norm IN ('huy','tra_hang')`, excluding `line_type` rows that are not products. This template was checked on the 30 training shops (see the Verification section).

## Principles

- **Rename only columns that are certain.** When unsure keep the original name, because a wrong name is worse than an odd one.
- **Canonical names are shared by every table.** `qty` in a sales table and a purchase table both mean "quantity in the row's unit". Which table is what is given by the *table role*.
- **Normalised values.**
  - Money is a VND number (`"1tr440"` becomes `1440000`).
  - Dates are `TIMESTAMP`.
  - A column parsed from strings gets an extra column `<name>__raw` that keeps the original value.
- **No business filtering in the data layer.** Cancelled / returned orders, time periods, service fees… stay intact; columns are only added so the query side can filter by the rulebook.
- **Traceable.** Every row has `_src_file`, `_src_sheet`, `_src_row` (the row number in Excel/CSV).

## List of canonical names

> The tables below with the header `| Tên chuẩn | Ý nghĩa |` ("canonical name | meaning") are **read by the code**: `loader/mapping.canonical_vocab()` parses them and the
> meanings are inserted into the LLM prompt. They are therefore kept in Vietnamese, exactly as the LLM sees them. Headings: Mã = codes; Tên, phân loại = names, classification;
> Đơn vị, số lượng = units, quantity; Tiền = money; Ngày giờ = dates and times; Trạng thái, ghi chú = status, notes; Khác = other.

### Codes

| Tên chuẩn | Ý nghĩa |
|---|---|
| `order_id` | mã đơn / số hoá đơn |
| `line_no` | số thứ tự dòng trong đơn / phiếu nhập |
| `product_id` | mã hàng |
| `customer_id` | mã khách; trống là khách lẻ |
| `purchase_id` | mã phiếu nhập |
| `payment_id` | mã phiếu thu |
| `return_id` | mã phiếu trả hàng |
| `supplier_id` | mã nhà cung cấp |
| `staff_id` | mã nhân viên |
| `shift_id`, `promo_id` | mã ca, mã khuyến mãi |
| `batch_id` | số lô |
| `imei` | IMEI máy |

### Names, classification

| Tên chuẩn | Ý nghĩa |
|---|---|
| `product_name` | tên hàng (như ghi trong file) |
| `customer_name` | tên khách (có thể kèm SĐT như trong file) |
| `supplier_name` | tên nhà cung cấp |
| `staff_name` | tên nhân viên / thợ (như ghi trong file) |
| `name` | tên chung, khi bảng không cho biết là tên của gì |
| `category` | nhóm hàng |
| `customer_group` | nhóm khách |
| `channel` | kênh bán |
| `payment_method` | hình thức thanh toán |
| `active_ingredient`, `is_prescription` | hoạt chất, thuốc kê đơn |
| `expense_item` | khoản / mục chi (sổ chi phí) |
| `active` | còn kinh doanh |
| `vehicle`, `service_desc` | xe (dòng xe và biển số ghi chung), các việc đã làm (shop dịch vụ) |

### Units, quantity

| Tên chuẩn | Ý nghĩa |
|---|---|
| `unit` | đơn vị của dòng (bán hoặc nhập) |
| `base_unit` | đơn vị cơ bản |
| `big_unit` | đơn vị lớn |
| `conv_factor` | số đơn vị cơ bản trong 1 đơn vị lớn |
| `qty` | số lượng theo đơn vị của dòng |
| `qty_base` | số lượng quy ra đơn vị cơ bản |
| `qty_returned_base` | số lượng khách trả lại (đơn vị cơ bản) |
| `qty_returned` | số lượng trả lại (theo đơn vị của dòng) |
| `stock_qty` | tồn kho (đơn vị cơ bản) tại ngày cuối dữ liệu |
| `stock_after` | tồn sau mỗi lần nhập/bán/trả (thẻ kho) |
| `guests` | số khách của đơn (nhà hàng) |

### Money

| Tên chuẩn | Ý nghĩa |
|---|---|
| `unit_price` | đơn giá trên dòng (bán, hoặc giá nhập một đơn vị trong bảng nhập) |
| `line_amount` | thành tiền của dòng, đã trừ giảm giá dòng |
| `line_discount` | tiền giảm của dòng |
| `unit_cost` | giá vốn một đơn vị bán, ghi trên dòng bán |
| `price` | giá bán hiện tại (danh mục) |
| `new_price`, `old_price`, `price_from` | lịch sử giá |
| `order_subtotal` | tổng tiền hàng của đơn, trước giảm giá đơn |
| `order_discount` | giảm giá cả đơn |
| `order_total` | khách phải trả cho cả đơn / cả phiếu (phiếu dịch vụ: tiền cả phiếu) |
| `shipping_fee` | phí giao |
| `paid`, `debt` | khách đã trả, còn nợ |
| `payment_amount` | tiền một lần thu |
| `purchase_total` | tổng tiền cả phiếu nhập |
| `commission_pct` | % hoa hồng của nhân viên / thợ |
| `expense_amount` | tiền một khoản chi (sổ chi phí) |

Cột nào có ô ghi `%` thì có thêm cột `<cột>_is_pct`: ô đó là số phần trăm, không phải tiền.

### Dates and times

| Tên chuẩn | Ý nghĩa |
|---|---|
| `order_date` | ngày bán / ngày đặt / ngày làm phiếu dịch vụ |
| `order_time` | thời điểm đặt đơn (ngày giờ) |
| `purchase_date` | ngày nhập hàng |
| `payment_date` | thời điểm thu |
| `return_date` | ngày trả hàng |
| `expiry_date` | hạn sử dụng |
| `delivery_date` | ngày giao |
| `due_date` | hạn thanh toán |
| `created_date` | ngày tạo trên phần mềm |
| `start_date` | ngày vào làm (nhân viên) |
| `expense_date` | ngày chi (sổ chi phí) |

### Status, notes

| Tên chuẩn | Ý nghĩa |
|---|---|
| `status` | trạng thái đơn gốc (trạng thái thanh toán, hoá đơn điện tử, khách thì **không** map vào đây) |
| `purchase_status` | trạng thái phiếu nhập |
| `order_note` | ghi chú của đơn. Một số shop ghi đơn huỷ / trả / đang giao ở đây |
| `note` | ghi chú khác |

### Other

| Tên chuẩn | Ý nghĩa |
|---|---|
| `delivery_round`, `driver`, `plate` | đợt giao, tài xế, biển số xe |
| `credit_days` | số ngày được nợ |
| `table_no`, `area` | bàn, khu vực (nhà hàng) |
| `phone`, `address` | liên hệ |

## The DuckDB file

### Data tables

- **A table name** is the original name with accents stripped, lowercase, joined by `_`. The original name is in `_meta_tables.label`.
- **Tables split over several files/sheets with the same columns are merged** into one table, for example a sales ledger by month/year. The columns `_src_file`, `_src_sheet` tell where a row came from.
- **Empty cells in the later rows of an order are filled** (rulebook §8), not across file/sheet boundaries. Order-level columns (order id, date, customer, status, note, channel, order money) have the **same value on every row of the order**. So:
  - To compute per order take **one value per order** (`ANY_VALUE` or `DISTINCT ON (order_id)`).
  - Do **not** sum these columns across rows.
- **Duplicate rows of the same order are dropped** (rulebook §8). A table that is not recorded per order is left as is. Dropped rows are listed in `_meta_dropped`.
- **Total rows are dropped**, including a total row without the word "Tổng" (the last row of a sheet whose numbers all equal the column totals).

### Columns added by build

| Column | Present in | Meaning |
|---|---|---|
| `status_norm` | `sales_lines`, `orders` | the **final status** of the order, one value for the whole order, the same in every table of the shop: `hoan_thanh` / `huy` / `tra_hang` / `dang_giao`. How it is taken: (1) the same product recorded several times (history) takes the latest by the time column, cancelled / returned are final so they always win; (2) different products, or the order table and the line table recording it differently: cancelled > returned > in delivery > completed (a partial return is a returned order, §1); (3) a return recorded only in another table (return quantity > 0 in a batch / line table, a return slip with an order id) makes the order `tra_hang`, except a cancelled order. With no status column it is inferred from `order_note`; an empty note means completed. A status the rules do not recognise is left empty (the LLM hook: `loader/llm.py`). The original status stays in the `status` / `order_note` column; day-based events (delivery, return, payment) are in the corresponding tables and date columns |
| `is_latest_row` | sales tables that record status history | present only when the same (order, product) is recorded with a **different status at a different time** (the history of one product line): TRUE on the latest row, FALSE on old rows. **When summing money / quantity filter on `is_latest_row`** (a table without this column counts every row). Several rows with the same status (extra serving of the same dish) are real separate sales, all TRUE |
| `is_walk_in` | `sales_lines`, `orders` | a walk-in customer. A shop with customer codes: no code recorded. Without codes: an empty name or one starting with "khách lẻ / vãng lai / KL" |
| `customer_key`, `product_key` | wherever there is a name | a name key with accents, case, spaces and punctuation removed; used to merge "names that differ only in spelling". `product_key` is computed from `product_full` if present. `customer_key` keeps the number written with the name (a different number is two customers, §3) |
| `customer_name_clean`, `phone_in_name` | where a customer name is written with a phone number | the name with the phone number removed ("Lê Hải Hà - 0327 210 166" -> "Lê Hải Hà") to match customer names in questions; the number is split out. An answer that writes the name with the phone number as in the file is also graded correct (§9) |
| `product_full` | sales / purchase rows of an abbreviated ledger | the full name looked up from the abbreviation table; if it cannot be found the written name is kept |
| `category` | sales rows of an abbreviated ledger | the product group taken from the abbreviation table's group column (if the sales row does not have one) |
| `line_type` | `sales_lines` | `hang` / `phi_phuc_vu` / `voucher` / `giam_gia` / `dung_diem` / `phi_giao` / `thue` / `phi_khac`. Anything other than `hang` is not a product. Rulebook: a service fee still counts as revenue; an order discount, delivery fee, VAT do not |
| `qty_base`, `qty_base_src`, `base_unit` | sales / purchase rows missing `qty_base` | `qty` × the factor in `_unit_conversions`. `qty_base_src`: `quy_doi`, `co_ban`, `mac_dinh` (no conversion row: factor 1, rulebook §10), `khong_ro` (quantity missing), `khong_khop` (the product has conversions but the row's unit is odd: left empty, **not** guessed) |
| `purchase_cancelled` | where `purchase_status` exists | the purchase slip is cancelled |
| `plate`, `staff_key`, `is_warranty` | service shops | a licence plate split from the vehicle cell; the worker's short name (without "A.", "anh"…); a warranty slip |

### Meta tables and shared tables

| Table | Contents |
|---|---|
| `_meta_shop` | shop, `data_end`, `has_customer_id`, description, `industry` (inferred), `industry_evidence`, `rulebook_section` (the matching §6 section name), `conventions` (JSON, only when built with `--llm`: the README conventions read — `money_scale` already multiplied to dong, `status_column` the column the README says holds the status, `walk_in_labels`, `not_recorded` information the README states the ledger does not record (entries that contradict a real column removed), `other` other conventions) |
| `_meta_tables` | `table_name`, `role`, `role_desc`, original name, source file, row count, duplicate rows dropped, cells filled |
| `_meta_columns` | for each column: the name in the DB, original name, canonical name, evidence source (`map_source`: `readme_col`, `name_dict`, `builtin`, `value_pattern`, `value_rule`, `llm`), type, README description, note. When built with the LLM, an unusual column (unmapped) carries the note `LLM: <column meaning>` |
| `_meta_conventions` | the README's conventions section |
| `_meta_dropped` | dropped rows: total rows (with / without a label), repeated headers, duplicate rows of the same order |
| `_unit_conversions` | `product_id`, `product_name`, `product_key`, `unit` (large unit), `factor` (to the base unit), `base_unit`, `source`. A conversion row that has only a name (README, a conversion table without codes) gets `product_id` filled from the catalogue when the name matches exactly one code, and vice versa |
| `_product_aliases` | `alias`, `product_name`, `source`: other ways to call a product (abbreviations in the ledger, the catalogue's "Tên gọi khác" column), to match names in questions |
| `_calendar` | the split's `calendar.csv`: solar date, weekday, lunar day/month/year, leap, holidays |
| `_tet_periods` | `tet_year`, `start_date`, `end_date`: the Tet period per rulebook §2 (23rd of the 12th lunar month → end of the 10th of the 1st lunar month) |
| `_rulebook` | the split's rulebook cut by section: `level`, `heading`, `path`, `text` |

### Table roles (`_meta_tables.role`)

| Role | Meaning |
|---|---|
| `sales_lines` | sales product rows |
| `orders` | sales orders, one order per row (including a service-slip ledger with no codes) |
| `order_info` | extra order information |
| `sales_batches` | sales rows by batch |
| `payments` | payment receipts |
| `purchases` | purchase slips |
| `purchase_lines` | purchase product rows |
| `batches` | purchased batches |
| `products` | product catalogue |
| `drug_info` | drug information |
| `customers` | customer list |
| `suppliers` | suppliers |
| `staff` | staff |
| `expenses` | expense ledger: `expense_amount`, `expense_date`. Monthly net profit = gross profit − every expense of the month (§5) |
| `stock` | stock |
| `stock_moves` | stock card |
| `unit_conversions` | unit conversions |
| `price_history` | price history |
| `deliveries` | deliveries by batch |
| `debt_terms` | an order's debt terms |
| `debts` | debt ledger |
| `other` | other tables |

A role is assigned by rules from the canonical columns the table has. It is a hint, not an absolute guarantee.

## Verification

- `python -m pytest tests -q`: unit tests, including two anchor tests that lock the meaning of every status value and every README column description of train.
- `python -m datamate.tests.audit_mapping <data> <out> --leave family [--llm]` (a script not included in this repo): mapping accuracy on file layouts not yet seen (a whole shop family removed from the dictionary); `--llm` adds an LLM pass.
- `python -m datamate.tests.verify_answers <split> <DB folder>`: shared SQL relying only on canonical names, table roles and columns added by build, run on the pure-form train questions and compared with the answers. It covers revenue, order count, cancellations, cancellation rate, best sellers (including the Tet period), top customers, purchase money, number of deliveries, debt (total, number of customers owing, debt age, 3 age groups, top debtors, debt collected in a month, extra debt / payments of one customer), end-of-day stock (number of base units, and [whole large units, leftover base units]), batches / expiry (nearly expired batches, cost value, number of batches, products about to expire), out-of-stock products, average purchase price, stock value at cost, gross / net profit, the most profitable product, prescription drugs (revenue, ratio), average order value, the day / month of highest revenue, revenue and share by product group / customer group, quantity sold (one product, size M/L, by base unit), the selling price on day D, the price-increase effect; periods per §2 (week, quarter, last N days, from X to Y…). For unanswerable questions: check that the DB can **not** build the source. The formulas in the script are ways of reading the DB that were verified against the answers, and the query side may use them as templates:
  - debt: order total minus the payments in the **payment ledger** up to the end of day X (the order's "đã trả" column is not used), excluding cancelled orders and COD / marketplace orders (`payment_method`, `channel`); "hơn N ngày" (over N days) means debt age > N
  - stock: the stock table (`stock_qty`, a snapshot of the last data day) -> the stock card (the last `stock_after`) -> batches (purchased per batch − sold per batch + `qty_returned_base`)
  - number of deliveries: the number of (order id, delivery) pairs whose delivery date falls in the period
  - a batch's remaining stock = purchased per batch − sold per batch + `qty_returned_base`; cost value at the batch's own purchase price (purchase line amount / the batch's base quantity)
  - gross profit = §1 revenue − `qty` × `unit_cost` **recorded on the sales row**; net profit also subtracts the `expense_amount` of the `expenses` table in the same month; stock value uses the **catalogue's** `unit_cost`
  - prescription: `is_prescription` in `drug_info` by code; a sales row that records only the name is joined code -> catalogue name -> `product_key`
  - selling price on day D: `unit_price` × `qty` / `qty_base` (one base unit, before the line discount), including rows of cancelled / returned orders
  - group share: the denominator is the whole shop's revenue in the same period (including service fees and walk-in orders); customer groups follow `customer_group`, a walk-in order belongs to no group
