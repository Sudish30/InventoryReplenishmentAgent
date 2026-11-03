# Inventory Replenishment Agent (Group Exercise – Agentic AI in Supply Chain)

An agent that decides each day, per SKU, whether to place a purchase order or wait, using an EWMA demand forecast,
service-level safety stock, lead-time inventory projection, `min_order_qty` and an over-order cap. It is compared
against a fixed reorder-point baseline.

## Run
    pip install -r requirements.txt
    python InventoryReplenishmentAgent.py        # or open InventoryReplenishmentAgent.ipynb and Run All

If `data/sales.csv`, `data/inventory.csv`, `data/params.csv` are missing, they are generated automatically.
Tunable settings (alpha, service level override, review period, order cost, cap) are at the top of the code.

## Contents
- `InventoryReplenishmentAgent.ipynb` – notebook with design doc, code, daily agent log, evaluation, reflection, scaling note (outputs saved)
- `InventoryReplenishmentAgent.py` – same code as a script
- `data/` – sales.csv, inventory.csv, params.csv
- `docs/Design_Document_and_Scaling_Note.docx` – design doc (1 page) + scaling note
- `outputs/` – program_output.txt (full console output incl. daily agent log), charts, results summary

## Results (76 simulated days, 3 SKUs, $20 fixed cost per PO)
| | Baseline | Agent |
|---|---|---|
| Total cost | $8,857 | $6,162 |
| Fill rate | 95.9% | 98.5% |
| Stockout units | 273 | 99 |
