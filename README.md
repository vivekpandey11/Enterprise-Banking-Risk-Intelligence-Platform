# 🏦 Enterprise Banking Risk Intelligence Platform

<p align="center">

**An end-to-end banking risk intelligence and fraud analytics platform integrating transaction monitoring, credit risk, fraud detection, AML governance, machine learning, PostgreSQL, Excel reporting, and Power BI analytics.**

<br/>

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15%2B-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analytics-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![Excel](https://img.shields.io/badge/Excel-Executive_Reporting-217346?style=for-the-badge&logo=microsoftexcel&logoColor=white)

</p>

---


## 📋 1. Executive Summary

| Risk Domain | Purpose | Implementation |
|---|---|---|
| **Credit Risk** | Estimate borrower/customer default probability | Preprocessing, feature engineering, model training, threshold optimization, inference, evaluation, explainability |
| **Fraud Detection** | Detect suspicious customer/transaction behavior | Preprocessing, feature engineering, ML models, threshold optimization, inference, evaluation, explainability |
| **Transaction Fraud** | Detect high-risk individual transactions | Preprocessing, rule engine, ML training/inference, threshold optimization |
| **AML Monitoring** | Identify suspicious activity for AML review | Preprocessing, feature engineering, model training/inference, validation, explainability, reporting |

Also included: PostgreSQL banking/risk schemas, referential-integrity & data-quality validation, `.joblib` model artifacts, an integrated risk engine, a FastAPI risk API, executive Excel reporting, a one-page Power BI dashboard design, full design docs (business requirements, architecture, database design, data dictionary, dataset catalog), and automated Python syntax/test validation.

---

## 🎯 2. Business Problem

A customer can simultaneously be credit-risky, linked to suspicious transactions, subject to a fraud alert, exposed via a high-risk merchant/device, or flagged for AML/KYC concerns. Evaluating these signals separately fragments risk decisions.

**EBRIP objective:** answer *who/what is risky, why, how severe, and what to review first* by combining:

**Customer → Account → Transaction → Merchant → Device → KYC → Credit Risk → Fraud Risk → AML Risk → Integrated Risk Decision → Executive Reporting**

---

## ⚙️ 3. Key Capabilities

- **Credit Risk:** preprocessing, feature engineering, Logistic Regression / Random Forest / Gradient Boosting, model comparison, threshold optimization, inference, explainability, evaluation
- **Fraud Detection:** preprocessing, feature engineering, LR/RF/GB models, evaluation, threshold optimization, inference, explainability, fraud-alert integration
- **Transaction Fraud:** transaction-level preprocessing & feature engineering, rule-engine detection, LR/RF training, threshold optimization, inference, high-risk transaction identification
- **AML Monitoring:** preprocessing, feature engineering, LR/RF, threshold optimization, inference, validation, explainability, reporting
- **Integrated Risk Engine:** combines Credit → Fraud → Transaction Fraud → AML → KYC/Compliance signals into a Risk Score, Risk Tier, Alert Status, and Business Interpretation

---

## 🏗️ 4. High-Level Architecture

```text
Banking Data (Customers/Accounts/Transactions/Cards/Merchants/Devices/KYC/Branches)
        ↓
PostgreSQL (banking + risk schemas)
        ↓
Credit Risk | Fraud Detection | AML  →  each: Preprocessing → Features → ML Models
        ↓
Transaction Fraud Engine (Rules + ML inference)
        ↓
Integrated Risk Engine
        ↓
Risk API | Excel Reporting | Power BI
```

---

## 🗄️ 5. Database Design

Database name: `ebrip`

**Banking schema:** `banking.customers`, `banking.accounts`, `banking.transactions`, `banking.cards`, `banking.devices`, `banking.merchants`, `banking.branches`, `banking.customer_addresses`, `banking.customer_kyc`

**Risk schema:** `risk.credit_scores` (score storage), `risk.fraud_alerts` (alert/investigation info), `risk.aml_rules` (rule configuration)

### Populated validation snapshot
- **500 customers**, **500 accounts**, **10,000 transactions**, **500 cards**, **500 devices**, **500 merchants**, **5 branches**, **500 customer addresses**, **500 KYC records**, **1,073 fraud alerts**

`risk.credit_scores` and `risk.aml_rules` were empty at the latest snapshot — pipelines/reporting should not be read as a live production system.

---

## ✅ 6. Data Quality & Integrity

Checks performed: Transaction→Account, Transaction→Merchant, Transaction→Device, and Fraud Alert→Transaction orphan checks; NULL checks on transaction amount, timestamp, status, and account ID.

```text
Transactions → Accounts       : 0 orphan relationships
Transactions → Merchants      : 0 orphan relationships
Transactions → Devices        : 0 orphan relationships
Fraud Alerts → Transactions   : 0 orphan relationships
Transaction Amount NULL       : 0
Transaction Timestamp NULL    : 0
Transaction Status NULL       : 0
Transaction Account ID NULL   : 0
```

---

## 💳 7. Transaction Validation Snapshot

```text
Transactions               : 10,000
Total Transaction Amount   : 1,000,189,000
Average Transaction Amount : 100,018.90
Minimum Amount             : 111
Maximum Amount             : 199,976
```

**Status:** Success 9,867 | Failed 100 | Pending 33

**Channels (2,000 each):** Mobile, Branch, POS, ATM, Internet Banking

---

## 🚨 8. Fraud Validation Snapshot

```text
Fraud Alerts        : 1,073
Average Fraud Score : 0.6756
Minimum Score        : 0.5000
Maximum Score        : 1.0000
Open alerts          : 1,073
```

**Risk levels:** High 618 | Critical 455

This is demonstration data, not a real bank's operational fraud rate.

---

## 🪪 9. KYC / Compliance Snapshot

```text
KYC records      : 500
PAN verified     : 500
Aadhaar verified : 500
PEP flags        : 0
Sanctions flags  : 0
```

**Customer risk categories:** Low 400 | Medium 75 | High 25
**Merchant risk levels:** Low 400 | Medium 75 | High 25

---

## 🤖 10. Machine Learning Layer

```text
models/
├── aml/
├── credit_risk/
├── fraud/
└── transaction_fraud/
```

**Algorithms:** Logistic Regression, Random Forest, Gradient Boosting

**Lifecycle:** Raw Data → Validation → Preprocessing → Feature Engineering → Train/Val Split → Training → Evaluation → Threshold Optimization → Best Model → Inference → Risk Decision

Each major domain has its own trainer, evaluator, threshold-optimizer, predictor/inference, and explainability modules.

---

## 📊 11. Model Performance Snapshot

| Domain | Selected Model | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| Credit Risk | Gradient Boosting | 0.871633 | 0.409072 |
| Fraud Detection | Gradient Boosting | 0.868241 | 0.407716 |
| Transaction Fraud | Random Forest | 0.972772 | 0.807150 |
| AML Monitoring | Logistic Regression | 0.861643 | 0.000361 |

**AML governance note:** the AML validation set is highly imbalanced — only **1 positive AML case out of 20,000 validation rows**. AML threshold metrics must be treated as experimental, not production-grade, until evaluated on a larger, representative dataset. This is documented deliberately, not hidden.

---

## 🔍 12. Explainability

Explainability modules exist for Credit Risk, Fraud Detection, and AML — supporting model transparency, analyst review, risk-decision interpretation, feature-level reasoning, and governance discussions.

> A risk score should be explainable enough for an analyst to understand why a case was flagged.

---

## 📈 13. Executive Excel Reporting

`reports/EBRIP_Executive_Risk_Report.xlsx` includes: Executive Summary, Integrated Risk, Credit Risk, Fraud Detection, Transaction Fraud, AML Monitoring, Fraud Alerts, Model Performance, Data Dictionary.

Executive Summary contains **6 KPI cards**, **3 charts**, cross-domain model performance, a governance warning, business risk interpretation, and credit-risk segment distribution. Generated via `scripts/generate_executive_excel.py` and `scripts/build_executive_dashboard.py`.

---

## 📉 14. Power BI — One-Page Executive Risk Command Center

Designed as **one professional page**: one screen → complete banking risk picture → drill-free decision support. The `.pbix` is maintained separately from this README until final.

### Header
**Enterprise Banking Risk Intelligence Platform** — *Integrated Credit Risk • Fraud Detection • Transaction Fraud • AML Monitoring*, plus last refresh timestamp, data period, model version, and governance status.

### Row 1 — KPI cards (~8)
Total Customers, Total Transactions, Transaction Value, Fraud Alerts, Critical Fraud Alerts, Fraud Alert Rate, High-Risk Customers, AML/Compliance Alerts (combine Transaction Value + Count if space is tight).

### Row 2 — Risk distribution
- **Customer Risk Distribution** (donut) — legend `risk_category`, values Customer Count
- **Fraud Severity** (donut) — legend `risk_level`, values Fraud Alert Count
- **Transaction Status** (100% stacked column) — axis Transaction Status, values Transaction Count

(Max 2 donuts recommended — avoid decorative repetition.)

### Row 3 — Trends
- **Transaction & Risk Trend** (line): X = Transaction Date, Y = Transaction Count, secondary = Transaction Value
- **Fraud Alert Trend** (line/column combo): X = Date, column = Fraud Alerts, line = Average Fraud Score

### Row 4 — Domain comparison
- **Risk Domain Model Performance** (clustered bar): categories = Credit Risk / Fraud Detection / Transaction Fraud / AML; measures = ROC-AUC, PR-AUC (label clearly — AML PR-AUC is extremely low due to imbalance)
- **Transaction Channel Risk** (horizontal bar): axis = Channel, value = Transaction Value, tooltip = Count/Fraud Alerts/Rate

### Row 5 — Operational intelligence
- **Top Risk Drivers** (bar): Feature vs. Importance, from model feature-importance output
- **Fraud Alert Operations** (matrix): rows = Risk Level / Alert Status / Detection Source; values = Alert Count, Avg Fraud Score, Fraud Amount (conditional formatting for scanning)

### Row 6 — Governance panel
**MODEL GOVERNANCE:** Credit — validated · Fraud — validated · Transaction Fraud — validated · AML — experimental / insufficient positive validation cases.

### Recommended slicers (max, don't exceed)
Transaction Date, Transaction Channel, Transaction Type, Transaction Status, Fraud Risk Level, Customer Risk Category, Merchant Risk Level, State/Region. Never slice on PII (name, email, phone, PAN, Aadhaar, IP, account number).

### Privacy rule for all executive visuals
Never expose: PAN, Aadhaar, card number, mobile number, email, IP address, full account number. Use risk segments, masked identifiers, aggregated counts/exposure, and appropriate-level geography instead.

---

## 🧩 15. Power BI Data Model & Measures

**Reporting source:** `reporting.vw_powerbi_transaction_risk` — purpose-built Power BI transaction-risk view, currently **10,000 rows**. Expand with additional curated views when a metric can't be derived from it alone.

**Conceptual star model:**
```text
DimCustomer ─┬─ FactTransactions ─┬─ DimMerchant
DimAccount ──┘        │            ├─ DimDevice
                       │            ├─ DimCard
                       │            └─ FactFraudAlerts
DimCustomer ── KYC/Compliance
DimCustomer ── Credit Risk
```
- **FactTransactions:** transaction_id, account_id, merchant_id, device_id, date/time, type, channel, status, amount
- **FactFraudAlerts:** fraud_alert_id, transaction_id, alert timestamp, detection source, fraud score, risk level, alert status, resolution timestamp
- **DimCustomer:** customer_id, customer type, risk category, KYC status
- **DimMerchant:** merchant_id, category, city, state, country, risk level
- **DimAccount:** account_id, customer_id, branch_id, account type, status, balance
- **DimDevice:** device_id, device type, OS, trusted-device indicator
- **DimDate:** Date, Year, Quarter, Month, Month Number, Week, Day, Day of Week

Follow the transaction-centric design; avoid unnecessary many-to-many relationships. Build **measures** (not duplicated calculated columns) for aggregations, e.g.:

```dax
Total Transactions = COUNTROWS(Transactions)
Transaction Value = SUM(Transactions[transaction_amount])
Transaction Success Rate = DIVIDE([Successful Transactions], [Total Transactions])
Fraud Alerts = COUNTROWS(FraudAlerts)
Fraud Alert Rate = DIVIDE([Fraud Alerts], [Total Transactions])
Average Fraud Score = AVERAGE(FraudAlerts[fraud_score])
High Risk Customers = CALCULATE([Total Customers], Customers[risk_category] = "High")
```
Add equivalent measures for: medium/low-risk customers, PEP/sanctions customers, failed/pending transactions, open/resolved fraud alerts, transaction value by risk level, fraud exposure, and model metrics.

---

## 🔌 16. API Layer

- `src/api/risk_api.py` — exposes risk functionality for application/integration use
- `src/integration/risk_engine.py` — integrated risk engine

**Flow:** Application/Analyst → Risk API → Integrated Risk Engine → Domain Inference → Risk Decision

---

## 📁 17. Repository Structure

```text
Enterprise-Banking-Risk-Intelligence-Platform/
├── dashboards/{aml, integrated, powerbi}
├── data/quarantine/
├── database/{ddl/{schemas,tables}, dml, seeds}
├── docs/
│   ├── data/DATASET_CATALOG.md
│   └── design/{BUSINESS_REQUIREMENTS, DATABASE_DESIGN, DATA_DICTIONARY, SYSTEM_ARCHITECTURE}.md
├── models/{aml, credit_risk, fraud, transaction_fraud}
├── reports/EBRIP_Executive_Risk_Report.xlsx
├── scripts/{generate_executive_excel.py, build_executive_dashboard.py}
├── src/{api, config, dashboard, evaluation, explainability, feature_engineering,
│        fraud_detection, inference, integration, models, preprocessing,
│        reporting, tests, utils, validation}
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🔁 18. Reproducibility

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m compileall -q .\src .\tests .\scripts
python -m pytest -q
python ".\scripts\generate_executive_excel.py"
python ".\scripts\build_executive_dashboard.py"
```

---

## 🧪 19. Validation Status

```text
Python AST syntax audit : PASSED
Python compileall       : PASSED
Pytest                  : 8 passed, 1 warning
Database relationship   : PASSED
Transaction NULL checks : PASSED
Fraud alert integrity   : PASSED
Excel report generation : PASSED
Executive dashboard     : PASSED
```
The 1 warning is a Starlette/httpx deprecation notice, not a failing test.

---

## ⚠️ 20. Governance & Limitations

Portfolio/demonstration platform — **not** a production banking decision system.

- **AML data imbalance:** only 1 positive case in 20,000 validation rows; needs more representative data before production use.
- **Synthetic data:** database/model results are for engineering demonstration, not real customer or bank performance.
- **Model governance gaps for production:** would additionally need approval workflow, independent validation, drift monitoring, bias/fairness testing, champion/challenger models, version registry, audit trails, human-in-the-loop investigation, secure PII handling, and regulatory controls.
- **Security:** PAN, Aadhaar, email, mobile, IP, and account numbers should be masked/tokenized in any production deployment.

---

## 💪 21. What Makes This Project Strong

Presented as an **integrated risk intelligence system**, not four separate ML notebooks:

```text
Banking Data → PostgreSQL Data Model → Validation/Quality Controls
    → Preprocessing & Feature Engineering
    → [Credit Risk | Fraud | Transaction Fraud | AML]
    → Inference + Thresholding + Explainability
    → Integrated Risk Engine → Risk Alerts/Business Decisions
    → Excel Report + Power BI Command Center + REST API
```

Combines: Banking Domain Understanding + Relational DB Design + Data Quality/Validation + Feature Engineering + ML + Threshold Optimization + Inference + Explainability + Fraud Rule Engine + Integrated Risk Engine + API + Executive Reporting + Power BI + Model Governance.

Relevant roles: Data/Risk/Banking Analytics/Fraud Analytics/Credit Risk/AML Analyst, Business Analyst (Risk/Banking), Junior Data Scientist, ML Engineer (Risk Analytics), Python Developer (Risk/FinTech), Analytics Engineer, BI/Power BI Analyst.

---

## 🗣️ 22. Interview Story

> "I built an integrated banking risk intelligence platform combining credit risk, fraud detection, transaction fraud, and AML monitoring. I designed the PostgreSQL data model, built preprocessing/feature-engineering pipelines, trained and evaluated multiple ML models, optimized risk thresholds, built inference and explainability layers, integrated signals through a common risk engine, and created executive reporting via Excel and a one-page Power BI dashboard — while explicitly documenting the AML model-governance limitation caused by severe class imbalance."

**Five layers to walk through:** Business (Credit → Fraud → Transaction Fraud → AML) → Data (PostgreSQL, customer/account/transaction relationships) → ML (preprocessing, training, evaluation, thresholding, inference, explainability per domain) → Decisioning (integrated risk engine → alerts) → Reporting (Excel + Power BI).

---

## 📝 23. Resume-Ready Bullets

> **Built an end-to-end Enterprise Banking Risk Intelligence Platform** integrating credit risk scoring, fraud detection, transaction-fraud rule/model inference, AML monitoring, PostgreSQL risk data models, explainability, REST APIs, automated Excel reporting, and a Power BI executive risk command center.

> **Implemented model training, preprocessing, threshold optimization, evaluation, inference, and explainability** across credit risk, fraud, transaction fraud, and AML domains.

> **Designed a relational PostgreSQL banking data model** covering customers, accounts, cards, devices, merchants, branches, transactions, KYC, credit scores, fraud alerts, and AML rules, with validation and orphan-relationship checks.

> **Developed executive risk reporting** with portfolio KPIs, transaction intelligence, fraud severity, model-performance monitoring, customer risk segmentation, operational alerts, and governance indicators.

> **Implemented automated Python compilation and pytest validation**; test suite passes **8/8**, with AML governance limitations explicitly surfaced due to insufficient positive validation cases.

*(Use the Section 11 performance table for model metrics — treat as project-dataset validation, not production-bank performance.)*

---

## 🛡️ 24. Evidence-Based Claims Policy

Never invent production volumes · never call validation metrics "production accuracy" · never hide the AML class-imbalance limitation · never expose real-looking sensitive identifiers in dashboards · keep model metrics tied to actual evaluation outputs · keep dashboard KPIs tied to database/reporting views · keep Power BI measures dynamic · prefer reproducible scripts over manually edited outputs · keep backups/temp files out of Git · keep test/validation commands documented.

---

## ✔️ 25. Definition of Done

- [x] PostgreSQL banking schema, customer/account/transaction domain, KYC/merchant/device data, fraud alerts
- [x] Credit-risk, fraud, transaction-fraud, and AML model artifacts
- [x] Preprocessing, feature-engineering, training, threshold-optimization, evaluation, inference, explainability modules
- [x] Integrated risk engine, REST API, data-validation modules
- [x] Automated tests pass
- [x] Executive Excel report, executive dashboard script, Power BI reporting view
- [ ] Final Power BI one-page command center polished
- [ ] Final README committed after Power BI completion
- [ ] Final Power BI `.pbix` committed only after final dashboard verification

---

## 🚀 26. Final Portfolio Positioning

**EBRIP** demonstrates: **SQL + Data Engineering + Machine Learning + Risk Analytics + Explainability + APIs + Reporting + BI** in one banking-focused solution.

Best suited for entry-level/fresher roles: Data Analyst, Risk Analyst, Fraud Analytics Analyst, AML/Transaction Monitoring Analyst, Business Analyst (Risk/Banking), Junior Data Scientist, ML Engineer (Risk Analytics), Python Developer (Data/Risk Platforms), Analytics Engineer, BI/Power BI Analyst.

Should be described honestly as a **production-style demonstration platform**, not a deployed banking production system.
