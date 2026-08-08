\# Enterprise Banking Risk Intelligence Platform

\## Dataset Catalog



\### 1. Credit Risk Dataset



\*\*Dataset:\*\* Give Me Some Credit



\*\*Purpose:\*\*

Credit default risk prediction and customer risk segmentation.



\*\*Primary Target:\*\*

`SeriousDlqin2yrs`



\*\*Problem Type:\*\*

Binary classification.



\*\*Business Objective:\*\*

Estimate probability of serious delinquency and classify customers into risk bands.



\*\*Planned Models:\*\*

\- Logistic Regression

\- Random Forest

\- XGBoost



\*\*Explainability:\*\*

SHAP



\*\*Primary Metrics:\*\*

\- ROC-AUC

\- Precision

\- Recall

\- F1 Score

\- Brier Score

\- Calibration



\---



\### 2. Fraud Detection Dataset



\*\*Dataset:\*\* Credit Card Fraud Detection



\*\*Purpose:\*\*

Identify potentially fraudulent card transactions.



\*\*Primary Target:\*\*

`Class`



\*\*Problem Type:\*\*

Highly imbalanced binary classification.



\*\*Business Objective:\*\*

Detect fraudulent transactions while controlling false positives.



\*\*Planned Models:\*\*

\- Logistic Regression

\- Random Forest

\- XGBoost



\*\*Primary Metrics:\*\*

\- Precision

\- Recall

\- F1 Score

\- PR-AUC

\- False Positive Rate



\---



\### 3. AML Transaction Dataset



\*\*Dataset:\*\* IBM AML Transactions Dataset



\*\*Purpose:\*\*

Anti-Money Laundering transaction monitoring.



\*\*Problem Type:\*\*

Rule-based detection + anomaly detection.



\*\*Business Objective:\*\*

Identify suspicious transaction behaviour and generate compliance alerts.



\*\*Detection Techniques:\*\*

\- Transaction velocity

\- Structuring detection

\- High-value transactions

\- Cross-border activity

\- Unusual transaction patterns

\- Isolation Forest

\- Rule-based AML engine



\*\*Output:\*\*

AML alerts for compliance investigation.



\---



\## Data Architecture



External Dataset

&#x20;       |

&#x20;       v

Raw Layer

&#x20;       |

&#x20;       v

Validation

&#x20;       |

&#x20;       v

Staging Layer

&#x20;       |

&#x20;       v

Transformation

&#x20;       |

&#x20;       v

Processed Layer

&#x20;       |

&#x20;       v

PostgreSQL

&#x20;       |

&#x20;       +---- Analytics

&#x20;       +---- Machine Learning

&#x20;       +---- Risk Scoring

&#x20;       +---- Fraud Detection

&#x20;       +---- AML Monitoring

&#x20;       +---- Power BI

