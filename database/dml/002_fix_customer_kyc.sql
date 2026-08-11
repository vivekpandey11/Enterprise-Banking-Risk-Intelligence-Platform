INSERT INTO banking.customer_kyc
(
 customer_id,
 pan_number,
 aadhaar_number,
 pan_verified,
 aadhaar_verified,
 kyc_completion_date,
 kyc_expiry_date,
 risk_rating,
 pep_flag,
 sanctions_flag
)
SELECT
 c.customer_id,
 c.pan_number,
 '900000000000',
 TRUE,
 TRUE,
 CURRENT_DATE - ((c.customer_id % 1000)::int),
 CURRENT_DATE + 365,
 CASE
   WHEN c.customer_id % 20 = 0 THEN 'High'
   WHEN c.customer_id % 5 = 0 THEN 'Medium'
   ELSE 'Low'
 END,
 FALSE,
 FALSE
FROM banking.customers c
WHERE NOT EXISTS
(
    SELECT 1
    FROM banking.customer_kyc k
    WHERE k.customer_id = c.customer_id
);
