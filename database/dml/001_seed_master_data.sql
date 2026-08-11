-- EBRIP DEMO MASTER DATA SEED
-- Safe for empty development database.

BEGIN;

-- ------------------------------------------------------------
-- 1. BRANCHES
-- ------------------------------------------------------------

INSERT INTO banking.branches
(branch_code, branch_name, city, state, region, ifsc_code, branch_status)
VALUES
('EBR001','EBRIP Central Branch','Bhopal','Madhya Pradesh','Central','EBRIP000001','Active'),
('EBR002','EBRIP Indore Branch','Indore','Madhya Pradesh','Central','EBRIP000002','Active'),
('EBR003','EBRIP Mumbai Branch','Mumbai','Maharashtra','West','EBRIP000003','Active'),
('EBR004','EBRIP Delhi Branch','New Delhi','Delhi','North','EBRIP000004','Active'),
('EBR005','EBRIP Bengaluru Branch','Bengaluru','Karnataka','South','EBRIP000005','Active')
ON CONFLICT DO NOTHING;


-- ------------------------------------------------------------
-- 2. CUSTOMERS
-- ------------------------------------------------------------

INSERT INTO banking.customers
(
 customer_code,
 first_name,
 last_name,
 date_of_birth,
 gender,
 email,
 mobile_number,
 pan_number,
 customer_type,
 risk_category,
 kyc_status,
 record_status
)
SELECT
 'CUST' || LPAD(gs::text,6,'0'),
 'Customer',
 'Demo' || gs,
 DATE '1980-01-01' + ((gs * 137) % 12000),
 CASE WHEN gs % 2 = 0 THEN 'Male' ELSE 'Female' END,
 'customer' || gs || '@ebrip.demo',
 '9' || LPAD(gs::text,9,'0'),
 'ABCDE' || LPAD(gs::text,4,'0') || 'F',
 CASE WHEN gs % 10 = 0 THEN 'Corporate' ELSE 'Individual' END,
 CASE
   WHEN gs % 20 = 0 THEN 'High'
   WHEN gs % 5 = 0 THEN 'Medium'
   ELSE 'Low'
 END,
 'Verified',
 'Active'
FROM generate_series(1,500) gs
ON CONFLICT DO NOTHING;


-- ------------------------------------------------------------
-- 3. CUSTOMER ADDRESSES
-- ------------------------------------------------------------

INSERT INTO banking.customer_addresses
(
 customer_id,
 address_type,
 address_line_1,
 address_line_2,
 city,
 state,
 postal_code,
 country
)
SELECT
 customer_id,
 'Permanent',
 'EBRIP Demo Address ' || customer_id,
 NULL,
 CASE ((customer_id - 1) % 5)
   WHEN 0 THEN 'Bhopal'
   WHEN 1 THEN 'Indore'
   WHEN 2 THEN 'Mumbai'
   WHEN 3 THEN 'New Delhi'
   ELSE 'Bengaluru'
 END,
 CASE ((customer_id - 1) % 5)
   WHEN 0 THEN 'Madhya Pradesh'
   WHEN 1 THEN 'Madhya Pradesh'
   WHEN 2 THEN 'Maharashtra'
   WHEN 3 THEN 'Delhi'
   ELSE 'Karnataka'
 END,
 LPAD((100000 + customer_id)::text,6,'0'),
 'India'
FROM banking.customers
WHERE customer_code LIKE 'CUST%'
ON CONFLICT DO NOTHING;


-- ------------------------------------------------------------
-- 4. CUSTOMER KYC
-- ------------------------------------------------------------

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
 customer_id,
 pan_number,
 '900000000000' || LPAD(customer_id::text,0,'0'),
 TRUE,
 TRUE,
 CURRENT_DATE - ((customer_id % 1000)::int),
 CURRENT_DATE + 365,
 CASE
   WHEN customer_id % 20 = 0 THEN 'High'
   WHEN customer_id % 5 = 0 THEN 'Medium'
   ELSE 'Low'
 END,
 FALSE,
 FALSE
FROM banking.customers
WHERE customer_code LIKE 'CUST%'
ON CONFLICT DO NOTHING;


-- ------------------------------------------------------------
-- 5. ACCOUNTS
-- ------------------------------------------------------------

INSERT INTO banking.accounts
(
 account_number,
 customer_id,
 branch_id,
 account_type,
 account_status,
 opening_balance,
 current_balance,
 currency_code,
 opened_date
)
SELECT
 'AC' || LPAD(c.customer_id::text,10,'0'),
 c.customer_id,
 (
   SELECT branch_id
   FROM banking.branches
   ORDER BY branch_id
   OFFSET ((c.customer_id - 1) % 5)
   LIMIT 1
 ),
 CASE
   WHEN c.customer_id % 10 = 0 THEN 'Current'
   ELSE 'Savings'
 END,
 'Active',
 10000.00 + (c.customer_id * 125.50),
 10000.00 + (c.customer_id * 125.50),
 'INR',
 CURRENT_DATE - ((c.customer_id % 2000)::int)
FROM banking.customers c
WHERE c.customer_code LIKE 'CUST%'
ON CONFLICT DO NOTHING;

COMMIT;
