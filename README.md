# Customer Service Requests Analysis – NYC 311

Course-end Project 1 | Python for Data Science

---

## 1. Executive Summary

This project conducts an exhaustive end-to-end data science analysis of New York City's **311 Service Requests** (from 2010 to Present, focused on 2015 records). NYC 311 provides residents with a unified portal to submit municipal non-emergency service requests and citizen grievances across the five boroughs: The Bronx, Brooklyn, Manhattan, Queens, and Staten Island.

Utilizing Python, Pandas, NumPy, Matplotlib, Seaborn, and SciPy, this study performs systematic data wrangling, missing-value treatment, chronological anomaly filtering, exploratory data analysis (EDA), spatial density analysis across Brooklyn, resolution-time performance evaluation, and inferential statistical hypothesis testing ($\chi^2$ test of independence).

---

## 2. Project Directory Structure

```text
Customer_service_request_analysis/
│
├── data/
│   └── 311_Service_Requests_from_2010_to_Present.csv   # Raw dataset (300,698 records, 53 features)
│
├── notebooks/
│   └── NYC_311_Analysis.ipynb                         # Fully executed Jupyter Notebook with embedded outputs
│
├── output/
│   └── complaint_types_by_city.csv                    # City-wise complaint contingency matrix (49 cities × 24 types)
│
├── visualizations/
│   ├── monthly_complaints.png                         # Time-series trend of service requests
│   ├── city_complaints.png                            # Top 20 cities bar chart
│   ├── brooklyn_scatter.png                           # Geographic scatter plot of complaints in Brooklyn
│   ├── brooklyn_hexbin.png                            # Hexagonal binning density heatmap for Brooklyn
│   ├── top_complaints.png                             # Top 10 complaint categories across NYC
│   ├── city_top_complaints_heatmap.png                # Matrix heatmap of complaints across top cities
│   └── response_time.png                              # Average resolution time across complaint types
│
├── analysis.py                                        # Standalone end-to-end Python pipeline script
├── requirements.txt                                   # Python dependencies
└── README.md                                          # Project documentation and findings
```

---

## 3. Dataset Understanding & Profile

- **Total Records (Rows):** `300,698`
- **Total Variables (Columns):** `53`
- **Data Types:** A mix of string categoricals, timestamps, integer keys, floating-point coordinates, and geographical boundary identifiers.

### Top Variables with Null Values

Out of 53 variables, 35 columns contain missing values. High-missing attributes represent specialized domain fields (such as transit ferry routes or school-specific categories) that do not apply to general municipal service requests:

| Column Name | Missing Count | Missing Percentage | Implication / Treatment |
| :--- | :---: | :---: | :--- |
| `School or Citywide Complaint` | 300,698 | 100.00% | Completely empty; uninformative for municipal analysis. |
| `Vehicle Type` | 300,698 | 100.00% | Completely empty. |
| `Taxi Company Borough` | 300,698 | 100.00% | Empty (TLC complaints logged separately). |
| `Taxi Pick Up Location` | 300,698 | 100.00% | Empty. |
| `Garage Lot Name` | 300,698 | 100.00% | Empty. |
| `Ferry Direction` | 300,697 | 99.999% | Specialized ferry request field. |
| `Ferry Terminal Name` | 300,696 | 99.999% | Specialized ferry request field. |
| `Road Ramp` | 300,485 | 99.93% | Applies only to highway incidents. |
| `Bridge Highway Segment` | 300,485 | 99.93% | Applies only to highway incidents. |
| `Bridge Highway Name` | 300,455 | 99.92% | Applies only to highway incidents. |
| `Bridge Highway Direction` | 300,455 | 99.92% | Applies only to highway incidents. |
| `Landmark` | 300,349 | 99.88% | Rarely provided by callers. |
| `Intersection Street 2` | 257,336 | 85.58% | Missing when requests specify block addresses. |
| `Intersection Street 1` | 256,840 | 85.41% | Missing when requests specify block addresses. |
| `Cross Street 2` | 49,779 | 16.55% | Omitted for mid-block incidents. |
| `Closed Date` | 2,164 | 0.72% | Ongoing / unclosed requests at extract time. |
| `City` | 2,614 | 0.87% | Imputed as `'Unknown'`. |

---

## 4. Data Cleaning & Wrangling Pipeline

1. **Duplicate Detection:**
   - 0 duplicate rows detected across the 300,698 records (`df.duplicated().sum() == 0`).
2. **Missing Value Imputation:**
   - Categorical fields `Complaint Type`, `City`, and `Borough` were checked for missing values and imputed with `'Unknown'`. Text values were stripped of surrounding whitespace and normalized using title case to eliminate duplicate keys created by formatting inconsistencies.
3. **Geographic Precision:**
   - `Latitude` and `Longitude` were coerced to float types; records lacking valid coordinates were excluded from spatial mapping to prevent skewed projections.
4. **Datetime Chronology & Timeline Verification:**
   - `Created Date` and `Closed Date` were parsed into standard ISO datetime timestamps.
   - 0 records had missing creation timestamps.
   - 2,164 records had missing `Closed Date` (open/pending cases).
   - Validated timeline constraint: `Closed Date >= Created Date`. All closed records conformed to correct chronological sequence.
5. **Feature Engineering:**
   - Temporal features extracted: `Year`, `Month`, `Day`, `Day_Name`, and `Year_Month`.
   - Resolution duration: `Response_Time_Hours = (Closed Date - Created Date).total_seconds() / 3600.0`.

---

## 5. Detailed Findings & Exploratory Data Analysis

### 5.1 City-Wise Frequency Analysis

Complaints are heavily concentrated in the major boroughs/cities:

| Rank | City | Total Complaints | Percentage of Total |
| :---: | :--- | :---: | :---: |
| 1 | **Brooklyn** | 98,307 | 32.69% |
| 2 | **New York (Manhattan)** | 65,994 | 21.95% |
| 3 | **Bronx** | 40,702 | 13.54% |
| 4 | **Staten Island** | 12,343 | 4.10% |
| 5 | **Jamaica** | 7,296 | 2.43% |
| 6 | **Astoria** | 7,047 | 2.34% |
| 7 | **Flushing** | 5,971 | 1.99% |
| 8 | **Ridgewood** | 5,163 | 1.72% |
| 9 | **Corona** | 4,295 | 1.43% |
| 10 | **Woodside** | 3,664 | 1.22% |

> *Note: In NYC 311 records, Queens entries are recorded by individual postal towns (Jamaica, Astoria, Flushing, Ridgewood, Corona, Woodside), whereas Brooklyn, Bronx, and Manhattan are aggregated under their borough/city labels.*

### 5.2 Brooklyn Spatial Concentration (Scatter & Hexbin Analysis)

- **Total Geocoded Complaints in Brooklyn:** 98,113.
- **Scatter Plot Interpretation:** Displays complete spatial coverage along transit corridors and residential streets, with dense ribbons lining Flatbush Avenue, Atlantic Avenue, and Broadway.
- **Hexbin Density Interpretation:** Hexagonal binning highlights distinct hotspots with complaints exceeding 250+ per hexagonal cell:
  - **North Brooklyn:** Williamsburg and Bushwick (primarily nightlife noise and parking congestion).
  - **Central Brooklyn:** Crown Heights and Bedford-Stuyvesant (blocked driveways and street noise).
  - **South/Central Brooklyn:** Flatbush and Borough Park (severe parking and driveway blockage).

### 5.3 Major Complaint Types Across New York City

There are 24 distinct complaint types in the dataset. The **Top 10** categories account for **97.8%** of all calls:

| Rank | Complaint Type | Total Volume | Proportion | Cumulative |
| :---: | :--- | :---: | :---: | :---: |
| 1 | **Blocked Driveway** | 77,044 | 25.62% | 25.62% |
| 2 | **Illegal Parking** | 75,361 | 25.06% | 50.68% |
| 3 | **Noise - Street/Sidewalk** | 48,612 | 16.17% | 66.85% |
| 4 | **Noise - Commercial** | 35,577 | 11.83% | 78.68% |
| 5 | **Derelict Vehicle** | 17,718 | 5.89% | 84.57% |
| 6 | **Noise - Vehicle** | 17,083 | 5.68% | 90.25% |
| 7 | **Animal Abuse** | 7,778 | 2.59% | 92.84% |
| 8 | **Traffic** | 4,498 | 1.50% | 94.34% |
| 9 | **Homeless Encampment** | 4,416 | 1.47% | 95.81% |
| 10 | **Noise - Park** | 4,042 | 1.34% | 97.15% |

**Key Takeaways:**
1. **Automotive & Parking Dominance:** `Blocked Driveway` and `Illegal Parking` together exceed **50.6%** of all 311 calls.
2. **Noise Pollution:** Across all four noise categories (`Street/Sidewalk`, `Commercial`, `Vehicle`, `Park`), noise complaints total **105,314 calls (35.0%)**.

---

## 6. Response Time Analysis

Response time was calculated for all 298,534 resolved service requests.

### Average vs. Median Resolution Time (Hours)

Because resolution durations contain heavy right-tail outliers (cases delayed due to investigations or towing operations), comparing the arithmetic **Mean** with the **Median** reveals the skewness:

| Complaint Type | Resolved Count | Mean Response Time (Hours) | Median Response Time (Hours) | Skewness Note |
| :--- | :---: | :---: | :---: | :--- |
| **Animal in a Park** | 1 | 336.83 h | 336.83 h | Isolated single outlier case. |
| **Derelict Vehicle** | 17,588 | **7.36 h** | **4.12 h** | Requires towing and agency coordination. |
| **Graffiti** | 113 | **7.15 h** | **4.53 h** | Specialized cleaning/removal process. |
| **Agency Issues** | 6 | **5.26 h** | **4.90 h** | Administrative routing delay. |
| **Animal Abuse** | 7,768 | **5.21 h** | **3.30 h** | Requires ASPCA / NYPD on-site dispatch. |
| **Blocked Driveway** | 76,810 | **4.74 h** | **3.17 h** | Handled by local police precincts. |
| **Illegal Parking** | 74,532 | **4.50 h** | **2.93 h** | Enforced by traffic/patrol officers. |
| **Panhandling** | 305 | **4.37 h** | **2.67 h** | Outreach / patrol dispatch. |
| **Homeless Encampment**| 4,416 | **4.37 h** | **2.85 h** | Coordinated with DHS social services. |
| **Vending** | 3,795 | **4.01 h** | **2.66 h** | Sidewalk inspection. |
| **Noise - Street/Sidewalk**| 48,076 | **3.45 h** | **2.20 h** | High priority rapid police response. |
| **Traffic** | 4,496 | **3.45 h** | **1.99 h** | Immediate traffic safety dispatch. |
| **Noise - Commercial** | 35,247 | **3.15 h** | **2.07 h** | Patrol dispatch to commercial establishments. |

---

## 7. Statistical Hypothesis Testing: Chi-Square Test ($\chi^2$)

To determine whether complaint patterns are statistically dependent upon the borough from which they originate, we formulate a **Chi-Square Test of Independence**:

### Hypotheses
- **Null Hypothesis ($H_0$):** Complaint Type and Borough are **independent**. The likelihood of observing a specific complaint type does not vary across NYC boroughs.
- **Alternative Hypothesis ($H_1$):** Complaint Type and Borough are **dependent / associated**. Specific boroughs exhibit distinct distributions of complaint types.
- **Significance Level ($\alpha$):** $0.05$

### Test Formulation
$$\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$
Where:
- $O_{ij}$ = Observed complaint count for Borough $i$ and Complaint Type $j$.
- $E_{ij}$ = Expected count under the assumption of independence: $\frac{(\text{Row Total}_i \times \text{Column Total}_j)}{\text{Grand Total}}$.
- Degrees of Freedom: $\text{dof} = (r - 1) \times (c - 1) = (5 - 1) \times (10 - 1) = 36$.

### Contingency Matrix (Borough vs. Top 10 Complaint Types)

| Borough | Blocked Driveway | Illegal Parking | Noise - Street | Noise - Commercial | Derelict Vehicle | Noise - Vehicle | Animal Abuse | Traffic | Homeless Encampment | Noise - Park |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bronx** | 12,755 | 7,859 | 8,892 | 2,434 | 1,953 | 3,396 | 1,415 | 357 | 358 | 547 |
| **Brooklyn** | 28,148 | 27,462 | 21,273 | 11,463 | 6,179 | 5,177 | 2,394 | 1,086 | 1,213 | 1,555 |
| **Manhattan** | 2,073 | 12,132 | 14,550 | 14,552 | 252 | 5,485 | 1,525 | 1,551 | 2,778 | 1,211 |
| **Queens** | 31,644 | 21,982 | 4,409 | 6,075 | 8,110 | 2,615 | 1,876 | 1,305 | 408 | 637 |
| **Staten Island** | 2,142 | 1,761 | 819 | 433 | 1,766 | 356 | 557 | 198 | 34 | 67 |

### Statistical Results
- **Chi-Square Statistic ($\chi^2$):** **`64,425.3089`**
- **p-value:** **`0.0`** ($p < 10^{-16}$)
- **Degrees of Freedom ($dof$):** **`36`**
- **Decision:** Because $p \ll 0.05$, we **reject the null hypothesis $H_0$**.

### Domain Interpretation
The association between Borough and Complaint Type is overwhelmingly significant:
1. **Queens and Brooklyn** account for over **77.6% of all Blocked Driveway** complaints, reflecting car-centric residential zones with narrow shared driveways.
2. **Manhattan** records disproportionately high **Noise - Commercial** (14,552) and **Homeless Encampment** (2,778) calls, consistent with high-density commercial nightlife corridors and dense pedestrian zones.
3. **Queens** dominates **Derelict Vehicle** reports (8,110 out of 17,718 across NYC, ~45.8%), highlighting differences in vacant lot distribution and curbside parking patterns.

---

## 8. How to Run the Analysis

### Option A: Run via Python CLI
Ensure the virtual environment is activated, then run:

```bash
# Run the complete analysis pipeline
./venv/bin/python analysis.py
```

### Option B: Interactive Jupyter Notebook
Launch the Jupyter Notebook server:

```bash
./venv/bin/jupyter lab notebooks/NYC_311_Analysis.ipynb
# or
./venv/bin/jupyter notebook notebooks/NYC_311_Analysis.ipynb
```

All cells are pre-executed with fully formatted charts, data tables, and statistical summaries.

---

## 9. Deliverables Summary

1. `data/311_Service_Requests_from_2010_to_Present.csv`: Official uncompressed course dataset.
2. `notebooks/NYC_311_Analysis.ipynb`: Fully executed interactive Jupyter Notebook.
3. `analysis.py`: Modular, production-ready Python execution script.
4. `output/complaint_types_by_city.csv`: Complete city-by-complaint crosstabulation matrix.
5. `visualizations/`: High-resolution figures (300 DPI) for city complaints, Brooklyn spatial maps, complaint bar graphs, heatmap matrices, and resolution times.
6. `requirements.txt`: Standardized dependency file.
