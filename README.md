# Brain Percentile Grids API

A FastAPI-based service for calculating normative brain structure percentiles using GAMLSS (Generalized Additive Models for Location, Scale, and Shape) statistical modeling. This application enables researchers to:

1. Build reference models from neuroimaging data
2. Calculate percentiles for out-of-sample patients against these models
3. Identify potential abnormalities by comparing individual measurements to population norms

## Overview

The application uses R's GAMLSS package via rpy2 to fit flexible statistical models that capture the age-related changes in brain structure volumes. Unlike simple z-scores based on fixed means and standard deviations, GAMLSS models account for:

- **Location** (how the mean changes with age)
- **Scale** (how variability changes with age)
- **Shape** (how the distribution shape changes with age)

This provides more accurate percentile estimates across the age range.

## User Workflow

### 1. Authentication

Users must register and authenticate to access the API:

```
POST /api/auth/register - Create a new account
POST /api/auth/login    - Get JWT access token
```

All subsequent requests require the JWT token in the Authorization header.

### 2. Dataset Management

Users can create multiple named reference datasets (e.g., "Pediatric MRI Cohort", "Adult Controls"):

```
POST   /api/datasets            - Create a new dataset
GET    /api/datasets            - List all your datasets
GET    /api/datasets/{id}       - Get dataset details (with model info)
PATCH  /api/datasets/{id}       - Update dataset name/description
DELETE /api/datasets/{id}       - Delete dataset and all associated data/models
```

**Dataset Features:**
- Each user can have multiple independent datasets
- Datasets are isolated - the same patient can exist in different datasets
- Duplicate detection is scoped per-dataset (same patient + study_date cannot appear twice in ONE dataset)

### 3. Upload Reference Data

Upload patient data files (CSV or XLSX) to build reference models:

```
POST   /api/datasets/{id}/upload     - Upload patient data files
GET    /api/datasets/{id}/data       - View data summary
GET    /api/datasets/{id}/structures - List available brain structures
DELETE /api/datasets/{id}/data       - Clear all data from dataset
```

**Expected Data Format:**

Each file should contain columns for:
- `PatientID` - Unique patient identifier
- `BirthDate` - Patient birth date (for age calculation)
- `StudyDate` - Date of the scan
- Brain structure volume columns (e.g., `CerebralCortex`, `WhiteMatterTotal`, etc.)

### 4. Fit Reference Models

Once data is uploaded, fit GAMLSS models for each brain structure:

```
POST /api/datasets/{id}/fit        - Fit models (returns results when complete)
POST /api/datasets/{id}/fit/stream - Fit models with SSE progress updates
```

**What happens during fitting:**
1. For each brain structure, multiple GAMLSS distribution families are tested (NO, LOGNO, BCT, BCCG, etc.)
2. Models are ranked by BIC (Bayesian Information Criterion)
3. The best-fitting model is selected and persisted
4. Percentile curves are calculated for visualization

**Response includes:**
- Model convergence status
- Selected distribution family
- AIC/BIC scores
- Percentile curves for plotting

### 5. Calculate Out-of-Sample Patient Percentiles

With fitted models, calculate percentiles for new patients not in the reference set:

```
POST /api/datasets/{id}/calculate
```

**Upload patient files** (same format as reference data - CSV/XLSX):
- Files are processed using the same parser as reference data
- Data is NOT stored in the database - processed transiently
- Results are returned immediately

**Optional query parameter:**
- `structures` - List of structures to calculate (default: all available)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/api/datasets/1/calculate?structures=CerebralCortex&structures=WhiteMatterTotal" \
  -H "Authorization: Bearer <token>" \
  -F "files=@patient1.xlsx" \
  -F "files=@patient2.xlsx"
```

**Response includes:**
- Z-score for each structure
- Percentile (0-100) for each structure
- Error messages if calculation failed

**Important:** Out-of-sample patients are NOT stored in the database. They are processed transiently and results are returned immediately.

## Technical Architecture

```
app/
├── core/                          # Core statistical engine
│   ├── engine/
│   │   ├── model.py               # GAMLSS model wrapper
│   │   ├── selector.py            # Model selection logic
│   │   └── environment.py         # R environment setup
│   ├── data_processing/
│   │   └── process_input.py       # Data parsing and validation
│   └── resources/
│       ├── brain_structures.py    # Known brain structure definitions
│       └── model_candidates.py    # GAMLSS distribution families
│
├── fastapi/                       # FastAPI application
│   ├── main.py                    # App entry point
│   ├── config.py                  # Configuration settings
│   ├── db/
│   │   ├── database.py            # Database connection
│   │   └── models.py              # SQLModel definitions
│   ├── auth/
│   │   ├── security.py            # JWT token handling
│   │   └── dependencies.py        # Auth dependencies
│   ├── services/
│   │   ├── calculation.py         # GAMLSS fitting logic
│   │   ├── reference_data.py      # Reference data management
│   │   └── model_persistence.py   # Model save/load
│   └── routers/
│       ├── auth.py                # Authentication endpoints
│       ├── datasets.py            # Dataset CRUD
│       ├── data.py                # Data upload/management
│       └── calculations.py        # Model fitting & percentile calc
```

## Data Storage

### Database (SQLite/PostgreSQL)

- **Users** - Account credentials
- **ReferenceDatasets** - Named dataset metadata
- **PatientRecords** - Reference patient information
- **PatientStructureValues** - Brain volume measurements
- **FittedModels** - Model metadata (family, AIC, BIC, file path)

### File System

Fitted models are stored as R `.rds` files:
```
models/
└── user_{user_id}/
    └── dataset_{dataset_id}/
        ├── CerebralCortex.rds
        ├── WhiteMatterTotal.rds
        └── ...
```

## GAMLSS Distribution Families

The system tests multiple distribution families to find the best fit:

| Family | Description | Use Case |
|--------|-------------|----------|
| NO | Normal | Symmetric, constant variance |
| LOGNO | Log-Normal | Right-skewed, positive values |
| BCT | Box-Cox t | Flexible, handles skewness & kurtosis |
| BCCG | Box-Cox Cole-Green | Power exponential family |
| GG | Generalized Gamma | Very flexible right-skewed |
| GA | Gamma | Positive, right-skewed |
| IG | Inverse Gaussian | Positive, right-skewed |
| WEI | Weibull | Positive, flexible shape |

## Requirements

### System Requirements

- Python 3.11+
- R 4.0+ with packages:
  - `gamlss`
  - `gamlss.dist`

### Python Dependencies

```
fastapi[standard]
uvicorn[standard]
sqlmodel
pydantic-settings
rpy2
pandas
numpy
scipy
pyjwt[crypto]
pwdlib[argon2]
```

## Installation

1. **Install R and required packages:**
   ```bash
   # Ubuntu/Debian
   sudo apt install r-base r-base-dev

   # Install R packages
   R -e "install.packages(c('gamlss', 'gamlss.dist'))"
   ```

2. **Clone and setup Python environment:**
   ```bash
   git clone <repository-url>
   cd percentile-grids-fastapi

   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure environment (optional):**
   ```bash
   # Create .env file for custom settings
   echo "SECRET_KEY=your-secure-secret-key" > .env
   echo "DEBUG=true" >> .env
   ```

4. **Run the server:**
   ```bash
   uvicorn app.fastapi.main:app --reload
   ```

5. **Access the API:**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register` | POST | Register new user |
| `/api/auth/login` | POST | Login and get JWT |
| `/api/datasets` | GET | List user's datasets |
| `/api/datasets` | POST | Create new dataset |
| `/api/datasets/{id}` | GET | Get dataset details |
| `/api/datasets/{id}` | PATCH | Update dataset |
| `/api/datasets/{id}` | DELETE | Delete dataset |
| `/api/datasets/{id}/upload` | POST | Upload reference data |
| `/api/datasets/{id}/data` | GET | Get data summary |
| `/api/datasets/{id}/structures` | GET | List available structures |
| `/api/datasets/{id}/fit` | POST | Fit GAMLSS models |
| `/api/datasets/{id}/fit/stream` | POST | Fit with SSE progress |
| `/api/datasets/{id}/calculate` | POST | Calculate OOS percentiles |
| `/health` | GET | Health check |

## Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `DEBUG` | true | Enable debug mode |
| `SECRET_KEY` | - | JWT signing key |
| `DB_URL` | sqlite:///grids_database.db | Database URL |
| `UPLOAD_FOLDER` | ./uploads | Temp upload directory |
| `MODELS_DIR` | ./models | Model storage directory |
| `MAX_UPLOAD_SIZE` | 16MB | Maximum file upload size |
| `MAX_FILES_COUNT` | 300 | Max files per upload |
| `CORS_ORIGINS` | localhost:3000,5000 | Allowed CORS origins |

## Future Possibilities

- **Batch Processing**: Background task queue for large dataset processing
- **Model Versioning**: Keep multiple model versions per dataset
- **Export**: Download fitted models and percentile curves
- **Visualization API**: Generate percentile curve plots
- **Multi-tenancy**: Organization-level dataset sharing
- **Model Comparison**: Compare results across different reference datasets

## License

[Add license information]
