# Quick Start Guide - Running Praxis AI

## Prerequisites

- **Python 3.9+** installed
- **Node.js 18+** and **npm 8+** installed
- **Git** (optional, if cloning)

## Step 1: Install Python Dependencies

Open a terminal in the project root directory and run:

```bash
# Install Python dependencies
pip install -r requirements.txt
```

**Note:** If you're using a virtual environment (recommended):
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Then install dependencies
pip install -r requirements.txt
```

## Step 2: Install Frontend Dependencies

```bash
# Install root dependencies (if not already installed)
npm install

# Install frontend dependencies
cd frontend
npm install
cd ..
```

Or use the convenience script:
```bash
npm run install:all
```

## Step 3: Create Required Directories (if not exists)

The application will create these automatically, but you can create them manually:

```bash
mkdir -p data/uploads
mkdir -p data/reports
```

## Step 4: Run the Application

### Option A: Run Both Frontend and Backend Together (Recommended)

From the project root directory:

```bash
npm run dev
```

This will start:
- **Backend API** on `http://localhost:8000`
- **Frontend** on `http://localhost:8080`

### Option B: Run Separately

**Terminal 1 - Backend:**
```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

## Step 5: Access the Web Application

1. Open your web browser
2. Navigate to: **http://localhost:8080**
3. You should see the Praxis AI interface

## Step 6: Upload a Notebook

1. Click on the upload area or drag and drop a Jupyter notebook (`.ipynb` file)
2. Wait for the analysis to complete (this may take a few minutes)
3. View the generated research paper and industry analysis in the tabs

## Troubleshooting

### Issue: Python dependencies fail to install

**Solution:**
```bash
# Upgrade pip first
pip install --upgrade pip

# Try installing with --no-cache-dir
pip install --no-cache-dir -r requirements.txt
```

### Issue: Node modules fail to install

**Solution:**
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules frontend/node_modules
npm run install:all
```

### Issue: Port already in use

**Solution:**
- Backend: Change port in `backend/app/main.py` or kill process using port 8000
- Frontend: Change port in `frontend/vite.config.ts` or kill process using port 8080

### Issue: Gemini API errors

**Solution:**
- The API key is already configured in the code
- If you get API errors, check your internet connection
- Verify the API key is valid in `backend/app/core/llm_providers.py`

### Issue: Import errors

**Solution:**
```bash
# Make sure you're in the backend directory when running Python
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

## Verify Installation

### Check Backend:
```bash
# Test backend health endpoint
curl http://localhost:8000/health

# Should return: {"status": "healthy"}
```

### Check Frontend:
- Open browser to `http://localhost:8080`
- You should see the Praxis AI interface

## Environment Variables (Optional)

Create a `.env` file in the root directory if you want to override defaults:

```env
# Gemini API Key (already configured in code)
GEMINI_API_KEY=AIzaSyA9a3OHjxLiTO07EmUbqVEw3GbA0xZRMYo

# Optional: Enable full text scraping (slower but more comprehensive)
ENABLE_FULL_TEXT_SCRAPING=false

# Optional: Change log level
LOG_LEVEL=INFO
```

## API Documentation

Once the backend is running, you can access:
- **API Docs:** http://localhost:8000/docs (Swagger UI)
- **Alternative Docs:** http://localhost:8000/redoc (ReDoc)

## Next Steps

1. Upload a Jupyter notebook to test the system
2. Check the console logs for processing status
3. View the generated reports in the web interface
4. Explore the A2A agent APIs at `/api/v1/agents/`

## Need Help?

- Check the logs in the terminal for error messages
- Verify all dependencies are installed
- Make sure ports 8000 and 8080 are available
- Check that Python and Node.js versions are correct
