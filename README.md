# 🚀 AI Comment Analysis Tool

Transform hours of manual customer feedback analysis into minutes of AI-powered insights.

## Features

- **LDA Topic Modeling**: Automatically discovers hidden themes
- **GenAI Classification**: Uses Claude AI to name and categorize themes
- **Business Intelligence**: Priority-ranked actionable insights
- **Interactive Dashboard**: Professional UI with filters and chat
- **Export Options**: PDF reports, CSV data, shareable links

## Quick Start

### Local Development
```bash
npm install
npm start
```

### Environment Variables (Optional)
```bash
ANTHROPIC_API_KEY=sk-ant-your_anthropic_api_key_here
PORT=3000
```

### Deploy to CodeSandbox

1. Go to [codesandbox.io](https://codesandbox.io)
2. Click "Create Sandbox" → "Import from GitHub"
3. Or manually upload these files:
   - `package.json`
   - `server.js`
   - `public/index.html`

## Usage

1. Upload a CSV file with customer comments
2. Select the comment column
3. Click "Analyze" to run AI analysis
4. Explore results with interactive filters and chat

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/analyze` - Analyze comments (expects CSV file upload)

## Tech Stack

- **Backend**: Node.js, Express, Natural.js, Claude AI
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **AI/ML**: LDA Topic Modeling, K-Means Clustering, Claude Integration