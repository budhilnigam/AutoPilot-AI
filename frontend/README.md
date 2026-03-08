# AutoPilot AI Frontend

React + Vite frontend for the AutoPilot AI Multi-Agent SRE System.

## Features

- **3-Panel Layout:**
  - Left: Health check monitoring for all 6 agents and services
  - Center: SRE AI Copilot chatbot with default prompts
  - Right: Live alerts from CloudWatch and other sources

- **Technologies:**
  - React 18
  - Vite
  - Tailwind CSS
  - Lucid React Icons
  - Axios for API calls
  - WebSocket for real-time alerts

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Copy environment variables
cp .env.example .env

# Edit .env with your API endpoint
```

### Development

```bash
# Start development server
npm run dev

# Server will run on http://localhost:5173
```

### Build for Production

```bash
# Create production build
npm run build

# Preview production build
npm run preview
```

## Configuration

Edit `.env` file:

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## API Integration

The frontend connects to the FastAPI backend:

- REST API: `http://localhost:8000/api/*`
- WebSocket: `ws://localhost:8000/ws/alerts`

Make sure the backend is running before starting the frontend.
