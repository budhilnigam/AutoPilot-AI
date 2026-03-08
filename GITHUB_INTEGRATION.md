# GitHub Integration Guide

## Overview

The AutoPilot AI system now includes comprehensive GitHub integration for monitoring CI/CD pipelines, build performance, and repository health. The integration enables:

- **Real-time build monitoring** via GitHub Actions API
- **Build trend analysis** and regression detection
- **Failed build tracking** and diagnostics
- **Commit history analysis**
- **Repository health assessment**
- **CI/CD Agent integration** for intelligent build optimization

## Configuration

### Required Settings in `.env`

```env
# GitHub Personal Access Token (required for any GitHub functionality)
GITHUB_TOKEN=your_github_personal_access_token

# Repository to monitor
GITHUB_REPO_OWNER=your-github-username
GITHUB_REPO_NAME=your-repo-name
```

### How to Get a GitHub Personal Access Token

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token"
3. Select these scopes:
   - `repo` - Full control of private repositories
   - `workflow` - Read and write access to GitHub Actions workflows
   - `read:org` - Read org and team membership
4. Copy the token and paste it in `.env`

**Important**: Never commit your `.env` file to version control!

## Architecture

### Components

#### 1. **GitHubClient** (`services/github_client.py`)
Low-level GitHub API client with methods for:
- `get_workflow_runs()` - Fetch workflow executions
- `get_build_data()` - Extract detailed build information
- `get_commit_history()` - Retrieve commit logs
- `get_repository_info()` - Get repo metadata
- `analyze_build_trends()` - Trend analysis

#### 2. **GitHubService** (`services/github_service.py`)
High-level service wrapper providing:
- `get_repository_info()` - Repository metadata
- `get_recent_builds()` - Recent workflow runs
- `get_failed_builds()` - Failed runs
- `get_build_trends()` - Build trend analysis
- `get_build_health_summary()` - Health overview
- `get_commit_history()` - Commit analysis
- `is_configured()` - Configuration validation

#### 3. **CI/CD Agent** (`agents/cicd_agent.py`)
Intelligent agent that:
- Analyzes build performance data
- Detects build time regressions
- Identifies optimization opportunities
- Uses GitHub data for repository-wide analysis
- Provides AI-powered recommendations via Bedrock

## API Endpoints

All GitHub endpoints require proper configuration. If not configured, they return a 400 error with a helpful message.

### Repository Information
```
GET /api/github/repo/info
```
Returns repository metadata (description, stars, language, last update)

### Recent Builds
```
GET /api/github/builds/recent?limit=10
```
Returns recent workflow runs with build times and status

### Failed Builds
```
GET /api/github/builds/failed?limit=5
```
Returns recent failed workflow runs

### Build Trends
```
GET /api/github/builds/trends?days=7
```
Returns build trend analysis

### Build Health
```
GET /api/github/builds/health
```
Returns overall build health summary including:
- Success rate percentage
- Failed build count
- Average build time
- Health status (healthy/degraded/poor)

### Commit History
```
GET /api/github/commits?branch=main&limit=20
```
Returns commit history for a branch

### Workflow Runs
```
GET /api/github/workflows?limit=10
```
Returns all workflow runs with details

## Usage Examples

### Python

```python
from services.github_service import GitHubService
from config import config

# Initialize service
github_service = GitHubService()

# Check if configured
if github_service.is_configured():
    # Get repository info
    repo_info = github_service.get_repository_info()
    print(f"Repository: {repo_info['full_name']}")
    
    # Get build health
    health = github_service.get_build_health_summary()
    print(f"Build Success Rate: {health['success_rate_percent']}%")
    
    # Get recent failures
    failed_builds = github_service.get_failed_builds(limit=5)
    for build in failed_builds:
        print(f"Failed: {build.build_id} - {build.status}")
```

### API Calls

```bash
# Get build health
curl -X GET http://localhost:8000/api/github/builds/health

# Get recent builds
curl -X GET "http://localhost:8000/api/github/builds/recent?limit=10"

# Get repository info
curl -X GET http://localhost:8000/api/github/repo/info
```

### Frontend Integration

The framework provides hooks for the frontend to:

```javascript
// Fetch build health
const health = await fetch('/api/github/builds/health')
  .then(r => r.json());

// Display in dashboard
console.log(`Success Rate: ${health.success_rate_percent}%`);

// Track failed builds
const failed = await fetch('/api/github/builds/failed?limit=5')
  .then(r => r.json());
```

## CI/CD Agent Integration

The CI/CD Agent automatically uses GitHub data when available:

### Query Pattern 1: Repository Health
```
"Check the health of our GitHub builds"
```
The agent will:
1. Fetch recent builds from GitHub
2. Analyze success rates and trends
3. Identify patterns and recommendations
4. Provide actionable insights

### Query Pattern 2: Build Analysis
```
"Why are my builds taking longer than usual?"
```
The agent will:
1. Fetch historical build times
2. Compare against baselines
3. Detect regressions
4. Suggest optimizations

### Programmatic Analysis

```python
from agents.cicd_agent import CICDAgent

# Initialize agent (automatically uses GitHubService)
agent = CICDAgent()

# Analyze repository health
health_analysis = agent.analyze_github_repository_health()

# Process through planner agent
results = planner_agent.process_query(
    "Analyze our GitHub build pipeline",
    context={'github_data': health_analysis}
)
```

## Health Check Integration

The system health endpoint automatically checks GitHub connectivity:

```
GET /api/health/services
```

Response includes:
```json
{
  "service": "GitHub",
  "status": "healthy",
  "message": "GitHub client connected",
  "response_time_ms": 150
}
```

Status values:
- `healthy` - GitHub configured and accessible
- `degraded` - GitHub token missing or incomplete
- `unhealthy` - Connection failed

## Error Handling

All GitHub endpoints handle errors gracefully:

- **400 Bad Request**: Service not configured
- **500 Internal Server Error**: API call failed (rate limit, invalid token, network error)

Example error response:
```json
{
  "status": "error",
  "message": "GitHub service not configured. Set GITHUB_TOKEN, GITHUB_REPO_OWNER, and GITHUB_REPO_NAME in .env"
}
```

## Rate Limiting

GitHub API has rate limits:
- Authenticated requests: 5,000 per hour
- Unauthenticated: 60 per hour

The system handles rate limiting through:
- Token-based authentication (higher limits)
- Caching (reduce redundant calls)
- Error handling (graceful degradation)

## Production Setup

### Best Practices

1. **Use Environment Secrets**
   ```bash
   export GITHUB_TOKEN=$(aws secretsmanager get-secret-value --secret-id github-token --query SecretString --output text)
   ```

2. **Monitor Rate Limits**
   Subscribe to rate limit warnings and adjust polling intervals

3. **Filter Workflows**
   Specify specific workflows if you have many:
   ```python
   runs = github_client.get_workflow_runs(
       owner="myorg",
       repo="myrepo",
       workflow_id="build-and-test.yml"
   )
   ```

4. **Implement Caching**
   Some endpoints cache results (health check, repository info)

5. **Regular Audits**
   Review token permissions quarterly

## Troubleshooting

### GitHub service not configured
**Solution**: Ensure `.env` has:
```
GITHUB_TOKEN=your_token
GITHUB_REPO_OWNER=owner
GITHUB_REPO_NAME=repo
```

### 401 Unauthorized Errors
**Solution**: Check token validity
```bash
curl -H "Authorization: token YOUR_TOKEN" https://api.github.com/user
```

### No workflow runs returned
**Solution**: Verify repository has workflows configured

### High latency on build endpoints
**Solution**: 
- Increase limit parameter conservatively
- Check GitHub API status
- Verify network connectivity

## Future Enhancements

- [ ] Webhook support for real-time build notifications
- [ ] Build and PR analysis
- [ ] GitHub Discussions integration
- [ ] Team and organization level analysis
- [ ] Custom workflow metrics
- [ ] Caching layer for improved performance
- [ ] Rate limit tracking and alerts

## Support

For issues or questions:
1. Check configuration: `.env` exists with valid token
2. Verify GitHub API access: test token manually
3. Review logs: `LOG_LEVEL=DEBUG` in `.env`
4. Check health endpoint: `/api/health/services`
