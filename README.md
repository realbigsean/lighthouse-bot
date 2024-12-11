
**Set up**
```bash
cp example.env .env
```
Update your `.env` file.

**How to Run in Docker:**
1. Build the Docker image:
```bash
docker build -t consensus-bot:latest .
```
2. Run the container (ensure .env is in the same directory):
```bash
docker run -d --env-file .env --name consensus-bot consensus-bot:latest
```

**How to Use Test Mode:**
- In the `.env` file, set `TEST_MODE=true` to enable test mode. Test mode reduces delay times to allow rapid testing in various parts of the consensus polling cycle. For example:
```env
  TEST_MODE=true
```
