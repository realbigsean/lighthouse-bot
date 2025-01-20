import asyncio
import os
import re
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import uuid

import aiohttp
import pytz
from apscheduler.events import EVENT_JOB_ERROR
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from dotenv import load_dotenv
from github import Github
from mattermostdriver import Driver
from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Global variable to track the last time the bot started a consensus cycle
LAST_RUN_TIME = None

class ConsensusBot:
    """
    A bot that manages consensus-building for recurring meetings:
    - Finds meeting issues on GitHub
    - Sends DMs to participants for feedback
    - Aggregates and summarizes responses before regional calls
    - Handles scheduling across multiple time zones
    """

    def __init__(self, mattermost_url: str, mattermost_token: str, repo_name: str, user_list: list[str], scheduler: AsyncIOScheduler):
        """
        Initialize the bot with necessary connections and configurations.

        Args:
            mattermost_url: Base URL for Mattermost instance
            mattermost_token: Authentication token for Mattermost
            repo_name: GitHub repository name (format: "owner/repo")
            user_list: List of Mattermost usernames to include
            scheduler: The AsyncIOScheduler instance for scheduling tasks
        """
        # Validate inputs
        if not all([mattermost_url, mattermost_token, repo_name, user_list, scheduler]):
            raise ValueError("All configuration parameters must be provided")

        # Store scheduler instance
        self.scheduler = scheduler

        # Mattermost setup with timeout and retry configuration
        self.driver = Driver({
            'url': mattermost_url,
            'token': mattermost_token,
            'scheme': 'https',
            'port': 443,
            'timeout': 30
        })

        try:
            self.driver.login()
            self.bot_user_id = self.driver.users.get_user('me')['id']
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Mattermost connection: {e}")

        # GitHub setup
        try:
            self.github = Github(os.getenv("GITHUB_TOKEN"))
            self.repo = self.github.get_repo(repo_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GitHub connection: {e}")

        # Rate limiting and concurrency control
        self.session = aiohttp.ClientSession()
        self.rate_limiter = asyncio.Semaphore(5)

        # User management
        self.user_list = user_list
        self.user_ids = self._get_user_ids()

        # Cycle tracking
        self.active_cycles: Dict[str, Dict[str, Any]] = {}

        # OpenAI setup
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Test mode configuration
        self.test_mode = os.getenv("TEST_MODE", "false").lower() == "true"

    async def find_current_meeting_issue(self) -> Optional[Any]:
        """Find the most recent consensus layer meeting issue."""
        try:
            async with self.rate_limiter:
                issues = self.repo.get_issues(state='open', sort='updated', direction='desc')
                pattern = re.compile(r"^Consensus-layer Call \d+$")

                for issue in issues:
                    if pattern.match(issue.title):
                        logging.info("Found meeting issue: %s (Issue #%d)", issue.title, issue.number)
                        return issue

                logging.warning("No matching consensus meeting issue found.")
                return None
        except Exception as e:
            logging.error("Error finding meeting issue: %s", e)
            return None

    def _get_user_ids(self) -> dict:
        """Get Mattermost user IDs with caching."""
        cache = {}
        for username in self.user_list:
            try:
                user = self.driver.users.get_user_by_username(username)
                cache[username] = user['id']
            except Exception as e:
                logging.error(f"Failed to get user ID for {username}: {e}")
        return cache

    async def create_dm_channel(self, user_id: str) -> str:
        """Create DM channel with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with self.rate_limiter:
                    channel = self.driver.channels.create_direct_message_channel([user_id, self.bot_user_id])
                    return channel['id']
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1 * (attempt + 1))

    async def summarize_with_llm(self, text: str, prompt: str) -> str:
        """Use OpenAI to summarize text with retry logic."""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                async with self.rate_limiter:
                    response = await self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that summarizes technical discussions."},
                            {"role": "user", "content": prompt + "\n\n" + text}
                        ],
                        temperature=0.7,
                        max_tokens=700
                    )
                    return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error("Error calling OpenAI API: %s", e)
                    return "Error occurred while summarizing."
                await asyncio.sleep(1 * (attempt + 1))

    def parse_meeting_time(self, issue_body: str) -> datetime:
        """
        Extract meeting time from issue body.
        Expected format: Meeting Date/Time: Wednesday 2025/01/30 at 14:00 UTC
        """
        time_pattern = r"Meeting Date/Time: \[?(\w+ \d{4}/\d{1,2}/\d{1,2} at \d{2}:\d{2} UTC)\]?(?:\(.*\))?"
        match = re.search(time_pattern, issue_body)
        if not match:
            raise ValueError("Could not find meeting time in issue body")

        datetime_str = re.sub(r'^\w+\s+', '', match.group(1))
        try:
            return datetime.strptime(datetime_str, "%Y/%m/%d at %H:%M UTC").replace(tzinfo=pytz.UTC)
        except ValueError as e:
            logging.error("Failed to parse datetime: %s", e)
            raise

    def calculate_cycle_times(self, issue) -> Tuple[datetime, datetime, datetime]:
        """Calculate timing for all cycle phases."""
        try:
            if self.test_mode:
                now = datetime.now(pytz.UTC)
                # Example: set calls ~30 seconds in the future for easy testing
                call_time = now + timedelta(seconds=55980)
                emea_time = now + timedelta(seconds=30)
                ameristralia_time = now + timedelta(seconds=30)
            else:
                call_time = self.parse_meeting_time(issue.body)
                emea_time = call_time - timedelta(hours=29)
                ameristralia_time = call_time - timedelta(hours=16)

            return emea_time, call_time, ameristralia_time
        except Exception as e:
            logging.error(f"Error calculating cycle times: {e}")
            raise

    async def send_dm_to_users(self, message: str, cycle_id: str) -> None:
        """Send DMs with cycle tracking and error handling."""
        self.active_cycles[cycle_id] = {
            'messages': {},
            'start_time': datetime.now(pytz.UTC)
        }

        for username, user_id in self.user_ids.items():
            try:
                channel_id = await self.create_dm_channel(user_id)
                async with self.rate_limiter:
                    post = self.driver.posts.create_post({
                        'channel_id': channel_id,
                        'message': message
                    })

                self.active_cycles[cycle_id]['messages'][username] = {
                    'id': post['id'],
                    'channel_id': channel_id,
                    'create_at': post['create_at']
                }
            except Exception as e:
                logging.error(f"Failed to send DM to {username}: {e}")

    async def start_consensus_cycle(self):
        """Start a new consensus cycle with unique tracking ID."""
        cycle_id = str(uuid.uuid4())
        issue = await self.find_current_meeting_issue()

        if not issue:
            # If no issue was found and not in test mode, retry in 1 hour
            if not self.test_mode:
                run_time = datetime.now(pytz.UTC) + timedelta(hours=1)
                self.scheduler.add_job(
                    self.start_consensus_cycle,
                    'date',
                    run_date=run_time
                )
            return

        # Get and summarize issue content
        issue_text = issue.body + "\n\nComments:\n" + "\n".join(c.body for c in issue.get_comments())
        agenda = await self.summarize_with_llm(
            issue_text,
            "Extract and summarize the key agenda items from this GitHub issue and its comments. Format as bullet points."
        )

        try:
            emea_time, call_time, ameristralia_time = self.calculate_cycle_times(issue)
        except ValueError as e:
            logging.error(f"Failed to parse meeting time: {e}")
            return

        template = f"""
{agenda}

**Please provide your input on these agenda items.**

Schedule:
- EMEA Call: {emea_time.strftime('%Y-%m-%d %H:%M UTC')}
- Main Call: {call_time.strftime('%Y-%m-%d %H:%M UTC')}
- Ameristralia Call: {ameristralia_time.strftime('%Y-%m-%d %H:%M UTC')}

GitHub Issue: {issue.html_url}
"""
        await self.send_dm_to_users(template, cycle_id)

        # Schedule aggregations with cycle ID *directly* in the event loop.
        self.scheduler.add_job(
            self.aggregate_feedback,
            'date',
            run_date=emea_time,
            args=["EMEA", cycle_id]
        )
        self.scheduler.add_job(
            self.aggregate_feedback,
            'date',
            run_date=ameristralia_time,
            args=["Ameristralia", cycle_id]
        )

        # Schedule cleanup
        cleanup_time = call_time + timedelta(hours=1)
        self.scheduler.add_job(
            self.cleanup_cycle,
            'date',
            run_date=cleanup_time,
            args=[cycle_id]
        )

    async def fetch_user_responses(self, cycle_id: str) -> str:
        """Fetch responses for a specific cycle."""
        if cycle_id not in self.active_cycles:
            logging.error(f"No active cycle found with ID: {cycle_id}")
            return "No responses collected."

        cycle_info = self.active_cycles[cycle_id]
        all_responses = []

        for username, info in cycle_info['messages'].items():
            try:
                async with self.rate_limiter:
                    posts = self.driver.posts.get_posts_for_channel(info['channel_id'])

                responses = [
                    p['message'] for p in posts['posts'].values()
                    if (p['user_id'] != self.bot_user_id and
                        p['create_at'] > info['create_at'] and
                        not p.get('delete_at'))
                ]

                if responses:
                    all_responses.append(f"### {username}'s Input:\n" + "\n".join(responses))

            except Exception as e:
                logging.error(f"Failed to fetch responses for {username}: {e}")

        return "\n\n".join(all_responses) if all_responses else "No responses collected."

    async def aggregate_feedback(self, phase: str, cycle_id: str):
        """Aggregate and summarize feedback for a specific cycle phase."""
        responses = await self.fetch_user_responses(cycle_id)
        summary = await self.summarize_with_llm(
            responses,
            f"Summarize the key points and consensus from these responses for the {phase} call."
        )

        cycle_info = self.active_cycles.get(cycle_id, {})
        if not cycle_info:
            logging.error(f"No cycle info found for {cycle_id}")
            return

        for info in cycle_info['messages'].values():
            try:
                async with self.rate_limiter:
                    summary_post = self.driver.posts.create_post({
                        'channel_id': info['channel_id'],
                        'message': f"# Pre-{phase} Call Summary\n\n{summary}"
                    })

                    parsed_responses = self.parse_responses(responses)
                    for username, response in parsed_responses.items():
                        self.driver.posts.create_post({
                            'channel_id': info['channel_id'],
                            'message': f"### {username}'s Input:\n{response}",
                            'root_id': summary_post['id']
                        })
            except Exception as e:
                logging.error(f"Error posting summary: {e}")

    def parse_responses(self, responses_text: str) -> Dict[str, str]:
        """
        Dummy method for parsing structured input. Right now, it scans
        for blocks of the form '### <user>' and returns a dict.
        """
        results = {}
        pattern = r"### (.+?)'s Input:\n(.+?)(?=\n###|$)"
        matches = re.findall(pattern, responses_text, re.DOTALL)
        for user, content in matches:
            results[user.strip()] = content.strip()
        return results

    async def cleanup_cycle(self, cycle_id: str):
        """Clean up cycle data after completion."""
        if cycle_id in self.active_cycles:
            del self.active_cycles[cycle_id]
            logging.info(f"Cleaned up cycle {cycle_id}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

def setup_scheduler() -> AsyncIOScheduler:
    """Configure scheduler with AsyncIOExecutor and error handling."""
    jobstores = {
        'default': MemoryJobStore()
    }
    executors = {
        'default': AsyncIOExecutor()
    }
    loop = asyncio.get_event_loop()

    scheduler = AsyncIOScheduler(jobstores=jobstores, executors=executors, event_loop=loop)

    def job_error_listener(event):
        if event.code == EVENT_JOB_ERROR:
            logging.error(f"Job failed: {event.job_id}", exc_info=event.exception)

    scheduler.add_listener(job_error_listener, EVENT_JOB_ERROR)
    return scheduler

# -------------------------------------------------------------------
# New function to prevent running too soon after a previous run
# -------------------------------------------------------------------
async def maybe_start_consensus_cycle(bot):
    global LAST_RUN_TIME
    now = datetime.now(pytz.UTC)

    # For example, skip if last run was within the last 5 minutes
    min_interval = timedelta(minutes=5)

    if LAST_RUN_TIME and (now - LAST_RUN_TIME) < min_interval:
        logging.info(
            "Skipping start_consensus_cycle because the bot just ran within the last 5 minutes."
        )
        return

    await bot.start_consensus_cycle()
    LAST_RUN_TIME = now


async def main():
    load_dotenv()

    required_vars = ['MATTERMOST_URL', 'MATTERMOST_TOKEN', 'GITHUB_REPO', 'MATTERMOST_USERS']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Initialize scheduler first
    scheduler = setup_scheduler()

    # Create bot
    async with ConsensusBot(
        mattermost_url=os.getenv('MATTERMOST_URL'),
        mattermost_token=os.getenv('MATTERMOST_TOKEN'),
        repo_name=os.getenv('GITHUB_REPO'),
        user_list=os.getenv('MATTERMOST_USERS').split(','),
        scheduler=scheduler
    ) as bot:
        # Start scheduler
        scheduler.start()

        # Schedule a recurring job for every Monday at 14:00
        if not bot.test_mode:
            scheduler.add_job(
                maybe_start_consensus_cycle,
                'cron',
                day_of_week='mon',
                hour=14,
                args=[bot]
            )

        # Run on startup if enabled
        if os.getenv("RUN_ON_STARTUP", "false").lower() == "true":
            await maybe_start_consensus_cycle(bot)

        try:
            # Keep the script running forever
            await asyncio.Event().wait()
        except (KeyboardInterrupt, SystemExit):
            scheduler.shutdown()
            logging.info("Bot shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
