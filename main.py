import asyncio
import os
import re
import logging
from datetime import datetime, timedelta

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from github import Github
from mattermostdriver import Driver
from openai import AsyncOpenAI  # New Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ConsensusBot:
    def __init__(self, mattermost_url: str, mattermost_token: str, repo_name: str, user_list: list[str], test_mode: bool = False):
        # Mattermost setup
        self.driver = Driver({
            'url': mattermost_url,
            'token': mattermost_token,
            'scheme': 'https',
            'port': 443
        })
        self.driver.login()

        # Store the bot's user ID
        self.bot_user_id = self.driver.users.get_user('me')['id']

        # GitHub setup - anonymous access for public repo
        self.github = Github()
        self.repo = self.github.get_repo(repo_name)

        # Store user list and their IDs
        self.user_list = user_list
        self.user_ids = self._get_user_ids()

        # We will store the current issue and cycle data once found
        self.current_issue = None
        self.cycle_data = {}
        # message_info will store info about the initial post for each user: {username: {"id": post_id, "channel_id": ..., "create_at": ...}}
        self.message_info = {}

        # Test mode flag
        self.test_mode = test_mode
        logging.info("ConsensusBot initialized. Test mode: %s", self.test_mode)

        # OpenAI API Key (must be set in environment)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logging.error("OPENAI_API_KEY not found. LLM summarization will not work.")
        # Instantiate the AsyncOpenAI client
        self.client = AsyncOpenAI(api_key=self.openai_api_key)

    async def find_current_meeting_issue(self):
        """Find the most recent consensus or execution layer meeting issue"""
        try:
            issues = self.repo.get_issues(state='open', sort='updated', direction='desc')
            pattern = re.compile(r"^(Consensus-layer Call \d+|Execution Layer Meeting \d+)$")

            for issue in issues:
                if pattern.match(issue.title):
                    logging.info("Found meeting issue: %s (Issue #%d)", issue.title, issue.number)
                    return issue

            logging.warning("No matching meeting issue found.")
            return None
        except Exception as e:
            logging.error("Error finding meeting issue: %s", e)
            return None

    def _get_user_ids(self) -> dict:
        """Get Mattermost user IDs for all users in the list"""
        user_ids = {}
        for username in self.user_list:
            try:
                user = self.driver.users.get_user_by_username(username)
                user_ids[username] = user['id']
                logging.info("Found user ID for %s: %s", username, user['id'])
            except Exception as e:
                logging.warning("Could not find user %s: %s", username, e)
        return user_ids

    def create_dm_channel(self, user_id: str) -> str:
        """Create or get direct message channel with a user"""
        try:
            channel = self.driver.channels.create_direct_message_channel([user_id, self.bot_user_id])
            logging.info("DM channel created for user_id %s: channel_id %s", user_id, channel['id'])
            return channel['id']
        except Exception as e:
            logging.error("Error creating DM channel: %s", e)
            return None

    def get_issue_and_comments_text(self, issue):
        """Fetch issue body and comments, and return a combined text."""
        logging.info("Fetching issue body and comments for issue #%d", issue.number)
        issue_body = issue.body or "No issue body."
        combined_text = f"Issue Title: {issue.title}\n\nIssue Body:\n{issue_body}\n\nComments:\n"
        
        comments = issue.get_comments()
        for c in comments:
            combined_text += f"---\n{c.user.login} said:\n{c.body}\n"
        return combined_text

    async def summarize_agenda_from_llm(self, text: str) -> str:
        """Use the new OpenAI SDK to summarize agenda from given text."""
        if not self.openai_api_key:
            logging.error("OpenAI API key not set. Cannot summarize.")
            return "No agenda available (OpenAI API key missing)."
        try:
            logging.info("Sending content to OpenAI for agenda summarization...")
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts agenda items from a GitHub issue and its comments."},
                    {"role": "user", "content": (
                        "Please read the following GitHub issue and comments, and produce a summary of the agenda items. "
                        "If there are any relevant links in the issue description or comments, please include them next "
                        "to the related agenda items. Format the agenda as a bullet point list.\n\n" + text
                    )}
                ],
                temperature=0.7,
                max_tokens=700
            )
            agenda_summary = response.choices[0].message.content.strip()
            logging.info("Received agenda summary from OpenAI.")
            return agenda_summary
        except Exception as e:
            logging.error("Error calling OpenAI API for agenda: %s", e)
            return "Error occurred while summarizing agenda."

    async def get_agenda_from_issue(self, issue):
        """Get a summarized agenda of the issue and comments from LLM."""
        text = self.get_issue_and_comments_text(issue)
        agenda_summary = await self.summarize_agenda_from_llm(text)
        return agenda_summary

    async def send_dm_to_users(self, message: str) -> None:
        """Send a DM to all users and store post info."""
        for username, user_id in self.user_ids.items():
            try:
                channel_id = self.create_dm_channel(user_id)
                if channel_id:
                    post = self.driver.posts.create_post({
                        'channel_id': channel_id,
                        'message': message
                    })
                    # Store initial message info for later reference
                    self.message_info[username] = {
                        'id': post['id'],
                        'channel_id': channel_id,
                        'create_at': post['create_at']  # This is in milliseconds since epoch
                    }
                    logging.info("Message sent to %s in channel %s (post_id %s)", username, channel_id, post['id'])
            except Exception as e:
                logging.error("Error sending message to %s: %s", username, e)

    async def start_consensus_cycle(self):
        """Attempt to start the consensus cycle. If no issue or no agenda, retry after some time."""
        logging.info("Attempting to start consensus cycle...")
        issue = await self.find_current_meeting_issue()

        # In test mode, shorter delays
        no_issue_delay = 5 if self.test_mode else 3600  # 5s vs 1h
        no_agenda_delay = 5 if self.test_mode else 3600 # 5s vs 1h

        if not issue:
            logging.info("No issue found, will retry in %d seconds...", no_issue_delay)
            run_time = datetime.now(pytz.UTC) + timedelta(seconds=no_issue_delay)
            scheduler.add_job(self.start_consensus_cycle, 'date', run_date=run_time)
            return

        # Get agenda from LLM
        agenda = await self.get_agenda_from_issue(issue)
        if not agenda.strip():
            logging.info("Issue found but could not summarize agenda, will retry in %d seconds...", no_agenda_delay)
            run_time = datetime.now(pytz.UTC) + timedelta(seconds=no_agenda_delay)
            scheduler.add_job(self.start_consensus_cycle, 'date', run_date=run_time)
            return

        # We have an agenda and an issue
        self.current_issue = issue
        self.cycle_data = {
            'start_date': datetime.now(pytz.UTC),
            'deadline': datetime.now(pytz.UTC) + timedelta(days=7),
            'issue_number': issue.number,
            'status': 'collecting'
        }

        logging.info("Starting consensus gathering for issue #%d", issue.number)

        template = f"""
{agenda}

**Please provide your input on these agenda items in the next 24hrs.**
If there are issues you want to talk about that are not listed, please still provide feedback! Focus on things like fork inclusion/exclusion.

Prep Calls:
- APAC: {(datetime.now(pytz.UTC) + timedelta(days=1)).strftime("%Y-%m-%d %H:%M UTC+8")}
- Americas: {(datetime.now(pytz.UTC) + timedelta(days=1)).strftime("%Y-%m-%d %H:%M UTC-5")}

GitHub Issue: {issue.html_url}
"""
        await self.send_dm_to_users(template)

        reminder_delay = 10 if self.test_mode else 86400   # 10s vs 24h
        aggregation_delay = 15 if self.test_mode else 172800 # 15s vs 48h

        # Schedule a reminder
        reminder_time = datetime.now(pytz.UTC) + timedelta(seconds=reminder_delay)
        scheduler.add_job(self.send_reminder, 'date', run_date=reminder_time)
        logging.info("Scheduled reminder in %d seconds", reminder_delay)

        # Schedule aggregation
        aggregation_time = self.cycle_data['start_date'] + timedelta(seconds=aggregation_delay)
        scheduler.add_job(self.aggregate_feedback, 'date', run_date=aggregation_time)
        logging.info("Scheduled aggregation in %d seconds", aggregation_delay)

    def user_has_responded(self, username: str) -> bool:
        """
        Check if the user has responded with at least one message after the bot's initial message.
        This will consider ANY message in the DM channel posted by the user after the initial bot message.
        """
        if username not in self.message_info:
            return False

        channel_id = self.message_info[username]['channel_id']
        initial_timestamp = self.message_info[username]['create_at']
        try:
            posts_data = self.driver.posts.get_posts_for_channel(channel_id)
            for p_id, post_data in posts_data['posts'].items():
                # Check if the post is by the user and after the initial message timestamp
                if post_data['user_id'] != self.bot_user_id and post_data['create_at'] > initial_timestamp:
                    return True
        except Exception as e:
            logging.error("Error checking responses for user %s: %s", username, e)
        return False

    async def send_reminder(self):
        """Send reminder after the allotted time, but only if the user has NOT responded."""
        reminder = "⏰ Time is running short to provide your input! Please share your thoughts informally."
        logging.info("Sending reminder to users who haven't responded.")
        for username, info in self.message_info.items():
            # Check if user responded
            if not self.user_has_responded(username):
                try:
                    channel_id = info['channel_id']
                    msg_id = info['id']
                    self.driver.posts.create_post({
                        'channel_id': channel_id,
                        'message': reminder,
                        'root_id': msg_id
                    })
                    logging.info("Sent reminder to %s in channel %s", username, channel_id)
                except Exception as e:
                    logging.error("Error sending reminder to %s: %s", username, e)
            else:
                logging.info("Skipping reminder for %s as they have already responded.", username)

    def fetch_user_responses(self) -> str:
        """
        Fetch all responses from the DM channels.
        We'll gather all the posts made by users (excluding the bot) after the initial message.
        """
        all_responses = []
        for username, info in self.message_info.items():
            channel_id = info['channel_id']
            initial_timestamp = info['create_at']

            # Fetch all posts in channel
            try:
                posts_data = self.driver.posts.get_posts_for_channel(channel_id)
                for p_id, post_data in posts_data['posts'].items():
                    # Consider any user post after the initial bot message
                    if (post_data['user_id'] != self.bot_user_id 
                        and post_data['create_at'] > initial_timestamp):
                        msg = post_data['message'].strip()
                        if msg:
                            all_responses.append(f"From {username}: {msg}")
            except Exception as e:
                logging.error("Error fetching posts for user %s: %s", username, e)

        return "\n".join(all_responses) if all_responses else "No user responses collected."

    async def summarize_responses_with_llm(self, responses_text: str) -> str:
        """Use OpenAI to summarize the collected informal responses."""
        if not self.openai_api_key:
            logging.error("OpenAI API key not set. Cannot summarize responses.")
            return "No summary available (OpenAI API key missing)."
        try:
            logging.info("Sending user responses to OpenAI for summarization...")
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes user feedback."},
                    {"role": "user", "content": (
                        "Below are several informal inputs from different participants. "
                        "Please provide a concise summary of the main points, concerns, suggestions, and any emerging consensus. "
                        "Focus on key themes, not individual messages, and be neutral.\n\n" + responses_text
                    )}
                ],
                temperature=0.7,
                max_tokens=700
            )
            summary = response.choices[0].message.content.strip()
            logging.info("Received summary of user responses from OpenAI.")
            return summary
        except Exception as e:
            logging.error("Error calling OpenAI API for responses summary: %s", e)
            return "Error occurred while summarizing responses."

    async def aggregate_feedback(self):
        """Aggregate all feedback by summarizing the informal user responses via LLM."""
        logging.info("Aggregating informal user responses...")
        responses_text = self.fetch_user_responses()
        summary = await self.summarize_responses_with_llm(responses_text)

        # Send the summary to each user's DM thread
        for username, info in self.message_info.items():
            try:
                channel_id = info['channel_id']
                msg_id = info['id']
                self.driver.posts.create_post({
                    'channel_id': channel_id,
                    'message': summary,
                    'root_id': msg_id
                })
                logging.info("Sent aggregated summary to %s in channel %s", username, channel_id)
            except Exception as e:
                logging.error("Error sending summary to %s: %s", username, e)

        logging.info("Feedback aggregation complete.")

async def main():
    load_dotenv()
    users = os.getenv('MATTERMOST_USERS').split(',')
    mattermost_url = os.getenv('MATTERMOST_URL')
    mattermost_token = os.getenv('MATTERMOST_TOKEN')
    repo_name = os.getenv('GITHUB_REPO')

    # Set test_mode based on environment variable
    test_mode = os.getenv('TEST_MODE', 'false').lower() == 'true'
    logging.info("Test mode: %s", test_mode)

    bot = ConsensusBot(
        mattermost_url=mattermost_url,
        mattermost_token=mattermost_token,
        repo_name=repo_name,
        user_list=users,
        test_mode=test_mode
    )

    global scheduler
    scheduler = AsyncIOScheduler()

    # Start cycle every Monday at 9 AM UTC
    scheduler.add_job(
        bot.start_consensus_cycle,
        'cron',
        day_of_week='mon',
        hour=9
    )

    # Run it immediately on startup
    await bot.start_consensus_cycle()

    scheduler.start()

    # Keep running indefinitely
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())

