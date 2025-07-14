import os
from atproto import Client, models
from dotenv import load_dotenv
import datetime
import time

# Types of notifications:
#   Got new notification! Type: mention; from: did:plc:zjolxv4umxejkvq4pyuayp4q
#   Got new notification! Type: reply; from: did:plc:zjolxv4umxejkvq4pyuayp4q
#   Got new notification! Type: follow; from: did:plc:zjolxv4umxejkvq4pyuayp4q
#   Got new notification! Type: like; from: did:plc:sxhvvd27nuho3aqdfxybwt3d


# Load environment variables from .env file
load_dotenv()

# Retrieve credentials
BSKY_USERNAME = os.getenv('BSKY_USERNAME')
BSKY_APP_PASSWORD = os.getenv('BSKY_APP_PASSWORD')

# Initialize the Bluesky client
client = Client()

# Store the last seen notification cursor to avoid reprocessing old notifications
last_notification_cursor = None

def authenticate():
    """Authenticates the bot with Bluesky using the App Password."""
    if not BSKY_USERNAME or not BSKY_APP_PASSWORD:
        print("Error: Bluesky credentials not found in environment variables.")
        return False
    
    print(f"Attempting to log in as {BSKY_USERNAME} with password: `{BSKY_APP_PASSWORD}`...")
    try:
        client.login(BSKY_USERNAME, BSKY_APP_PASSWORD)
        print("Authentication successful.")
        return True
    except Exception as e:
        print(f"Authentication failed: {e}")
        return False

def post_skeet(text: str, reply_to=None):
    """Posts a new text update (skeet) to Bluesky, with optional reply functionality."""
    if client.me is None:
        print("Cannot post: Bot is not authenticated.")
        return

    print(f"Attempting to post: '{text}'")
    try:
        if reply_to:
            # Create a reply reference
            parent_uri = reply_to['uri']
            parent_cid = reply_to['cid']
            # Get the post's author and the root post's author (if it's a reply chain)
            # This is simplified. For full threading, you'd need to fetch the full thread
            # and set the root accordingly.
            reply_ref = models.AppBskyFeedPost.ReplyRef(
                parent=models.ComAtprotoUri.Record(uri=parent_uri, cid=parent_cid),
                root=reply_to.get('root', models.ComAtprotoUri.Record(uri=parent_uri, cid=parent_cid))
            )
            post_ref = client.send_post(text, reply=reply_ref)
        else:
            post_ref = client.send_post(text)

        client.like(post_ref) # Liking your own post is optional

        print("Post successful.")
        print(f"Post URI: {post_ref.uri}")
    except Exception as e:
        print(f"Failed to post: {e}")

def check_notifications():
    print("Checking for new notifications...")
    try:
        response = client.app.bsky.notification.list_notifications()

        for notification in response.notifications:
            if notification.is_read:
                continue

            print(f"Unknown notification.  Type: {notification.reason}; From: {notification.author.did}")
            if notification.reason == "mention":
                process_mention(notification)
            elif notification.reason == "reply":
                process_reply(notification)
            elif notification.reason == "like":
                process_like(notification)
            elif notification.reason == "follow":
                process_follow(notification)
            else:
                print(f"Unknown notification.  Type: {notification.reason}; From: {notification.author.did}")
    except Exception as e:
        print(f"Failed to check for mentions: {e}")


def check_mentions():
    """Checks for new mentions and processes them."""
    global last_notification_cursor
    print("Checking for new notifications...")
    try:
        # Fetch notifications. Use the cursor for pagination.
        # Limit to a reasonable number to avoid overwhelming the API or processing too many at once.
        notifications_response = client.app.bsky.notification.list_notifications(
            limit=50,
            cursor=last_notification_cursor
        )

        new_notifications = notifications_response.notifications

        if not new_notifications:
            print("No new notifications.")
            return

        # Sort notifications by creation time (newest first)
        new_notifications.sort(key=lambda n: n.indexedAt, reverse=True)

        for notif in new_notifications:
            # Check if it's a mention and it's a new notification (not already processed)
            # 'mention' is the type for a direct @-mention
            # 'reply' can also contain mentions, so you might want to process those too.
            if notif.reason == 'mention': # or notif.reason == 'reply':
                # The subject of the notification is the post that mentioned you
                mentioning_post_uri = notif.uri
                mentioning_post_cid = notif.cid
                mentioning_author_did = notif.author.did
                mentioning_author_handle = notif.author.handle

                # Extract the text of the mentioning post (if available in the record)
                # You might need to fetch the full record to get the text,
                # as `list_notifications` might not include the full content.
                # For simplicity, we'll assume the text is directly accessible for now.
                # In a real-world scenario, you'd do:
                # post_record = client.get_post_thread(mentioning_post_uri).thread.post.record
                # post_text = post_record.text

                # Example: Log the mention
                print(f"\n--- New Mention! ---")
                print(f"From: @{mentioning_author_handle} ({mentioning_author_did})")
                print(f"URI: {mentioning_post_uri}")
                # print(f"Text: {post_text}") # Uncomment after fetching post text
                print(f"Indexed At: {notif.indexedAt}")

                # Example: Reply to the mention
                reply_text = f"Thanks for the mention, @{mentioning_author_handle}! How can I help you?"
                # You'll need the URI and CID of the post being replied to
                reply_to_info = {
                    'uri': mentioning_post_uri,
                    'cid': mentioning_post_cid,
                    'root': notif.record # The record from the notification can serve as the root if it's the first in a new thread
                }
                post_skeet(reply_text, reply_to=reply_to_info)

        # Update the cursor to the latest notification's cursor to ensure we don't re-fetch
        # already processed notifications. The `cursor` in the response represents the
        # starting point for the *next* request.
        if notifications_response.cursor:
            last_notification_cursor = notifications_response.cursor
        elif new_notifications:
            # If no cursor is provided in the response but there are new notifications,
            # use the indexedAt of the oldest new notification as a crude fallback
            # for the next fetch's 'before' parameter if a proper cursor isn't found.
            # This is less reliable than the actual cursor.
            pass # Better to rely on `notifications_response.cursor`

    except Exception as e:
        print(f"Failed to check for mentions: {e}")

def main():
    if authenticate():
        # Example usage: Post a simple message
        #message = "Hello Bluesky! My Name is Al!  This is my first post on BlueSky!  #firstpost"
        #post_skeet(message)

        print("\nBot is now running and checking for mentions every 60 seconds...")
        while True:
            check_mentions()
            time.sleep(60) # Check every 60 seconds (adjust as needed)

if __name__ == "__main__":
    main()
