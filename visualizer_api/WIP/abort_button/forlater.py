import asyncio
from sqlalchemy import asc, or_
from my_database_models import MessageSql  # Replace with your actual SQLAlchemy models

# Global abort flag (this flag can be set via an API endpoint to abort fetching)
abort_flag = False

# Your original get_data function
async def get_data(request):
    global abort_flag  # Ensure we are checking the global abort flag
    
    # Early checks that do not need to access the database
    if request.password != valid_password:
        return {
            "success": False, 
            "message": "Wrong password.", 
            "reload": True
        }, 0, 0, 0, 0
    
    if request.house_alias == '':
        return {
            "success": False, 
            "message": "Please enter a house alias.", 
            "reload": True
        }, 0, 0, 0, 0
    
    if (request.end_ms - request.start_ms) / 1000 / 60 / 60 / 24 > 31:
        return {
            "success": False,
            "message": "The time difference between the start and end date exceeds the authorized limit (31 days).", 
            "reload": False
        }, 0, 0, 0, 0
    
    session = Session()
    
    try:
        messages = await asyncio.to_thread(fetch_messages_from_db, session, request)
        
        if abort_flag:
            return {"success": False, "message": "Data fetch aborted.", "reload": False}, 0, 0, 0, 0

        if not messages:
            return {
                "success": False, 
                "message": f"No data found for house '{request.house_alias}' in the selected timeframe.", 
                "reload": False
            }, 0, 0, 0, 0

        # Continue processing channels...
        channels = {}
        for message in messages:
            if abort_flag:
                return {"success": False, "message": "Data fetch aborted.", "reload": False}, 0, 0, 0, 0

            # Process each message and channel...
            for channel in message.payload['ChannelReadingList']:
                if abort_flag:
                    return {"success": False, "message": "Data fetch aborted.", "reload": False}, 0, 0, 0, 0
                # Collect channel data...
                channel_name = channel.get('ChannelName', 'unknown')
                if channel_name not in channels:
                    channels[channel_name] = {
                        'values': channel['ValueList'],
                        'times': channel['ScadaReadTimeUnixMsList']
                    }
                else:
                    channels[channel_name]['values'].extend(channel['ValueList'])
                    channels[channel_name]['times'].extend(channel['ScadaReadTimeUnixMsList'])

        return "", channels, [], [], []
    
    except Exception as e:
        return {"success": False, "message": f"An error occurred: {str(e)}", "reload": False}, 0, 0, 0, 0

# Fetch messages from DB with abort check after each batch of results
async def fetch_messages_from_db(session, request):
    """Run the DB query inside a thread and allow periodic abort checks."""
    global abort_flag
    
    # The query to fetch messages based on the time range and house alias
    query = session.query(MessageSql).filter(
        MessageSql.from_alias.like(f'%{request.house_alias}%'),
        or_(
            MessageSql.message_type_name == "batched.readings",
            MessageSql.message_type_name == "report"
        ),
        MessageSql.message_persisted_ms >= request.start_ms,
        MessageSql.message_persisted_ms <= request.end_ms,
    ).order_by(asc(MessageSql.message_persisted_ms))

    batch_size = 100  # Number of rows to fetch at once
    offset = 0  # Start from the beginning
    all_messages = []

    while True:
        # Fetch a batch of messages
        batch = query.limit(batch_size).offset(offset).all()

        # If no more messages are returned, we're done
        if not batch:
            break

        all_messages.extend(batch)
        offset += batch_size  # Move to the next batch

        # Check if we need to abort
        if abort_flag:
            print("Aborting data fetch!")
            break
    
    return all_messages
