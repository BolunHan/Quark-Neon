import json
from collections.abc import Callable
from typing import Any

import redis

from quark.base import GlobalStatics, ConfigDict
from . import LOGGER
from threading import Thread

LOGGER = LOGGER.getChild('RedisServer')
CONFIG: ConfigDict = GlobalStatics.CONFIG


class Server(object):
    def __init__(self, redis_host: str = None, redis_port: int = None, redis_password: str = None):
        self.redis_host = redis_host if redis_host is not None else CONFIG.get_config('Application.Redis.HOST', default='localhost')
        self.redis_port = redis_port if redis_port is not None else CONFIG.get_config('Application.Redis.PORT', default=6379)
        self.redis_password = redis_password if redis_password is not None else CONFIG.get_config('Application.Redis.PASSWORD', default=None)

        self.redis_conn = None

    def __del__(self):
        self.stop()
        super().__del__()

    def on_data(self, topic: str, data: str | bytes | dict | Any):
        """
        Handle incoming data, determine its type, and serialize if necessary.
        Save the data to a history topic and publish it to a realtime topic.

        Args:
            topic (str): The base topic name.
            data (str | bytes | dict | Any): The data to process and publish.
        """
        h_topic = f'{topic}.history'
        r_topic = f'{topic}.realtime'

        match data:
            case str():
                # Data is already a string
                serialized_data = data
            case bytes():
                # Data is already in bytes; decode to string for compatibility
                serialized_data = data
            case dict():
                # Serialize dictionary as JSON
                serialized_data = json.dumps(data)
            case _:
                # For other types, attempt to convert to a string representation
                serialized_data = str(data)

        # Save to history topic
        self.redis_conn.rpush(h_topic, serialized_data)

        # Publish to realtime topic
        self.redis_conn.publish(r_topic, serialized_data)

        if GlobalStatics.DEBUG_MODE:
            LOGGER.debug(f"Data processed and sent: {serialized_data}")

    def start(self):
        """Initialize the Redis connection."""

        if self.redis_conn is not None:
            LOGGER.info('Already connected to Redis service at {self.redis_host}:{self.redis_port}!')
            return

        try:
            if self.redis_password:
                self.redis_conn = redis.StrictRedis(host=self.redis_host, port=self.redis_port, password=self.redis_password, decode_responses=True)
            else:
                self.redis_conn = redis.StrictRedis(host=self.redis_host, port=self.redis_port, decode_responses=True)

            LOGGER.info(f"Connected to Redis service at {self.redis_host}:{self.redis_port}.")
        except Exception as _:
            LOGGER.error(f"Failed to connect Redis service at {self.redis_host}:{self.redis_port}.")

    def stop(self):
        """Clean up and disconnect the Redis connection."""
        if self.redis_conn:
            LOGGER.info("Server stopping. Disconnecting from Redis...")
            self.redis_conn.close()
            self.redis_conn = None

    def clean(self, topic: str):
        """
        Drop all data from the Redis history and realtime channels for the given topic.

        Args:
            topic (str): The base topic name to clean up.
        """
        h_topic = f'{topic}.history'

        # Delete the history list for the topic
        self.redis_conn.delete(h_topic)

        LOGGER.info(f"Cleaned all data for topic: {topic}")

    @property
    def connected(self) -> bool:
        if self.redis_conn is None:
            return False
        return True


class Client(object):
    def __init__(self, topic: str | list[str] = None, callback: Callable | list[Callable] | dict[str, Callable] = None, redis_host: str = None, redis_port: int = None, redis_password: str = None):
        self.redis_host = redis_host if redis_host is not None else CONFIG.get_config('Application.Redis.HOST', default='localhost')
        self.redis_port = redis_port if redis_port is not None else CONFIG.get_config('Application.Redis.PORT', default=6379)
        self.redis_password = redis_password if redis_password is not None else CONFIG.get_config('Application.Redis.PASSWORD', default=None)

        self.topic: set[str] = set()
        self.callback: dict[str, Callable[[str | bytes], None]] = {}
        self.redis_conn = None
        self.redis_pubsub = None
        self.pubsub_thread = None

        if isinstance(topic, str) and callable(callback):
            self.subscribe(topic=topic, callback=callback)
        elif isinstance(topic, list) and isinstance(callback, list):
            for _topic, _callback in zip(topic, callback):
                self.subscribe(topic=_topic, callback=_callback)
        elif isinstance(topic, list) and isinstance(callback, dict):
            for _topic in topic:
                self.subscribe(topic=_topic, callback=callback[_topic])
        elif topic is None and callback is None:
            pass
        else:
            raise TypeError(f'Can not add {topic=} and {callback=} to subscription.')

    def subscribe(self, topic: str, callback: Callable[[str | bytes], None]):
        if topic in self.topic:
            raise ValueError(f'Topic {topic} already registered')

        self.topic.add(topic)
        self.callback[topic] = callback

        # Subscript to the topic
        self.redis_pubsub.subscribe(f'{topic}.realtime')
        LOGGER.info(f"Subscribed to topics: {', '.join(self.topic)}.")

        # Fetch and process historical data
        LOGGER.info(f"Fetching history for topic: {topic}.")
        self.fetch_history(topic)

    def fetch_history(self, topic: str):
        """
        Fetch historical data from Redis for the specified topic and process it with the callback.

        Args:
            topic (str): The topic to fetch historical data for.
        """
        if not self.connected:
            raise ConnectionError("Redis connection not established. Call start() before using the client.")

        h_topic = f'{topic}.history'
        data_list = self.redis_conn.lrange(h_topic, 0, -1)

        LOGGER.info(f"Fetched {len(data_list)} historical messages for topic {topic}")
        for data in data_list:
            self.on_data(topic=topic, msg=data)

    def on_data(self, topic: str, msg: str | bytes):
        """
        Process received data using the registered callback for the topic.

        Args:
            topic (str): The topic of the data.
            msg (str | bytes): The received data.
        """
        if topic in self.callback:
            self.callback[topic](msg)
        else:
            raise ValueError(f"No callback registered for topic: {topic}")

    def start(self):
        """Initialize the Redis connection and start listening to the channel."""

        if self.redis_password:
            self.redis_conn = redis.StrictRedis(host=self.redis_host, port=self.redis_port, password=self.redis_password, decode_responses=True)
        else:
            self.redis_conn = redis.StrictRedis(host=self.redis_host, port=self.redis_port, decode_responses=True)
        self.redis_pubsub = self.redis_conn.pubsub()
        LOGGER.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
        # self.redis_pubsub.psubscribe('*.realtime')
        # Start listening to the Redis channel
        self.pubsub_thread = Thread(target=self.listen_to_channel)
        self.pubsub_thread.start()

    def stop(self):
        """Clean up and disconnect the Redis connection."""
        if not self.redis_conn:
            LOGGER.info("Client already stopped!")

        LOGGER.info("Client stopping. Disconnecting from Redis...")
        self.redis_conn.close()
        self.redis_conn = None
        self.redis_pubsub = None

    def listen_to_channel(self):
        """Subscribe to the Redis channel for new structured data and print incoming messages."""
        while self.connected:
            try:
                # Process messages
                message = self.redis_pubsub.get_message(timeout=30)

                if GlobalStatics.DEBUG_MODE:
                    LOGGER.debug(f"Received message: {message}")

                if message is None:
                    continue

                if message['type'] == 'message':
                    topic = message['channel'].replace('.realtime', '')
                    self.on_data(topic, message['data'])
            except redis.exceptions.ConnectionError:
                LOGGER.info("Redis connection closed. Listener thread exiting gracefully.")
            except Exception as e:
                LOGGER.error(f"Unexpected error in listener thread: {e}")

    @property
    def connected(self) -> bool:
        if self.redis_conn is None:
            return False
        return True
