from ..auto import tqdm as tqdm_auto
from .utils_worker import MonoWorker
from _typeshed import Incomplete

__all__ = ['DiscordIO', 'tqdm_discord', 'tdrange', 'tqdm', 'trange']

class DiscordIO(MonoWorker):
    """Non-blocking file-like IO using a Discord Bot."""
    API: str
    UA: Incomplete
    token: Incomplete
    channel_id: Incomplete
    session: Incomplete
    text: Incomplete
    def __init__(self, token, channel_id) -> None:
        """Creates a new message in the given `channel_id`."""
    _message_id: Incomplete
    @property
    def message_id(self): ...
    def write(self, s):
        """Replaces internal `message_id`'s text with `s`."""
    def delete(self):
        """Deletes internal `message_id`."""

class tqdm_discord(tqdm_auto):
    """
    Standard `tqdm.auto.tqdm` but also sends updates to a Discord Bot.
    May take a few seconds to create (`__init__`).

    - create a discord bot (not public, no requirement of OAuth2 code
      grant, only send message permissions) & invite it to a channel:
      <https://discordpy.readthedocs.io/en/latest/discord.html>
    - copy the bot `{token}` & `{channel_id}` and paste below

    >>> from tqdm.contrib.discord import tqdm, trange
    >>> for i in tqdm(iterable, token='{token}', channel_id='{channel_id}'):
    ...     ...
    """
    dio: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """
        Parameters
        ----------
        token  : str, required. Discord bot token
            [default: ${TQDM_DISCORD_TOKEN}].
        channel_id  : int, required. Discord channel ID
            [default: ${TQDM_DISCORD_CHANNEL_ID}].

        See `tqdm.auto.tqdm.__init__` for other parameters.
        """
    def display(self, **kwargs) -> None: ...
    def clear(self, *args, **kwargs) -> None: ...
    def close(self) -> None: ...

def tdrange(*args, **kwargs):
    """Shortcut for `tqdm.contrib.discord.tqdm(range(*args), **kwargs)`."""
tqdm = tqdm_discord
trange = tdrange
