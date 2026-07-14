import asyncio



async def read_message(reader: asyncio.StreamReader) -> str | None:
    """
    Reads data from the stream. Returns the decoded string,
    or None if the connection was closed by the peer.
    """
    line = await reader.readline()  # reads until b'\n'
    if not line:
        return None
    return line.decode('utf-8').rstrip('\n')


async def send_message(writer: asyncio.StreamWriter, message: str) -> None:
    """
    Encodes and sends a message, then flushes the buffer.
    """
    print("DATA: ", message)
    writer.write((message + '\n').encode('utf-8'))
    await writer.drain()