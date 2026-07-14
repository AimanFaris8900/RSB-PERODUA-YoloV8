import asyncio
from network_utils import read_message, send_message
import connection_state

print(f"TCP connection_state module id: {id(connection_state)}")

async def handle_client(reader, writer):
    addr = writer.get_extra_info('peername')
    print(f"[*] New connection from {addr}")

    connection_state.current_reader = reader
    connection_state.current_writer = writer

    try:
        # keep the connection open; someone else will read/write it
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        connection_state.current_reader = None
        connection_state.current_writer = None
        writer.close()
        await writer.wait_closed()
        print(f"[*] Connection closed: {addr}")

async def main():
    try:
        print("Server starting...")
        server = await asyncio.start_server(handle_client, '0.0.0.0', 8000)
        async with server:
            await server.serve_forever()
    finally:
        print("program exit")

if __name__ == '__main__':
    asyncio.run(main())