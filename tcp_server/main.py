import orbbec_service as ob
import tcp_async_service as tcp
import asyncio
from tcp_async_service import main as start_server
from network_utils import read_message, send_message
import json
import connection_state
from orbbec_service import camera_data_stream, start_camera_pipeline


print(f"MAIN connection_state module id: {id(connection_state)}")
async def handle_incoming(reader):
    """
    Independent read loop — reacts to whatever comes from the client.
    """
    while True:
        msg = await read_message(reader)
        if msg is None:
            print("Client disconnected (read side)")
            break
        print(f"Received: {msg}")
        # do whatever you want with incoming data here
        # e.g. update robot target pose, trigger a pick action, etc.


async def push_status(writer, data: str):
    """
    Call this on-demand, whenever YOU decide something needs to be sent.
    Not tied to reading at all.
    """
    # message = json.dumps(status)
    await send_message(writer, data)

async def main():
    # kickstart the server
    server_task = asyncio.create_task(start_server())

    print("MAIN STARTED...")
    await asyncio.sleep(15)

    # start camera pipeline
    pipeline = start_camera_pipeline()

    #init angle
    j1 = -1
    j2 = 0
    # start telling robot to start program/start rotate J1
    move = f"R,J,{j1},{j2},0,0,0,0"
    await push_status(connection_state.current_writer, move)

    # start read realtime camera data and port center
    while True:
        
        frame_size ,bb_box = camera_data_stream(pipeline)
        print(bb_box)

        # move = f"R,J,-1,0,0,0,0,0"
        # await push_status(connection_state.current_writer, move)

        if bb_box:
            yAxis = frame_center_height - bb_box[1]

            # check whether the port is on top or bottom
            if yAxis > 0:
                # top
                j2 = -1
            else:
                # bottom
                j2 = 1

            print("BB_BOX: ", bb_box)
            frame_center_width = int(frame_size[0]/2)
            frame_center_height = int(frame_size[1]/2)

            # check if bb X axis is equal to the frame center X axis
            if bb_box[0] >= (frame_center_width-15) and bb_box[0] <= (frame_center_width+15):
                print("STOP J1")
                j1 = 0
                # stop = "S,J1,0"
                # await push_status(connection_state.current_writer, stop)
                break

            # check if bb X axis is equal to the frame center X axis
            if bb_box[0] >= (frame_center_height-15) and bb_box[0] <= (frame_center_height+15):
                print("STOP J2")
                j2 = 0
                # stop = "S,J1,0"
                # await push_status(connection_state.current_writer, stop)
                break

        move = f"R,J,{j1},{j2},0,0,0,0"
        await push_status(connection_state.current_writer, move)

        await asyncio.sleep(0.1)

async def test_tcp():
    server_task = asyncio.create_task(start_server())

    print("wait 5 secs")
    await asyncio.sleep(5)

    stop = "R,J1,0"
    await push_status(connection_state.current_writer, stop)

    print("closing 5 secs")
    await asyncio.sleep(5)


if __name__ == "__main__":
    # asyncio.run(test_tcp())
    asyncio.run(main())