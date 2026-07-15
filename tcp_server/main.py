import orbbec_service as ob
import tcp_async_service as tcp
import asyncio
from tcp_async_service import main as start_server
from network_utils import read_message, send_message
import json
import time
import connection_state
from orbbec_service import camera_data_stream, start_camera_pipeline, get_port_bbox, get_depth_data

MIN_DEPTH_RANGE = 520 # mm

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

async def start_move_to_port_sequence(pipeline, align_filter, x_offset=4, z_offset=5, diff=15,x_point = -100):
    #init angle
    action = "M"
    # x_point = -100
    z_point = -0
    x_stop = False
    z_stop = False

    # start telling robot to start program/start rotate x_point
    move = f"{action},J,{x_point},0,{z_point},0,0,0"
    await push_status(connection_state.current_writer, move)

    # start read realtime camera data and port center
    while True:
        color_frame = camera_data_stream(pipeline, align_filter,)
        frame_size ,bb_box = get_port_bbox(color_frame)
        print(bb_box)

        # move = f"R,J,-1,0,0,0,0,0"
        # await push_status(connection_state.current_writer, move)

        if bb_box:
            frame_center_width = int(frame_size[0]/2)
            frame_center_height = int(frame_size[1]/2)

            zAxis = frame_center_height - bb_box[1]
            xAxis = frame_center_width - bb_box[0]

            # x axis
            if not x_stop:
                if xAxis > 0:
                    # top
                    x_point = x_offset
                else:
                    # bottom
                    x_point = x_offset * -1

            # check whether the port is on top or bottom
            # z axis
            if not z_stop:
                if zAxis > 0:
                    # top
                    z_point = z_offset
                else:
                    # bottom
                    z_point = z_offset * -1

            print("BB_BOX: ", bb_box)
        
            # check if bb X axis is equal to the frame center X axis
            if bb_box[0] >= (frame_center_width-diff) and bb_box[0] <= (frame_center_width+diff):
                print("STOP x_point")
                x_point = 0
                x_stop = True
                # stop = "S,x_point,0"
                # await push_status(connection_state.current_writer, stop)
                # break

            # check if bb X axis is equal to the frame center X axis
            if bb_box[1] >= (frame_center_height-diff) and bb_box[1] <= (frame_center_height+diff):
                print("STOP z_point")
                z_point = 0
                z_stop = True
                # stop = "S,x_point,0"
                # await push_status(connection_state.current_writer, stop)
                # break

        move = f"{action},J,{x_point},0,{z_point},0,0,0"
        await push_status(connection_state.current_writer, move)

        if z_stop and x_stop:
            action = "B"
            move = f"{action},J,{x_point},0,{z_point},0,0,0"
            await push_status(connection_state.current_writer, move)
            print("PROGRAM END")
            break

        await asyncio.sleep(0.1)


async def maintain_port_distance_sequence(pipeline, align_filter):
    action = "D"
    y = 10
    y_stop = False

    # start telling robot to start program/start rotate x_point
    move = f"{action},L,0,{y},0,0,0,0"
    await push_status(connection_state.current_writer, move)

    while True:
        depth_frame = camera_data_stream(pipeline, align_filter, depth=True)
        center_dist = get_depth_data(depth_frame)

        if center_dist >= MIN_DEPTH_RANGE:
            print("STOP Y")
            y = 0
            y_stop = True

        move = f"{action},L,0,{y},0,0,0,0"
        await push_status(connection_state.current_writer, move)

        if y_stop:
            action = "B"
            move = f"{action},L,0,{y},0,0,0,0"
            await push_status(connection_state.current_writer, move)
            break

        await asyncio.sleep(0.1)

async def small_adjustment():
    pass

async def main():
    # kickstart the server
    server_task = asyncio.create_task(start_server())

    print("MAIN STARTED...")
    await asyncio.sleep(15)

    # start camera pipeline
    pipeline, align_filter = start_camera_pipeline()

    await start_move_to_port_sequence(pipeline, align_filter)

    time.sleep(1)

    print("START BACKING UP")
    await maintain_port_distance_sequence(pipeline, align_filter)

    time.sleep(1)

    print("START SMALL ADJUSTMENT")
    #small adjustment move towards center
    await start_move_to_port_sequence(pipeline, align_filter,
                                      x_offset=0.1,
                                      z_offset=0.1,
                                      diff=0,
                                      x_point=0)
    



async def test_tcp():
    server_task = asyncio.create_task(start_server())

    print("wait 5 secs")
    await asyncio.sleep(5)

    stop = "R,x_point,0"
    await push_status(connection_state.current_writer, stop)

    print("closing 5 secs")
    await asyncio.sleep(5)


if __name__ == "__main__":
    # asyncio.run(test_tcp())
    asyncio.run(main())