import cv2
import asyncio
from tcp_async_service import main as start_server
from network_utils import read_message, send_message
import json
import time
import connection_state
from orbbec_service import camera_data_stream, start_camera_pipeline, get_port_bbox, get_depth_data, get_pixel_depth, calculate_center, calculate_gradient

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
            return msg
        print(f"Received: {msg}")

        return msg
            # do whatever you want with incoming data here
            # e.g. update robot target pose, trigger a pick action, etc.

async def push_status(writer, data: str):
    """
    Call this on-demand, whenever YOU decide something needs to be sent.
    Not tied to reading at all.
    """
    # message = json.dumps(status)
    await send_message(writer, data)

async def start_move_to_port_sequence(pipeline, align_filter, x_offset=4, z_offset=4, diff=4,x_point = 170):
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
        color_frame, depth_frame = camera_data_stream(pipeline, align_filter, depth=True)
        frame_size ,bb_box = get_port_bbox(color_frame, depth_frame)
        print(bb_box)

        # move = f"R,J,-1,0,0,0,0,0"
        # await push_status(connection_state.current_writer, move)

        if bb_box:
            # camera center
            frame_center_width = int(frame_size[0]/2)
            frame_center_height = int(frame_size[1]/2)

            # bbox x and z axis
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
            for i in range(3):
                action = "B"
                move = f"{action},J,{x_point},0,{z_point},0,0,0"
                await push_status(connection_state.current_writer, move)
                await asyncio.sleep(0.01)
            print("PROGRAM END")
            break

        await asyncio.sleep(0.1)

async def maintain_port_distance_sequence(pipeline, align_filter):
    action = "D"
    y = 5
    y_stop = False

    # start telling robot to start program/start rotate x_point
    move = f"{action},L,0,{y},0,0,0,0"
    await push_status(connection_state.current_writer, move)

    while True:
        color_frame, depth_frame = camera_data_stream(pipeline, align_filter, depth=True)
        depth_mm, width, height = get_depth_data(depth_frame)

        #calculate center point
        cy, cx = height // 2, width // 2
        center_dist = get_pixel_depth(depth_mm, cy, cx)

        if center_dist != 0.0:
            if center_dist >= MIN_DEPTH_RANGE:
                y = 0
                diff_dist = center_dist - MIN_DEPTH_RANGE

                y = diff_dist * -1
                y_stop = True
                print("STOP Y")

        move = f"{action},L,0,{y},0,0,0,0"
        await push_status(connection_state.current_writer, move)

        if y_stop:
            action = "B"
            move = f"{action},L,0,{y},0,0,0,0"
            await push_status(connection_state.current_writer, move)
            time.sleep(3)
            print("END Y")
            break

        await asyncio.sleep(0.1)

async def set_tool_coord_to_cover(pipeline, align_filter):
    action = "T"
    y_point = 0
    feedback = ""

    color_frame, depth_frame = camera_data_stream(pipeline, align_filter, depth=True)
    depth_mm, width, height = get_depth_data(depth_frame)

    #calculate center point
    cy, cx = height // 2, width // 2
    center_dist = get_pixel_depth(depth_mm, cy, cx)
    print("PORT DISTANCE: ", center_dist)


    for i in range(3):
        move = f"{action},L,0,0,{center_dist},0,0,0"
        await push_status(connection_state.current_writer, move)

        await asyncio.sleep(0.08)

    action = "B"
    move = f"{action},L,0,0,{center_dist},0,0,0"
    await push_status(connection_state.current_writer, move)

    # print("SEND SET TCP OFFSET TO COVER")
    # feedback = await handle_incoming(connection_state.current_reader)
    # print("RECEIVED FEEDBACK")
    
async def pivot_perpendicular(pipeline, align_filter, offset_ry = 1):
    print("PIVOT STARTED")
    action = "P"
    ry = 0.1
    ry_stop = False
    move = f"{action},J,0,0,0,0,{ry},0"
    await push_status(connection_state.current_writer, move)

    while True:
        color_frame, depth_frame = camera_data_stream(pipeline, align_filter, depth=True)
        depth_mm, width, height = get_depth_data(depth_frame)

        frame_size ,center_port, bb_result = get_port_bbox(color_frame, depth_frame, bb_result=True)
        frame_width = frame_size[0]
        frame_height = frame_size[1]
        center_width = int(frame_width/2)
        center_height = int(frame_height/2)

        if bb_result:
            bounding_box = bb_result[0]
            center_bb = calculate_center(bounding_box)

            # 2 Depth Points
            depth_left_dot = int(((center_width - int(bounding_box[0]))/2)+int(bounding_box[0]))
            depth_right_dot = int(((int(bounding_box[2]) - center_width)/2)+center_width)

            #2 Depth points distance
            depth_left_dist = get_pixel_depth(depth_mm, cy=center_bb[1], cx=depth_left_dot)
            depth_right_dist = get_pixel_depth(depth_mm, cy=center_bb[1], cx=depth_right_dot)

            gradient = calculate_gradient(lx= depth_left_dot, ly= depth_left_dist,
                                      rx= depth_right_dot, ry= depth_right_dist)
            
            print("GRADIENT VALUE: ", gradient)
            
            if gradient == 0.0:
                ry = 0
                ry_stop = True
            elif gradient > 0.0:
                ry = offset_ry * -1
            elif gradient < 0.0:
                ry = offset_ry

        move = f"{action},J,0,0,0,0,{ry},0"    
        await push_status(connection_state.current_writer, move)
        print("PIVOT: ", move)

        if ry_stop:
            action = "B"
            move = f"{action},L,0,0,0,0,{ry},0"
            await push_status(connection_state.current_writer, move)
            time.sleep(3)
            print("END PIVOT")
            break

        await asyncio.sleep(0.1)

#----------------------------------------------------------------------------

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
                                      x_offset=0.15,
                                      z_offset=0.15,
                                      diff=0,
                                      x_point=0)
    
    cv2.destroyAllWindows()
    
    time.sleep(5)

    await set_tool_coord_to_cover(pipeline, align_filter)

    time.sleep(4)

    await pivot_perpendicular(pipeline, align_filter, offset_ry=0.2)

    cv2.destroyAllWindows()



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