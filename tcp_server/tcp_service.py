import socket

def main():
    # 1. Create a TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 2. Bind to a specific address (host and port)
    host = '0.0.0.0' # Local machine
    port = 65432
    server_socket.bind((host, port))

    # 3. Start listening for incoming connections
    server_socket.listen()
    print(f"Server listening on {host}:{port}...")

    # 4. Accept a connection
    conn, address = server_socket.accept()
    print(f"Connected by {address}")

    # 5. Receive and send data
    with conn:
        while True:
            data = conn.recv(1024) # Buffer size of 1024 bytes
            if not data:
                break
            print(f"Received: {data.decode('utf-8')}")
            new_data = f"Hellooo from server echo: {data.decode('utf-8')}"
            conn.sendall(new_data.encode()) # Echoing data back

    # 6. Close socket
    server_socket.close()

if __name__ == "__main__":
    main()