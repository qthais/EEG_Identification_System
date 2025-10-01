// lib/socket.ts
import { io, Socket } from "socket.io-client";

const socketUrl = process.env.NEXT_PUBLIC_SOCKET_URL || "http://localhost:8000";

export const socket: Socket = io(socketUrl, {
  transports: ["websocket"],
  autoConnect: true,
});
