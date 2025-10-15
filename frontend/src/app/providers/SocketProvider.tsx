"use client";

import React, { createContext, useEffect, useState } from 'react'
import { socket } from '../lib/src';
import { Socket } from 'socket.io-client';
import toast from 'react-hot-toast';
interface RetrainEvent {
  filename: string;
  status: "complete" | "failed";
  test_accuracy?: number;
  error?: string;
}

interface SocketContextType {
  socket: Socket | null;
  isConnected: boolean;
  transport: string;
  latestRetrainEvent?: RetrainEvent | null;
}


export const SocketContext = createContext<SocketContextType>({
  socket: null,
  isConnected: false,
  transport: "N/A",
});

const SocketProvider = ({ children }: { children: React.ReactNode }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [transport, setTransport] = useState("N/A");
  const [latestRetrainEvent, setLatestRetrainEvent] = useState<RetrainEvent | null>(null);
  useEffect(() => {
    function onConnect() {
      console.log("✅ Connected to Socket.IO server");
      setIsConnected(true);
      setTransport(socket.io.engine.transport.name);

      // handle upgrade from polling → websocket
      socket.io.engine.on("upgrade", (newTransport) => {
        setTransport(newTransport.name);
      });
    }

    function onDisconnect() {
      console.log("❌ Disconnected");
      setIsConnected(false);
      setTransport("N/A");
    }

    function onRetrainComplete(data: { filename: string; test_accuracy: number }) {
      console.log('onRetrainComplete', data)
      toast.success(`🎉 Retrain done: ${data.filename}, acc: ${data.test_accuracy}`);
      setLatestRetrainEvent({ filename: data.filename, status: "complete", test_accuracy: data.test_accuracy });
    }

    function onRetrainFailed(data: { filename: string; error: string }) {
      console.log('onRetrainFailed', data)
      toast.error(`❌ Retrain failed: ${data.error}`);
      setLatestRetrainEvent({ filename: data.filename, status: "failed", error: data.error });
    }

    // bind events
    socket.on("connect", onConnect);
    socket.on("disconnect", onDisconnect);
    socket.on("retrain_complete", onRetrainComplete);
    socket.on("retrain_failed", onRetrainFailed);

    // cleanup
    return () => {
      socket.off("connect", onConnect);
      socket.off("disconnect", onDisconnect);
      socket.off("retrain_complete", onRetrainComplete);
      socket.off("retrain_failed", onRetrainFailed);
    };
  }, []);

  return (
    <SocketContext.Provider value={{ socket, isConnected, transport, latestRetrainEvent }}>
      {children}
    </SocketContext.Provider>
  );
}

export default SocketProvider