import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import axios from "axios";
import React, { useState, useEffect } from "react";
import ClipLoader from "react-spinners/ClipLoader";

const App = () => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<
    { type: "user" | "bot"; content: string }[]
  >([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    chrome.runtime.onMessage.addListener((msg) => {
      if (msg.action === "openPopupWithText") {
        chrome.storage.local.get("selectedText", (data) => {
          setInput(data.selectedText);
        });
      }
    });
  }, []);

  const handleIngest = async () => {
    const { data } = await chrome.storage.local.get("data");
    const stored = await chrome.storage.local.get("email");

    try {
      if (!stored.email) {
  console.error("Email not yet available.");
  return;
}
      const response = await axios.post("http://localhost:8000/ingest", {
        type: data?.type,
        url: data?.url,
        email: data?.email,
        video_id: data?.video_id
      });
      setMessages((prev) => {
        const updated = [...prev, { type: "bot" as const, content: "Ingestion started. Please wait for 4-5 minutes" }];
        return updated.length > 15 ? updated.slice(-15) : updated;
      });
    } catch (error) {
      console.error("Error during ingestion:", error);
      setMessages((prev) => {
        const updated = [...prev, { type: "bot" as const, content: "Error during ingestion. Please try again." }];
        return updated.length > 15 ? updated.slice(-15) : updated;
      });
    }
  };

  const handleAsk = async () => {
    if (!input.trim()) return;

    const { data: data1 } = await chrome.storage.local.get("data");

    let data;
    if (data1?.type !== "doc") {
      data = {
        type: "transcript",
        url: data1?.video_id,
        email: data1?.email || "",
      };
    } else {
      data = {
        type: "doc",
        url: data1?.url,
        email: data1?.email || "",
      };
    }

    // Add user message
    setMessages((prev) => {
      const updated = [...prev, { type: "user" as const, content: input }];
      return updated.length > 15 ? updated.slice(-15) : updated;
    });

    // Add loading spinner message
    setLoading(true);
    setMessages((prev) => {
      const updated = [...prev, { type: "bot" as const, content: "__LOADING__" }];
      return updated.length > 15 ? updated.slice(-15) : updated;
    });

    try {
      const response = await axios.post("http://localhost:8000/query_rag_answer", {
        query: input,
        url: data?.url,
        email: data?.email,
      });

      // Replace loader with actual answer
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          type: "bot" as const,
          content: response.data.answer,
        };
        return updated.length > 15 ? updated.slice(-15) : updated;
      });
    } catch (error) {
      console.error("Error during query:", error);
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          type: "bot" as const,
          content: "Error during query. Please try again.",
        };
        return updated.length > 15 ? updated.slice(-15) : updated;
      });
    } finally {
      setLoading(false);
      setInput("");
    }
  };

  return (
    <div className="w-[400px] h-[600px] bg-black text-white flex flex-col p-4 gap-4">
      <h1 className="text-2xl font-bold text-orange-600">Documentation RAG</h1>

      <div className="flex-1 overflow-y-auto flex flex-col gap-2 bg-zinc-900 p-2 rounded">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`p-2 rounded max-w-[80%] ${
              msg.type === "user"
                ? "bg-orange-600 self-end"
                : "bg-zinc-700 self-start"
            }`}
          >
            {msg.content === "__LOADING__" ? (
              <div className="flex items-center gap-2">
                <ClipLoader size={18} color="#FFA500" />
                <span className="text-sm text-gray-300">Bot is thinking...</span>
              </div>
            ) : (
              msg.content
            )}
          </div>
        ))}
      </div>

      <div className="flex gap-2">
        <Input
          placeholder="Type your question..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="bg-zinc-800 text-white"
        />
        <Button onClick={handleAsk}>Ask</Button>
        <Button onClick={handleIngest}>Ingest Data</Button>
      </div>
    </div>
  );
};

export default App;

