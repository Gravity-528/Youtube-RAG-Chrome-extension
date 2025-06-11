import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import React, { useState } from "react";

const App = () => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<
    { type: "user" | "bot"; content: string }[]
  >([]);

  const handleAsk = () => {
    if (!input.trim()) return;

    setMessages((prev) => [
      ...prev,
      { type: "user", content: input },
      { type: "bot", content: "This is a dummy response." }
    ]);
    setInput("");
  };

  return (
    <div className="w-[400px] h-[600px] bg-black text-white flex flex-col p-4 gap-4">
      <h1 className="text-2xl font-bold text-orange-600">YouTube RAG</h1>

      <div className="flex-1 overflow-y-auto flex flex-col gap-2 bg-zinc-900 p-2 rounded">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`p-2 rounded max-w-[80%] ${
              msg.type === "user" ? "bg-orange-600 self-end" : "bg-zinc-700 self-start"
            }`}
          >
            {msg.content}
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
      </div>
    </div>
  );
};

export default App;

