import axios from "axios";

chrome.runtime.onInstalled.addListener(() => {
    chrome.identity.getAuthToken({ interactive: true }, async (token) => {
        if (chrome.runtime.lastError) {
            console.error("Error getting auth token:", chrome.runtime.lastError);
        } else {
            console.log("Auth token received:", token);
        }
        try {
            const response = await axios.get('https://www.googleapis.com/oauth2/v3/userinfo', {
                headers: {
                    Authorization: `Bearer ${token}`
                }
            });
            console.log("User info response:", response.data);
            chrome.storage.local.set({ email: response.data.email }, () => {
                if (chrome.runtime.lastError) {
                    console.error("Error saving email:", chrome.runtime.lastError);
                } else {
                    console.log("Email saved successfully!");
                }
            });
        } catch (error) {
            console.error("Error during authentication:", error);
        }
    });
});



chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
    if (changeInfo.status === "complete" && tab.url && tab.url.includes("youtube.com/watch")) {
        const urlParams = new URLSearchParams(new URL(tab.url).search);
        const videoId = urlParams.get("v");

        if (videoId) {
            console.log("videoid", videoId);
        }

        chrome.storage.local.get("email", async (result) => {
            const email = result.email || "";
            if (!email) {
                console.warn("Email not available yet, skipping state save.");
                return;
            }

            const state = {
                type: "transcript",
                url: tab.url,
                email: email,
                video_id: videoId
            };

            chrome.storage.local.set({ data: state }, () => {
                if (chrome.runtime.lastError) {
                    console.error("Error saving state:", chrome.runtime.lastError);
                } else {
                    console.log("State saved successfully!");
                }
            });

            chrome.tabs.sendMessage(tabId, {
                action: "videoId",
                videoId: videoId
            }, (response) => {
                if (chrome.runtime.lastError) {
                    console.error("Error sending message:", chrome.runtime.lastError);
                } else {
                    console.log("Response", response);
                }
            });
        });

    } else if (changeInfo.status === "complete" && tab.url) {
        chrome.storage.local.get("email", async (result) => {
            const email = result.email || "";
            if (!email) {
                console.warn("Email not available yet, skipping state save.");
                return;
            }

            const state = {
                type: "doc",
                url: tab.url,
                email: email,
                video_id: ""
            };

            chrome.storage.local.set({ data: state }, () => {
                if (chrome.runtime.lastError) {
                    console.error("Error saving state:", chrome.runtime.lastError);
                } else {
                    console.log("State saved successfully!");
                }
            });

            chrome.tabs.sendMessage(tabId, {
                action: "others",
                url: tab.url
            }, (response) => {
                if (chrome.runtime.lastError) {
                    console.error("Error sending message:", chrome.runtime.lastError);
                } else {
                    console.log("Response", response);
                }
            });
        });
    }
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "openPopupWithText") {
        console.log("Received openPopupWithText request:", request.text);
        sendResponse({ status: "success", text: request.text });
    }else if (request.action === "StoreQuery") {
        console.log("Received StoreQuery request:", request.text);

        // const { data: data1 } = await chrome.storage.local.get("data");
        // let data = {
        // type: "doc",
        // url: data1?.url,
        // email: data1?.email || "",
        // };

        // // console.log("Data retrieved from storage:", data);
        
        // try{
        // const response = await axios.post("http://localhost:8000/query_rag_answer", {
        //   query: request.text,
        //   url: data?.url,
        //   email: data?.email,
        // });
        // // console.log("Response from StoreQuery:", response);
        // sendResponse({ status: "success",explanation:response.data.answer });

        // }catch(error){
        //     console.error("Error during StoreQuery:", error);
        //     sendResponse({ status: "error", message: "Error during StoreQuery." });
        //     return;
        // }   
    (async () => {
      try {
        const { data: data1 } = await chrome.storage.local.get("data");
        const data = {
          type: "doc",
          url: data1?.url,
          email: data1?.email || "",
        };

        const response = await axios.post("https://youtube-rag-chrome-extension.onrender.com/query_rag_answer", {
          query: request.text,
          url: data.url,
          email: data.email,
        });

        sendResponse({ status: "success", explanation: response.data.answer });
      } catch (error) {
        console.error("Error during StoreQuery:", error);
        sendResponse({ status: "error", message: "Error during StoreQuery." });
      }
    })();
        return true; // Keep the message channel open for sendResponse  
    } 
});

