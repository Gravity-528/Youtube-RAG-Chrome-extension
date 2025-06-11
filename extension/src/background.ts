chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === "complete" && tab.url && tab.url.includes("youtube.com/watch")) {
        console.log("Back scr", tab.url);

        const urlParams = new URLSearchParams(new URL(tab.url).search);
        const videoId = urlParams.get("v");

        if (videoId) {
            console.log("videoid", videoId);
        }
        
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

    } else if (changeInfo.status === "complete" && tab.url) {
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
    }
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "openPopupWithText") {
        console.log("Received openPopupWithText request:", request.text);
        sendResponse({ status: "success", text: request.text });
    }else if(request.action==="StoreQuery"){
        console.log("Received StoreQuery request:", request.query);
        chrome.storage.local.set({ selectedText: request.query }, () => {
            if (chrome.runtime.lastError) {
                console.error("Storage error:", chrome.runtime.lastError);
            } else {
                console.log("Query saved successfully!");
            }
        });
        sendResponse({ status: "success" });
    }
});
