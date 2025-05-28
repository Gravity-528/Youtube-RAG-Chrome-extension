chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === "complete" && tab.url && tab.url.includes("youtube.com/watch")) {
        console.log("Back scr", tab.url);

        const urlParams = new URLSearchParams(new URL(tab.url).search);
        const videoId = urlParams.get("v");

        if (videoId) {
            console.log("Extracted videoid", videoId);
        }
        
        chrome.tabs.sendMessage(tabId, { 
            action: "videoId", 
            videoId: videoId 
        }, (response) => {
            if (chrome.runtime.lastError) {
                console.error("Error sending message:", chrome.runtime.lastError);
            } else {
                console.log("Response from content script:", response);
            }
        });

    }
});
